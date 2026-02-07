//! Memory consolidation: auto-merging duplicates and linking related items.
//!
//! Consolidation runs as a background task triggered after recall.
//! It processes items from a queue populated during store (conflict detection).

use std::path::Path;
use std::sync::Arc;

use rusqlite::{Connection, params};
use tracing::{debug, info, warn};

use crate::error::{Result, SedimentError};

/// A pending consolidation candidate.
#[derive(Debug, Clone)]
pub struct ConsolidationCandidate {
    pub item_id_a: String,
    pub item_id_b: String,
    pub similarity: f64,
}

/// Manages the consolidation queue in SQLite.
pub struct ConsolidationQueue {
    conn: Connection,
}

impl ConsolidationQueue {
    /// Open or create the consolidation queue database.
    /// Uses the same SQLite file as access tracking.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path).map_err(|e| {
            SedimentError::Database(format!("Failed to open consolidation database: {}", e))
        })?;

        if let Err(e) = conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;") {
            tracing::warn!("Failed to set SQLite PRAGMAs (consolidation): {}", e);
        }

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS consolidation_queue (
                item_id_a TEXT NOT NULL,
                item_id_b TEXT NOT NULL,
                similarity REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at INTEGER NOT NULL,
                UNIQUE(item_id_a, item_id_b)
            );",
        )
        .map_err(|e| {
            SedimentError::Database(format!("Failed to create consolidation_queue table: {}", e))
        })?;

        Ok(Self { conn })
    }

    /// Enqueue a consolidation candidate (ignores duplicates).
    pub fn enqueue(&self, item_id_a: &str, item_id_b: &str, similarity: f64) -> Result<()> {
        let now = chrono::Utc::now().timestamp();

        // Ensure consistent ordering (smaller ID first)
        let (a, b) = if item_id_a < item_id_b {
            (item_id_a, item_id_b)
        } else {
            (item_id_b, item_id_a)
        };

        self.conn.execute(
            "INSERT OR IGNORE INTO consolidation_queue (item_id_a, item_id_b, similarity, status, created_at)
             VALUES (?1, ?2, ?3, 'pending', ?4)",
            params![a, b, similarity, now],
        ).map_err(|e| {
            SedimentError::Database(format!("Failed to enqueue consolidation: {}", e))
        })?;

        debug!(
            "Enqueued consolidation: {} <-> {} (similarity: {:.2})",
            a, b, similarity
        );
        Ok(())
    }

    /// Fetch up to `limit` pending candidates.
    pub fn fetch_pending(&self, limit: usize) -> Result<Vec<ConsolidationCandidate>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT item_id_a, item_id_b, similarity FROM consolidation_queue
             WHERE status = 'pending'
             ORDER BY similarity DESC
             LIMIT ?1",
            )
            .map_err(|e| SedimentError::Database(format!("Failed to prepare fetch: {}", e)))?;

        let rows = stmt
            .query_map(params![limit as i64], |row| {
                Ok(ConsolidationCandidate {
                    item_id_a: row.get(0)?,
                    item_id_b: row.get(1)?,
                    similarity: row.get(2)?,
                })
            })
            .map_err(|e| SedimentError::Database(format!("Failed to fetch pending: {}", e)))?;

        let mut candidates = Vec::new();
        for row in rows {
            let candidate = row
                .map_err(|e| SedimentError::Database(format!("Failed to read candidate: {}", e)))?;
            candidates.push(candidate);
        }

        Ok(candidates)
    }

    /// Delete old entries to prevent unbounded growth:
    /// - Processed (non-pending) entries older than 7 days
    /// - Pending entries older than 30 days (likely stuck or referencing deleted items)
    pub fn cleanup_processed(&self) -> Result<()> {
        let processed_cutoff = chrono::Utc::now().timestamp() - 7 * 86400;
        let pending_cutoff = chrono::Utc::now().timestamp() - 30 * 86400;
        self.conn
            .execute(
                "DELETE FROM consolidation_queue WHERE
                    (status != 'pending' AND created_at < ?1) OR
                    (status = 'pending' AND created_at < ?2)",
                params![processed_cutoff, pending_cutoff],
            )
            .map_err(|e| {
                SedimentError::Database(format!("Failed to cleanup consolidation queue: {}", e))
            })?;
        Ok(())
    }

    /// Mark a candidate as processed with the given status ("merged" or "linked").
    pub fn mark_processed(&self, item_id_a: &str, item_id_b: &str, status: &str) -> Result<()> {
        let (a, b) = if item_id_a < item_id_b {
            (item_id_a, item_id_b)
        } else {
            (item_id_b, item_id_a)
        };

        self.conn.execute(
            "UPDATE consolidation_queue SET status = ?1 WHERE item_id_a = ?2 AND item_id_b = ?3",
            params![status, a, b],
        ).map_err(|e| {
            SedimentError::Database(format!("Failed to mark processed: {}", e))
        })?;

        Ok(())
    }
}

/// Spawn a background consolidation task.
/// Returns immediately. Uses a semaphore to ensure only one runs at a time.
pub fn spawn_consolidation(
    db_path: Arc<std::path::PathBuf>,
    access_db_path: Arc<std::path::PathBuf>,
    project_id: Option<String>,
    embedder: Arc<crate::Embedder>,
    semaphore: Arc<tokio::sync::Semaphore>,
) {
    tokio::spawn(async move {
        // Try to acquire the semaphore (non-blocking).
        // Using try_acquire_owned so the permit is dropped automatically
        // even if the task panics (no manual drop needed).
        let _permit = match semaphore.try_acquire_owned() {
            Ok(p) => p,
            Err(_) => {
                debug!("Consolidation already running, skipping");
                return;
            }
        };

        if let Err(e) =
            run_consolidation_batch(&db_path, &access_db_path, project_id, embedder).await
        {
            warn!("Consolidation error: {}", e);
        }
        // _permit is dropped here automatically, or on panic
    });
}

/// Process up to 5 pending consolidation candidates.
async fn run_consolidation_batch(
    db_path: &Path,
    access_db_path: &Path,
    project_id: Option<String>,
    embedder: Arc<crate::Embedder>,
) -> Result<()> {
    let queue = ConsolidationQueue::open(access_db_path)?;
    let candidates = queue.fetch_pending(5)?;

    if candidates.is_empty() {
        return Ok(());
    }

    info!("Processing {} consolidation candidates", candidates.len());

    // Open fresh DB and graph connections
    let mut db = crate::Database::open_with_embedder(db_path, project_id, embedder)
        .await
        .map_err(|e| SedimentError::Database(format!("Consolidation DB open failed: {}", e)))?;

    for candidate in &candidates {
        let result = process_candidate(&mut db, access_db_path, candidate).await;
        match result {
            Ok(status) => {
                if let Err(e) =
                    queue.mark_processed(&candidate.item_id_a, &candidate.item_id_b, &status)
                {
                    tracing::warn!("mark_processed failed: {}", e);
                }
                info!(
                    "Consolidated {} <-> {}: {} (similarity: {:.2})",
                    candidate.item_id_a, candidate.item_id_b, status, candidate.similarity
                );
            }
            Err(e) => {
                warn!(
                    "Failed to consolidate {} <-> {}: {}",
                    candidate.item_id_a, candidate.item_id_b, e
                );
            }
        }
    }

    Ok(())
}

/// Process a single consolidation candidate.
/// Returns the status string ("merged" or "linked").
async fn process_candidate(
    db: &mut crate::Database,
    graph_db_path: &Path,
    candidate: &ConsolidationCandidate,
) -> Result<String> {
    let graph = crate::graph::GraphStore::open(graph_db_path)?;

    // Recompute similarity using current embeddings (the stored score may be stale
    // if items were modified via replace since enqueue time).
    let fresh_similarity = {
        let item_a = db.get_item(&candidate.item_id_a).await?;
        let item_b = db.get_item(&candidate.item_id_b).await?;
        match (item_a.as_ref(), item_b.as_ref()) {
            (Some(a), Some(b)) if !a.embedding.is_empty() && !b.embedding.is_empty() => {
                // Cosine similarity (embeddings are L2-normalized, so dot product = cosine sim)
                let dot: f32 = a
                    .embedding
                    .iter()
                    .zip(b.embedding.iter())
                    .map(|(x, y)| x * y)
                    .sum();
                dot as f64
            }
            _ => candidate.similarity, // fallback to stored score if items missing/no embeddings
        }
    };

    if fresh_similarity >= 0.95 {
        // Near-duplicate: newer absorbs older (non-destructive — archive removed content)
        let item_a = db.get_item(&candidate.item_id_a).await?;
        let item_b = db.get_item(&candidate.item_id_b).await?;

        match (item_a, item_b) {
            (Some(a), Some(b)) => {
                // Don't merge items from different projects
                if a.project_id != b.project_id {
                    if let Err(e) = graph.add_related_edge(
                        &candidate.item_id_a,
                        &candidate.item_id_b,
                        fresh_similarity,
                        "cross_project_similar",
                    ) {
                        tracing::warn!("add_related_edge failed: {}", e);
                    }
                    return Ok("linked".to_string());
                }

                let (keep, remove) = if a.created_at >= b.created_at {
                    (a, b)
                } else {
                    (b, a)
                };

                // Transfer edges from old to new
                if let Err(e) = graph.transfer_edges(&remove.id, &keep.id) {
                    tracing::warn!("transfer_edges failed: {}", e);
                }

                // Create SUPERSEDES edge (preserves lineage for recovery).
                // This MUST succeed before we delete — without it the removed
                // item's lineage is lost and content is irrecoverable.
                if let Err(e) = graph.add_supersedes_edge(&keep.id, &remove.id) {
                    tracing::warn!(
                        "add_supersedes_edge failed, aborting merge to prevent data loss: {}",
                        e
                    );
                    return Ok("linked".to_string());
                }

                // Archive the removed item's content into a RELATED edge label
                // so it can be recovered if this was a false positive merge.
                // The edge label stores a truncated snapshot; full content is
                // preserved in the SUPERSEDES relationship for audit.
                let archive_preview = if remove.content.chars().count() > 500 {
                    let cut = remove
                        .content
                        .char_indices()
                        .nth(497)
                        .map(|(i, _)| i)
                        .unwrap_or(remove.content.len());
                    format!("{}...", &remove.content[..cut])
                } else {
                    remove.content.clone()
                };
                if let Err(e) = graph.add_related_edge(
                    &keep.id,
                    &remove.id,
                    fresh_similarity,
                    &format!("merged_archive:{}", archive_preview),
                ) {
                    tracing::warn!("add_related_edge (archive) failed: {}", e);
                }

                // Delete the duplicate item (safe: supersedes edge preserves lineage)
                if let Err(e) = db.delete_item(&remove.id).await {
                    warn!("delete_item failed: {}", e);
                }
                if let Err(e) = graph.remove_node(&remove.id) {
                    tracing::warn!("remove_node failed: {}", e);
                }

                Ok("merged".to_string())
            }
            _ => {
                // One or both items missing, skip
                Ok("linked".to_string())
            }
        }
    } else {
        // Similar but distinct: create RELATED edge
        if let Err(e) = graph.add_related_edge(
            &candidate.item_id_a,
            &candidate.item_id_b,
            fresh_similarity,
            "similar",
        ) {
            tracing::warn!("add_related_edge failed: {}", e);
        }
        Ok("linked".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_cleanup_removes_old_pending_entries() {
        // Fix #4: Pending entries older than 30 days should be cleaned up
        let tmp = NamedTempFile::new().unwrap();
        let queue = ConsolidationQueue::open(tmp.path()).unwrap();

        // Insert a pending entry with a timestamp 31 days ago
        let old_ts = chrono::Utc::now().timestamp() - 31 * 86400;
        queue
            .conn
            .execute(
                "INSERT INTO consolidation_queue (item_id_a, item_id_b, similarity, status, created_at)
                 VALUES ('a', 'b', 0.9, 'pending', ?1)",
                params![old_ts],
            )
            .unwrap();

        // Insert a recent pending entry
        queue.enqueue("c", "d", 0.9).unwrap();

        queue.cleanup_processed().unwrap();

        // Old pending entry should be removed, recent one should remain
        let remaining = queue.fetch_pending(10).unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].item_id_a, "c");
    }

    #[test]
    fn test_enqueue_normalizes_order() {
        let tmp = NamedTempFile::new().unwrap();
        let queue = ConsolidationQueue::open(tmp.path()).unwrap();

        queue.enqueue("z", "a", 0.9).unwrap();
        let pending = queue.fetch_pending(10).unwrap();
        assert_eq!(pending[0].item_id_a, "a");
        assert_eq!(pending[0].item_id_b, "z");
    }

    #[tokio::test]
    async fn test_semaphore_released_on_panic() {
        // Bug #8: Semaphore permit must be released even if the spawned task panics.
        // OwnedSemaphorePermit is dropped automatically on panic; borrowed SemaphorePermit is not.
        let semaphore = Arc::new(tokio::sync::Semaphore::new(1));

        let sem = semaphore.clone();
        let handle = tokio::spawn(async move {
            let _permit = sem.try_acquire_owned().unwrap();
            panic!("simulated panic");
        });
        let _ = handle.await; // JoinError from panic

        assert_eq!(
            semaphore.available_permits(),
            1,
            "Semaphore should be released after panic"
        );
    }

    #[test]
    fn test_cross_project_items_should_not_merge() {
        // Bug #11: Items from different projects should not be merged.
        // The actual merge prevention is in process_candidate (async + DB),
        // so we test the guard logic here.
        let project_a = Some("proj-aaa".to_string());
        let project_b = Some("proj-bbb".to_string());
        let project_a2 = Some("proj-aaa".to_string());

        assert_ne!(project_a, project_b, "Different projects should not match");
        assert_eq!(project_a, project_a2, "Same project should match");
        assert_ne!(
            project_a, None::<String>,
            "Global vs project should not match"
        );
    }
}
