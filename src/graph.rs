//! Graph store using SQLite for relationship tracking between memories.
//!
//! Provides a graph layer alongside LanceDB for tracking
//! relationships like RELATED, SUPERSEDES, and CO_ACCESSED between items.

use std::path::Path;

use rusqlite::{Connection, params};
use tracing::debug;

use crate::error::{Result, SedimentError};

/// A relationship between two memory items in the graph.
#[derive(Debug, Clone)]
pub struct Edge {
    pub target_id: String,
    pub rel_type: String,
    pub strength: f64,
}

/// A co-access relationship between two memory items.
#[derive(Debug, Clone)]
pub struct CoAccessEdge {
    pub target_id: String,
    pub count: i64,
}

/// Full connection info for a memory item.
#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    pub target_id: String,
    pub rel_type: String,
    pub strength: f64,
    pub count: Option<i64>,
}

/// SQLite-backed graph store for memory relationships.
pub struct GraphStore {
    conn: Connection,
}

impl GraphStore {
    /// Open or create the graph store using the given SQLite database path.
    /// Shares the same file as access.db.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path).map_err(|e| {
            SedimentError::Database(format!("Failed to open graph database: {}", e))
        })?;

        if let Err(e) = conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;") {
            tracing::warn!("Failed to set SQLite PRAGMAs (graph): {}", e);
        }

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS graph_nodes (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS graph_edges (
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                strength REAL NOT NULL DEFAULT 0.0,
                rel_type TEXT NOT NULL DEFAULT '',
                count INTEGER NOT NULL DEFAULT 0,
                last_at INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                UNIQUE(from_id, to_id, edge_type)
            );

            CREATE INDEX IF NOT EXISTS idx_edges_from ON graph_edges(from_id);
            CREATE INDEX IF NOT EXISTS idx_edges_to ON graph_edges(to_id);",
        )
        .map_err(|e| SedimentError::Database(format!("Failed to create graph tables: {}", e)))?;

        Ok(Self { conn })
    }

    /// Add a Memory node to the graph.
    pub fn add_node(&self, id: &str, project_id: Option<&str>, created_at: i64) -> Result<()> {
        let pid = project_id.unwrap_or("");

        self.conn
            .execute(
                "INSERT OR IGNORE INTO graph_nodes (id, project_id, created_at) VALUES (?1, ?2, ?3)",
                params![id, pid, created_at],
            )
            .map_err(|e| SedimentError::Database(format!("Failed to add node: {}", e)))?;

        debug!("Added graph node: {}", id);
        Ok(())
    }

    /// Ensure a node exists in the graph. Creates it if missing (for backfill).
    pub fn ensure_node_exists(
        &self,
        id: &str,
        project_id: Option<&str>,
        created_at: i64,
    ) -> Result<()> {
        self.add_node(id, project_id, created_at)
    }

    /// Remove a Memory node and all its edges from the graph.
    pub fn remove_node(&self, id: &str) -> Result<()> {
        self.conn
            .execute(
                "DELETE FROM graph_edges WHERE from_id = ?1 OR to_id = ?1",
                params![id],
            )
            .map_err(|e| SedimentError::Database(format!("Failed to remove edges: {}", e)))?;

        self.conn
            .execute("DELETE FROM graph_nodes WHERE id = ?1", params![id])
            .map_err(|e| SedimentError::Database(format!("Failed to remove node: {}", e)))?;

        debug!("Removed graph node: {}", id);
        Ok(())
    }

    /// Add a RELATED edge between two Memory nodes.
    pub fn add_related_edge(
        &self,
        from_id: &str,
        to_id: &str,
        strength: f64,
        rel_type: &str,
    ) -> Result<()> {
        let now = chrono::Utc::now().timestamp();

        self.conn
            .execute(
                "INSERT OR IGNORE INTO graph_edges (from_id, to_id, edge_type, strength, rel_type, created_at)
                 VALUES (?1, ?2, 'related', ?3, ?4, ?5)",
                params![from_id, to_id, strength, rel_type, now],
            )
            .map_err(|e| SedimentError::Database(format!("Failed to add related edge: {}", e)))?;

        debug!(
            "Added RELATED edge: {} -> {} ({})",
            from_id, to_id, rel_type
        );
        Ok(())
    }

    /// Add a SUPERSEDES edge from new_id to old_id.
    pub fn add_supersedes_edge(&self, new_id: &str, old_id: &str) -> Result<()> {
        let now = chrono::Utc::now().timestamp();

        self.conn
            .execute(
                "INSERT OR IGNORE INTO graph_edges (from_id, to_id, edge_type, strength, created_at)
                 VALUES (?1, ?2, 'supersedes', 1.0, ?3)",
                params![new_id, old_id, now],
            )
            .map_err(|e| SedimentError::Database(format!("Failed to add supersedes edge: {}", e)))?;

        debug!("Added SUPERSEDES edge: {} -> {}", new_id, old_id);
        Ok(())
    }

    /// Get 1-hop neighbors of the given item IDs via RELATED or SUPERSEDES edges.
    /// Returns (neighbor_id, rel_type, strength) tuples.
    ///
    /// Note on parameter binding: SQLite reuses the same positional parameters (?1..?N)
    /// across all three IN clauses and the CASE expression. This is correct because
    /// SQLite binds by position, so the same parameter set is applied to each reference.
    pub fn get_neighbors(
        &self,
        ids: &[&str],
        min_strength: f64,
    ) -> Result<Vec<(String, String, f64)>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let placeholders: Vec<String> = (1..=ids.len()).map(|i| format!("?{}", i)).collect();
        let ph = placeholders.join(",");
        let strength_idx = ids.len() + 1;

        let sql = format!(
            "SELECT
                CASE WHEN from_id IN ({ph}) THEN to_id ELSE from_id END AS neighbor,
                CASE WHEN edge_type = 'related' THEN rel_type ELSE 'supersedes' END AS rtype,
                strength
             FROM graph_edges
             WHERE (from_id IN ({ph}) OR to_id IN ({ph}))
               AND edge_type IN ('related', 'supersedes')
               AND strength >= ?{strength_idx}"
        );

        let mut stmt = self.conn.prepare(&sql).map_err(|e| {
            SedimentError::Database(format!("Failed to prepare neighbors query: {}", e))
        })?;

        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        for id in ids {
            param_values.push(Box::new(id.to_string()));
        }
        param_values.push(Box::new(min_strength));

        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|b| b.as_ref()).collect();

        let rows = stmt
            .query_map(params_ref.as_slice(), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, f64>(2)?,
                ))
            })
            .map_err(|e| SedimentError::Database(format!("Failed to query neighbors: {}", e)))?;

        // Filter out input IDs from results so we never return a query ID as its own neighbor
        let input_set: std::collections::HashSet<&str> = ids.iter().copied().collect();
        let mut results = Vec::new();
        for row in rows {
            let r = row
                .map_err(|e| SedimentError::Database(format!("Failed to read neighbor: {}", e)))?;
            if !input_set.contains(r.0.as_str()) {
                results.push(r);
            }
        }

        Ok(results)
    }

    /// Record co-access between pairs of item IDs.
    /// Creates or increments CO_ACCESSED edges.
    pub fn record_co_access(&self, item_ids: &[String]) -> Result<()> {
        if item_ids.len() < 2 {
            return Ok(());
        }

        let item_ids = if item_ids.len() > 3 {
            &item_ids[..3]
        } else {
            item_ids
        };

        let now = chrono::Utc::now().timestamp();

        for i in 0..item_ids.len() {
            for j in (i + 1)..item_ids.len() {
                // Normalize edge direction: smaller ID always goes first to prevent
                // duplicate edges (A,B) and (B,A) from accumulating separately.
                let (a, b) = if item_ids[i] <= item_ids[j] {
                    (&item_ids[i], &item_ids[j])
                } else {
                    (&item_ids[j], &item_ids[i])
                };

                self.conn
                    .execute(
                        "INSERT INTO graph_edges (from_id, to_id, edge_type, count, last_at, created_at)
                         VALUES (?1, ?2, 'co_accessed', 1, ?3, ?3)
                         ON CONFLICT(from_id, to_id, edge_type)
                         DO UPDATE SET count = count + 1, last_at = ?3",
                        params![a, b, now],
                    )
                    .map_err(|e| {
                        SedimentError::Database(format!("Failed to record co-access: {}", e))
                    })?;
            }
        }

        Ok(())
    }

    /// Get items that are frequently co-accessed with the given IDs.
    /// Returns (neighbor_id, co_access_count) tuples.
    pub fn get_co_accessed(&self, ids: &[&str], min_count: i64) -> Result<Vec<(String, i64)>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let placeholders: Vec<String> = (1..=ids.len()).map(|i| format!("?{}", i)).collect();
        let ph = placeholders.join(",");
        let min_idx = ids.len() + 1;

        let sql = format!(
            "SELECT
                CASE WHEN from_id IN ({ph}) THEN to_id ELSE from_id END AS neighbor,
                count
             FROM graph_edges
             WHERE (from_id IN ({ph}) OR to_id IN ({ph}))
               AND edge_type = 'co_accessed'
               AND count >= ?{min_idx}
             ORDER BY count DESC"
        );

        let mut stmt = self.conn.prepare(&sql).map_err(|e| {
            SedimentError::Database(format!("Failed to prepare co-access query: {}", e))
        })?;

        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        for id in ids {
            param_values.push(Box::new(id.to_string()));
        }
        param_values.push(Box::new(min_count));

        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|b| b.as_ref()).collect();

        let rows = stmt
            .query_map(params_ref.as_slice(), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .map_err(|e| SedimentError::Database(format!("Failed to query co-access: {}", e)))?;

        let mut results = Vec::new();
        for row in rows {
            let r = row
                .map_err(|e| SedimentError::Database(format!("Failed to read co-access: {}", e)))?;
            results.push(r);
        }

        // Deduplicate by target_id, keeping highest count
        results.sort_by(|a, b| b.1.cmp(&a.1));
        let mut seen = std::collections::HashSet::new();
        results.retain(|(id, _)| seen.insert(id.clone()));

        Ok(results)
    }

    /// Transfer all edges from one node to another (used during consolidation merge).
    pub fn transfer_edges(&self, from_id: &str, to_id: &str) -> Result<()> {
        // Get all RELATED edges connected to from_id (excluding edges to to_id)
        let mut stmt = self
            .conn
            .prepare(
                "SELECT from_id, to_id, strength, rel_type, created_at
             FROM graph_edges
             WHERE (from_id = ?1 OR to_id = ?1)
               AND edge_type = 'related'
               AND from_id != ?2 AND to_id != ?2",
            )
            .map_err(|e| {
                SedimentError::Database(format!("Failed to prepare transfer query: {}", e))
            })?;

        let edges: Vec<(String, f64, String, i64)> = stmt
            .query_map(params![from_id, to_id], |row| {
                let fid: String = row.get(0)?;
                let tid: String = row.get(1)?;
                let neighbor = if fid == from_id { tid } else { fid };
                Ok((neighbor, row.get(2)?, row.get(3)?, row.get(4)?))
            })
            .map_err(|e| {
                SedimentError::Database(format!("Failed to query edges for transfer: {}", e))
            })?
            .filter_map(|r| match r {
                Ok(v) => Some(v),
                Err(e) => {
                    tracing::warn!("transfer_edges: failed to read row: {}", e);
                    None
                }
            })
            .collect();

        // Create edges on the new node
        for (neighbor, strength, rel_type, _) in &edges {
            if let Err(e) = self.add_related_edge(to_id, neighbor, *strength, rel_type) {
                tracing::warn!("transfer edge to {} failed: {}", neighbor, e);
            }
        }

        Ok(())
    }

    /// Detect triangles of RELATED items (for clustering).
    /// Returns sets of 3 item IDs that form triangles.
    pub fn detect_clusters(&self) -> Result<Vec<(String, String, String)>> {
        let mut stmt = self
            .conn
            .prepare(
                "WITH biedges AS (
                SELECT from_id AS a, to_id AS b FROM graph_edges WHERE edge_type = 'related'
                UNION ALL
                SELECT to_id AS a, from_id AS b FROM graph_edges WHERE edge_type = 'related'
            )
            SELECT DISTINCT e1.a, e1.b, e2.b
            FROM biedges e1
            JOIN biedges e2 ON e1.b = e2.a
            JOIN biedges e3 ON e2.b = e3.a AND e3.b = e1.a
            WHERE e1.a < e1.b AND e1.b < e2.b
            LIMIT 50",
            )
            .map_err(|e| SedimentError::Database(format!("Failed to detect clusters: {}", e)))?;

        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|e| SedimentError::Database(format!("Failed to read clusters: {}", e)))?;

        let mut clusters = Vec::new();
        for r in rows.flatten() {
            clusters.push(r);
        }

        Ok(clusters)
    }

    /// Get full connection info for an item (all edge types).
    pub fn get_full_connections(&self, item_id: &str) -> Result<Vec<ConnectionInfo>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT
                CASE WHEN from_id = ?1 THEN to_id ELSE from_id END AS neighbor,
                edge_type,
                strength,
                rel_type,
                count
             FROM graph_edges
             WHERE from_id = ?1 OR to_id = ?1",
            )
            .map_err(|e| {
                SedimentError::Database(format!("Failed to prepare connections query: {}", e))
            })?;

        let rows = stmt
            .query_map(params![item_id], |row| {
                let neighbor: String = row.get(0)?;
                let edge_type: String = row.get(1)?;
                let strength: f64 = row.get(2)?;
                let rel_type_val: String = row.get(3)?;
                let count: i64 = row.get(4)?;

                let display_type = match edge_type.as_str() {
                    "related" => rel_type_val.clone(),
                    "supersedes" => "supersedes".to_string(),
                    "co_accessed" => "co_accessed".to_string(),
                    _ => edge_type.clone(),
                };

                Ok(ConnectionInfo {
                    target_id: neighbor,
                    rel_type: display_type,
                    strength,
                    count: if edge_type == "co_accessed" {
                        Some(count)
                    } else {
                        None
                    },
                })
            })
            .map_err(|e| SedimentError::Database(format!("Failed to query connections: {}", e)))?;

        let mut connections = Vec::new();
        for row in rows {
            let r = row.map_err(|e| {
                SedimentError::Database(format!("Failed to read connection: {}", e))
            })?;
            connections.push(r);
        }

        Ok(connections)
    }

    /// Get the edge count for an item (total number of edges of all types).
    pub fn get_edge_count(&self, item_id: &str) -> Result<u32> {
        let count: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM graph_edges WHERE from_id = ?1 OR to_id = ?1",
                params![item_id],
                |row| row.get(0),
            )
            .map_err(|e| SedimentError::Database(format!("Failed to count edges: {}", e)))?;

        Ok(count as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn open_test_graph() -> GraphStore {
        let tmp = NamedTempFile::new().unwrap();
        GraphStore::open(tmp.path()).unwrap()
    }

    #[test]
    fn test_get_neighbors_excludes_input_ids() {
        // Fix #7: get_neighbors should never return an input ID as a neighbor
        let graph = open_test_graph();
        let now = chrono::Utc::now().timestamp();
        graph.add_node("A", Some("proj"), now).unwrap();
        graph.add_node("B", Some("proj"), now).unwrap();
        graph.add_node("C", Some("proj"), now).unwrap();

        // Create edges A-B, B-C
        graph.add_related_edge("A", "B", 0.9, "test").unwrap();
        graph.add_related_edge("B", "C", 0.9, "test").unwrap();

        // Query neighbors for [A, B] — should only return C, not A or B
        let neighbors = graph.get_neighbors(&["A", "B"], 0.0).unwrap();
        let neighbor_ids: Vec<&str> = neighbors.iter().map(|(id, _, _)| id.as_str()).collect();
        assert!(neighbor_ids.contains(&"C"));
        assert!(!neighbor_ids.contains(&"A"));
        assert!(!neighbor_ids.contains(&"B"));
    }

    #[test]
    fn test_co_access_normalized_direction() {
        // Fix #8: co-access edges should be normalized so (A,B) and (B,A) don't create duplicates
        let graph = open_test_graph();
        let now = chrono::Utc::now().timestamp();
        graph.add_node("Z", Some("proj"), now).unwrap();
        graph.add_node("A", Some("proj"), now).unwrap();

        // Record co-access with Z before A (Z > A lexicographically)
        graph
            .record_co_access(&["Z".to_string(), "A".to_string()])
            .unwrap();
        // Record again with reversed order
        graph
            .record_co_access(&["A".to_string(), "Z".to_string()])
            .unwrap();

        // Should only have 1 edge with count=2 (not 2 edges with count=1 each)
        let count: i64 = graph
            .conn
            .query_row(
                "SELECT COUNT(*) FROM graph_edges WHERE edge_type = 'co_accessed'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1, "Should have exactly 1 co-access edge");

        let edge_count: i64 = graph
            .conn
            .query_row(
                "SELECT count FROM graph_edges WHERE edge_type = 'co_accessed'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(edge_count, 2, "Edge count should be 2 (incremented twice)");
    }

    #[test]
    fn test_transfer_edges_preserves_relationships() {
        // Fix #6: transfer_edges should move edges from old node to new node
        let graph = open_test_graph();
        let now = chrono::Utc::now().timestamp();
        graph.add_node("old", Some("proj"), now).unwrap();
        graph.add_node("new", Some("proj"), now).unwrap();
        graph.add_node("friend", Some("proj"), now).unwrap();

        graph
            .add_related_edge("old", "friend", 0.9, "test")
            .unwrap();

        // Transfer edges from old to new
        graph.transfer_edges("old", "new").unwrap();

        // New node should now have edge to friend
        let neighbors = graph.get_neighbors(&["new"], 0.0).unwrap();
        assert!(
            !neighbors.is_empty(),
            "New node should have inherited edges"
        );
        let neighbor_ids: Vec<&str> = neighbors.iter().map(|(id, _, _)| id.as_str()).collect();
        assert!(neighbor_ids.contains(&"friend"));
    }
}
