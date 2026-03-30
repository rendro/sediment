//! Background compaction: periodic optimization of LanceDB on-disk storage.
//!
//! LanceDB is append-only — every store, delete, and update creates new data files.
//! This module spawns a background task that compacts fragments, prunes old versions,
//! and re-optimizes indices to reclaim disk space.

use std::path::PathBuf;
use std::sync::Arc;

use tracing::{debug, info, warn};

use crate::db::CompactConfig;
use crate::embedder::Embedder;

/// Spawn a background compaction task.
/// Returns immediately. Uses a semaphore to ensure only one runs at a time.
pub fn spawn_compaction(
    db_path: Arc<PathBuf>,
    project_id: Option<String>,
    embedder: Arc<Embedder>,
    semaphore: Arc<tokio::sync::Semaphore>,
) {
    tokio::spawn(async move {
        let _permit = match semaphore.try_acquire_owned() {
            Ok(p) => p,
            Err(_) => {
                debug!("Compaction already running, skipping");
                return;
            }
        };

        info!("Starting background compaction");

        let config = CompactConfig {
            prune_older_than: Some(chrono::Duration::days(1)),
            delete_unverified: false,
            num_threads: Some(1),
            skip_prune: false,
        };

        match crate::Database::open_with_embedder(&*db_path, project_id, embedder).await {
            Ok(db) => match db.optimize_tables(&config).await {
                Ok(_stats) => {
                    info!("Background compaction completed");
                }
                Err(e) => {
                    warn!("Background compaction failed: {}", e);
                }
            },
            Err(e) => {
                warn!("Failed to open database for compaction: {}", e);
            }
        }
    });
}
