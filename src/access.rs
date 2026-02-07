//! Access tracking for memory decay scoring.
//!
//! Uses a SQLite sidecar database to track access counts and timestamps,
//! enabling freshness and frequency-based scoring without modifying LanceDB.

use std::collections::HashMap;
use std::path::Path;

use rusqlite::{Connection, params};

use crate::error::{Result, SedimentError};

/// Combined access and validation data for decay scoring.
#[derive(Debug, Clone)]
pub struct DecayData {
    pub access_count: u32,
    pub last_accessed_at: i64,
    pub created_at: i64,
    pub validation_count: u32,
}

/// Tracks item access history in SQLite for decay scoring.
pub struct AccessTracker {
    conn: Connection,
}

impl AccessTracker {
    /// Open or create the access tracking database.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path).map_err(|e| {
            SedimentError::Database(format!("Failed to open access database: {}", e))
        })?;

        if let Err(e) = conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;") {
            tracing::warn!("Failed to set SQLite PRAGMAs (access): {}", e);
        }

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS access_log (
                item_id TEXT PRIMARY KEY,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            );",
        )
        .map_err(|e| {
            SedimentError::Database(format!("Failed to create access_log table: {}", e))
        })?;

        // Idempotent schema migration: add validation_count column
        if let Err(e) = conn.execute_batch(
            "ALTER TABLE access_log ADD COLUMN validation_count INTEGER NOT NULL DEFAULT 0;",
        ) {
            let msg = e.to_string();
            if !msg.contains("duplicate column") {
                tracing::warn!("access_log migration unexpected error: {}", msg);
            }
        }

        Ok(Self { conn })
    }

    /// Record an access for an item. If no record exists, creates one with the given created_at.
    pub fn record_access(&self, item_id: &str, created_at: i64) -> Result<()> {
        let now = chrono::Utc::now().timestamp();
        self.conn
            .execute(
                "INSERT INTO access_log (item_id, access_count, last_accessed_at, created_at)
                 VALUES (?1, 1, ?2, ?3)
                 ON CONFLICT(item_id) DO UPDATE SET
                     access_count = access_count + 1,
                     last_accessed_at = ?2",
                params![item_id, now, created_at],
            )
            .map_err(|e| SedimentError::Database(format!("Failed to record access: {}", e)))?;
        Ok(())
    }

    /// Record a validation (replace/confirm) for an item.
    pub fn record_validation(&self, item_id: &str, created_at: i64) -> Result<()> {
        let now = chrono::Utc::now().timestamp();
        self.conn
            .execute(
                "INSERT INTO access_log (item_id, access_count, last_accessed_at, created_at, validation_count)
                 VALUES (?1, 0, ?2, ?3, 1)
                 ON CONFLICT(item_id) DO UPDATE SET
                     validation_count = validation_count + 1,
                     last_accessed_at = ?2",
                params![item_id, now, created_at],
            )
            .map_err(|e| {
                SedimentError::Database(format!("Failed to record validation: {}", e))
            })?;
        Ok(())
    }

    /// Get validation counts for multiple items in a single query.
    pub fn get_validation_counts(&self, item_ids: &[&str]) -> Result<HashMap<String, u32>> {
        if item_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let placeholders: Vec<String> = item_ids
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect();
        let sql = format!(
            "SELECT item_id, COALESCE(validation_count, 0) FROM access_log WHERE item_id IN ({})",
            placeholders.join(", ")
        );

        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| SedimentError::Database(format!("Failed to prepare query: {}", e)))?;

        let params: Vec<&dyn rusqlite::types::ToSql> = item_ids
            .iter()
            .map(|id| id as &dyn rusqlite::types::ToSql)
            .collect();

        let rows = stmt
            .query_map(params.as_slice(), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, u32>(1)?))
            })
            .map_err(|e| {
                SedimentError::Database(format!("Failed to query validation counts: {}", e))
            })?;

        let mut map = HashMap::new();
        for row in rows {
            let (id, count) = row.map_err(|e| {
                SedimentError::Database(format!("Failed to read validation count: {}", e))
            })?;
            map.insert(id, count);
        }

        Ok(map)
    }

    /// Get combined access and validation data for decay scoring in a single query.
    pub fn get_decay_data(&self, item_ids: &[&str]) -> Result<HashMap<String, DecayData>> {
        if item_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let placeholders: Vec<String> = item_ids
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect();
        let sql = format!(
            "SELECT item_id, access_count, last_accessed_at, created_at, COALESCE(validation_count, 0) FROM access_log WHERE item_id IN ({})",
            placeholders.join(", ")
        );

        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| SedimentError::Database(format!("Failed to prepare query: {}", e)))?;

        let params: Vec<&dyn rusqlite::types::ToSql> = item_ids
            .iter()
            .map(|id| id as &dyn rusqlite::types::ToSql)
            .collect();

        let rows = stmt
            .query_map(params.as_slice(), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    DecayData {
                        access_count: row.get::<_, u32>(1)?,
                        last_accessed_at: row.get::<_, i64>(2)?,
                        created_at: row.get::<_, i64>(3)?,
                        validation_count: row.get::<_, u32>(4)?,
                    },
                ))
            })
            .map_err(|e| SedimentError::Database(format!("Failed to query decay data: {}", e)))?;

        let mut map = HashMap::new();
        for row in rows {
            let (id, data) = row.map_err(|e| {
                SedimentError::Database(format!("Failed to read decay data: {}", e))
            })?;
            map.insert(id, data);
        }

        Ok(map)
    }
}

#[cfg(test)]
/// Record of access history for a single item.
#[derive(Debug, Clone)]
pub struct AccessRecord {
    pub access_count: u32,
    pub last_accessed_at: i64,
    pub created_at: i64,
}

#[cfg(test)]
impl AccessTracker {
    /// Get validation count for an item.
    pub fn get_validation_count(&self, item_id: &str) -> Result<u32> {
        let count: u32 = self
            .conn
            .query_row(
                "SELECT COALESCE(validation_count, 0) FROM access_log WHERE item_id = ?1",
                params![item_id],
                |row| row.get(0),
            )
            .unwrap_or(0);
        Ok(count)
    }

    /// Get access records for a batch of item IDs.
    pub fn get_accesses(&self, item_ids: &[&str]) -> Result<HashMap<String, AccessRecord>> {
        if item_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let placeholders: Vec<String> = item_ids
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect();
        let sql = format!(
            "SELECT item_id, access_count, last_accessed_at, created_at FROM access_log WHERE item_id IN ({})",
            placeholders.join(", ")
        );

        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| SedimentError::Database(format!("Failed to prepare query: {}", e)))?;

        let params: Vec<&dyn rusqlite::types::ToSql> = item_ids
            .iter()
            .map(|id| id as &dyn rusqlite::types::ToSql)
            .collect();

        let rows = stmt
            .query_map(params.as_slice(), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    AccessRecord {
                        access_count: row.get::<_, u32>(1)?,
                        last_accessed_at: row.get::<_, i64>(2)?,
                        created_at: row.get::<_, i64>(3)?,
                    },
                ))
            })
            .map_err(|e| SedimentError::Database(format!("Failed to query accesses: {}", e)))?;

        let mut map = HashMap::new();
        for row in rows {
            let (id, record) = row.map_err(|e| {
                SedimentError::Database(format!("Failed to read access record: {}", e))
            })?;
            map.insert(id, record);
        }

        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_open_creates_table() {
        let tmp = NamedTempFile::new().unwrap();
        let tracker = AccessTracker::open(tmp.path()).unwrap();
        // Should not error on second open
        drop(tracker);
        let _tracker2 = AccessTracker::open(tmp.path()).unwrap();
    }

    #[test]
    fn test_record_and_get_access() {
        let tmp = NamedTempFile::new().unwrap();
        let tracker = AccessTracker::open(tmp.path()).unwrap();

        let created = 1700000000i64;
        tracker.record_access("item1", created).unwrap();
        tracker.record_access("item1", created).unwrap();
        tracker.record_access("item2", created).unwrap();

        let records = tracker.get_accesses(&["item1", "item2", "item3"]).unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records["item1"].access_count, 2);
        assert_eq!(records["item1"].created_at, created);
        assert_eq!(records["item2"].access_count, 1);
        assert!(!records.contains_key("item3"));
    }

    #[test]
    fn test_get_accesses_empty() {
        let tmp = NamedTempFile::new().unwrap();
        let tracker = AccessTracker::open(tmp.path()).unwrap();
        let records = tracker.get_accesses(&[]).unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn test_record_validation_on_new_item() {
        // Fix #5: validation should be recorded on the new (replacement) item
        let tmp = NamedTempFile::new().unwrap();
        let tracker = AccessTracker::open(tmp.path()).unwrap();

        let created = 1700000000i64;
        // Record validation on a new item (not pre-existing)
        tracker.record_validation("new-item", created).unwrap();
        tracker.record_validation("new-item", created).unwrap();

        let count = tracker.get_validation_count("new-item").unwrap();
        assert_eq!(
            count, 2,
            "Validation count should be 2 after two record_validation calls"
        );

        // Item that was never validated should have 0
        let count = tracker.get_validation_count("other-item").unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_get_validation_counts() {
        let tmp = NamedTempFile::new().unwrap();
        let tracker = AccessTracker::open(tmp.path()).unwrap();

        let created = 1700000000i64;
        tracker.record_validation("item-a", created).unwrap();
        tracker.record_validation("item-a", created).unwrap();
        tracker.record_validation("item-b", created).unwrap();

        let counts = tracker
            .get_validation_counts(&["item-a", "item-b", "item-c"])
            .unwrap();

        // Batch results should match individual lookups
        assert_eq!(counts.get("item-a").copied().unwrap_or(0), 2);
        assert_eq!(counts.get("item-b").copied().unwrap_or(0), 1);
        // item-c has no record, should not be in map
        assert!(!counts.contains_key("item-c"));

        // Verify consistency with single-item method
        assert_eq!(
            counts.get("item-a").copied().unwrap_or(0),
            tracker.get_validation_count("item-a").unwrap()
        );
    }
}
