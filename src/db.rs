//! Database module using LanceDB for vector storage
//!
//! Provides a simple interface for storing and searching items
//! using LanceDB's native vector search capabilities.

use std::path::PathBuf;
use std::sync::Arc;

/// Sanitize a string value for use in LanceDB SQL filter expressions.
///
/// LanceDB uses DataFusion as its SQL engine. Since `only_if()` doesn't support
/// parameterized queries, we must escape string literals. This function handles:
/// - Null bytes: stripped (could truncate strings in some parsers)
/// - Backslashes: escaped to prevent escape sequence injection
/// - Single quotes: doubled per SQL standard
/// - Semicolons: stripped to prevent statement injection
/// - Comment sequences: `--`, `/*`, and `*/` stripped to prevent comment injection
fn sanitize_sql_string(s: &str) -> String {
    s.replace('\0', "")
        .replace('\\', "\\\\")
        .replace('\'', "''")
        .replace(';', "")
        .replace("--", "")
        .replace("/*", "")
        .replace("*/", "")
}

/// Validate that a string looks like a valid item/project ID (UUID hex + hyphens).
/// Returns true if the string only contains safe characters for SQL interpolation.
/// Use this as an additional guard before `sanitize_sql_string` for ID fields.
fn is_valid_id(s: &str) -> bool {
    !s.is_empty() && s.len() <= 64 && s.chars().all(|c| c.is_ascii_hexdigit() || c == '-')
}

use arrow_array::{
    Array, BooleanArray, FixedSizeListArray, Float32Array, Int32Array, Int64Array, RecordBatch,
    RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use chrono::{TimeZone, Utc};
use futures::TryStreamExt;
use lancedb::Table;
use lancedb::connect;
use lancedb::index::scalar::FullTextSearchQuery;
use lancedb::query::{ExecutableQuery, QueryBase};
use tracing::{debug, info};

use crate::boost_similarity;
use crate::chunker::{ChunkingConfig, chunk_content};
use crate::document::ContentType;
use crate::embedder::{EMBEDDING_DIM, Embedder};
use crate::error::{Result, SedimentError};
use crate::item::{Chunk, ConflictInfo, Item, ItemFilters, SearchResult, StoreResult};

/// Threshold for auto-chunking (in characters)
const CHUNK_THRESHOLD: usize = 1000;

/// Similarity threshold for conflict detection
const CONFLICT_SIMILARITY_THRESHOLD: f32 = 0.85;

/// Maximum number of conflicts to return
const CONFLICT_SEARCH_LIMIT: usize = 5;

/// Maximum number of chunks per item to prevent CPU exhaustion during embedding.
/// With default config (800 char chunks), 1MB content produces ~1250 chunks.
/// Cap at 200 to bound embedding time while covering most legitimate content.
const MAX_CHUNKS_PER_ITEM: usize = 200;

/// Maximum number of chunks to embed in a single model forward pass.
/// Bounds peak memory usage while still batching efficiently.
const EMBEDDING_BATCH_SIZE: usize = 32;

/// Maximum additive FTS boost applied to vector similarity scores.
/// The top FTS result (by BM25 score) gets this full boost after power-law scaling.
/// Grid-searched over [0.04, 0.12] × γ [1.0, 4.0]; 0.12 gives the best composite
/// score (R@1=45.5%, R@5=78.0%, MRR=59.3%, nDCG@5=57.4%).
const FTS_BOOST_MAX: f32 = 0.12;

/// Power-law exponent for BM25 score normalization.
/// γ=1.0 is linear (uniform boost distribution), higher values concentrate
/// boost on top FTS hits. γ=2.0 balances R@1 precision with R@5/MRR breadth.
/// Higher γ (e.g. 4.0) improves R@1 by +0.5pp but costs -4pp R@5.
const FTS_GAMMA: f32 = 2.0;

/// Row count threshold below which vector search bypasses the index (brute-force).
/// Brute-force is both faster and more accurate for small tables.
const VECTOR_INDEX_THRESHOLD: usize = 5000;

/// Database wrapper for LanceDB
pub struct Database {
    db: lancedb::Connection,
    embedder: Arc<Embedder>,
    project_id: Option<String>,
    items_table: Option<Table>,
    chunks_table: Option<Table>,
    /// FTS boost parameters — overridable via env vars in bench builds only.
    fts_boost_max: f32,
    fts_gamma: f32,
}

/// Database statistics
#[derive(Debug, Default, Clone)]
pub struct DatabaseStats {
    pub item_count: usize,
    pub chunk_count: usize,
}

/// Current schema version. Increment when making breaking schema changes.
const SCHEMA_VERSION: i32 = 2;

// Arrow schema builders
fn item_schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("project_id", DataType::Utf8, true),
        Field::new("is_chunked", DataType::Boolean, false),
        Field::new("created_at", DataType::Int64, false), // Unix timestamp
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                EMBEDDING_DIM as i32,
            ),
            false,
        ),
    ])
}

fn chunk_schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("item_id", DataType::Utf8, false),
        Field::new("chunk_index", DataType::Int32, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("context", DataType::Utf8, true),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                EMBEDDING_DIM as i32,
            ),
            false,
        ),
    ])
}

impl Database {
    /// Open or create a database at the given path
    pub async fn open(path: impl Into<PathBuf>) -> Result<Self> {
        Self::open_with_project(path, None).await
    }

    /// Open or create a database at the given path with a project ID
    pub async fn open_with_project(
        path: impl Into<PathBuf>,
        project_id: Option<String>,
    ) -> Result<Self> {
        let embedder = Arc::new(Embedder::new()?);
        Self::open_with_embedder(path, project_id, embedder).await
    }

    /// Open or create a database with a pre-existing embedder.
    ///
    /// This constructor is useful for connection pooling scenarios where
    /// the expensive embedder should be loaded once and shared across
    /// multiple database connections.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the database directory
    /// * `project_id` - Optional project ID for scoped operations
    /// * `embedder` - Shared embedder instance
    pub async fn open_with_embedder(
        path: impl Into<PathBuf>,
        project_id: Option<String>,
        embedder: Arc<Embedder>,
    ) -> Result<Self> {
        let path = path.into();
        info!("Opening database at {:?}", path);

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                SedimentError::Database(format!("Failed to create database directory: {}", e))
            })?;
        }

        let db = connect(path.to_str().ok_or_else(|| {
            SedimentError::Database("Database path contains invalid UTF-8".to_string())
        })?)
        .execute()
        .await
        .map_err(|e| SedimentError::Database(format!("Failed to connect to database: {}", e)))?;

        // In bench builds, allow overriding FTS params via environment variables
        // for parameter sweeps. Production builds always use the compiled defaults.
        #[cfg(feature = "bench")]
        let (fts_boost_max, fts_gamma) = {
            let boost = std::env::var("SEDIMENT_FTS_BOOST_MAX")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(FTS_BOOST_MAX);
            let gamma = std::env::var("SEDIMENT_FTS_GAMMA")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(FTS_GAMMA);
            (boost, gamma)
        };
        #[cfg(not(feature = "bench"))]
        let (fts_boost_max, fts_gamma) = (FTS_BOOST_MAX, FTS_GAMMA);

        let mut database = Self {
            db,
            embedder,
            project_id,
            items_table: None,
            chunks_table: None,
            fts_boost_max,
            fts_gamma,
        };

        database.ensure_tables().await?;
        database.ensure_vector_index().await?;

        Ok(database)
    }

    /// Set the current project ID for scoped operations
    pub fn set_project_id(&mut self, project_id: Option<String>) {
        self.project_id = project_id;
    }

    /// Get the current project ID
    pub fn project_id(&self) -> Option<&str> {
        self.project_id.as_deref()
    }

    /// Ensure all required tables exist, migrating schema if needed
    async fn ensure_tables(&mut self) -> Result<()> {
        // Check for existing tables
        let mut table_names = self
            .db
            .table_names()
            .execute()
            .await
            .map_err(|e| SedimentError::Database(format!("Failed to list tables: {}", e)))?;

        // Recover from interrupted migration if staging table exists
        if table_names.contains(&"items_migrated".to_string()) {
            info!("Detected interrupted migration, recovering...");
            self.recover_interrupted_migration(&table_names).await?;
            // Re-fetch table names after recovery
            table_names =
                self.db.table_names().execute().await.map_err(|e| {
                    SedimentError::Database(format!("Failed to list tables: {}", e))
                })?;
        }

        // Check if migration is needed (items table exists but has old schema)
        if table_names.contains(&"items".to_string()) {
            let needs_migration = self.check_needs_migration().await?;
            if needs_migration {
                info!("Migrating database schema to version {}", SCHEMA_VERSION);
                self.migrate_schema().await?;
            }
        }

        // Items table
        if table_names.contains(&"items".to_string()) {
            self.items_table =
                Some(self.db.open_table("items").execute().await.map_err(|e| {
                    SedimentError::Database(format!("Failed to open items: {}", e))
                })?);
        }

        // Chunks table
        if table_names.contains(&"chunks".to_string()) {
            self.chunks_table =
                Some(self.db.open_table("chunks").execute().await.map_err(|e| {
                    SedimentError::Database(format!("Failed to open chunks: {}", e))
                })?);
        }

        Ok(())
    }

    /// Check if the database needs migration by checking for old schema columns
    async fn check_needs_migration(&self) -> Result<bool> {
        let table = self.db.open_table("items").execute().await.map_err(|e| {
            SedimentError::Database(format!("Failed to open items for check: {}", e))
        })?;

        let schema = table
            .schema()
            .await
            .map_err(|e| SedimentError::Database(format!("Failed to get schema: {}", e)))?;

        // Old schema has 'tags' column, new schema doesn't
        let has_tags = schema.fields().iter().any(|f| f.name() == "tags");
        Ok(has_tags)
    }

    /// Recover from an interrupted migration.
    ///
    /// The staging table `items_migrated` indicates a migration was in progress.
    /// We determine the state and recover:
    /// - Case A: `items_migrated` exists, `items` does not → migration completed
    ///   but cleanup didn't finish. Copy data to new `items`, drop staging.
    /// - Case B: both exist, `items` has old schema (has `tags`) → migration never
    ///   completed. Drop staging (old data is still intact), migration will re-run.
    /// - Case C: both exist, `items` has new schema → migration completed but
    ///   staging cleanup didn't finish. Just drop staging.
    async fn recover_interrupted_migration(&mut self, table_names: &[String]) -> Result<()> {
        let has_items = table_names.contains(&"items".to_string());

        if !has_items {
            // Case A: items was dropped, items_migrated has the data
            info!("Recovery case A: restoring items from items_migrated");
            let staging = self
                .db
                .open_table("items_migrated")
                .execute()
                .await
                .map_err(|e| {
                    SedimentError::Database(format!("Failed to open staging table: {}", e))
                })?;

            let results = staging
                .query()
                .execute()
                .await
                .map_err(|e| SedimentError::Database(format!("Recovery query failed: {}", e)))?
                .try_collect::<Vec<_>>()
                .await
                .map_err(|e| SedimentError::Database(format!("Recovery collect failed: {}", e)))?;

            let schema = Arc::new(item_schema());
            let new_table = self
                .db
                .create_empty_table("items", schema.clone())
                .execute()
                .await
                .map_err(|e| {
                    SedimentError::Database(format!("Failed to create items table: {}", e))
                })?;

            if !results.is_empty() {
                let batches = RecordBatchIterator::new(results.into_iter().map(Ok), schema);
                new_table
                    .add(Box::new(batches))
                    .execute()
                    .await
                    .map_err(|e| {
                        SedimentError::Database(format!("Failed to restore items: {}", e))
                    })?;
            }

            self.db.drop_table("items_migrated").await.map_err(|e| {
                SedimentError::Database(format!("Failed to drop staging table: {}", e))
            })?;
            info!("Recovery case A completed");
        } else {
            // Both exist — check if items has old or new schema
            let has_old_schema = self.check_needs_migration().await?;

            if has_old_schema {
                // Case B: migration never completed, old data intact. Drop staging.
                info!("Recovery case B: dropping incomplete staging table");
                self.db.drop_table("items_migrated").await.map_err(|e| {
                    SedimentError::Database(format!("Failed to drop staging table: {}", e))
                })?;
                // Migration will re-run in ensure_tables
            } else {
                // Case C: migration completed, just cleanup staging
                info!("Recovery case C: dropping leftover staging table");
                self.db.drop_table("items_migrated").await.map_err(|e| {
                    SedimentError::Database(format!("Failed to drop staging table: {}", e))
                })?;
            }
        }

        Ok(())
    }

    /// Migrate from old schema to new schema using atomic staging table pattern.
    ///
    /// Steps:
    /// 1. Read all rows from old "items" table
    /// 2. Convert to new schema
    /// 3. Verify row counts match
    /// 4. Create "items_migrated" staging table with new data
    /// 5. Verify staging row count
    /// 6. Drop old "items" (data safe in staging)
    /// 7. Create new "items" from staging data
    /// 8. Drop staging table
    ///
    /// If crash occurs at any point, `recover_interrupted_migration` handles it.
    async fn migrate_schema(&mut self) -> Result<()> {
        info!("Starting schema migration...");

        // Step 1: Read all items from old table
        let old_table = self
            .db
            .open_table("items")
            .execute()
            .await
            .map_err(|e| SedimentError::Database(format!("Failed to open old items: {}", e)))?;

        let results = old_table
            .query()
            .execute()
            .await
            .map_err(|e| SedimentError::Database(format!("Migration query failed: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| SedimentError::Database(format!("Migration collect failed: {}", e)))?;

        // Step 2: Convert to new format
        let mut new_batches = Vec::new();
        for batch in &results {
            let converted = self.convert_batch_to_new_schema(batch)?;
            new_batches.push(converted);
        }

        // Step 3: Verify row counts
        let old_count: usize = results.iter().map(|b| b.num_rows()).sum();
        let new_count: usize = new_batches.iter().map(|b| b.num_rows()).sum();
        if old_count != new_count {
            return Err(SedimentError::Database(format!(
                "Migration row count mismatch: old={}, new={}",
                old_count, new_count
            )));
        }
        info!("Migrating {} items to new schema", old_count);

        // Step 4: Drop stale staging table if exists (from previous failed attempt)
        let table_names = self
            .db
            .table_names()
            .execute()
            .await
            .map_err(|e| SedimentError::Database(format!("Failed to list tables: {}", e)))?;
        if table_names.contains(&"items_migrated".to_string()) {
            self.db.drop_table("items_migrated").await.map_err(|e| {
                SedimentError::Database(format!("Failed to drop stale staging: {}", e))
            })?;
        }

        // Step 5: Create staging table with migrated data
        let schema = Arc::new(item_schema());
        let staging_table = self
            .db
            .create_empty_table("items_migrated", schema.clone())
            .execute()
            .await
            .map_err(|e| {
                SedimentError::Database(format!("Failed to create staging table: {}", e))
            })?;

        if !new_batches.is_empty() {
            let batches = RecordBatchIterator::new(new_batches.into_iter().map(Ok), schema.clone());
            staging_table
                .add(Box::new(batches))
                .execute()
                .await
                .map_err(|e| {
                    SedimentError::Database(format!("Failed to insert into staging: {}", e))
                })?;
        }

        // Step 6: Verify staging row count
        let staging_count = staging_table
            .count_rows(None)
            .await
            .map_err(|e| SedimentError::Database(format!("Failed to count staging rows: {}", e)))?;
        if staging_count != old_count {
            // Staging is incomplete — drop it and bail
            let _ = self.db.drop_table("items_migrated").await;
            return Err(SedimentError::Database(format!(
                "Staging row count mismatch: expected {}, got {}",
                old_count, staging_count
            )));
        }

        // Step 7: Drop old items (data is safe in staging)
        self.db.drop_table("items").await.map_err(|e| {
            SedimentError::Database(format!("Failed to drop old items table: {}", e))
        })?;

        // Step 8: Create new items from staging data
        let staging_data = staging_table
            .query()
            .execute()
            .await
            .map_err(|e| SedimentError::Database(format!("Failed to read staging: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| SedimentError::Database(format!("Failed to collect staging: {}", e)))?;

        let new_table = self
            .db
            .create_empty_table("items", schema.clone())
            .execute()
            .await
            .map_err(|e| {
                SedimentError::Database(format!("Failed to create new items table: {}", e))
            })?;

        if !staging_data.is_empty() {
            let batches = RecordBatchIterator::new(staging_data.into_iter().map(Ok), schema);
            new_table
                .add(Box::new(batches))
                .execute()
                .await
                .map_err(|e| {
                    SedimentError::Database(format!("Failed to insert migrated items: {}", e))
                })?;
        }

        // Step 9: Drop staging table (cleanup)
        self.db
            .drop_table("items_migrated")
            .await
            .map_err(|e| SedimentError::Database(format!("Failed to drop staging table: {}", e)))?;

        info!("Schema migration completed successfully");
        Ok(())
    }

    /// Convert a batch from old schema to new schema
    fn convert_batch_to_new_schema(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let schema = Arc::new(item_schema());

        // Extract columns from old batch (handle missing columns gracefully)
        let id_col = batch
            .column_by_name("id")
            .ok_or_else(|| SedimentError::Database("Missing id column".to_string()))?
            .clone();

        let content_col = batch
            .column_by_name("content")
            .ok_or_else(|| SedimentError::Database("Missing content column".to_string()))?
            .clone();

        let project_id_col = batch
            .column_by_name("project_id")
            .ok_or_else(|| SedimentError::Database("Missing project_id column".to_string()))?
            .clone();

        let is_chunked_col = batch
            .column_by_name("is_chunked")
            .ok_or_else(|| SedimentError::Database("Missing is_chunked column".to_string()))?
            .clone();

        let created_at_col = batch
            .column_by_name("created_at")
            .ok_or_else(|| SedimentError::Database("Missing created_at column".to_string()))?
            .clone();

        let vector_col = batch
            .column_by_name("vector")
            .ok_or_else(|| SedimentError::Database("Missing vector column".to_string()))?
            .clone();

        RecordBatch::try_new(
            schema,
            vec![
                id_col,
                content_col,
                project_id_col,
                is_chunked_col,
                created_at_col,
                vector_col,
            ],
        )
        .map_err(|e| SedimentError::Database(format!("Failed to create migrated batch: {}", e)))
    }

    /// Ensure vector and FTS indexes exist on tables with enough rows.
    ///
    /// LanceDB requires at least 256 rows before creating a vector index.
    /// Once created, the vector index converts brute-force scans to HNSW/IVF-PQ.
    /// The FTS index enables full-text search for hybrid retrieval.
    async fn ensure_vector_index(&self) -> Result<()> {
        const MIN_ROWS_FOR_INDEX: usize = 256;

        for (name, table_opt) in [("items", &self.items_table), ("chunks", &self.chunks_table)] {
            if let Some(table) = table_opt {
                let row_count = table.count_rows(None).await.unwrap_or(0);

                // Check existing indices
                let indices = table.list_indices().await.unwrap_or_default();

                // Vector index (requires 256+ rows)
                if row_count >= MIN_ROWS_FOR_INDEX {
                    let has_vector_index = indices
                        .iter()
                        .any(|idx| idx.columns.contains(&"vector".to_string()));

                    if !has_vector_index {
                        info!(
                            "Creating vector index on {} table ({} rows)",
                            name, row_count
                        );
                        match table
                            .create_index(&["vector"], lancedb::index::Index::Auto)
                            .execute()
                            .await
                        {
                            Ok(_) => info!("Vector index created on {} table", name),
                            Err(e) => {
                                // Non-fatal: brute-force search still works
                                tracing::warn!("Failed to create vector index on {}: {}", name, e);
                            }
                        }
                    }
                }

                // FTS index on content column (no minimum row requirement)
                if row_count > 0 {
                    let has_fts_index = indices
                        .iter()
                        .any(|idx| idx.columns.contains(&"content".to_string()));

                    if !has_fts_index {
                        info!("Creating FTS index on {} table ({} rows)", name, row_count);
                        match table
                            .create_index(
                                &["content"],
                                lancedb::index::Index::FTS(Default::default()),
                            )
                            .execute()
                            .await
                        {
                            Ok(_) => info!("FTS index created on {} table", name),
                            Err(e) => {
                                // Non-fatal: vector-only search still works
                                tracing::warn!("Failed to create FTS index on {}: {}", name, e);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Get or create the items table
    async fn get_items_table(&mut self) -> Result<&Table> {
        if self.items_table.is_none() {
            let schema = Arc::new(item_schema());
            let table = self
                .db
                .create_empty_table("items", schema)
                .execute()
                .await
                .map_err(|e| {
                    SedimentError::Database(format!("Failed to create items table: {}", e))
                })?;
            self.items_table = Some(table);
        }
        Ok(self.items_table.as_ref().unwrap())
    }

    /// Get or create the chunks table
    async fn get_chunks_table(&mut self) -> Result<&Table> {
        if self.chunks_table.is_none() {
            let schema = Arc::new(chunk_schema());
            let table = self
                .db
                .create_empty_table("chunks", schema)
                .execute()
                .await
                .map_err(|e| {
                    SedimentError::Database(format!("Failed to create chunks table: {}", e))
                })?;
            self.chunks_table = Some(table);
        }
        Ok(self.chunks_table.as_ref().unwrap())
    }

    // ==================== Item Operations ====================

    /// Store an item with automatic chunking for long content
    ///
    /// Returns a `StoreResult` containing the new item ID and any potential conflicts
    /// (items with similarity >= 0.85 to the new content).
    pub async fn store_item(&mut self, mut item: Item) -> Result<StoreResult> {
        // Set project_id if not already set and we have a current project
        if item.project_id.is_none() {
            item.project_id = self.project_id.clone();
        }

        // Determine if we need to chunk (by character count, not byte count,
        // so multi-byte UTF-8 content isn't prematurely chunked)
        let should_chunk = item.content.chars().count() > CHUNK_THRESHOLD;
        item.is_chunked = should_chunk;

        // Generate item embedding
        let embedding_text = item.embedding_text();
        let embedding = self.embedder.embed(&embedding_text)?;
        item.embedding = embedding;

        // Store the item
        let table = self.get_items_table().await?;
        let batch = item_to_batch(&item)?;
        let batches = RecordBatchIterator::new(vec![Ok(batch)], Arc::new(item_schema()));

        table
            .add(Box::new(batches))
            .execute()
            .await
            .map_err(|e| SedimentError::Database(format!("Failed to store item: {}", e)))?;

        // If chunking is needed, create and store chunks with rollback on failure
        if should_chunk {
            let content_type = detect_content_type(&item.content);
            let config = ChunkingConfig::default();
            let mut chunk_results = chunk_content(&item.content, content_type, &config);

            // Cap chunk count to prevent CPU exhaustion from pathological inputs
            if chunk_results.len() > MAX_CHUNKS_PER_ITEM {
                tracing::warn!(
                    "Chunk count {} exceeds limit {}, truncating",
                    chunk_results.len(),
                    MAX_CHUNKS_PER_ITEM
                );
                chunk_results.truncate(MAX_CHUNKS_PER_ITEM);
            }

            if let Err(e) = self.store_chunks(&item.id, &chunk_results).await {
                // Rollback: remove the parent item (and any partial chunks)
                let _ = self.delete_item(&item.id).await;
                return Err(e);
            }

            debug!(
                "Stored item: {} with {} chunks",
                item.id,
                chunk_results.len()
            );
        } else {
            debug!("Stored item: {} (no chunking)", item.id);
        }

        // Detect conflicts after storing (informational only, avoids TOCTOU race)
        // Uses pre-computed embedding to avoid re-embedding the same content.
        let potential_conflicts = self
            .find_similar_items_by_vector(
                &item.embedding,
                Some(&item.id),
                CONFLICT_SIMILARITY_THRESHOLD,
                CONFLICT_SEARCH_LIMIT,
            )
            .await
            .unwrap_or_default();

        Ok(StoreResult {
            id: item.id,
            potential_conflicts,
        })
    }

    /// Store chunks for an item using batched embedding and a single LanceDB write.
    async fn store_chunks(
        &mut self,
        item_id: &str,
        chunk_results: &[crate::chunker::ChunkResult],
    ) -> Result<()> {
        let embedder = self.embedder.clone();
        let chunks_table = self.get_chunks_table().await?;

        // Batch embed all chunks in sub-batches to bound memory
        let chunk_texts: Vec<&str> = chunk_results.iter().map(|cr| cr.content.as_str()).collect();
        let mut all_embeddings = Vec::with_capacity(chunk_texts.len());
        for batch_start in (0..chunk_texts.len()).step_by(EMBEDDING_BATCH_SIZE) {
            let batch_end = (batch_start + EMBEDDING_BATCH_SIZE).min(chunk_texts.len());
            let batch_embeddings = embedder.embed_batch(&chunk_texts[batch_start..batch_end])?;
            all_embeddings.extend(batch_embeddings);
        }

        // Build all chunks with their embeddings
        let mut all_chunk_batches = Vec::with_capacity(chunk_results.len());
        for (i, (chunk_result, embedding)) in chunk_results.iter().zip(all_embeddings).enumerate() {
            let mut chunk = Chunk::new(item_id, i, &chunk_result.content);
            if let Some(ctx) = &chunk_result.context {
                chunk = chunk.with_context(ctx);
            }
            chunk.embedding = embedding;
            all_chunk_batches.push(chunk_to_batch(&chunk)?);
        }

        // Single LanceDB write for all chunks
        if !all_chunk_batches.is_empty() {
            let schema = Arc::new(chunk_schema());
            let batches = RecordBatchIterator::new(all_chunk_batches.into_iter().map(Ok), schema);
            chunks_table
                .add(Box::new(batches))
                .execute()
                .await
                .map_err(|e| SedimentError::Database(format!("Failed to store chunks: {}", e)))?;
        }

        Ok(())
    }

    /// Run a full-text search on the items table and return a map of item_id → BM25 score.
    /// Returns None if FTS is unavailable (no index or query fails).
    async fn fts_rank_items(
        &self,
        table: &Table,
        query: &str,
        limit: usize,
    ) -> Option<std::collections::HashMap<String, f32>> {
        let fts_query =
            FullTextSearchQuery::new(query.to_string()).columns(Some(vec!["content".to_string()]));

        let fts_results = table
            .query()
            .full_text_search(fts_query)
            .limit(limit)
            .execute()
            .await
            .ok()?
            .try_collect::<Vec<_>>()
            .await
            .ok()?;

        let mut scores = std::collections::HashMap::new();
        for batch in fts_results {
            let ids = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())?;
            let bm25_scores = batch
                .column_by_name("_score")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());
            for i in 0..ids.len() {
                if !ids.is_null(i) {
                    let score = bm25_scores.map(|s| s.value(i)).unwrap_or(0.0);
                    scores.insert(ids.value(i).to_string(), score);
                }
            }
        }
        Some(scores)
    }

    /// Search items by semantic similarity
    pub async fn search_items(
        &mut self,
        query: &str,
        limit: usize,
        filters: ItemFilters,
    ) -> Result<Vec<SearchResult>> {
        // Cap limit to prevent overflow in limit*2 and limit*3 multiplications below
        let limit = limit.min(1000);
        // Retry vector index creation if it failed previously
        self.ensure_vector_index().await?;

        // Generate query embedding
        let query_embedding = self.embedder.embed(query)?;
        let min_similarity = filters.min_similarity.unwrap_or(0.3);

        // We need to search both items and chunks, then merge results
        let mut results_map: std::collections::HashMap<String, (SearchResult, f32)> =
            std::collections::HashMap::new();

        // Search items table directly (for non-chunked items and chunked items)
        if let Some(table) = &self.items_table {
            let row_count = table.count_rows(None).await.unwrap_or(0);
            let base_query = table
                .vector_search(query_embedding.clone())
                .map_err(|e| SedimentError::Database(format!("Failed to build search: {}", e)))?;
            let query_builder = if row_count < VECTOR_INDEX_THRESHOLD {
                base_query.bypass_vector_index().limit(limit * 2)
            } else {
                base_query.refine_factor(10).limit(limit * 2)
            };

            let results = query_builder
                .execute()
                .await
                .map_err(|e| SedimentError::Database(format!("Search failed: {}", e)))?
                .try_collect::<Vec<_>>()
                .await
                .map_err(|e| {
                    SedimentError::Database(format!("Failed to collect results: {}", e))
                })?;

            // Collect vector search results with similarity scores
            let mut vector_items: Vec<(Item, f32)> = Vec::new();
            for batch in results {
                let items = batch_to_items(&batch)?;
                let distances = batch
                    .column_by_name("_distance")
                    .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

                for (i, item) in items.into_iter().enumerate() {
                    let distance = distances.map(|d| d.value(i)).unwrap_or(0.0);
                    let similarity = 1.0 / (1.0 + distance);
                    if similarity >= min_similarity {
                        vector_items.push((item, similarity));
                    }
                }
            }

            // Run FTS search for keyword-based ranking signal
            let fts_ranking = self.fts_rank_items(table, query, limit * 2).await;

            // Compute max BM25 score for normalization (once, before the loop)
            let max_bm25 = fts_ranking
                .as_ref()
                .and_then(|scores| scores.values().cloned().reduce(f32::max))
                .unwrap_or(1.0)
                .max(f32::EPSILON);

            // Apply additive FTS boost before project boosting.
            // Uses BM25-normalized scores with power-law concentration:
            //   boost = fts_boost_max * (bm25 / max_bm25)^gamma
            // gamma > 1 concentrates boost on top FTS hits (like rank-based)
            // while staying grounded in actual BM25 relevance scores.
            for (item, similarity) in vector_items {
                let fts_boost = fts_ranking.as_ref().map_or(0.0, |scores| {
                    scores.get(&item.id).map_or(0.0, |&bm25_score| {
                        self.fts_boost_max * (bm25_score / max_bm25).powf(self.fts_gamma)
                    })
                });
                let boosted_similarity = boost_similarity(
                    similarity + fts_boost,
                    item.project_id.as_deref(),
                    self.project_id.as_deref(),
                );

                let result = SearchResult::from_item(&item, boosted_similarity);
                results_map
                    .entry(item.id.clone())
                    .or_insert((result, boosted_similarity));
            }
        }

        // Search chunks table (for chunked items)
        if let Some(chunks_table) = &self.chunks_table {
            let chunk_row_count = chunks_table.count_rows(None).await.unwrap_or(0);
            let chunk_base_query = chunks_table.vector_search(query_embedding).map_err(|e| {
                SedimentError::Database(format!("Failed to build chunk search: {}", e))
            })?;
            let chunk_results = if chunk_row_count < VECTOR_INDEX_THRESHOLD {
                chunk_base_query.bypass_vector_index().limit(limit * 3)
            } else {
                chunk_base_query.refine_factor(10).limit(limit * 3)
            }
            .execute()
            .await
            .map_err(|e| SedimentError::Database(format!("Chunk search failed: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| {
                SedimentError::Database(format!("Failed to collect chunk results: {}", e))
            })?;

            // Group chunks by item and find best chunk for each item
            let mut chunk_matches: std::collections::HashMap<String, (String, f32)> =
                std::collections::HashMap::new();

            for batch in chunk_results {
                let chunks = batch_to_chunks(&batch)?;
                let distances = batch
                    .column_by_name("_distance")
                    .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

                for (i, chunk) in chunks.into_iter().enumerate() {
                    let distance = distances.map(|d| d.value(i)).unwrap_or(0.0);
                    let similarity = 1.0 / (1.0 + distance);

                    if similarity < min_similarity {
                        continue;
                    }

                    // Keep track of best matching chunk per item
                    chunk_matches
                        .entry(chunk.item_id.clone())
                        .and_modify(|(content, best_sim)| {
                            if similarity > *best_sim {
                                *content = chunk.content.clone();
                                *best_sim = similarity;
                            }
                        })
                        .or_insert((chunk.content.clone(), similarity));
                }
            }

            // Batch-fetch parent items for all chunk matches in a single query
            let chunk_item_ids: Vec<&str> = chunk_matches.keys().map(|id| id.as_str()).collect();
            let parent_items = self.get_items_batch(&chunk_item_ids).await?;
            let parent_map: std::collections::HashMap<&str, &Item> = parent_items
                .iter()
                .map(|item| (item.id.as_str(), item))
                .collect();

            for (item_id, (excerpt, chunk_similarity)) in chunk_matches {
                if let Some(item) = parent_map.get(item_id.as_str()) {
                    // Apply project boosting
                    let boosted_similarity = boost_similarity(
                        chunk_similarity,
                        item.project_id.as_deref(),
                        self.project_id.as_deref(),
                    );

                    let result =
                        SearchResult::from_item_with_excerpt(item, boosted_similarity, excerpt);

                    // Update if this chunk-based result is better
                    results_map
                        .entry(item_id)
                        .and_modify(|(existing, existing_sim)| {
                            if boosted_similarity > *existing_sim {
                                *existing = result.clone();
                                *existing_sim = boosted_similarity;
                            }
                        })
                        .or_insert((result, boosted_similarity));
                }
            }
        }

        // Sort by boosted similarity (which already includes FTS boost + project boost)
        // and truncate to the requested limit.
        let mut search_results: Vec<SearchResult> =
            results_map.into_values().map(|(sr, _)| sr).collect();
        search_results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        search_results.truncate(limit);

        Ok(search_results)
    }

    /// Find items similar to the given content (for conflict detection)
    ///
    /// This searches the items table directly by content embedding to find
    /// potentially conflicting items before storing new content.
    pub async fn find_similar_items(
        &mut self,
        content: &str,
        min_similarity: f32,
        limit: usize,
    ) -> Result<Vec<ConflictInfo>> {
        let embedding = self.embedder.embed(content)?;
        self.find_similar_items_by_vector(&embedding, None, min_similarity, limit)
            .await
    }

    /// Find items similar to a pre-computed embedding vector (avoids re-embedding).
    ///
    /// If `exclude_id` is provided, results matching that ID are filtered out.
    pub async fn find_similar_items_by_vector(
        &self,
        embedding: &[f32],
        exclude_id: Option<&str>,
        min_similarity: f32,
        limit: usize,
    ) -> Result<Vec<ConflictInfo>> {
        let table = match &self.items_table {
            Some(t) => t,
            None => return Ok(Vec::new()),
        };

        let row_count = table.count_rows(None).await.unwrap_or(0);
        let base_query = table
            .vector_search(embedding.to_vec())
            .map_err(|e| SedimentError::Database(format!("Failed to build search: {}", e)))?;
        let results = if row_count < VECTOR_INDEX_THRESHOLD {
            base_query.bypass_vector_index().limit(limit)
        } else {
            base_query.refine_factor(10).limit(limit)
        }
        .execute()
        .await
        .map_err(|e| SedimentError::Database(format!("Search failed: {}", e)))?
        .try_collect::<Vec<_>>()
        .await
        .map_err(|e| SedimentError::Database(format!("Failed to collect results: {}", e)))?;

        let mut conflicts = Vec::new();

        for batch in results {
            let items = batch_to_items(&batch)?;
            let distances = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            for (i, item) in items.into_iter().enumerate() {
                if exclude_id.is_some_and(|eid| eid == item.id) {
                    continue;
                }

                let distance = distances.map(|d| d.value(i)).unwrap_or(0.0);
                let similarity = 1.0 / (1.0 + distance);

                if similarity >= min_similarity {
                    conflicts.push(ConflictInfo {
                        id: item.id,
                        content: item.content,
                        similarity,
                    });
                }
            }
        }

        // Sort by similarity descending
        conflicts.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(conflicts)
    }

    /// List items with optional filters
    pub async fn list_items(
        &mut self,
        _filters: ItemFilters,
        limit: Option<usize>,
        scope: crate::ListScope,
    ) -> Result<Vec<Item>> {
        let table = match &self.items_table {
            Some(t) => t,
            None => return Ok(Vec::new()),
        };

        let mut filter_parts = Vec::new();

        // Apply scope filter
        match scope {
            crate::ListScope::Project => {
                if let Some(ref pid) = self.project_id {
                    if !is_valid_id(pid) {
                        return Err(SedimentError::Database(
                            "Invalid project_id for list filter".to_string(),
                        ));
                    }
                    filter_parts.push(format!("project_id = '{}'", sanitize_sql_string(pid)));
                } else {
                    // No project context: return empty rather than silently listing all items
                    return Ok(Vec::new());
                }
            }
            crate::ListScope::Global => {
                filter_parts.push("project_id IS NULL".to_string());
            }
            crate::ListScope::All => {
                // No additional filter
            }
        }

        let mut query = table.query();

        if !filter_parts.is_empty() {
            let filter_str = filter_parts.join(" AND ");
            query = query.only_if(filter_str);
        }

        if let Some(l) = limit {
            query = query.limit(l);
        }

        let results = query
            .execute()
            .await
            .map_err(|e| SedimentError::Database(format!("Query failed: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| SedimentError::Database(format!("Failed to collect: {}", e)))?;

        let mut items = Vec::new();
        for batch in results {
            items.extend(batch_to_items(&batch)?);
        }

        Ok(items)
    }

    /// Get an item by ID
    pub async fn get_item(&self, id: &str) -> Result<Option<Item>> {
        if !is_valid_id(id) {
            return Ok(None);
        }
        let table = match &self.items_table {
            Some(t) => t,
            None => return Ok(None),
        };

        let results = table
            .query()
            .only_if(format!("id = '{}'", sanitize_sql_string(id)))
            .limit(1)
            .execute()
            .await
            .map_err(|e| SedimentError::Database(format!("Query failed: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| SedimentError::Database(format!("Failed to collect: {}", e)))?;

        for batch in results {
            let items = batch_to_items(&batch)?;
            if let Some(item) = items.into_iter().next() {
                return Ok(Some(item));
            }
        }

        Ok(None)
    }

    /// Get multiple items by ID in a single query
    pub async fn get_items_batch(&self, ids: &[&str]) -> Result<Vec<Item>> {
        let table = match &self.items_table {
            Some(t) => t,
            None => return Ok(Vec::new()),
        };

        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let quoted: Vec<String> = ids
            .iter()
            .filter(|id| is_valid_id(id))
            .map(|id| format!("'{}'", sanitize_sql_string(id)))
            .collect();
        if quoted.is_empty() {
            return Ok(Vec::new());
        }
        let filter = format!("id IN ({})", quoted.join(", "));

        let results = table
            .query()
            .only_if(filter)
            .execute()
            .await
            .map_err(|e| SedimentError::Database(format!("Batch query failed: {}", e)))?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| SedimentError::Database(format!("Failed to collect batch: {}", e)))?;

        let mut items = Vec::new();
        for batch in results {
            items.extend(batch_to_items(&batch)?);
        }

        Ok(items)
    }

    /// Delete an item and its chunks.
    /// Returns `true` if the item existed, `false` if it was not found.
    pub async fn delete_item(&self, id: &str) -> Result<bool> {
        if !is_valid_id(id) {
            return Ok(false);
        }
        // Check if item exists first
        let table = match &self.items_table {
            Some(t) => t,
            None => return Ok(false),
        };

        let exists = self.get_item(id).await?.is_some();
        if !exists {
            return Ok(false);
        }

        // Delete item first — if this fails, chunks remain valid.
        // Reversing the order avoids leaving an is_chunked=true item with
        // no chunks, which would produce incorrect search results.
        table
            .delete(&format!("id = '{}'", sanitize_sql_string(id)))
            .await
            .map_err(|e| SedimentError::Database(format!("Delete failed: {}", e)))?;

        // Delete orphaned chunks (best-effort — orphaned chunks are benign
        // since they are only found via item_id join on the now-deleted item).
        if let Some(chunks_table) = &self.chunks_table
            && let Err(e) = chunks_table
                .delete(&format!("item_id = '{}'", sanitize_sql_string(id)))
                .await
        {
            tracing::warn!("Failed to delete chunks for item {}: {}", id, e);
        }

        Ok(true)
    }

    /// Get database statistics
    pub async fn stats(&self) -> Result<DatabaseStats> {
        let mut stats = DatabaseStats::default();

        if let Some(table) = &self.items_table {
            stats.item_count = table
                .count_rows(None)
                .await
                .map_err(|e| SedimentError::Database(format!("Count failed: {}", e)))?;
        }

        if let Some(table) = &self.chunks_table {
            stats.chunk_count = table
                .count_rows(None)
                .await
                .map_err(|e| SedimentError::Database(format!("Count failed: {}", e)))?;
        }

        Ok(stats)
    }
}

// ==================== Project ID Migration ====================

/// Migrate all LanceDB items from one project ID to another.
///
/// Used when a project's ID changes (e.g., UUID→git root commit hash).
/// Updates the `project_id` column in-place on both items and chunks tables.
pub async fn migrate_project_id(
    db_path: &std::path::Path,
    old_id: &str,
    new_id: &str,
) -> Result<u64> {
    if !is_valid_id(old_id) || !is_valid_id(new_id) {
        return Err(SedimentError::Database(
            "Invalid project ID for migration".to_string(),
        ));
    }

    let db = connect(db_path.to_str().ok_or_else(|| {
        SedimentError::Database("Database path contains invalid UTF-8".to_string())
    })?)
    .execute()
    .await
    .map_err(|e| SedimentError::Database(format!("Failed to connect for migration: {}", e)))?;

    let table_names = db
        .table_names()
        .execute()
        .await
        .map_err(|e| SedimentError::Database(format!("Failed to list tables: {}", e)))?;

    let mut total_updated = 0u64;

    if table_names.contains(&"items".to_string()) {
        let table =
            db.open_table("items").execute().await.map_err(|e| {
                SedimentError::Database(format!("Failed to open items table: {}", e))
            })?;

        let updated = table
            .update()
            .only_if(format!("project_id = '{}'", sanitize_sql_string(old_id)))
            .column("project_id", format!("'{}'", sanitize_sql_string(new_id)))
            .execute()
            .await
            .map_err(|e| SedimentError::Database(format!("Failed to migrate items: {}", e)))?;

        total_updated += updated;
        info!(
            "Migrated {} items from project {} to {}",
            updated, old_id, new_id
        );
    }

    Ok(total_updated)
}

// ==================== Decay Scoring ====================

/// Compute a decay-adjusted score for a search result.
///
/// Formula: `similarity * freshness * frequency`
/// - freshness = 1.0 / (1.0 + age_days / 30.0)  (half-life ~30 days)
/// - frequency = 1.0 + 0.1 * ln(1 + access_count)
///
/// `last_accessed_at` and `created_at` are unix timestamps.
/// If no access record exists, pass `access_count=0` and use `created_at` for age.
pub fn score_with_decay(
    similarity: f32,
    now: i64,
    created_at: i64,
    access_count: u32,
    last_accessed_at: Option<i64>,
) -> f32 {
    // Guard against NaN/Inf from corrupted data
    if !similarity.is_finite() {
        return 0.0;
    }

    let reference_time = last_accessed_at.unwrap_or(created_at);
    let age_secs = (now - reference_time).max(0) as f64;
    let age_days = age_secs / 86400.0;

    let freshness = 1.0 / (1.0 + age_days / 30.0);
    let frequency = 1.0 + 0.1 * (1.0 + access_count as f64).ln();

    let result = similarity * (freshness * frequency) as f32;
    if result.is_finite() { result } else { 0.0 }
}

// ==================== Helper Functions ====================

/// Detect content type for smart chunking
fn detect_content_type(content: &str) -> ContentType {
    let trimmed = content.trim();

    // Check for JSON
    if ((trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']')))
        && serde_json::from_str::<serde_json::Value>(trimmed).is_ok()
    {
        return ContentType::Json;
    }

    // Check for YAML (common patterns)
    // Require either a "---" document separator or multiple lines matching "key: value"
    // where "key" is a simple identifier (no spaces before colon, no URLs).
    if trimmed.contains(":\n") || trimmed.contains(": ") || trimmed.starts_with("---") {
        let lines: Vec<&str> = trimmed.lines().take(10).collect();
        let yaml_key_count = lines
            .iter()
            .filter(|line| {
                let l = line.trim();
                // A YAML key line: starts with a word-like key followed by ': '
                // Excludes URLs (://), empty lines, comments, prose (key must be identifier-like)
                !l.is_empty()
                    && !l.starts_with('#')
                    && !l.contains("://")
                    && l.contains(": ")
                    && l.split(": ").next().is_some_and(|key| {
                        let k = key.trim_start_matches("- ");
                        !k.is_empty()
                            && k.chars()
                                .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
                    })
            })
            .count();
        // Require at least 2 YAML-like key lines or starts with ---
        if yaml_key_count >= 2 || (trimmed.starts_with("---") && yaml_key_count >= 1) {
            return ContentType::Yaml;
        }
    }

    // Check for Markdown (has headers)
    if trimmed.contains("\n# ") || trimmed.starts_with("# ") || trimmed.contains("\n## ") {
        return ContentType::Markdown;
    }

    // Check for code (common patterns at start of lines to avoid false positives
    // from English prose like "let me explain" or "import regulations")
    let code_patterns = [
        "fn ",
        "pub fn ",
        "def ",
        "class ",
        "function ",
        "const ",
        "let ",
        "var ",
        "import ",
        "export ",
        "struct ",
        "impl ",
        "trait ",
    ];
    let has_code_pattern = trimmed.lines().any(|line| {
        let l = line.trim();
        code_patterns.iter().any(|p| l.starts_with(p))
    });
    if has_code_pattern {
        return ContentType::Code;
    }

    ContentType::Text
}

// ==================== Arrow Conversion Helpers ====================

fn item_to_batch(item: &Item) -> Result<RecordBatch> {
    let schema = Arc::new(item_schema());

    let id = StringArray::from(vec![item.id.as_str()]);
    let content = StringArray::from(vec![item.content.as_str()]);
    let project_id = StringArray::from(vec![item.project_id.as_deref()]);
    let is_chunked = BooleanArray::from(vec![item.is_chunked]);
    let created_at = Int64Array::from(vec![item.created_at.timestamp()]);

    let vector = create_embedding_array(&item.embedding)?;

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id),
            Arc::new(content),
            Arc::new(project_id),
            Arc::new(is_chunked),
            Arc::new(created_at),
            Arc::new(vector),
        ],
    )
    .map_err(|e| SedimentError::Database(format!("Failed to create batch: {}", e)))
}

fn batch_to_items(batch: &RecordBatch) -> Result<Vec<Item>> {
    let mut items = Vec::new();

    let id_col = batch
        .column_by_name("id")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| SedimentError::Database("Missing id column".to_string()))?;

    let content_col = batch
        .column_by_name("content")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| SedimentError::Database("Missing content column".to_string()))?;

    let project_id_col = batch
        .column_by_name("project_id")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());

    let is_chunked_col = batch
        .column_by_name("is_chunked")
        .and_then(|c| c.as_any().downcast_ref::<BooleanArray>());

    let created_at_col = batch
        .column_by_name("created_at")
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>());

    let vector_col = batch
        .column_by_name("vector")
        .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>());

    for i in 0..batch.num_rows() {
        let id = id_col.value(i).to_string();
        let content = content_col.value(i).to_string();

        let project_id = project_id_col.and_then(|c| {
            if c.is_null(i) {
                None
            } else {
                Some(c.value(i).to_string())
            }
        });

        let is_chunked = is_chunked_col.map(|c| c.value(i)).unwrap_or(false);

        let created_at = created_at_col
            .map(|c| {
                Utc.timestamp_opt(c.value(i), 0)
                    .single()
                    .unwrap_or_else(Utc::now)
            })
            .unwrap_or_else(Utc::now);

        let embedding = vector_col
            .and_then(|col| {
                let value = col.value(i);
                value
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .map(|arr| arr.values().to_vec())
            })
            .unwrap_or_default();

        let item = Item {
            id,
            content,
            embedding,
            project_id,
            is_chunked,
            created_at,
        };

        items.push(item);
    }

    Ok(items)
}

fn chunk_to_batch(chunk: &Chunk) -> Result<RecordBatch> {
    let schema = Arc::new(chunk_schema());

    let id = StringArray::from(vec![chunk.id.as_str()]);
    let item_id = StringArray::from(vec![chunk.item_id.as_str()]);
    let chunk_index = Int32Array::from(vec![i32::try_from(chunk.chunk_index).unwrap_or(i32::MAX)]);
    let content = StringArray::from(vec![chunk.content.as_str()]);
    let context = StringArray::from(vec![chunk.context.as_deref()]);

    let vector = create_embedding_array(&chunk.embedding)?;

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id),
            Arc::new(item_id),
            Arc::new(chunk_index),
            Arc::new(content),
            Arc::new(context),
            Arc::new(vector),
        ],
    )
    .map_err(|e| SedimentError::Database(format!("Failed to create batch: {}", e)))
}

fn batch_to_chunks(batch: &RecordBatch) -> Result<Vec<Chunk>> {
    let mut chunks = Vec::new();

    let id_col = batch
        .column_by_name("id")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| SedimentError::Database("Missing id column".to_string()))?;

    let item_id_col = batch
        .column_by_name("item_id")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| SedimentError::Database("Missing item_id column".to_string()))?;

    let chunk_index_col = batch
        .column_by_name("chunk_index")
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())
        .ok_or_else(|| SedimentError::Database("Missing chunk_index column".to_string()))?;

    let content_col = batch
        .column_by_name("content")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| SedimentError::Database("Missing content column".to_string()))?;

    let context_col = batch
        .column_by_name("context")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());

    for i in 0..batch.num_rows() {
        let id = id_col.value(i).to_string();
        let item_id = item_id_col.value(i).to_string();
        let chunk_index = chunk_index_col.value(i) as usize;
        let content = content_col.value(i).to_string();
        let context = context_col.and_then(|c| {
            if c.is_null(i) {
                None
            } else {
                Some(c.value(i).to_string())
            }
        });

        let chunk = Chunk {
            id,
            item_id,
            chunk_index,
            content,
            embedding: Vec::new(),
            context,
        };

        chunks.push(chunk);
    }

    Ok(chunks)
}

fn create_embedding_array(embedding: &[f32]) -> Result<FixedSizeListArray> {
    let values = Float32Array::from(embedding.to_vec());
    let field = Arc::new(Field::new("item", DataType::Float32, true));

    FixedSizeListArray::try_new(field, EMBEDDING_DIM as i32, Arc::new(values), None)
        .map_err(|e| SedimentError::Database(format!("Failed to create vector: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_with_decay_fresh_item() {
        let now = 1700000000i64;
        let created = now; // just created
        let score = score_with_decay(0.8, now, created, 0, None);
        // freshness = 1.0, frequency = 1.0 + 0.1 * ln(1) = 1.0
        let expected = 0.8 * 1.0 * 1.0;
        assert!((score - expected).abs() < 0.001, "got {}", score);
    }

    #[test]
    fn test_score_with_decay_30_day_old() {
        let now = 1700000000i64;
        let created = now - 30 * 86400; // 30 days old
        let score = score_with_decay(0.8, now, created, 0, None);
        // freshness = 1/(1+1) = 0.5, frequency = 1.0
        let expected = 0.8 * 0.5;
        assert!((score - expected).abs() < 0.001, "got {}", score);
    }

    #[test]
    fn test_score_with_decay_frequent_access() {
        let now = 1700000000i64;
        let created = now - 30 * 86400;
        let last_accessed = now; // just accessed
        let score = score_with_decay(0.8, now, created, 10, Some(last_accessed));
        // freshness = 1.0 (just accessed), frequency = 1.0 + 0.1 * ln(11) ≈ 1.2397
        let freq = 1.0 + 0.1 * (11.0_f64).ln();
        let expected = 0.8 * 1.0 * freq as f32;
        assert!((score - expected).abs() < 0.01, "got {}", score);
    }

    #[test]
    fn test_score_with_decay_old_and_unused() {
        let now = 1700000000i64;
        let created = now - 90 * 86400; // 90 days old
        let score = score_with_decay(0.8, now, created, 0, None);
        // freshness = 1/(1+3) = 0.25
        let expected = 0.8 * 0.25;
        assert!((score - expected).abs() < 0.001, "got {}", score);
    }

    #[test]
    fn test_sanitize_sql_string_escapes_quotes_and_backslashes() {
        assert_eq!(sanitize_sql_string("hello"), "hello");
        assert_eq!(sanitize_sql_string("it's"), "it''s");
        assert_eq!(sanitize_sql_string(r"a\'b"), r"a\\''b");
        assert_eq!(sanitize_sql_string(r"path\to\file"), r"path\\to\\file");
    }

    #[test]
    fn test_sanitize_sql_string_strips_null_bytes() {
        assert_eq!(sanitize_sql_string("abc\0def"), "abcdef");
        assert_eq!(sanitize_sql_string("\0' OR 1=1 --"), "'' OR 1=1 ");
        // Block comment close should also be stripped
        assert_eq!(sanitize_sql_string("*/ OR 1=1"), " OR 1=1");
        assert_eq!(sanitize_sql_string("clean"), "clean");
    }

    #[test]
    fn test_sanitize_sql_string_strips_semicolons() {
        assert_eq!(
            sanitize_sql_string("a; DROP TABLE items"),
            "a DROP TABLE items"
        );
        assert_eq!(sanitize_sql_string("normal;"), "normal");
    }

    #[test]
    fn test_sanitize_sql_string_strips_comments() {
        // Line comments (-- stripped, leaving extra space)
        assert_eq!(sanitize_sql_string("val' -- comment"), "val''  comment");
        // Block comments (/* and */ both stripped)
        assert_eq!(sanitize_sql_string("val' /* block */"), "val''  block ");
        // Nested attempts
        assert_eq!(sanitize_sql_string("a--b--c"), "abc");
        // Standalone */ without matching /*
        assert_eq!(sanitize_sql_string("injected */ rest"), "injected  rest");
        // Only */
        assert_eq!(sanitize_sql_string("*/"), "");
    }

    #[test]
    fn test_sanitize_sql_string_adversarial_inputs() {
        // Classic SQL injection
        assert_eq!(
            sanitize_sql_string("'; DROP TABLE items;--"),
            "'' DROP TABLE items"
        );
        // Unicode escapes (should pass through harmlessly)
        assert_eq!(
            sanitize_sql_string("hello\u{200B}world"),
            "hello\u{200B}world"
        );
        // Empty string
        assert_eq!(sanitize_sql_string(""), "");
        // Only special chars
        assert_eq!(sanitize_sql_string("\0;\0"), "");
    }

    #[test]
    fn test_is_valid_id() {
        // Valid UUIDs
        assert!(is_valid_id("550e8400-e29b-41d4-a716-446655440000"));
        assert!(is_valid_id("abcdef0123456789"));
        // Invalid
        assert!(!is_valid_id(""));
        assert!(!is_valid_id("'; DROP TABLE items;--"));
        assert!(!is_valid_id("hello world"));
        assert!(!is_valid_id("abc\0def"));
        // Too long
        assert!(!is_valid_id(&"a".repeat(65)));
    }

    #[test]
    fn test_detect_content_type_yaml_not_prose() {
        // Fix #11: Prose with colons should NOT be detected as YAML
        let prose = "Dear John:\nI wanted to write you about something.\nSubject: important matter";
        let detected = detect_content_type(prose);
        assert_ne!(
            detected,
            ContentType::Yaml,
            "Prose with colons should not be detected as YAML"
        );

        // Actual YAML should still be detected
        let yaml = "server: localhost\nport: 8080\ndatabase: mydb";
        let detected = detect_content_type(yaml);
        assert_eq!(detected, ContentType::Yaml);
    }

    #[test]
    fn test_detect_content_type_yaml_with_separator() {
        let yaml = "---\nname: test\nversion: 1.0";
        let detected = detect_content_type(yaml);
        assert_eq!(detected, ContentType::Yaml);
    }

    #[test]
    fn test_chunk_threshold_uses_chars_not_bytes() {
        // Bug #12: CHUNK_THRESHOLD should compare character count, not byte count.
        // 500 emoji chars = 2000 bytes. Should NOT exceed 1000-char threshold.
        let emoji_content = "😀".repeat(500);
        assert_eq!(emoji_content.chars().count(), 500);
        assert_eq!(emoji_content.len(), 2000); // 4 bytes per emoji

        let should_chunk = emoji_content.chars().count() > CHUNK_THRESHOLD;
        assert!(
            !should_chunk,
            "500 chars should not exceed 1000-char threshold"
        );

        // 1001 chars should trigger chunking
        let long_content = "a".repeat(1001);
        let should_chunk = long_content.chars().count() > CHUNK_THRESHOLD;
        assert!(should_chunk, "1001 chars should exceed 1000-char threshold");
    }

    #[test]
    fn test_schema_version() {
        // Ensure schema version is set
        let version = SCHEMA_VERSION;
        assert!(version >= 2, "Schema version should be at least 2");
    }

    /// Build the old item schema (v1) that included a `tags` column.
    fn old_item_schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("project_id", DataType::Utf8, true),
            Field::new("tags", DataType::Utf8, true), // removed in v2
            Field::new("is_chunked", DataType::Boolean, false),
            Field::new("created_at", DataType::Int64, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    EMBEDDING_DIM as i32,
                ),
                false,
            ),
        ])
    }

    /// Create a RecordBatch with old schema for migration testing.
    fn old_item_batch(id: &str, content: &str) -> RecordBatch {
        let schema = Arc::new(old_item_schema());
        let vector_values = Float32Array::from(vec![0.0f32; EMBEDDING_DIM]);
        let vector_field = Arc::new(Field::new("item", DataType::Float32, true));
        let vector = FixedSizeListArray::try_new(
            vector_field,
            EMBEDDING_DIM as i32,
            Arc::new(vector_values),
            None,
        )
        .unwrap();

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![id])),
                Arc::new(StringArray::from(vec![content])),
                Arc::new(StringArray::from(vec![None::<&str>])), // project_id
                Arc::new(StringArray::from(vec![None::<&str>])), // tags
                Arc::new(BooleanArray::from(vec![false])),
                Arc::new(Int64Array::from(vec![1700000000i64])),
                Arc::new(vector),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    #[ignore] // requires model download
    async fn test_check_needs_migration_detects_old_schema() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("data");

        // Create a LanceDB connection and insert an old-schema table
        let db_conn = lancedb::connect(db_path.to_str().unwrap())
            .execute()
            .await
            .unwrap();

        let schema = Arc::new(old_item_schema());
        let batch = old_item_batch("test-id-1", "old content");
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        db_conn
            .create_table("items", Box::new(batches))
            .execute()
            .await
            .unwrap();

        // Build a Database struct manually (without ensure_tables to skip auto-migration)
        let db = Database {
            db: db_conn,
            embedder: Arc::new(Embedder::new().unwrap()),
            project_id: None,
            items_table: None,
            chunks_table: None,
            fts_boost_max: FTS_BOOST_MAX,
            fts_gamma: FTS_GAMMA,
        };

        let needs_migration = db.check_needs_migration().await.unwrap();
        assert!(
            needs_migration,
            "Old schema with tags column should need migration"
        );
    }

    #[tokio::test]
    #[ignore] // requires model download
    async fn test_check_needs_migration_false_for_new_schema() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("data");

        // Create a LanceDB connection with new schema
        let db_conn = lancedb::connect(db_path.to_str().unwrap())
            .execute()
            .await
            .unwrap();

        let schema = Arc::new(item_schema());
        db_conn
            .create_empty_table("items", schema)
            .execute()
            .await
            .unwrap();

        let db = Database {
            db: db_conn,
            embedder: Arc::new(Embedder::new().unwrap()),
            project_id: None,
            items_table: None,
            chunks_table: None,
            fts_boost_max: FTS_BOOST_MAX,
            fts_gamma: FTS_GAMMA,
        };

        let needs_migration = db.check_needs_migration().await.unwrap();
        assert!(!needs_migration, "New schema should not need migration");
    }

    #[tokio::test]
    #[ignore] // requires model download
    async fn test_migrate_schema_preserves_data() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("data");

        // Create a LanceDB connection with old schema and 2 rows
        let db_conn = lancedb::connect(db_path.to_str().unwrap())
            .execute()
            .await
            .unwrap();

        let schema = Arc::new(old_item_schema());
        let batch1 = old_item_batch("id-aaa", "first item content");
        let batch2 = old_item_batch("id-bbb", "second item content");
        let batches = RecordBatchIterator::new(vec![Ok(batch1), Ok(batch2)], schema);
        db_conn
            .create_table("items", Box::new(batches))
            .execute()
            .await
            .unwrap();
        drop(db_conn);

        // Open via Database (triggers auto-migration in ensure_tables)
        let embedder = Arc::new(Embedder::new().unwrap());
        let db = Database::open_with_embedder(&db_path, None, embedder)
            .await
            .unwrap();

        // Verify migration happened: no tags column
        let needs_migration = db.check_needs_migration().await.unwrap();
        assert!(
            !needs_migration,
            "Schema should be migrated (no tags column)"
        );

        // Verify data preserved
        let item_a = db.get_item("id-aaa").await.unwrap();
        assert!(item_a.is_some(), "Item id-aaa should be preserved");
        assert_eq!(item_a.unwrap().content, "first item content");

        let item_b = db.get_item("id-bbb").await.unwrap();
        assert!(item_b.is_some(), "Item id-bbb should be preserved");
        assert_eq!(item_b.unwrap().content, "second item content");

        // Verify row count
        let stats = db.stats().await.unwrap();
        assert_eq!(stats.item_count, 2, "Should have 2 items after migration");
    }

    #[tokio::test]
    #[ignore] // requires model download
    async fn test_recover_case_a_only_staging() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("data");

        // Create only items_migrated (staging) table, no items table
        let db_conn = lancedb::connect(db_path.to_str().unwrap())
            .execute()
            .await
            .unwrap();

        let schema = Arc::new(item_schema());
        let vector_values = Float32Array::from(vec![0.0f32; EMBEDDING_DIM]);
        let vector_field = Arc::new(Field::new("item", DataType::Float32, true));
        let vector = FixedSizeListArray::try_new(
            vector_field,
            EMBEDDING_DIM as i32,
            Arc::new(vector_values),
            None,
        )
        .unwrap();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["staging-id"])),
                Arc::new(StringArray::from(vec!["staging content"])),
                Arc::new(StringArray::from(vec![None::<&str>])),
                Arc::new(BooleanArray::from(vec![false])),
                Arc::new(Int64Array::from(vec![1700000000i64])),
                Arc::new(vector),
            ],
        )
        .unwrap();

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        db_conn
            .create_table("items_migrated", Box::new(batches))
            .execute()
            .await
            .unwrap();
        drop(db_conn);

        // Open via Database — recovery should restore items from staging
        let embedder = Arc::new(Embedder::new().unwrap());
        let db = Database::open_with_embedder(&db_path, None, embedder)
            .await
            .unwrap();

        // Verify item was recovered
        let item = db.get_item("staging-id").await.unwrap();
        assert!(item.is_some(), "Item should be recovered from staging");
        assert_eq!(item.unwrap().content, "staging content");

        // Verify staging table was cleaned up
        let table_names = db.db.table_names().execute().await.unwrap();
        assert!(
            !table_names.contains(&"items_migrated".to_string()),
            "Staging table should be dropped"
        );
    }

    #[tokio::test]
    #[ignore] // requires model download
    async fn test_recover_case_b_both_old_schema() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("data");

        // Create both items (old schema) and items_migrated
        let db_conn = lancedb::connect(db_path.to_str().unwrap())
            .execute()
            .await
            .unwrap();

        // items with old schema
        let old_schema = Arc::new(old_item_schema());
        let batch = old_item_batch("old-id", "old content");
        let batches = RecordBatchIterator::new(vec![Ok(batch)], old_schema);
        db_conn
            .create_table("items", Box::new(batches))
            .execute()
            .await
            .unwrap();

        // items_migrated (leftover from failed migration)
        let new_schema = Arc::new(item_schema());
        db_conn
            .create_empty_table("items_migrated", new_schema)
            .execute()
            .await
            .unwrap();
        drop(db_conn);

        // Open via Database — recovery should drop staging, then re-run migration
        let embedder = Arc::new(Embedder::new().unwrap());
        let db = Database::open_with_embedder(&db_path, None, embedder)
            .await
            .unwrap();

        // Verify migration completed (no tags column)
        let needs_migration = db.check_needs_migration().await.unwrap();
        assert!(!needs_migration, "Should have migrated after recovery");

        // Verify data preserved
        let item = db.get_item("old-id").await.unwrap();
        assert!(
            item.is_some(),
            "Item should be preserved through recovery + migration"
        );

        // Verify staging dropped
        let table_names = db.db.table_names().execute().await.unwrap();
        assert!(
            !table_names.contains(&"items_migrated".to_string()),
            "Staging table should be dropped"
        );
    }

    #[tokio::test]
    #[ignore] // requires model download
    async fn test_recover_case_c_both_new_schema() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("data");

        // Create items (new schema) and leftover items_migrated
        let db_conn = lancedb::connect(db_path.to_str().unwrap())
            .execute()
            .await
            .unwrap();

        let new_schema = Arc::new(item_schema());

        // items with new schema
        let vector_values = Float32Array::from(vec![0.0f32; EMBEDDING_DIM]);
        let vector_field = Arc::new(Field::new("item", DataType::Float32, true));
        let vector = FixedSizeListArray::try_new(
            vector_field,
            EMBEDDING_DIM as i32,
            Arc::new(vector_values),
            None,
        )
        .unwrap();

        let batch = RecordBatch::try_new(
            new_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["new-id"])),
                Arc::new(StringArray::from(vec!["new content"])),
                Arc::new(StringArray::from(vec![None::<&str>])),
                Arc::new(BooleanArray::from(vec![false])),
                Arc::new(Int64Array::from(vec![1700000000i64])),
                Arc::new(vector),
            ],
        )
        .unwrap();

        let batches = RecordBatchIterator::new(vec![Ok(batch)], new_schema.clone());
        db_conn
            .create_table("items", Box::new(batches))
            .execute()
            .await
            .unwrap();

        // Leftover staging table
        db_conn
            .create_empty_table("items_migrated", new_schema)
            .execute()
            .await
            .unwrap();
        drop(db_conn);

        // Open via Database — recovery should just drop staging
        let embedder = Arc::new(Embedder::new().unwrap());
        let db = Database::open_with_embedder(&db_path, None, embedder)
            .await
            .unwrap();

        // Verify data intact
        let item = db.get_item("new-id").await.unwrap();
        assert!(item.is_some(), "Item should be untouched");
        assert_eq!(item.unwrap().content, "new content");

        // Verify staging dropped
        let table_names = db.db.table_names().execute().await.unwrap();
        assert!(
            !table_names.contains(&"items_migrated".to_string()),
            "Staging table should be dropped"
        );
    }

    #[tokio::test]
    #[ignore] // requires model download
    async fn test_list_items_rejects_invalid_project_id() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("data");
        let malicious_pid = "'; DROP TABLE items;--".to_string();

        let mut db = Database::open_with_project(&db_path, Some(malicious_pid))
            .await
            .unwrap();

        let result = db
            .list_items(ItemFilters::new(), Some(10), crate::ListScope::Project)
            .await;

        assert!(result.is_err(), "Should reject invalid project_id");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Invalid project_id"),
            "Error should mention invalid project_id, got: {}",
            err_msg
        );
    }
}
