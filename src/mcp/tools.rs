//! MCP Tool definitions for Sediment
//!
//! 5 tools: store, recall, list, forget, connections

use std::sync::Arc;

use chrono::DateTime;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::access::AccessTracker;
use crate::consolidation::{ConsolidationQueue, spawn_consolidation};
use crate::db::score_with_decay;
use crate::graph::GraphStore;
use crate::item::{Item, ItemFilters};
use crate::retry::{RetryConfig, with_retry};
use crate::{Database, ListScope, StoreScope};

use super::protocol::{CallToolResult, Tool};
use super::server::ServerContext;

/// Spawn a background task with panic logging. If the task panics, the panic
/// is caught and logged as an error instead of silently disappearing.
fn spawn_logged(name: &'static str, fut: impl std::future::Future<Output = ()> + Send + 'static) {
    tokio::spawn(async move {
        let result = tokio::task::spawn(fut).await;
        if let Err(e) = result {
            tracing::error!("Background task '{}' panicked: {:?}", name, e);
        }
    });
}

/// Get all available tools (5 total)
pub fn get_tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "store".to_string(),
            description: "Store content for later retrieval. Use for preferences, facts, reference material, docs, or any information worth remembering. Long content is automatically chunked for better search.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to store"
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional title (recommended for long content)"
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Tags for categorization"
                    },
                    "source": {
                        "type": "string",
                        "description": "Source attribution (e.g., URL, file path, 'conversation')"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Custom JSON metadata"
                    },
                    "expires_at": {
                        "type": "string",
                        "description": "ISO datetime when this should expire (optional)"
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["project", "global"],
                        "default": "project",
                        "description": "Where to store: 'project' (current project) or 'global' (all projects)"
                    },
                    "replace": {
                        "type": "string",
                        "description": "ID of an existing item to replace (stores new item first, then deletes old)"
                    },
                    "related": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "IDs of related items to link in the knowledge graph"
                    }
                },
                "required": ["content"]
            }),
        },
        Tool {
            name: "recall".to_string(),
            description: "Search stored content by semantic similarity. Returns matching items with relevant excerpts for chunked content.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for (semantic search)"
                    },
                    "limit": {
                        "type": "number",
                        "default": 5,
                        "description": "Maximum number of results"
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Filter by tags (any match)"
                    },
                    "min_similarity": {
                        "type": "number",
                        "default": 0.3,
                        "description": "Minimum similarity threshold (0.0-1.0). Lower values return more results."
                    }
                },
                "required": ["query"]
            }),
        },
        Tool {
            name: "list".to_string(),
            description: "List stored items with optional filtering.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Filter by tags"
                    },
                    "limit": {
                        "type": "number",
                        "default": 10,
                        "description": "Maximum number of results"
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["project", "global", "all"],
                        "default": "project",
                        "description": "Which items to list: 'project', 'global', or 'all'"
                    }
                }
            }),
        },
        Tool {
            name: "forget".to_string(),
            description: "Delete a stored item by its ID.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The item ID to delete"
                    }
                },
                "required": ["id"]
            }),
        },
        Tool {
            name: "connections".to_string(),
            description: "Show the relationship graph for a stored item. Returns all connections including related items, superseded items, and frequently co-accessed items.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The item ID to show connections for"
                    }
                },
                "required": ["id"]
            }),
        },
    ]
}

// ========== Parameter Structs ==========

#[derive(Debug, Deserialize)]
pub struct StoreParams {
    pub content: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub expires_at: Option<String>,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub replace: Option<String>,
    #[serde(default)]
    pub related: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct RecallParams {
    pub query: String,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub min_similarity: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct ListParams {
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub scope: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ForgetParams {
    pub id: String,
}

#[derive(Debug, Deserialize)]
pub struct ConnectionsParams {
    pub id: String,
}

// ========== Recall Configuration ==========

/// Controls which graph and scoring features are enabled during recall.
/// Used by benchmarks to measure the impact of individual features.
pub struct RecallConfig {
    pub enable_graph_backfill: bool,
    pub enable_graph_expansion: bool,
    pub enable_co_access: bool,
    pub enable_decay_scoring: bool,
    pub enable_background_tasks: bool,
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            enable_graph_backfill: true,
            enable_graph_expansion: true,
            enable_co_access: true,
            enable_decay_scoring: true,
            enable_background_tasks: true,
        }
    }
}

/// Result of a recall pipeline execution (for benchmark consumption).
pub struct RecallResult {
    pub results: Vec<crate::item::SearchResult>,
    pub graph_expanded: Vec<Value>,
    pub suggested: Vec<Value>,
}

// ========== Tool Execution ==========

pub async fn execute_tool(ctx: &ServerContext, name: &str, args: Option<Value>) -> CallToolResult {
    let config = RetryConfig::default();
    let args_for_retry = args.clone();

    let result = with_retry(&config, || {
        let ctx_ref = ctx;
        let name_ref = name;
        let args_clone = args_for_retry.clone();

        async move {
            // Open fresh connection with shared embedder
            let mut db = Database::open_with_embedder(
                &ctx_ref.db_path,
                ctx_ref.project_id.clone(),
                ctx_ref.embedder.clone(),
            )
            .await
            .map_err(|e| format!("Failed to open database: {}", e))?;

            // Open access tracker
            let tracker = AccessTracker::open(&ctx_ref.access_db_path)
                .map_err(|e| format!("Failed to open access tracker: {}", e))?;

            // Open graph store (shares access.db)
            let graph = GraphStore::open(&ctx_ref.access_db_path)
                .map_err(|e| format!("Failed to open graph store: {}", e))?;

            let result = match name_ref {
                "store" => execute_store(&mut db, &tracker, &graph, ctx_ref, args_clone).await,
                "recall" => execute_recall(&mut db, &tracker, &graph, ctx_ref, args_clone).await,
                "list" => execute_list(&mut db, args_clone).await,
                "forget" => execute_forget(&mut db, &graph, args_clone).await,
                "connections" => execute_connections(&mut db, &graph, args_clone).await,
                _ => return Ok(CallToolResult::error(format!("Unknown tool: {}", name_ref))),
            };

            if result.is_error.unwrap_or(false)
                && let Some(content) = result.content.first()
                && is_retryable_error(&content.text)
            {
                return Err(content.text.clone());
            }

            Ok(result)
        }
    })
    .await;

    match result {
        Ok(call_result) => call_result,
        Err(e) => CallToolResult::error(format!("Operation failed after retries: {}", e)),
    }
}

fn is_retryable_error(error_msg: &str) -> bool {
    let retryable_patterns = [
        "connection",
        "timeout",
        "temporarily unavailable",
        "resource busy",
        "lock",
        "I/O error",
        "Failed to open",
        "Failed to connect",
    ];

    let lower = error_msg.to_lowercase();
    retryable_patterns
        .iter()
        .any(|p| lower.contains(&p.to_lowercase()))
}

// ========== Tool Implementations ==========

async fn execute_store(
    db: &mut Database,
    tracker: &AccessTracker,
    graph: &GraphStore,
    ctx: &ServerContext,
    args: Option<Value>,
) -> CallToolResult {
    let params: StoreParams = match args {
        Some(v) => match serde_json::from_value(v) {
            Ok(p) => p,
            Err(e) => return CallToolResult::error(format!("Invalid parameters: {}", e)),
        },
        None => return CallToolResult::error("Missing parameters"),
    };

    // Reject oversized content to prevent OOM during embedding/chunking (1MB limit)
    const MAX_CONTENT_BYTES: usize = 1_000_000;
    if params.content.len() > MAX_CONTENT_BYTES {
        return CallToolResult::error(format!(
            "Content too large: {} bytes (max {} bytes)",
            params.content.len(),
            MAX_CONTENT_BYTES
        ));
    }

    // Parse scope
    let scope = params
        .scope
        .as_deref()
        .map(|s| s.parse::<StoreScope>())
        .transpose();

    let scope = match scope {
        Ok(s) => s.unwrap_or(StoreScope::Project),
        Err(e) => return CallToolResult::error(e),
    };

    // Parse expires_at if provided
    let expires_at = if let Some(ref exp_str) = params.expires_at {
        match DateTime::parse_from_rfc3339(exp_str) {
            Ok(dt) => Some(dt.with_timezone(&chrono::Utc)),
            Err(e) => return CallToolResult::error(format!("Invalid expires_at: {}", e)),
        }
    } else {
        None
    };

    // Validate that the item to replace exists (actual deletion deferred until after store)
    let replaced_id = if let Some(ref replace_id) = params.replace {
        match db.get_item(replace_id).await {
            Ok(Some(_)) => Some(replace_id.clone()),
            Ok(None) => {
                return CallToolResult::error(format!(
                    "Cannot replace: item not found: {}",
                    replace_id
                ));
            }
            Err(e) => {
                return CallToolResult::error(format!("Failed to look up item for replace: {}", e));
            }
        }
    } else {
        None
    };

    // Build item
    let mut tags = params.tags.unwrap_or_default();
    let mut item = Item::new(&params.content).with_tags(tags.clone());

    if let Some(title) = params.title {
        item = item.with_title(title);
    }

    if let Some(source) = params.source {
        item = item.with_source(source);
    }

    // Build metadata with provenance
    let mut metadata = params.metadata.unwrap_or(json!({}));
    if let Some(obj) = metadata.as_object_mut() {
        let mut provenance = json!({
            "v": 1,
            "project_path": ctx.cwd.to_string_lossy()
        });
        if let Some(ref rid) = replaced_id {
            provenance["supersedes"] = json!(rid);
        }
        obj.insert("_provenance".to_string(), provenance);
    }
    item = item.with_metadata(metadata);

    if let Some(exp) = expires_at {
        item = item.with_expires_at(exp);
    }

    // Set project_id based on scope
    if scope == StoreScope::Project
        && let Some(project_id) = db.project_id()
    {
        item = item.with_project_id(project_id);
    }

    // Auto-tag inference (Phase 4a): if no user tags, infer from similar items
    if tags.is_empty()
        && let Ok(similar) = db.find_similar_items(&params.content, 0.85, 5).await
    {
        let mut tag_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for conflict in &similar {
            if let Some(similar_item) = db.get_item(&conflict.id).await.ok().flatten() {
                for tag in &similar_item.tags {
                    if !tag.starts_with("auto:") {
                        *tag_counts.entry(tag.clone()).or_insert(0) += 1;
                    }
                }
            }
        }
        // If 2+ similar items share a tag, auto-apply it
        let auto_tags: Vec<String> = tag_counts
            .into_iter()
            .filter(|(_, count)| *count >= 2)
            .map(|(tag, _)| format!("auto:{}", tag))
            .collect();
        if !auto_tags.is_empty() {
            tags = item.tags.clone();
            tags.extend(auto_tags);
            item = item.with_tags(tags);
        }
    }

    match db.store_item(item).await {
        Ok(store_result) => {
            let new_id = store_result.id.clone();

            // Create graph node
            let now = chrono::Utc::now().timestamp();
            let project_id = db.project_id().map(|s| s.to_string());
            if let Err(e) = graph.add_node(&new_id, project_id.as_deref(), now) {
                tracing::warn!("graph add_node failed: {}", e);
            }

            // Complete replace: now that the new item is stored, delete the old one
            // (store-before-delete ensures no data loss on crash)
            if let Some(ref old_id) = replaced_id {
                // Record validation on the NEW item (the replacement is a "confirmed" version)
                let now_ts = chrono::Utc::now().timestamp();
                if let Err(e) = tracker.record_validation(&new_id, now_ts) {
                    tracing::warn!("record_validation failed: {}", e);
                }
                // Transfer graph edges from old node to new node before removing old node
                if let Err(e) = graph.transfer_edges(old_id, &new_id) {
                    tracing::warn!("transfer_edges failed: {}", e);
                }
                // Create SUPERSEDES edge before removing old node
                if let Err(e) = graph.add_supersedes_edge(&new_id, old_id) {
                    tracing::warn!("add_supersedes_edge failed: {}", e);
                }
                // Delete old item from LanceDB
                if let Err(e) = db.delete_item(old_id).await {
                    tracing::warn!("delete_item failed: {}", e);
                }
                // Remove old graph node (and its remaining edges)
                if let Err(e) = graph.remove_node(old_id) {
                    tracing::warn!("remove_node failed: {}", e);
                }
            }

            // Create RELATED edges if specified
            if let Some(ref related_ids) = params.related {
                for rid in related_ids {
                    if let Err(e) = graph.add_related_edge(&new_id, rid, 1.0, "user_linked") {
                        tracing::warn!("add_related_edge failed: {}", e);
                    }
                }
            }

            // Enqueue consolidation candidates from conflicts
            if !store_result.potential_conflicts.is_empty()
                && let Ok(queue) = ConsolidationQueue::open(&ctx.access_db_path)
            {
                for conflict in &store_result.potential_conflicts {
                    if let Err(e) = queue.enqueue(&new_id, &conflict.id, conflict.similarity as f64)
                    {
                        tracing::warn!("enqueue consolidation failed: {}", e);
                    }
                }
            }

            let mut result = json!({
                "success": true,
                "id": new_id,
                "message": format!("Stored in {} scope", scope)
            });

            if !store_result.potential_conflicts.is_empty() {
                let conflicts: Vec<Value> = store_result
                    .potential_conflicts
                    .iter()
                    .map(|c| {
                        json!({
                            "id": c.id,
                            "content": c.content,
                            "similarity": format!("{:.2}", c.similarity)
                        })
                    })
                    .collect();
                result["potential_conflicts"] = json!(conflicts);
            }

            CallToolResult::success(serde_json::to_string_pretty(&result).unwrap())
        }
        Err(e) => CallToolResult::error(format!("Failed to store: {}", e)),
    }
}

/// Core recall pipeline, extracted for benchmarking.
///
/// Performs: vector search, optional decay scoring, optional graph backfill,
/// optional 1-hop graph expansion, and optional co-access suggestions.
pub async fn recall_pipeline(
    db: &mut Database,
    tracker: &AccessTracker,
    graph: &GraphStore,
    query: &str,
    limit: usize,
    filters: ItemFilters,
    config: &RecallConfig,
) -> std::result::Result<RecallResult, String> {
    let mut results = db
        .search_items(query, limit, filters)
        .await
        .map_err(|e| format!("Search failed: {}", e))?;

    if results.is_empty() {
        return Ok(RecallResult {
            results: Vec::new(),
            graph_expanded: Vec::new(),
            suggested: Vec::new(),
        });
    }

    // Lazy graph backfill (uses project_id from SearchResult, no extra queries)
    if config.enable_graph_backfill {
        for result in &results {
            if let Err(e) = graph.ensure_node_exists(
                &result.id,
                result.project_id.as_deref(),
                result.created_at.timestamp(),
            ) {
                tracing::warn!("ensure_node_exists failed: {}", e);
            }
        }
    }

    // Decay scoring
    if config.enable_decay_scoring {
        let item_ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        let access_records = tracker.get_accesses(&item_ids).unwrap_or_default();
        let now = chrono::Utc::now().timestamp();

        for result in &mut results {
            let created_at = result.created_at.timestamp();
            let (access_count, last_accessed) = match access_records.get(&result.id) {
                Some(rec) => (rec.access_count, Some(rec.last_accessed_at)),
                None => (0, None),
            };

            let base_score = score_with_decay(
                result.similarity,
                now,
                created_at,
                access_count,
                last_accessed,
            );

            let validation_count = tracker.get_validation_count(&result.id).unwrap_or(0);
            let edge_count = graph.get_edge_count(&result.id).unwrap_or(0);
            let trust_bonus =
                1.0 + 0.05 * (1.0 + validation_count as f64).ln() as f32 + 0.02 * edge_count as f32;

            result.similarity = (base_score * trust_bonus).min(1.0);
        }

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // Record access
    for result in &results {
        let created_at = result.created_at.timestamp();
        if let Err(e) = tracker.record_access(&result.id, created_at) {
            tracing::warn!("record_access failed: {}", e);
        }
    }

    // Graph expansion
    let existing_ids: std::collections::HashSet<String> =
        results.iter().map(|r| r.id.clone()).collect();

    let mut graph_expanded = Vec::new();
    if config.enable_graph_expansion {
        let top_ids: Vec<&str> = results.iter().take(5).map(|r| r.id.as_str()).collect();
        if let Ok(neighbors) = graph.get_neighbors(&top_ids, 0.5) {
            // Collect neighbor IDs not already in results, then batch fetch
            let neighbor_info: Vec<(String, String)> = neighbors
                .into_iter()
                .filter(|(id, _, _)| !existing_ids.contains(id))
                .map(|(id, rel_type, _)| (id, rel_type))
                .collect();

            let neighbor_ids: Vec<&str> = neighbor_info.iter().map(|(id, _)| id.as_str()).collect();
            if let Ok(items) = db.get_items_batch(&neighbor_ids).await {
                let item_map: std::collections::HashMap<&str, &Item> =
                    items.iter().map(|item| (item.id.as_str(), item)).collect();

                for (neighbor_id, rel_type) in &neighbor_info {
                    if let Some(item) = item_map.get(neighbor_id.as_str()) {
                        let sr = crate::item::SearchResult::from_item(item, 0.05);
                        graph_expanded.push(json!({
                            "id": sr.id,
                            "content": sr.content,
                            "similarity": "graph",
                            "created": sr.created_at.to_rfc3339(),
                            "graph_expanded": true,
                            "rel_type": rel_type,
                        }));
                    }
                }
            }
        }
    }

    // Co-access suggestions (batch fetch)
    let mut suggested = Vec::new();
    if config.enable_co_access {
        let top3_ids: Vec<&str> = results.iter().take(3).map(|r| r.id.as_str()).collect();
        if let Ok(co_accessed) = graph.get_co_accessed(&top3_ids, 3) {
            let co_info: Vec<(String, i64)> = co_accessed
                .into_iter()
                .filter(|(id, _)| !existing_ids.contains(id))
                .collect();

            let co_ids: Vec<&str> = co_info.iter().map(|(id, _)| id.as_str()).collect();
            if let Ok(items) = db.get_items_batch(&co_ids).await {
                let item_map: std::collections::HashMap<&str, &Item> =
                    items.iter().map(|item| (item.id.as_str(), item)).collect();

                for (co_id, co_count) in &co_info {
                    if let Some(item) = item_map.get(co_id.as_str()) {
                        suggested.push(json!({
                            "id": item.id,
                            "content": truncate(&item.content, 100),
                            "reason": format!("frequently recalled with result (co-accessed {} times)", co_count),
                        }));
                    }
                }
            }
        }
    }

    Ok(RecallResult {
        results,
        graph_expanded,
        suggested,
    })
}

async fn execute_recall(
    db: &mut Database,
    tracker: &AccessTracker,
    graph: &GraphStore,
    ctx: &ServerContext,
    args: Option<Value>,
) -> CallToolResult {
    let params: RecallParams = match args {
        Some(v) => match serde_json::from_value(v) {
            Ok(p) => p,
            Err(e) => return CallToolResult::error(format!("Invalid parameters: {}", e)),
        },
        None => return CallToolResult::error("Missing parameters"),
    };

    let limit = params.limit.unwrap_or(5).min(100);
    let min_similarity = params.min_similarity.unwrap_or(0.3);

    let mut filters = ItemFilters::new().with_min_similarity(min_similarity);

    if let Some(tags) = params.tags {
        filters = filters.with_tags(tags);
    }

    let config = RecallConfig::default();

    let recall_result =
        match recall_pipeline(db, tracker, graph, &params.query, limit, filters, &config).await {
            Ok(r) => r,
            Err(e) => return CallToolResult::error(e),
        };

    if recall_result.results.is_empty() {
        return CallToolResult::success("No items found matching your query.");
    }

    let results = &recall_result.results;

    let formatted: Vec<Value> = results
        .iter()
        .map(|r| {
            let mut obj = json!({
                "id": r.id,
                "content": r.content,
                "similarity": format!("{:.2}", r.similarity),
                "created": r.created_at.to_rfc3339(),
            });

            if let Some(ref excerpt) = r.relevant_excerpt {
                obj["relevant_excerpt"] = json!(excerpt);
            }
            if !r.tags.is_empty() {
                obj["tags"] = json!(r.tags);
            }
            if let Some(ref source) = r.source {
                obj["source"] = json!(source);
            }

            // Cross-project flag (Phase 3c) — uses cached project_id/metadata from SearchResult
            if let Some(ref current_pid) = ctx.project_id
                && let Some(ref item_pid) = r.project_id
                && item_pid != current_pid
            {
                obj["cross_project"] = json!(true);
                if let Some(ref meta) = r.metadata
                    && let Some(prov) = meta.get("_provenance")
                    && let Some(pp) = prov.get("project_path")
                {
                    obj["project_path"] = pp.clone();
                }
            }

            // Related IDs from graph (Phase 1d)
            if let Ok(neighbors) = graph.get_neighbors(&[r.id.as_str()], 0.5) {
                let related: Vec<String> = neighbors.iter().map(|(id, _, _)| id.clone()).collect();
                if !related.is_empty() {
                    obj["related_ids"] = json!(related);
                }
            }

            obj
        })
        .collect();

    let mut result_json = json!({
        "count": results.len(),
        "results": formatted
    });

    if !recall_result.graph_expanded.is_empty() {
        result_json["graph_expanded"] = json!(recall_result.graph_expanded);
    }

    if !recall_result.suggested.is_empty() {
        result_json["suggested"] = json!(recall_result.suggested);
    }

    // Fire-and-forget: background consolidation (Phase 2b)
    spawn_consolidation(
        Arc::new(ctx.db_path.clone()),
        Arc::new(ctx.access_db_path.clone()),
        ctx.project_id.clone(),
        ctx.embedder.clone(),
        ctx.consolidation_semaphore.clone(),
    );

    // Fire-and-forget: co-access recording (Phase 3a)
    let result_ids: Vec<String> = results.iter().map(|r| r.id.clone()).collect();
    let access_db_path = ctx.access_db_path.clone();
    spawn_logged("co_access", async move {
        if let Ok(g) = GraphStore::open(&access_db_path) {
            if let Err(e) = g.record_co_access(&result_ids) {
                tracing::warn!("record_co_access failed: {}", e);
            }
        } else {
            tracing::warn!("co_access: failed to open graph store");
        }
    });

    // Periodic maintenance: every 10th recall
    let run_count = ctx
        .recall_count
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if run_count % 10 == 9 {
        // Clustering
        let access_db_path = ctx.access_db_path.clone();
        spawn_logged("clustering", async move {
            if let Ok(g) = GraphStore::open(&access_db_path)
                && let Ok(clusters) = g.detect_clusters()
            {
                for (a, b, c) in &clusters {
                    let label = format!("cluster-{}", &a[..8.min(a.len())]);
                    if let Err(e) = g.add_related_edge(a, b, 0.8, &label) {
                        tracing::warn!("cluster add_related_edge failed: {}", e);
                    }
                    if let Err(e) = g.add_related_edge(b, c, 0.8, &label) {
                        tracing::warn!("cluster add_related_edge failed: {}", e);
                    }
                    if let Err(e) = g.add_related_edge(a, c, 0.8, &label) {
                        tracing::warn!("cluster add_related_edge failed: {}", e);
                    }
                }
                if !clusters.is_empty() {
                    tracing::info!("Detected {} clusters", clusters.len());
                }
            }
        });

        // Expired item cleanup
        let db_path = ctx.db_path.clone();
        let project_id = ctx.project_id.clone();
        let embedder = ctx.embedder.clone();
        spawn_logged("cleanup_expired", async move {
            match Database::open_with_embedder(&db_path, project_id, embedder).await {
                Ok(db) => {
                    if let Err(e) = db.cleanup_expired().await {
                        tracing::warn!("cleanup_expired failed: {}", e);
                    }
                }
                Err(e) => tracing::warn!("cleanup_expired: failed to open db: {}", e),
            }
        });

        // Consolidation queue cleanup: purge old processed entries
        let access_db_path2 = ctx.access_db_path.clone();
        spawn_logged("consolidation_cleanup", async move {
            if let Ok(q) = crate::consolidation::ConsolidationQueue::open(&access_db_path2)
                && let Err(e) = q.cleanup_processed()
            {
                tracing::warn!("consolidation queue cleanup failed: {}", e);
            }
        });
    }

    CallToolResult::success(serde_json::to_string_pretty(&result_json).unwrap())
}

async fn execute_list(db: &mut Database, args: Option<Value>) -> CallToolResult {
    let params: ListParams =
        args.and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or(ListParams {
                tags: None,
                limit: None,
                scope: None,
            });

    let limit = params.limit.unwrap_or(10).min(100);

    let mut filters = ItemFilters::new();

    if let Some(tags) = params.tags {
        filters = filters.with_tags(tags);
    }

    let scope = params
        .scope
        .as_deref()
        .map(|s| s.parse::<ListScope>())
        .transpose();

    let scope = match scope {
        Ok(s) => s.unwrap_or(ListScope::Project),
        Err(e) => return CallToolResult::error(e),
    };

    match db.list_items(filters, Some(limit), scope).await {
        Ok(items) => {
            if items.is_empty() {
                CallToolResult::success("No items stored yet.")
            } else {
                let formatted: Vec<Value> = items
                    .iter()
                    .map(|item| {
                        let content_preview = truncate(&item.content, 100);
                        let mut obj = json!({
                            "id": item.id,
                            "content": content_preview,
                            "created": item.created_at.to_rfc3339(),
                        });

                        if let Some(ref title) = item.title {
                            obj["title"] = json!(title);
                        }
                        if !item.tags.is_empty() {
                            obj["tags"] = json!(item.tags);
                        }
                        if item.is_chunked {
                            obj["chunked"] = json!(true);
                        }

                        obj
                    })
                    .collect();

                let result = json!({
                    "count": items.len(),
                    "items": formatted
                });

                CallToolResult::success(serde_json::to_string_pretty(&result).unwrap())
            }
        }
        Err(e) => CallToolResult::error(format!("Failed to list items: {}", e)),
    }
}

async fn execute_forget(
    db: &mut Database,
    graph: &GraphStore,
    args: Option<Value>,
) -> CallToolResult {
    let params: ForgetParams = match args {
        Some(v) => match serde_json::from_value(v) {
            Ok(p) => p,
            Err(e) => return CallToolResult::error(format!("Invalid parameters: {}", e)),
        },
        None => return CallToolResult::error("Missing parameters"),
    };

    match db.delete_item(&params.id).await {
        Ok(true) => {
            // Remove from graph
            if let Err(e) = graph.remove_node(&params.id) {
                tracing::warn!("remove_node failed: {}", e);
            }

            let result = json!({
                "success": true,
                "message": format!("Deleted item: {}", params.id)
            });
            CallToolResult::success(serde_json::to_string_pretty(&result).unwrap())
        }
        Ok(false) => CallToolResult::error(format!("Item not found: {}", params.id)),
        Err(e) => CallToolResult::error(format!("Failed to delete: {}", e)),
    }
}

async fn execute_connections(
    db: &mut Database,
    graph: &GraphStore,
    args: Option<Value>,
) -> CallToolResult {
    let params: ConnectionsParams = match args {
        Some(v) => match serde_json::from_value(v) {
            Ok(p) => p,
            Err(e) => return CallToolResult::error(format!("Invalid parameters: {}", e)),
        },
        None => return CallToolResult::error("Missing parameters"),
    };

    // Verify item exists
    match db.get_item(&params.id).await {
        Ok(None) => return CallToolResult::error(format!("Item not found: {}", params.id)),
        Err(e) => return CallToolResult::error(format!("Failed to get item: {}", e)),
        Ok(Some(_)) => {}
    }

    match graph.get_full_connections(&params.id) {
        Ok(connections) => {
            // Batch fetch all connected items
            let target_ids: Vec<&str> = connections.iter().map(|c| c.target_id.as_str()).collect();
            let items = db.get_items_batch(&target_ids).await.unwrap_or_default();
            let item_map: std::collections::HashMap<&str, &Item> =
                items.iter().map(|item| (item.id.as_str(), item)).collect();

            let mut conn_json: Vec<Value> = Vec::new();

            for conn in &connections {
                let mut obj = json!({
                    "id": conn.target_id,
                    "type": conn.rel_type,
                    "strength": conn.strength,
                });

                if let Some(count) = conn.count {
                    obj["count"] = json!(count);
                }

                // Add content preview from batch
                if let Some(item) = item_map.get(conn.target_id.as_str()) {
                    obj["content_preview"] = json!(truncate(&item.content, 80));
                }

                conn_json.push(obj);
            }

            let result = json!({
                "item_id": params.id,
                "connections": conn_json
            });

            CallToolResult::success(serde_json::to_string_pretty(&result).unwrap())
        }
        Err(e) => CallToolResult::error(format!("Failed to get connections: {}", e)),
    }
}

// ========== Utilities ==========

fn truncate(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else if max_len <= 3 {
        // Not enough room for "..." suffix; just take max_len chars
        s.chars().take(max_len).collect()
    } else {
        let cut = s
            .char_indices()
            .nth(max_len - 3)
            .map(|(i, _)| i)
            .unwrap_or(s.len());
        format!("{}...", &s[..cut])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_small_max_len() {
        // Bug #25: truncate should not panic when max_len < 3
        assert_eq!(truncate("hello", 0), "");
        assert_eq!(truncate("hello", 1), "h");
        assert_eq!(truncate("hello", 2), "he");
        assert_eq!(truncate("hello", 3), "hel");
        assert_eq!(truncate("hi", 3), "hi"); // shorter than max, no truncation
        assert_eq!(truncate("hello", 5), "hello");
        assert_eq!(truncate("hello!", 5), "he...");
    }

    #[test]
    fn test_truncate_unicode() {
        assert_eq!(truncate("héllo wörld", 5), "hé...");
        assert_eq!(truncate("日本語テスト", 4), "日...");
    }
}
