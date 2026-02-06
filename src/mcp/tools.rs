//! MCP Tool definitions for Sediment
//!
//! 4 tools: store, recall, list, forget

use std::sync::Arc;

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

/// Get all available tools (4 total)
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
                    "scope": {
                        "type": "string",
                        "enum": ["project", "global"],
                        "default": "project",
                        "description": "Where to store: 'project' (current project) or 'global' (all projects)"
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
                    }
                },
                "required": ["query"]
            }),
        },
        Tool {
            name: "list".to_string(),
            description: "List stored items.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
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
    ]
}

// ========== Parameter Structs ==========

#[derive(Debug, Deserialize)]
pub struct StoreParams {
    pub content: String,
    #[serde(default)]
    pub scope: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RecallParams {
    pub query: String,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct ListParams {
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub scope: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ForgetParams {
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
    /// Raw (pre-decay/trust) similarity scores, keyed by item ID
    pub raw_similarities: std::collections::HashMap<String, f32>,
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
            .map_err(|e| sanitize_err("Failed to open database", e))?;

            // Open access tracker
            let tracker = AccessTracker::open(&ctx_ref.access_db_path)
                .map_err(|e| sanitize_err("Failed to open access tracker", e))?;

            // Open graph store (shares access.db)
            let graph = GraphStore::open(&ctx_ref.access_db_path)
                .map_err(|e| sanitize_err("Failed to open graph store", e))?;

            let result = match name_ref {
                "store" => execute_store(&mut db, &tracker, &graph, ctx_ref, args_clone).await,
                "recall" => execute_recall(&mut db, &tracker, &graph, ctx_ref, args_clone).await,
                "list" => execute_list(&mut db, args_clone).await,
                "forget" => execute_forget(&mut db, &graph, ctx_ref, args_clone).await,
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
        Err(e) => {
            tracing::error!("Operation failed after retries: {}", e);
            CallToolResult::error("Operation failed after retries")
        }
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
    _tracker: &AccessTracker,
    graph: &GraphStore,
    ctx: &ServerContext,
    args: Option<Value>,
) -> CallToolResult {
    let params: StoreParams = match args {
        Some(v) => match serde_json::from_value(v) {
            Ok(p) => p,
            Err(e) => {
                tracing::debug!("Parameter validation failed: {}", e);
                return CallToolResult::error("Invalid parameters");
            }
        },
        None => return CallToolResult::error("Missing parameters"),
    };

    // Reject oversized content to prevent OOM during embedding/chunking.
    // Intentionally byte-based (not char-based): memory allocation is proportional
    // to byte length, so this is the correct metric for OOM prevention.
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

    // Build item
    let mut item = Item::new(&params.content);

    // Set project_id based on scope
    if scope == StoreScope::Project
        && let Some(project_id) = db.project_id()
    {
        item = item.with_project_id(project_id);
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

            CallToolResult::success(
                serde_json::to_string_pretty(&result)
                    .unwrap_or_else(|e| format!("{{\"error\": \"serialization failed: {}\"}}", e)),
            )
        }
        Err(e) => sanitized_error("Failed to store item", e),
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
            raw_similarities: std::collections::HashMap::new(),
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

    // Decay scoring — preserve raw similarity for transparency
    let mut raw_similarities: std::collections::HashMap<String, f32> =
        std::collections::HashMap::new();
    if config.enable_decay_scoring {
        let item_ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        let access_records = tracker.get_accesses(&item_ids).unwrap_or_default();
        let validation_counts = tracker.get_validation_counts(&item_ids).unwrap_or_default();
        let edge_counts = graph.get_edge_counts(&item_ids).unwrap_or_default();
        let now = chrono::Utc::now().timestamp();

        for result in &mut results {
            raw_similarities.insert(result.id.clone(), result.similarity);

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

            let validation_count = validation_counts.get(&result.id).copied().unwrap_or(0);
            let edge_count = edge_counts.get(&result.id).copied().unwrap_or(0);
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
                        let mut entry = json!({
                            "id": sr.id,
                            "similarity": "graph",
                            "created": sr.created_at.to_rfc3339(),
                            "graph_expanded": true,
                            "rel_type": rel_type,
                        });
                        // Only include content for same-project or global items
                        let same_project = match (db.project_id(), item.project_id.as_deref()) {
                            (Some(current), Some(item_pid)) => current == item_pid,
                            (_, None) => true,
                            _ => false,
                        };
                        if same_project {
                            entry["content"] = json!(sr.content);
                        } else {
                            entry["cross_project"] = json!(true);
                        }
                        graph_expanded.push(entry);
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
                        let same_project = match (db.project_id(), item.project_id.as_deref()) {
                            (Some(current), Some(item_pid)) => current == item_pid,
                            (_, None) => true,
                            _ => false,
                        };
                        let mut entry = json!({
                            "id": item.id,
                            "reason": format!("frequently recalled with result (co-accessed {} times)", co_count),
                        });
                        if same_project {
                            entry["content"] = json!(truncate(&item.content, 100));
                        } else {
                            entry["cross_project"] = json!(true);
                        }
                        suggested.push(entry);
                    }
                }
            }
        }
    }

    Ok(RecallResult {
        results,
        graph_expanded,
        suggested,
        raw_similarities,
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
            Err(e) => {
                tracing::debug!("Parameter validation failed: {}", e);
                return CallToolResult::error("Invalid parameters");
            }
        },
        None => return CallToolResult::error("Missing parameters"),
    };

    // Reject oversized queries to prevent OOM during tokenization.
    // The model truncates to 512 tokens (~2KB of English text), so anything
    // beyond 10KB is wasted processing.
    const MAX_QUERY_BYTES: usize = 10_000;
    if params.query.len() > MAX_QUERY_BYTES {
        return CallToolResult::error(format!(
            "Query too large: {} bytes (max {} bytes)",
            params.query.len(),
            MAX_QUERY_BYTES
        ));
    }

    let limit = params.limit.unwrap_or(5).min(100);

    let filters = ItemFilters::new();

    let config = RecallConfig::default();

    let recall_result =
        match recall_pipeline(db, tracker, graph, &params.query, limit, filters, &config).await {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("Recall failed: {}", e);
                return CallToolResult::error("Search failed");
            }
        };

    if recall_result.results.is_empty() {
        return CallToolResult::success("No items found matching your query.");
    }

    let results = &recall_result.results;

    // Batch-fetch neighbors for all result IDs
    let all_result_ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
    let neighbors_map = graph
        .get_neighbors_mapped(&all_result_ids, 0.5)
        .unwrap_or_default();

    let formatted: Vec<Value> = results
        .iter()
        .map(|r| {
            let mut obj = json!({
                "id": r.id,
                "content": r.content,
                "similarity": format!("{:.2}", r.similarity),
                "created": r.created_at.to_rfc3339(),
            });

            // Include raw (pre-decay) similarity when decay scoring was applied
            if let Some(&raw_sim) = recall_result.raw_similarities.get(&r.id)
                && (raw_sim - r.similarity).abs() > 0.001
            {
                obj["raw_similarity"] = json!(format!("{:.2}", raw_sim));
            }

            if let Some(ref excerpt) = r.relevant_excerpt {
                obj["relevant_excerpt"] = json!(excerpt);
            }

            // Cross-project flag
            if let Some(ref current_pid) = ctx.project_id
                && let Some(ref item_pid) = r.project_id
                && item_pid != current_pid
            {
                obj["cross_project"] = json!(true);
            }

            // Related IDs from graph (batch lookup)
            if let Some(related) = neighbors_map.get(&r.id)
                && !related.is_empty()
            {
                obj["related_ids"] = json!(related);
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
        .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
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

    CallToolResult::success(
        serde_json::to_string_pretty(&result_json)
            .unwrap_or_else(|e| format!("{{\"error\": \"serialization failed: {}\"}}", e)),
    )
}

async fn execute_list(db: &mut Database, args: Option<Value>) -> CallToolResult {
    let params: ListParams =
        args.and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or(ListParams {
                limit: None,
                scope: None,
            });

    let limit = params.limit.unwrap_or(10).min(100);

    let filters = ItemFilters::new();

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

                CallToolResult::success(
                    serde_json::to_string_pretty(&result).unwrap_or_else(|e| {
                        format!("{{\"error\": \"serialization failed: {}\"}}", e)
                    }),
                )
            }
        }
        Err(e) => sanitized_error("Failed to list items", e),
    }
}

async fn execute_forget(
    db: &mut Database,
    graph: &GraphStore,
    ctx: &ServerContext,
    args: Option<Value>,
) -> CallToolResult {
    let params: ForgetParams = match args {
        Some(v) => match serde_json::from_value(v) {
            Ok(p) => p,
            Err(e) => {
                tracing::debug!("Parameter validation failed: {}", e);
                return CallToolResult::error("Invalid parameters");
            }
        },
        None => return CallToolResult::error("Missing parameters"),
    };

    // Access control: verify the item belongs to the current project (or is global)
    if let Some(ref current_pid) = ctx.project_id {
        match db.get_item(&params.id).await {
            Ok(Some(item)) => {
                if let Some(ref item_pid) = item.project_id
                    && item_pid != current_pid
                {
                    return CallToolResult::error(format!(
                        "Cannot delete item {} from a different project",
                        params.id
                    ));
                }
            }
            Ok(None) => return CallToolResult::error(format!("Item not found: {}", params.id)),
            Err(e) => {
                return sanitized_error("Failed to look up item", e);
            }
        }
    }

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
            CallToolResult::success(
                serde_json::to_string_pretty(&result)
                    .unwrap_or_else(|e| format!("{{\"error\": \"serialization failed: {}\"}}", e)),
            )
        }
        Ok(false) => CallToolResult::error(format!("Item not found: {}", params.id)),
        Err(e) => sanitized_error("Failed to delete item", e),
    }
}

// ========== Utilities ==========

/// Log a detailed internal error and return a sanitized message to the MCP client.
/// This prevents leaking file paths, database internals, or OS details.
fn sanitized_error(context: &str, err: impl std::fmt::Display) -> CallToolResult {
    tracing::error!("{}: {}", context, err);
    CallToolResult::error(context.to_string())
}

/// Like `sanitized_error` but returns a String for use inside `map_err` chains.
fn sanitize_err(context: &str, err: impl std::fmt::Display) -> String {
    tracing::error!("{}: {}", context, err);
    context.to_string()
}

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

    // ========== Integration Tests ==========

    use std::path::PathBuf;
    use std::sync::Mutex;
    use tokio::sync::Semaphore;

    /// Create a ServerContext with temp dirs for integration testing.
    async fn setup_test_context() -> (ServerContext, tempfile::TempDir) {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("data");
        let access_db_path = tmp.path().join("access.db");

        let embedder = Arc::new(crate::Embedder::new().unwrap());
        let project_id = Some("test-project-00000001".to_string());

        let ctx = ServerContext {
            db_path,
            access_db_path,
            project_id,
            embedder,
            cwd: PathBuf::from("."),
            consolidation_semaphore: Arc::new(Semaphore::new(1)),
            recall_count: std::sync::atomic::AtomicU64::new(0),
            rate_limit: Mutex::new(super::super::server::RateLimitState {
                window_start_ms: 0,
                count: 0,
            }),
        };

        (ctx, tmp)
    }

    #[tokio::test]
    #[ignore] // requires model download
    async fn test_store_and_recall_roundtrip() {
        let (ctx, _tmp) = setup_test_context().await;

        // Store an item
        let store_result = execute_tool(
            &ctx,
            "store",
            Some(json!({ "content": "Rust is a systems programming language" })),
        )
        .await;
        assert!(
            store_result.is_error.is_none(),
            "Store should succeed: {:?}",
            store_result.content
        );

        // Recall by query
        let recall_result = execute_tool(
            &ctx,
            "recall",
            Some(json!({ "query": "systems programming language" })),
        )
        .await;
        assert!(recall_result.is_error.is_none(), "Recall should succeed");

        let text = &recall_result.content[0].text;
        assert!(
            text.contains("Rust is a systems programming language"),
            "Recall should return stored content, got: {}",
            text
        );
    }

    #[tokio::test]
    #[ignore] // requires model download
    async fn test_store_and_list() {
        let (ctx, _tmp) = setup_test_context().await;

        // Store 2 items
        execute_tool(
            &ctx,
            "store",
            Some(json!({ "content": "First item for listing" })),
        )
        .await;
        execute_tool(
            &ctx,
            "store",
            Some(json!({ "content": "Second item for listing" })),
        )
        .await;

        // List items
        let list_result = execute_tool(&ctx, "list", Some(json!({ "scope": "project" }))).await;
        assert!(list_result.is_error.is_none(), "List should succeed");

        let text = &list_result.content[0].text;
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["count"], 2, "Should list 2 items");
    }

    #[tokio::test]
    #[ignore] // requires model download
    async fn test_store_conflict_detection() {
        let (ctx, _tmp) = setup_test_context().await;

        // Store first item
        execute_tool(
            &ctx,
            "store",
            Some(json!({ "content": "The quick brown fox jumps over the lazy dog" })),
        )
        .await;

        // Store nearly identical item
        let result = execute_tool(
            &ctx,
            "store",
            Some(json!({ "content": "The quick brown fox jumps over the lazy dog" })),
        )
        .await;
        assert!(result.is_error.is_none(), "Store should succeed");

        let text = &result.content[0].text;
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert!(
            parsed.get("potential_conflicts").is_some(),
            "Should detect conflict for near-duplicate content, got: {}",
            text
        );
    }

    #[tokio::test]
    #[ignore] // requires model download
    async fn test_forget_removes_item() {
        let (ctx, _tmp) = setup_test_context().await;

        // Store an item
        let store_result = execute_tool(
            &ctx,
            "store",
            Some(json!({ "content": "Item to be forgotten" })),
        )
        .await;
        let text = &store_result.content[0].text;
        let parsed: Value = serde_json::from_str(text).unwrap();
        let item_id = parsed["id"].as_str().unwrap().to_string();

        // Forget it
        let forget_result = execute_tool(&ctx, "forget", Some(json!({ "id": item_id }))).await;
        assert!(forget_result.is_error.is_none(), "Forget should succeed");

        // List should be empty
        let list_result = execute_tool(&ctx, "list", Some(json!({ "scope": "project" }))).await;
        let text = &list_result.content[0].text;
        assert!(
            text.contains("No items stored yet"),
            "Should have no items after forget, got: {}",
            text
        );
    }

    #[tokio::test]
    #[ignore] // requires model download
    async fn test_recall_empty_db() {
        let (ctx, _tmp) = setup_test_context().await;

        let result = execute_tool(&ctx, "recall", Some(json!({ "query": "anything" }))).await;
        assert!(
            result.is_error.is_none(),
            "Recall on empty DB should not error"
        );

        let text = &result.content[0].text;
        assert!(
            text.contains("No items found"),
            "Should indicate no items found, got: {}",
            text
        );
    }

    #[tokio::test]
    #[ignore] // requires model download
    async fn test_store_rejects_oversized_content() {
        let (ctx, _tmp) = setup_test_context().await;

        let large_content = "x".repeat(1_100_000); // >1MB
        let result = execute_tool(&ctx, "store", Some(json!({ "content": large_content }))).await;
        assert!(
            result.is_error == Some(true),
            "Should reject oversized content"
        );

        let text = &result.content[0].text;
        assert!(
            text.contains("too large"),
            "Error should mention size, got: {}",
            text
        );
    }

    #[tokio::test]
    #[ignore] // requires model download
    async fn test_recall_rejects_oversized_query() {
        let (ctx, _tmp) = setup_test_context().await;

        let large_query = "x".repeat(11_000); // >10KB
        let result = execute_tool(&ctx, "recall", Some(json!({ "query": large_query }))).await;
        assert!(
            result.is_error == Some(true),
            "Should reject oversized query"
        );

        let text = &result.content[0].text;
        assert!(
            text.contains("too large"),
            "Error should mention size, got: {}",
            text
        );
    }

    #[tokio::test]
    #[ignore] // requires model download
    async fn test_store_missing_params() {
        let (ctx, _tmp) = setup_test_context().await;

        // No params at all
        let result = execute_tool(&ctx, "store", None).await;
        assert!(result.is_error == Some(true), "Should error with no params");

        // Empty object (missing required 'content')
        let result = execute_tool(&ctx, "store", Some(json!({}))).await;
        assert!(
            result.is_error == Some(true),
            "Should error with missing content"
        );
    }
}
