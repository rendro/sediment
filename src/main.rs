//! Sediment MCP Server
//!
//! Semantic memory for AI agents - local-first, MCP-native.
//! Run this binary to start the MCP server.

use std::io::Read;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use serde_json::json;
use tracing_subscriber::{EnvFilter, fmt};

#[derive(Parser)]
#[command(name = "sediment")]
#[command(about = "Semantic memory for AI agents - local-first, MCP-native")]
#[command(version)]
#[command(after_long_help = "\
Environment variables:
  SEDIMENT_DB              Override database path (default: ~/.sediment/data).
                           Useful for ephemeral/isolated storage:
                             SEDIMENT_DB=/tmp/task-xyz sediment store \"...\"
  SEDIMENT_EMBEDDING_MODEL Override embedding model (default: all-MiniLM-L6-v2).
                           Options: all-MiniLM-L6-v2, bge-small-en-v1.5
")]
struct Cli {
    /// Database path [default: ~/.sediment/data]
    #[arg(long, global = true, env = "SEDIMENT_DB")]
    db: Option<PathBuf>,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Output JSON instead of human-readable text
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Set up Claude Code integration - writes CLAUDE.md instructions
    Init {
        /// Write instructions to local project CLAUDE.md (non-interactive)
        #[arg(long, conflicts_with = "global")]
        local: bool,

        /// Write instructions to global ~/.claude/CLAUDE.md (non-interactive)
        #[arg(long, conflicts_with = "local")]
        global: bool,
    },

    /// Show database statistics (item count, chunk count)
    Stats,

    /// List stored items
    List {
        /// Maximum number of items to show
        #[arg(short, long, default_value = "20")]
        limit: usize,

        /// Scope: "project" (default), "global", or "all"
        #[arg(short, long, default_value = "project")]
        scope: String,
    },

    /// Store content for later retrieval
    Store {
        /// Content to store (use "-" to read from stdin)
        content: String,

        /// Scope: "project" (default) or "global"
        #[arg(short, long, default_value = "project")]
        scope: String,

        /// ID of an existing item to replace
        #[arg(long)]
        replace: Option<String>,
    },

    /// Search stored content by semantic similarity
    Recall {
        /// Search query
        query: String,

        /// Maximum number of results
        #[arg(short, long, default_value = "5")]
        limit: usize,
    },

    /// Delete a stored item by its ID
    Forget {
        /// Item ID to delete
        id: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging to stderr (stdout is for MCP protocol)
    // For CLI commands, only show logs if verbose is enabled
    // For MCP server (no command), always show info logs
    let filter = if cli.verbose {
        EnvFilter::new("sediment=debug")
    } else if cli.command.is_none() {
        // MCP server mode - show info logs
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("sediment=info"))
    } else {
        // CLI command mode - suppress logs unless verbose
        EnvFilter::new("sediment=warn")
    };

    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .without_time()
        .with_writer(std::io::stderr)
        .init();

    match cli.command {
        None => run_mcp_server(cli.db),
        Some(Commands::Init { local, global }) => run_init(local, global),
        Some(Commands::Stats) => run_stats(cli.db),
        Some(Commands::List { limit, scope }) => run_list(cli.db, limit, &scope, cli.json),
        Some(Commands::Store {
            content,
            scope,
            replace,
        }) => run_store(cli.db, &content, &scope, replace, cli.json),
        Some(Commands::Recall { query, limit }) => run_recall(cli.db, &query, limit, cli.json),
        Some(Commands::Forget { id }) => run_forget(cli.db, &id, cli.json),
    }
}

/// Run the MCP server (default behavior)
fn run_mcp_server(db_override: Option<PathBuf>) -> Result<()> {
    // Get database path
    let db_path = db_override.unwrap_or_else(sediment::central_db_path);

    // Auto-detect project context from current directory
    // If no .git or .sediment marker found, use cwd as project root
    let cwd = std::env::current_dir().ok();
    let project_root = cwd
        .as_deref()
        .map(|dir| sediment::find_project_root(dir).unwrap_or_else(|| dir.to_path_buf()));
    let project_id = project_root
        .as_ref()
        .and_then(|root| sediment::get_or_create_project_id(root).ok());

    // Check for pending project ID migration (UUID → git hash)
    if let Some(ref root) = project_root
        && let Some(old_id) = sediment::pending_migration(root)
        && let Some(ref new_id) = project_id
    {
        tracing::info!("Migrating project ID: {} → {}", old_id, new_id);

        // Migrate LanceDB items
        let rt = tokio::runtime::Runtime::new()?;
        if let Err(e) = rt.block_on(sediment::db::migrate_project_id(&db_path, &old_id, new_id)) {
            tracing::warn!("Failed to migrate LanceDB items: {}", e);
        }

        // Migrate graph nodes
        let sediment_dir = db_path.parent().unwrap_or(&db_path);
        let access_db_path = sediment_dir.join("access.db");
        if let Ok(graph) = sediment::graph::GraphStore::open(&access_db_path)
            && let Err(e) = graph.migrate_project_id(&old_id, new_id)
        {
            tracing::warn!("Failed to migrate graph nodes: {}", e);
        }

        // Clear migration marker
        if let Err(e) = sediment::clear_migration_marker(root) {
            tracing::warn!("Failed to clear migration marker: {}", e);
        }
    }

    tracing::info!("Starting Sediment MCP server");
    tracing::info!("Database: {:?}", db_path);

    if let Some(ref root) = project_root {
        tracing::info!("Project: {:?}", root);
    }

    if let Some(ref pid) = project_id {
        tracing::info!("Project ID: {}", pid);
    }

    // Run MCP server
    sediment::mcp::run(&db_path, project_id)?;

    Ok(())
}

/// Check if a file contains Sediment instructions
fn has_sediment_instructions(path: &PathBuf) -> bool {
    if path.exists()
        && let Ok(content) = std::fs::read_to_string(path)
    {
        return content.contains("mcp__sediment__");
    }
    false
}

/// Initialize Claude Code integration
fn run_init(use_local: bool, use_global: bool) -> Result<()> {
    use dialoguer::{Confirm, Select};

    let cwd = std::env::current_dir()?;

    // Find or create project root
    let project_root = sediment::find_project_root(&cwd).unwrap_or_else(|| cwd.clone());

    println!("Initializing Sediment for: {}", project_root.display());

    // Create .sediment directory and project ID
    let sediment_dir = sediment::init_project(&project_root)?;
    println!("Created: {}", sediment_dir.display());

    // Determine CLAUDE.md paths
    let local_claude_md = project_root.join("CLAUDE.md");
    let global_claude_md = dirs::home_dir()
        .map(|h| h.join(".claude").join("CLAUDE.md"))
        .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?;

    // Check if instructions already exist in either location
    let local_has_instructions = has_sediment_instructions(&local_claude_md);
    let global_has_instructions = has_sediment_instructions(&global_claude_md);

    // If both already have instructions, we're done
    if local_has_instructions && global_has_instructions {
        println!("Sediment instructions already present in both local and global CLAUDE.md");
        println!("\nSediment initialized successfully!");
        println!("The MCP server will now auto-detect this project.");
        return Ok(());
    }

    // Determine target based on flags or interactive prompt
    let claude_md_path = if use_local {
        if local_has_instructions {
            println!("Local CLAUDE.md already contains Sediment instructions");
            println!("\nSediment initialized successfully!");
            println!("The MCP server will now auto-detect this project.");
            return Ok(());
        }
        Some(local_claude_md)
    } else if use_global {
        if global_has_instructions {
            println!("Global CLAUDE.md already contains Sediment instructions");
            println!("\nSediment initialized successfully!");
            println!("The MCP server will now auto-detect this project.");
            return Ok(());
        }
        Some(global_claude_md)
    } else {
        // Interactive mode - build options with status indicators
        let local_status = if local_has_instructions {
            " [already added]"
        } else {
            ""
        };
        let global_status = if global_has_instructions {
            " [already added]"
        } else {
            ""
        };

        let options = &[
            format!("Local  ({}){}", local_claude_md.display(), local_status),
            format!("Global ({}){}", global_claude_md.display(), global_status),
            "Skip".to_string(),
        ];

        // Default to the first option that doesn't already have instructions
        let default_idx = if local_has_instructions { 1 } else { 0 };

        let selection = Select::new()
            .with_prompt("Where should Sediment instructions be added?")
            .items(options)
            .default(default_idx)
            .interact()?;

        match selection {
            0 => {
                if local_has_instructions {
                    println!("Local CLAUDE.md already contains Sediment instructions");
                    None
                } else {
                    Some(local_claude_md)
                }
            }
            1 => {
                if global_has_instructions {
                    println!("Global CLAUDE.md already contains Sediment instructions");
                    None
                } else {
                    Some(global_claude_md)
                }
            }
            _ => None,
        }
    };

    let Some(claude_md_path) = claude_md_path else {
        println!("\nSediment initialized successfully!");
        println!("The MCP server will now auto-detect this project.");
        return Ok(());
    };

    let instructions = generate_claude_md_instructions();

    // Ensure parent directory exists for global path
    if let Some(parent) = claude_md_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    if claude_md_path.exists() {
        // Read existing content
        let existing = std::fs::read_to_string(&claude_md_path)?;

        // In non-interactive mode, always append; in interactive mode, ask
        let should_append = if use_local || use_global {
            true
        } else {
            Confirm::new()
                .with_prompt("CLAUDE.md exists. Append Sediment instructions?")
                .default(true)
                .interact()?
        };

        if should_append {
            let new_content = format!("{}\n\n{}", existing.trim(), instructions);
            std::fs::write(&claude_md_path, new_content)?;
            println!("Updated: {}", claude_md_path.display());
        } else {
            println!("Skipped CLAUDE.md update");
        }
    } else {
        std::fs::write(&claude_md_path, &instructions)?;
        println!("Created: {}", claude_md_path.display());
    }

    println!("\nSediment initialized successfully!");
    println!("The MCP server will now auto-detect this project.");

    Ok(())
}

/// Show database statistics
fn run_stats(db_override: Option<PathBuf>) -> Result<()> {
    let db_path = db_override.unwrap_or_else(sediment::central_db_path);

    println!("Database: {}", db_path.display());

    if !db_path.exists() {
        println!("\nDatabase does not exist yet.");
        println!("It will be created when you first store an item.");
        return Ok(());
    }

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let db = sediment::Database::open(&db_path).await?;
        let stats = db.stats().await?;

        println!("\nStatistics:");
        println!("  Items:  {}", stats.item_count);
        println!("  Chunks: {}", stats.chunk_count);

        Ok(())
    })
}

/// Shared context for CLI commands that need database access.
struct CliContext {
    db_path: PathBuf,
    access_db_path: PathBuf,
    project_id: Option<String>,
}

/// Build CLI context: resolve DB path, detect project, derive project ID.
fn cli_context(db_override: Option<PathBuf>) -> CliContext {
    let db_path = db_override.unwrap_or_else(sediment::central_db_path);
    let sediment_dir = db_path.parent().unwrap_or(&db_path);
    let access_db_path = sediment_dir.join("access.db");

    let cwd = std::env::current_dir().ok();
    let project_root = cwd
        .as_deref()
        .map(|dir| sediment::find_project_root(dir).unwrap_or_else(|| dir.to_path_buf()));
    let project_id = project_root
        .as_ref()
        .and_then(|root| sediment::get_or_create_project_id(root).ok());

    CliContext {
        db_path,
        access_db_path,
        project_id,
    }
}

/// List stored items
fn run_list(
    db_override: Option<PathBuf>,
    limit: usize,
    scope: &str,
    output_json: bool,
) -> Result<()> {
    let ctx = cli_context(db_override);

    if !ctx.db_path.exists() {
        if output_json {
            println!("{}", json!({"count": 0, "items": []}));
        } else {
            println!("Database does not exist yet.");
        }
        return Ok(());
    }

    let scope = scope
        .parse::<sediment::ListScope>()
        .map_err(|e| anyhow::anyhow!(e))?;

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let mut db = sediment::Database::open_with_project(&ctx.db_path, ctx.project_id).await?;
        let items = db.list_items(Some(limit), scope).await?;

        if output_json {
            let formatted: Vec<serde_json::Value> = items
                .iter()
                .map(|item| {
                    let mut obj = json!({
                        "id": item.id,
                        "content": item.content,
                        "created": item.created_at.to_rfc3339(),
                    });
                    if item.project_id.is_some() {
                        obj["scope"] = json!("project");
                    } else {
                        obj["scope"] = json!("global");
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
            println!("{}", serde_json::to_string_pretty(&result)?);
            return Ok(());
        }

        if items.is_empty() {
            println!("No items stored.");
            return Ok(());
        }

        println!("Stored items ({}):\n", items.len());

        for item in items {
            let scope = if item.project_id.is_some() {
                "project"
            } else {
                "global"
            };

            // Show truncated content
            let content_preview: String = item
                .content
                .chars()
                .take(80)
                .collect::<String>()
                .replace('\n', " ");
            let ellipsis = if item.content.chars().count() > 80 {
                "..."
            } else {
                ""
            };

            println!("  {} ({})", item.id, scope);
            println!("    {}{}", content_preview, ellipsis);
            println!();
        }

        Ok(())
    })
}

/// Store content
fn run_store(
    db_override: Option<PathBuf>,
    content: &str,
    scope: &str,
    replace: Option<String>,
    output_json: bool,
) -> Result<()> {
    let ctx = cli_context(db_override);

    // Read content from stdin if "-"
    let content = if content == "-" {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        buf
    } else {
        content.to_string()
    };

    if content.trim().is_empty() {
        anyhow::bail!("Content must not be empty");
    }

    let scope = scope
        .parse::<sediment::StoreScope>()
        .map_err(|e| anyhow::anyhow!(e))?;

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let mut db = sediment::Database::open_with_project(&ctx.db_path, ctx.project_id.clone()).await?;

        let mut item = sediment::Item::new(&content);

        // Set project_id based on scope
        if scope == sediment::StoreScope::Project
            && let Some(ref project_id) = ctx.project_id
        {
            item = item.with_project_id(project_id);
        }

        let store_result = db.store_item(item).await?;
        let new_id = store_result.id.clone();

        // Create graph node
        let graph = sediment::graph::GraphStore::open(&ctx.access_db_path)?;
        let now = chrono::Utc::now().timestamp();
        if let Err(e) = graph.add_node(&new_id, ctx.project_id.as_deref(), now) {
            tracing::warn!("graph add_node failed: {}", e);
        }

        // Enqueue consolidation candidates from conflicts
        if !store_result.potential_conflicts.is_empty()
            && let Ok(queue) =
                sediment::consolidation::ConsolidationQueue::open(&ctx.access_db_path)
        {
            for conflict in &store_result.potential_conflicts {
                if let Err(e) =
                    queue.enqueue(&new_id, &conflict.id, conflict.similarity as f64)
                {
                    tracing::warn!("enqueue consolidation failed: {}", e);
                }
            }
        }

        // Handle replace: delete old item, preserve graph lineage
        let mut replaced = false;
        if let Some(ref old_id) = replace {
            if !sediment::db::is_valid_id(old_id) {
                tracing::warn!("replace ID is not valid: {}", old_id);
            } else {
                if let Err(e) = graph.add_supersedes_edge(&new_id, old_id) {
                    tracing::warn!("replace: add_supersedes_edge failed: {}", e);
                }
                if let Err(e) = graph.transfer_edges(old_id, &new_id) {
                    tracing::warn!("replace: transfer_edges failed: {}", e);
                }

                // Record validation on the new item
                if let Ok(tracker) =
                    sediment::access::AccessTracker::open(&ctx.access_db_path)
                {
                    let created_at = chrono::Utc::now().timestamp();
                    if let Err(e) = tracker.record_validation(&new_id, created_at) {
                        tracing::warn!("replace: record_validation failed: {}", e);
                    }
                }

                match db.delete_item(old_id).await {
                    Ok(true) => replaced = true,
                    Ok(false) => tracing::warn!("replace: old item not found: {}", old_id),
                    Err(e) => tracing::warn!("replace: delete_item failed: {}", e),
                }
                if let Err(e) = graph.remove_node(old_id) {
                    tracing::warn!("replace: remove_node failed: {}", e);
                }
            }
        }

        if output_json {
            let mut result = json!({
                "success": true,
                "id": new_id,
                "scope": scope.to_string(),
            });

            if replaced {
                result["replaced_id"] = json!(replace);
            }

            if !store_result.potential_conflicts.is_empty() {
                let conflicts: Vec<serde_json::Value> = store_result
                    .potential_conflicts
                    .iter()
                    .map(|c| json!({"id": c.id, "content": c.content, "similarity": format!("{:.2}", c.similarity)}))
                    .collect();
                result["potential_conflicts"] = json!(conflicts);
            }

            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            let header = if replaced {
                format!("Stored (replaced {}):", replace.as_deref().unwrap_or(""))
            } else {
                "Stored:".to_string()
            };
            println!("{}", header);
            println!("  ID:    {}", new_id);
            println!("  Scope: {}", scope);

            if !store_result.potential_conflicts.is_empty() {
                println!("\n  Potential conflicts:");
                for c in &store_result.potential_conflicts {
                    let preview: String = c.content.chars().take(60).collect::<String>().replace('\n', " ");
                    println!("    {} (similarity: {:.2})", c.id, c.similarity);
                    println!("      {}", preview);
                }
            }
        }

        Ok(())
    })
}

/// Search stored content
fn run_recall(
    db_override: Option<PathBuf>,
    query: &str,
    limit: usize,
    output_json: bool,
) -> Result<()> {
    let ctx = cli_context(db_override);

    if !ctx.db_path.exists() {
        if output_json {
            println!("{}", json!({"count": 0, "results": []}));
        } else {
            println!("No items found matching your query.");
        }
        return Ok(());
    }

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let mut db =
            sediment::Database::open_with_project(&ctx.db_path, ctx.project_id.clone()).await?;

        let tracker = sediment::access::AccessTracker::open(&ctx.access_db_path)?;
        let graph = sediment::graph::GraphStore::open(&ctx.access_db_path)?;

        let filters = sediment::ItemFilters::new();
        let config = sediment::mcp::tools::RecallConfig {
            enable_background_tasks: false,
            ..Default::default()
        };

        let recall_result = sediment::mcp::tools::recall_pipeline(
            &mut db, &tracker, &graph, query, limit, filters, &config,
        )
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

        if output_json {
            // Batch-fetch neighbors for related_ids
            let all_result_ids: Vec<&str> = recall_result
                .results
                .iter()
                .map(|r| r.id.as_str())
                .collect();
            let neighbors_map = graph
                .get_neighbors_mapped(&all_result_ids, 0.5)
                .unwrap_or_default();

            let formatted: Vec<serde_json::Value> = recall_result
                .results
                .iter()
                .map(|r| {
                    let mut obj = json!({
                        "id": r.id,
                        "content": r.content,
                        "similarity": format!("{:.2}", r.similarity),
                        "created": r.created_at.to_rfc3339(),
                    });
                    if let Some(&raw_sim) = recall_result.raw_similarities.get(&r.id)
                        && (raw_sim - r.similarity).abs() > 0.001
                    {
                        obj["raw_similarity"] = json!(format!("{:.2}", raw_sim));
                    }
                    if let Some(ref excerpt) = r.relevant_excerpt {
                        obj["relevant_excerpt"] = json!(excerpt);
                    }
                    if let Some(ref current_pid) = ctx.project_id
                        && let Some(ref item_pid) = r.project_id
                        && item_pid != current_pid
                    {
                        obj["cross_project"] = json!(true);
                    }
                    if let Some(related) = neighbors_map.get(&r.id)
                        && !related.is_empty()
                    {
                        obj["related_ids"] = json!(related);
                    }
                    obj
                })
                .collect();

            let mut result = json!({
                "count": recall_result.results.len(),
                "results": formatted,
            });

            if !recall_result.graph_expanded.is_empty() {
                result["graph_expanded"] = json!(recall_result.graph_expanded);
            }
            if !recall_result.suggested.is_empty() {
                result["suggested"] = json!(recall_result.suggested);
            }

            println!("{}", serde_json::to_string_pretty(&result)?);
            return Ok(());
        }

        // Human-readable output
        if recall_result.results.is_empty() {
            println!("No items found matching your query.");
            return Ok(());
        }

        println!("Results ({}):\n", recall_result.results.len());
        for r in &recall_result.results {
            println!("  {} (similarity: {:.2})", r.id, r.similarity);

            // Show raw similarity if decay scoring changed it
            if let Some(&raw_sim) = recall_result.raw_similarities.get(&r.id)
                && (raw_sim - r.similarity).abs() > 0.001
            {
                println!("    raw similarity: {:.2}", raw_sim);
            }

            let content_preview: String = r
                .content
                .chars()
                .take(80)
                .collect::<String>()
                .replace('\n', " ");
            let ellipsis = if r.content.chars().count() > 80 {
                "..."
            } else {
                ""
            };
            println!("    {}{}", content_preview, ellipsis);

            if let Some(ref excerpt) = r.relevant_excerpt {
                let excerpt_preview: String = excerpt
                    .chars()
                    .take(80)
                    .collect::<String>()
                    .replace('\n', " ");
                let ellipsis = if excerpt.chars().count() > 80 {
                    "..."
                } else {
                    ""
                };
                println!("    excerpt: {}{}", excerpt_preview, ellipsis);
            }

            println!();
        }

        if !recall_result.graph_expanded.is_empty() {
            println!("Graph-expanded:");
            for entry in &recall_result.graph_expanded {
                if let Some(id) = entry.get("id").and_then(|v| v.as_str()) {
                    let rel = entry
                        .get("rel_type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("related");
                    println!("  {} (via {})", id, rel);
                }
            }
            println!();
        }

        if !recall_result.suggested.is_empty() {
            println!("Suggested:");
            for entry in &recall_result.suggested {
                if let Some(id) = entry.get("id").and_then(|v| v.as_str()) {
                    let reason = entry.get("reason").and_then(|v| v.as_str()).unwrap_or("");
                    println!("  {} — {}", id, reason);
                }
            }
            println!();
        }

        Ok(())
    })
}

/// Delete a stored item
fn run_forget(db_override: Option<PathBuf>, id: &str, output_json: bool) -> Result<()> {
    let ctx = cli_context(db_override);

    if !ctx.db_path.exists() {
        anyhow::bail!("Database does not exist yet.");
    }

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let db =
            sediment::Database::open_with_project(&ctx.db_path, ctx.project_id.clone()).await?;

        // Access control: verify the item belongs to the current project or is global
        if let Some(ref current_pid) = ctx.project_id {
            match db.get_item(id).await {
                Ok(Some(item)) => {
                    if let Some(ref item_pid) = item.project_id
                        && item_pid != current_pid
                    {
                        anyhow::bail!("Cannot delete item {} from a different project", id);
                    }
                }
                Ok(None) => anyhow::bail!("Item not found: {}", id),
                Err(e) => anyhow::bail!("Failed to look up item: {}", e),
            }
        }

        match db.delete_item(id).await {
            Ok(true) => {
                // Remove from graph
                let graph = sediment::graph::GraphStore::open(&ctx.access_db_path)?;
                if let Err(e) = graph.remove_node(id) {
                    tracing::warn!("remove_node failed: {}", e);
                }

                if output_json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json!({
                            "success": true,
                            "message": format!("Deleted item: {}", id),
                        }))?
                    );
                } else {
                    println!("Deleted item: {}", id);
                }
            }
            Ok(false) => anyhow::bail!("Item not found: {}", id),
            Err(e) => anyhow::bail!("Failed to delete item: {}", e),
        }

        Ok(())
    })
}

/// Generate CLAUDE.md instructions for Sediment
fn generate_claude_md_instructions() -> String {
    r#"# Sediment Memory System

Use `mcp__sediment__store`, `mcp__sediment__recall`, `mcp__sediment__list`, and `mcp__sediment__forget` for persistent memory.

## Proactive Usage

- **Recall at conversation start** when context might exist (preferences, prior decisions, project conventions)
- **Store immediately** when the user states a preference, makes a decision, or you learn something important about the codebase
- Use `scope: "global"` for cross-project preferences; default `"project"` scope for everything else

## What to Store

- User preferences and working style
- Architectural decisions and rationale
- Project conventions (naming, patterns, tools)
- Non-obvious learnings about the codebase

## What NOT to Store

- Transient task context (current debugging session, temporary workarounds)
- Information already in CLAUDE.md or README
- Obvious facts derivable from code
"#
    .to_string()
}
