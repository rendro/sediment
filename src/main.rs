//! Sediment MCP Server
//!
//! Semantic memory for AI agents - local-first, MCP-native.
//! Run this binary to start the MCP server.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::{EnvFilter, fmt};

#[derive(Parser)]
#[command(name = "sediment")]
#[command(about = "Semantic memory for AI agents - local-first, MCP-native")]
#[command(version)]
struct Cli {
    /// Database path (defaults to ~/.sediment/data)
    #[arg(long, global = true, env = "SEDIMENT_DB")]
    db: Option<PathBuf>,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

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

    /// List stored items for debugging
    List {
        /// Maximum number of items to show
        #[arg(short, long, default_value = "20")]
        limit: usize,
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
        Some(Commands::List { limit }) => run_list(cli.db, limit),
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

/// List stored items
fn run_list(db_override: Option<PathBuf>, limit: usize) -> Result<()> {
    let db_path = db_override.unwrap_or_else(sediment::central_db_path);

    if !db_path.exists() {
        println!("Database does not exist yet.");
        return Ok(());
    }

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let mut db = sediment::Database::open(&db_path).await?;
        let filters = sediment::ItemFilters::default();
        let items = db
            .list_items(filters, Some(limit), sediment::ListScope::All)
            .await?;

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
