//! Sediment: Semantic memory for AI agents
//!
//! A local-first, MCP-native vector database for AI agent memory.
//!
//! ## Features
//!
//! - **Embedded storage** - LanceDB-powered, directory-based, no server required
//! - **Local embeddings** - Uses `all-MiniLM-L6-v2` locally, no API keys needed
//! - **MCP-native** - 4 tools for seamless LLM integration
//! - **Project-aware** - Scoped memories with automatic project detection
//! - **Auto-chunking** - Long content is automatically chunked for better search

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use uuid::Uuid;

pub mod access;
pub mod chunker;
pub mod consolidation;
pub mod db;
pub mod document;
pub mod embedder;
pub mod error;
pub mod graph;
pub mod item;
pub mod mcp;
pub mod retry;

pub use chunker::{ChunkResult, ChunkingConfig, chunk_content};
pub use db::Database;
pub use document::ContentType;
pub use embedder::{EMBEDDING_DIM, Embedder};
pub use error::{Result, SedimentError};
pub use item::{Chunk, ConflictInfo, Item, ItemFilters, SearchResult, StoreResult};
pub use retry::{RetryConfig, with_retry};

/// Scope for storing items
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StoreScope {
    /// Store in project-local scope (with project_id)
    #[default]
    Project,
    /// Store in global scope (no project_id)
    Global,
}

impl std::str::FromStr for StoreScope {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "project" => Ok(StoreScope::Project),
            "global" => Ok(StoreScope::Global),
            _ => Err(format!(
                "Invalid store scope: {}. Use 'project' or 'global'",
                s
            )),
        }
    }
}

impl std::fmt::Display for StoreScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreScope::Project => write!(f, "project"),
            StoreScope::Global => write!(f, "global"),
        }
    }
}

/// Scope for listing items (recall always searches all with boosting)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ListScope {
    /// List only project-local items
    #[default]
    Project,
    /// List only global items
    Global,
    /// List all items
    All,
}

impl std::str::FromStr for ListScope {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "project" => Ok(ListScope::Project),
            "global" => Ok(ListScope::Global),
            "all" => Ok(ListScope::All),
            _ => Err(format!(
                "Invalid list scope: {}. Use 'project', 'global', or 'all'",
                s
            )),
        }
    }
}

impl std::fmt::Display for ListScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ListScope::Project => write!(f, "project"),
            ListScope::Global => write!(f, "global"),
            ListScope::All => write!(f, "all"),
        }
    }
}

/// Get the central database path.
///
/// Returns `~/.sediment/data` or the path specified in `SEDIMENT_DB` environment variable.
/// Note: LanceDB uses a directory, not a single file.
pub fn central_db_path() -> PathBuf {
    if let Ok(path) = std::env::var("SEDIMENT_DB") {
        return PathBuf::from(path);
    }

    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".sediment")
        .join("data")
}

/// Get the default global database path (alias for central_db_path for backwards compatibility)
pub fn default_db_path() -> PathBuf {
    central_db_path()
}

/// Project configuration stored in `.sediment/config`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    /// Unique project identifier (UUID)
    pub project_id: String,
}

impl Default for ProjectConfig {
    fn default() -> Self {
        Self {
            project_id: Uuid::new_v4().to_string(),
        }
    }
}

/// Get or create the project ID for a given project root.
///
/// The project ID is stored in `<project_root>/.sediment/config`.
/// If no config exists, a new UUID is generated and saved.
pub fn get_or_create_project_id(project_root: &Path) -> std::io::Result<String> {
    let sediment_dir = project_root.join(".sediment");
    let config_path = sediment_dir.join("config");

    // Try to read existing config
    if config_path.exists() {
        let content = std::fs::read_to_string(&config_path)?;
        if let Ok(config) = serde_json::from_str::<ProjectConfig>(&content) {
            return Ok(config.project_id);
        }
    }

    // Create new config with generated UUID
    let config = ProjectConfig::default();

    // Ensure .sediment directory exists
    std::fs::create_dir_all(&sediment_dir)?;

    // Write to a temp file first, then atomically rename to prevent TOCTOU races
    // where two concurrent processes both see the file as missing and write different UUIDs.
    let content =
        serde_json::to_string_pretty(&config).map_err(|e| std::io::Error::other(e.to_string()))?;
    let tmp_path = sediment_dir.join(format!("config.tmp.{}", std::process::id()));
    std::fs::write(&tmp_path, &content)?;

    // Atomic rename: on Unix this is atomic. The first writer wins; subsequent renames
    // just overwrite with a different UUID but that's acceptable since no data existed yet.
    // After rename, re-read to get whichever UUID actually persisted.
    if let Err(e) = std::fs::rename(&tmp_path, &config_path) {
        // Clean up temp file on failure
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }

    // Re-read to return the UUID that actually persisted (could be from another process)
    let final_content = std::fs::read_to_string(&config_path)?;
    if let Ok(final_config) = serde_json::from_str::<ProjectConfig>(&final_content) {
        Ok(final_config.project_id)
    } else {
        Ok(config.project_id)
    }
}

/// Apply similarity boosting based on project context.
///
/// - Same project: 1.15x boost (capped at 1.0)
/// - Different project: 0.95x penalty
/// - Global or no context: no change
pub fn boost_similarity(
    base: f32,
    mem_project: Option<&str>,
    current_project: Option<&str>,
) -> f32 {
    match (mem_project, current_project) {
        (Some(m), Some(c)) if m == c => (base * 1.15).min(1.0), // Same project: boost
        (Some(_), Some(_)) => base * 0.95,                      // Different project: slight penalty
        _ => base,                                              // Global or no context
    }
}

/// Find the project root by walking up from the given path.
///
/// Looks for directories containing `.sediment/` or `.git/` markers.
/// Returns `None` if no project root is found.
pub fn find_project_root(start: &Path) -> Option<PathBuf> {
    let mut current = start.to_path_buf();

    // If start is a file, use its parent directory
    if current.is_file() {
        current = current.parent()?.to_path_buf();
    }

    let mut depth = 0;
    loop {
        if depth >= 100 {
            return None;
        }
        depth += 1;

        // Check for .sediment directory first (explicit project marker)
        if current.join(".sediment").is_dir() {
            return Some(current);
        }

        // Check for .git directory as fallback
        if current.join(".git").exists() {
            return Some(current);
        }

        // Move to parent directory; stop at filesystem root
        match current.parent() {
            Some(parent) if parent == current => return None,
            Some(parent) => current = parent.to_path_buf(),
            None => return None,
        }
    }
}

/// Initialize a project directory for Sediment.
///
/// Creates the `.sediment/` directory in the specified path and generates a project ID.
pub fn init_project(project_root: &Path) -> std::io::Result<PathBuf> {
    let sediment_dir = project_root.join(".sediment");
    std::fs::create_dir_all(&sediment_dir)?;

    // Generate project ID
    get_or_create_project_id(project_root)?;

    Ok(sediment_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_scope_default_is_project() {
        // Fix #17: ListScope::default() should be Project, matching the tool schema default
        assert_eq!(ListScope::default(), ListScope::Project);
    }

    #[test]
    fn test_store_scope_default_is_project() {
        assert_eq!(StoreScope::default(), StoreScope::Project);
    }

    #[test]
    fn test_project_config_idempotent() {
        // Fix #18: get_or_create_project_id should return the same ID on repeated calls
        let tmp = tempfile::TempDir::new().unwrap();
        let id1 = get_or_create_project_id(tmp.path()).unwrap();
        let id2 = get_or_create_project_id(tmp.path()).unwrap();
        assert_eq!(id1, id2, "Repeated calls should return the same project ID");
    }

    #[test]
    fn test_boost_similarity() {
        assert!((boost_similarity(0.5, Some("p1"), Some("p1")) - 0.575).abs() < 0.001);
        assert!((boost_similarity(0.5, Some("p1"), Some("p2")) - 0.475).abs() < 0.001);
        assert!((boost_similarity(0.5, None, Some("p1")) - 0.5).abs() < 0.001);
    }
}
