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
use std::process::Command;
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
    /// Unique project identifier (git root commit hash or UUID)
    pub project_id: String,
    /// How the project ID was derived: "git-root-commit" or "uuid"
    #[serde(default = "default_source")]
    pub source: String,
}

fn default_source() -> String {
    "uuid".to_string()
}

/// Derive a stable project ID from the git repository's initial (root) commit hash.
///
/// Returns `Ok(Some(hash))` if the project root is inside a git repo with at least one commit.
/// Returns `Ok(None)` if git is not installed, the directory is not a git repo, or there are no commits.
pub fn derive_git_root_commit(project_root: &Path) -> std::io::Result<Option<String>> {
    let output = match Command::new("git")
        .args(["rev-list", "--max-parents=0", "HEAD"])
        .current_dir(project_root)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
    {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };

    if !output.status.success() {
        return Ok(None);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let hash = stdout.lines().next().unwrap_or("").trim();

    // Validate: must be non-empty hex, at most 64 chars (covers SHA-1 40 and SHA-256 64)
    if !hash.is_empty()
        && hash.len() <= 64
        && hash.chars().all(|c| c.is_ascii_hexdigit())
    {
        Ok(Some(hash.to_string()))
    } else {
        Ok(None)
    }
}

/// Get or create the project ID for a given project root.
///
/// The project ID is stored in `<project_root>/.sediment/config`.
/// If the project is inside a git repo, the ID is derived from the root commit hash
/// for stability across clones. Falls back to a random UUID for non-git directories.
pub fn get_or_create_project_id(project_root: &Path) -> std::io::Result<String> {
    let sediment_dir = project_root.join(".sediment");
    let config_path = sediment_dir.join("config");

    // Try to read existing config
    if config_path.exists() {
        let content = std::fs::read_to_string(&config_path)?;
        if let Ok(config) = serde_json::from_str::<ProjectConfig>(&content) {
            if config.source == "git-root-commit" {
                // Already derived from git — trust it
                return Ok(config.project_id);
            }

            // Source is "uuid" (or missing field from old config) — try to upgrade to git
            if let Ok(Some(git_hash)) = derive_git_root_commit(project_root) {
                let new_config = ProjectConfig {
                    project_id: git_hash.clone(),
                    source: "git-root-commit".to_string(),
                };
                write_config_atomic(&sediment_dir, &config_path, &new_config)?;
                return Ok(git_hash);
            }

            // Git derivation failed — keep existing UUID
            return Ok(config.project_id);
        }
    }

    // No existing config — create new one
    std::fs::create_dir_all(&sediment_dir)?;

    let config = if let Ok(Some(git_hash)) = derive_git_root_commit(project_root) {
        ProjectConfig {
            project_id: git_hash,
            source: "git-root-commit".to_string(),
        }
    } else {
        ProjectConfig {
            project_id: Uuid::new_v4().to_string(),
            source: "uuid".to_string(),
        }
    };

    write_config_atomic(&sediment_dir, &config_path, &config)?;

    // Re-read to return the ID that actually persisted (could be from another process)
    let final_content = std::fs::read_to_string(&config_path)?;
    if let Ok(final_config) = serde_json::from_str::<ProjectConfig>(&final_content) {
        Ok(final_config.project_id)
    } else {
        Ok(config.project_id)
    }
}

/// Write a ProjectConfig atomically via temp file + rename.
fn write_config_atomic(
    sediment_dir: &Path,
    config_path: &Path,
    config: &ProjectConfig,
) -> std::io::Result<()> {
    let content =
        serde_json::to_string_pretty(config).map_err(|e| std::io::Error::other(e.to_string()))?;
    let tmp_path = sediment_dir.join(format!("config.tmp.{}", std::process::id()));
    std::fs::write(&tmp_path, &content)?;

    if let Err(e) = std::fs::rename(&tmp_path, config_path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }
    Ok(())
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

    #[test]
    fn test_project_config_backward_compat() {
        // Config JSON without 'source' field should deserialize with source="uuid"
        let json = r#"{"project_id": "550e8400-e29b-41d4-a716-446655440000"}"#;
        let config: ProjectConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.source, "uuid");
    }

    #[test]
    #[ignore] // requires git
    fn test_derive_git_root_commit_in_repo() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path();

        // git init + commit
        Command::new("git").args(["init"]).current_dir(dir).output().unwrap();
        Command::new("git").args(["config", "user.email", "test@test.com"]).current_dir(dir).output().unwrap();
        Command::new("git").args(["config", "user.name", "Test"]).current_dir(dir).output().unwrap();
        Command::new("git").args(["commit", "--allow-empty", "-m", "init"]).current_dir(dir).output().unwrap();

        let result = derive_git_root_commit(dir).unwrap();
        assert!(result.is_some(), "Should return root commit hash");
        let hash = result.unwrap();
        assert_eq!(hash.len(), 40, "SHA-1 hash should be 40 chars");
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()), "Should be hex");
    }

    #[test]
    #[ignore] // requires git
    fn test_derive_git_root_commit_no_commits() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path();

        Command::new("git").args(["init"]).current_dir(dir).output().unwrap();

        let result = derive_git_root_commit(dir).unwrap();
        assert!(result.is_none(), "Repo with no commits should return None");
    }

    #[test]
    fn test_derive_git_root_commit_no_git() {
        let tmp = tempfile::TempDir::new().unwrap();
        let result = derive_git_root_commit(tmp.path()).unwrap();
        assert!(result.is_none(), "Non-git directory should return None");
    }

    #[test]
    #[ignore] // requires git
    fn test_project_id_from_git_root_commit() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path();

        Command::new("git").args(["init"]).current_dir(dir).output().unwrap();
        Command::new("git").args(["config", "user.email", "test@test.com"]).current_dir(dir).output().unwrap();
        Command::new("git").args(["config", "user.name", "Test"]).current_dir(dir).output().unwrap();
        Command::new("git").args(["commit", "--allow-empty", "-m", "init"]).current_dir(dir).output().unwrap();

        let project_id = get_or_create_project_id(dir).unwrap();
        let expected = derive_git_root_commit(dir).unwrap().unwrap();
        assert_eq!(project_id, expected, "Project ID should be the git root commit hash");

        // Verify config source
        let config_content = std::fs::read_to_string(dir.join(".sediment/config")).unwrap();
        let config: ProjectConfig = serde_json::from_str(&config_content).unwrap();
        assert_eq!(config.source, "git-root-commit");
    }

    #[test]
    #[ignore] // requires git
    fn test_project_id_migration_uuid_to_git() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path();

        // Write a UUID-based config first
        let sediment_dir = dir.join(".sediment");
        std::fs::create_dir_all(&sediment_dir).unwrap();
        let old_uuid = "550e8400-e29b-41d4-a716-446655440000";
        let old_config = format!(r#"{{"project_id": "{}"}}"#, old_uuid);
        std::fs::write(sediment_dir.join("config"), &old_config).unwrap();

        // Now create a git repo with a commit
        Command::new("git").args(["init"]).current_dir(dir).output().unwrap();
        Command::new("git").args(["config", "user.email", "test@test.com"]).current_dir(dir).output().unwrap();
        Command::new("git").args(["config", "user.name", "Test"]).current_dir(dir).output().unwrap();
        Command::new("git").args(["commit", "--allow-empty", "-m", "init"]).current_dir(dir).output().unwrap();

        // Calling get_or_create_project_id should migrate to git hash
        let project_id = get_or_create_project_id(dir).unwrap();
        let git_hash = derive_git_root_commit(dir).unwrap().unwrap();
        assert_eq!(project_id, git_hash, "Should migrate to git hash");

        // Config should now have git-root-commit source
        let config_content = std::fs::read_to_string(sediment_dir.join("config")).unwrap();
        let config: ProjectConfig = serde_json::from_str(&config_content).unwrap();
        assert_eq!(config.source, "git-root-commit");
    }
}
