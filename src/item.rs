//! Unified Item type for semantic storage
//!
//! Items unify memories and documents into a single concept with automatic chunking.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A unified item stored in Sediment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Item {
    /// Unique identifier (UUID)
    pub id: String,
    /// The actual content
    pub content: String,
    /// Vector embedding (not serialized to JSON output)
    #[serde(skip)]
    pub embedding: Vec<f32>,
    /// Project ID (None for global items)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// Whether this item was chunked (internal)
    pub is_chunked: bool,
    /// When this item was created
    pub created_at: DateTime<Utc>,
}

impl Item {
    /// Create a new item with content
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content: content.into(),
            embedding: Vec::new(),
            project_id: None,
            is_chunked: false,
            created_at: Utc::now(),
        }
    }

    /// Set the project ID
    pub fn with_project_id(mut self, project_id: impl Into<String>) -> Self {
        self.project_id = Some(project_id.into());
        self
    }

    /// Set the embedding
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = embedding;
        self
    }

    /// Override the creation timestamp (benchmark builds only)
    #[cfg(feature = "bench")]
    pub fn with_created_at(mut self, created_at: DateTime<Utc>) -> Self {
        self.created_at = created_at;
        self
    }

    /// Mark as chunked
    pub fn with_chunked(mut self, is_chunked: bool) -> Self {
        self.is_chunked = is_chunked;
        self
    }

    /// Get the text to embed for this item
    /// For chunked items: first ~500 chars
    /// For non-chunked items: full content
    pub fn embedding_text(&self) -> String {
        if self.is_chunked {
            self.content.chars().take(500).collect()
        } else {
            self.content.clone()
        }
    }
}

/// A chunk of an item (internal, not exposed to MCP)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique identifier (UUID)
    pub id: String,
    /// Parent item ID
    pub item_id: String,
    /// Index of this chunk within the item (0-based)
    pub chunk_index: usize,
    /// The chunk content
    pub content: String,
    /// Vector embedding of the chunk (not serialized)
    #[serde(skip)]
    pub embedding: Vec<f32>,
    /// Optional context (e.g., header path, function name)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

impl Chunk {
    /// Create a new chunk
    pub fn new(item_id: impl Into<String>, chunk_index: usize, content: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            item_id: item_id.into(),
            chunk_index,
            content: content.into(),
            embedding: Vec::new(),
            context: None,
        }
    }

    /// Set context
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Set the embedding
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = embedding;
        self
    }
}

/// Result from storing an item
#[derive(Debug, Clone, Serialize)]
pub struct StoreResult {
    /// The ID of the newly stored item
    pub id: String,
    /// Potentially conflicting items (high similarity)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub potential_conflicts: Vec<ConflictInfo>,
}

/// Information about a potential conflict
#[derive(Debug, Clone, Serialize)]
pub struct ConflictInfo {
    /// The ID of the conflicting item
    pub id: String,
    /// The content of the conflicting item
    pub content: String,
    /// Similarity score (0.0-1.0)
    pub similarity: f32,
}

/// Result from a search query
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    /// The matching item's id
    pub id: String,
    /// Content (full if short, or preview if chunked)
    pub content: String,
    /// Most relevant chunk content (if chunked)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relevant_excerpt: Option<String>,
    /// Similarity score (0.0-1.0, higher is more similar)
    pub similarity: f32,
    /// When created
    pub created_at: DateTime<Utc>,
    /// Project ID (not serialized, used internally for cross-project checks and graph backfill)
    #[serde(skip)]
    pub project_id: Option<String>,
}

impl SearchResult {
    /// Create from an item (non-chunked)
    pub fn from_item(item: &Item, similarity: f32) -> Self {
        Self {
            id: item.id.clone(),
            content: item.content.clone(),
            relevant_excerpt: None,
            similarity,
            created_at: item.created_at,
            project_id: item.project_id.clone(),
        }
    }

    /// Create from an item with chunk excerpt
    pub fn from_item_with_excerpt(item: &Item, similarity: f32, excerpt: String) -> Self {
        // For chunked items, show a preview of the content
        let content: String = item.content.chars().take(100).collect();
        Self {
            id: item.id.clone(),
            content,
            relevant_excerpt: Some(excerpt),
            similarity,
            created_at: item.created_at,
            project_id: item.project_id.clone(),
        }
    }
}

/// Filters for search/list queries
#[derive(Debug, Default, Clone)]
pub struct ItemFilters {
    /// Minimum similarity threshold (0.0-1.0)
    pub min_similarity: Option<f32>,
}

impl ItemFilters {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_min_similarity(mut self, min_similarity: f32) -> Self {
        self.min_similarity = Some(min_similarity);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_item_creation() {
        let item = Item::new("Test content").with_project_id("project-123");

        assert_eq!(item.content, "Test content");
        assert_eq!(item.project_id, Some("project-123".to_string()));
        assert!(!item.is_chunked);
    }

    #[test]
    fn test_embedding_text_short() {
        let item = Item::new("Short content");
        assert_eq!(item.embedding_text(), "Short content");
    }

    #[test]
    fn test_embedding_text_chunked() {
        let item = Item::new("a".repeat(1000)).with_chunked(true);
        let text = item.embedding_text();
        assert_eq!(text.len(), 500);
    }

    #[test]
    fn test_chunk_creation() {
        let chunk = Chunk::new("item-123", 0, "Chunk content").with_context("## Header");

        assert_eq!(chunk.item_id, "item-123");
        assert_eq!(chunk.chunk_index, 0);
        assert_eq!(chunk.content, "Chunk content");
        assert_eq!(chunk.context, Some("## Header".to_string()));
    }

    #[test]
    fn test_search_result_from_item() {
        let item = Item::new("Test content");

        let result = SearchResult::from_item(&item, 0.95);
        assert_eq!(result.content, "Test content");
        assert_eq!(result.similarity, 0.95);
        assert!(result.relevant_excerpt.is_none());
    }

    #[test]
    fn test_search_result_with_excerpt() {
        let item = Item::new("Long content here").with_chunked(true);

        let result = SearchResult::from_item_with_excerpt(&item, 0.85, "relevant part".to_string());
        assert_eq!(result.content, "Long content here");
        assert_eq!(result.relevant_excerpt, Some("relevant part".to_string()));
    }

    #[test]
    fn test_store_result_serialization() {
        let result = StoreResult {
            id: "abc123".to_string(),
            potential_conflicts: vec![],
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("abc123"));
        // Empty conflicts should not be serialized
        assert!(!json.contains("potential_conflicts"));
    }

    #[test]
    fn test_store_result_with_conflicts() {
        let result = StoreResult {
            id: "new-id".to_string(),
            potential_conflicts: vec![ConflictInfo {
                id: "old-id".to_string(),
                content: "Old content".to_string(),
                similarity: 0.92,
            }],
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("new-id"));
        assert!(json.contains("potential_conflicts"));
        assert!(json.contains("old-id"));
        assert!(json.contains("0.92"));
    }

    #[test]
    fn test_conflict_info_serialization() {
        let conflict = ConflictInfo {
            id: "conflict-123".to_string(),
            content: "Conflicting content".to_string(),
            similarity: 0.87,
        };

        let json = serde_json::to_string(&conflict).unwrap();
        assert!(json.contains("conflict-123"));
        assert!(json.contains("Conflicting content"));
        assert!(json.contains("0.87"));
    }
}
