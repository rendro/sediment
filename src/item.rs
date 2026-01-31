//! Unified Item type for semantic storage
//!
//! Items unify memories and documents into a single concept with automatic chunking.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
    /// Optional title (recommended for long content)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Tags for categorization
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Source attribution
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Custom JSON metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    /// Project ID (None for global items)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// Whether this item was chunked (internal)
    pub is_chunked: bool,
    /// When this item expires (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<DateTime<Utc>>,
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
            title: None,
            tags: Vec::new(),
            source: None,
            metadata: None,
            project_id: None,
            is_chunked: false,
            expires_at: None,
            created_at: Utc::now(),
        }
    }

    /// Set the title
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Set the source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Set custom metadata
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Set the project ID
    pub fn with_project_id(mut self, project_id: impl Into<String>) -> Self {
        self.project_id = Some(project_id.into());
        self
    }

    /// Set expiration time
    pub fn with_expires_at(mut self, expires_at: DateTime<Utc>) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Set the embedding
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = embedding;
        self
    }

    /// Mark as chunked
    pub fn with_chunked(mut self, is_chunked: bool) -> Self {
        self.is_chunked = is_chunked;
        self
    }

    /// Get the text to embed for this item
    /// For chunked items: title + first ~500 chars
    /// For non-chunked items: full content
    pub fn embedding_text(&self) -> String {
        if self.is_chunked {
            let preview: String = self.content.chars().take(500).collect();
            match &self.title {
                Some(title) => format!("{} {}", title, preview),
                None => preview,
            }
        } else {
            self.content.clone()
        }
    }

    /// Check if this item has expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            Utc::now() > expires_at
        } else {
            false
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
    /// Content (full if short, or title if chunked)
    pub content: String,
    /// Most relevant chunk content (if chunked)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relevant_excerpt: Option<String>,
    /// Composite score (similarity * freshness * frequency * trust_bonus, capped at 1.0)
    pub similarity: f32,
    /// Raw cosine similarity before decay/trust adjustments (None if no adjustments applied)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_similarity: Option<f32>,
    /// Tags
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Source attribution
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// When created
    pub created_at: DateTime<Utc>,
    /// Project ID (not serialized, used internally for cross-project checks and graph backfill)
    #[serde(skip)]
    pub project_id: Option<String>,
    /// Metadata (not serialized, used internally for cross-project provenance)
    #[serde(skip)]
    pub metadata: Option<Value>,
}

impl SearchResult {
    /// Create from an item (non-chunked)
    pub fn from_item(item: &Item, similarity: f32) -> Self {
        Self {
            id: item.id.clone(),
            content: item.content.clone(),
            relevant_excerpt: None,
            similarity,
            raw_similarity: None,
            tags: item.tags.clone(),
            source: item.source.clone(),
            created_at: item.created_at,
            project_id: item.project_id.clone(),
            metadata: item.metadata.clone(),
        }
    }

    /// Create from an item with chunk excerpt
    pub fn from_item_with_excerpt(item: &Item, similarity: f32, excerpt: String) -> Self {
        let content = item.title.clone().unwrap_or_else(|| {
            // For chunked items without title, show a preview
            item.content.chars().take(100).collect()
        });
        Self {
            id: item.id.clone(),
            content,
            relevant_excerpt: Some(excerpt),
            similarity,
            raw_similarity: None,
            tags: item.tags.clone(),
            source: item.source.clone(),
            created_at: item.created_at,
            project_id: item.project_id.clone(),
            metadata: item.metadata.clone(),
        }
    }
}

/// Filters for search/list queries
#[derive(Debug, Default, Clone)]
pub struct ItemFilters {
    /// Filter by tags (any match)
    pub tags: Option<Vec<String>>,
    /// Minimum similarity threshold (0.0-1.0)
    pub min_similarity: Option<f32>,
    /// Include expired items
    pub include_expired: bool,
}

impl ItemFilters {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = Some(tags);
        self
    }

    pub fn with_min_similarity(mut self, min_similarity: f32) -> Self {
        self.min_similarity = Some(min_similarity);
        self
    }

    pub fn include_expired(mut self, include: bool) -> Self {
        self.include_expired = include;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_item_creation() {
        let item = Item::new("Test content")
            .with_title("Test Title")
            .with_tags(vec!["tag1".to_string(), "tag2".to_string()])
            .with_source("test-source")
            .with_project_id("project-123");

        assert_eq!(item.content, "Test content");
        assert_eq!(item.title, Some("Test Title".to_string()));
        assert_eq!(item.tags, vec!["tag1", "tag2"]);
        assert_eq!(item.source, Some("test-source".to_string()));
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
        let item = Item::new("a".repeat(1000))
            .with_title("My Title")
            .with_chunked(true);
        let text = item.embedding_text();
        assert!(text.starts_with("My Title "));
        assert!(text.len() < 600);
    }

    #[test]
    fn test_item_expiration() {
        let expired = Item::new("Expired").with_expires_at(Utc::now() - chrono::Duration::hours(1));
        assert!(expired.is_expired());

        let valid = Item::new("Valid").with_expires_at(Utc::now() + chrono::Duration::hours(1));
        assert!(!valid.is_expired());
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
        let item = Item::new("Test content")
            .with_tags(vec!["test".to_string()])
            .with_source("test");

        let result = SearchResult::from_item(&item, 0.95);
        assert_eq!(result.content, "Test content");
        assert_eq!(result.similarity, 0.95);
        assert!(result.relevant_excerpt.is_none());
    }

    #[test]
    fn test_search_result_with_excerpt() {
        let item = Item::new("Long content here")
            .with_title("Document Title")
            .with_chunked(true);

        let result = SearchResult::from_item_with_excerpt(&item, 0.85, "relevant part".to_string());
        assert_eq!(result.content, "Document Title");
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
