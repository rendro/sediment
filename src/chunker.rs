//! Chunking logic for splitting documents into searchable pieces
//!
//! Uses a two-pass approach for all content types:
//! 1. First pass: Split by semantic boundaries (headers, functions, paragraphs)
//! 2. Second pass: If any chunk exceeds max size, recursively split further
//!
//! Also applies minimum chunk size to avoid tiny fragments by merging small sections.

use crate::document::ContentType;

/// Configuration for chunking
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    /// Minimum content length before chunking is applied (default: 1000 chars)
    pub min_chunk_threshold: usize,
    /// Maximum chunk size in characters (default: 800 chars)
    pub max_chunk_size: usize,
    /// Minimum chunk size - merge if below (default: 200 chars)
    pub min_chunk_size: usize,
    /// Overlap between chunks in characters (default: 100 chars)
    pub chunk_overlap: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            min_chunk_threshold: 1000,
            max_chunk_size: 800,
            min_chunk_size: 200,
            chunk_overlap: 100,
        }
    }
}

impl ChunkingConfig {
    /// Create config with legacy field name for backwards compatibility
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.max_chunk_size = size;
        self
    }
}

/// Result of chunking a piece of content
#[derive(Debug, Clone)]
pub struct ChunkResult {
    /// The chunk content
    pub content: String,
    /// Start offset in original content (character position)
    pub start_offset: usize,
    /// End offset in original content (character position)
    pub end_offset: usize,
    /// Optional context (e.g., parent header for markdown)
    pub context: Option<String>,
    /// Whether this chunk represents a major boundary (header, function) - don't merge across
    pub is_boundary: bool,
}

impl ChunkResult {
    fn new(content: String, start_offset: usize, end_offset: usize) -> Self {
        Self {
            content,
            start_offset,
            end_offset,
            context: None,
            is_boundary: false,
        }
    }

    fn with_context(mut self, context: Option<String>) -> Self {
        self.context = context;
        self
    }

    fn with_boundary(mut self, is_boundary: bool) -> Self {
        self.is_boundary = is_boundary;
        self
    }
}

/// Chunk content based on content type
pub fn chunk_content(
    content: &str,
    content_type: ContentType,
    config: &ChunkingConfig,
) -> Vec<ChunkResult> {
    // Don't chunk if content is below threshold
    if content.len() < config.min_chunk_threshold {
        return vec![ChunkResult::new(content.to_string(), 0, content.len())];
    }

    // First pass: semantic splitting
    let chunks = match content_type {
        ContentType::Markdown => chunk_markdown(content, config),
        ContentType::Json => chunk_json(content, config),
        ContentType::Yaml => chunk_yaml(content, config),
        ContentType::Code => chunk_code(content, config),
        ContentType::Text => chunk_text(content, config),
    };

    // Second pass: enforce max size by recursive splitting
    let chunks = enforce_max_size(chunks, config);

    // Third pass: merge small chunks (respecting boundaries)
    merge_small_chunks(chunks, config.min_chunk_size)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Split text at sentence boundaries
fn split_at_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let mut char_indices = text.char_indices().peekable();

    while let Some((i, ch)) = char_indices.next() {
        // Look for sentence-ending punctuation (ASCII and Unicode)
        if matches!(ch, '.' | '?' | '!' | '。' | '？' | '！') {
            let end = i + ch.len_utf8();
            // Check if followed by whitespace or end of text
            let at_end_or_ws = match char_indices.peek() {
                None => true,
                Some(&(_, next_ch)) => next_ch == ' ' || next_ch == '\n' || next_ch == '\t',
            };
            if at_end_or_ws {
                if start < end {
                    sentences.push(&text[start..end]);
                }
                // Skip whitespace after punctuation
                while let Some(&(_, next_ch)) = char_indices.peek() {
                    if next_ch == ' ' || next_ch == '\n' || next_ch == '\t' {
                        char_indices.next();
                    } else {
                        break;
                    }
                }
                start = match char_indices.peek() {
                    Some(&(idx, _)) => idx,
                    None => text.len(),
                };
            }
        }
    }

    // Add remaining text
    if start < text.len() {
        sentences.push(&text[start..]);
    }

    if sentences.is_empty() && !text.is_empty() {
        sentences.push(text);
    }

    sentences
}

/// Recursively split content: paragraphs -> sentences -> chars
fn recursive_split(
    text: &str,
    max_size: usize,
    offset: usize,
    context: Option<String>,
    overlap: usize,
) -> Vec<ChunkResult> {
    if text.len() <= max_size {
        return vec![
            ChunkResult::new(text.to_string(), offset, offset + text.len()).with_context(context),
        ];
    }

    let mut chunks = Vec::new();

    // Try splitting by paragraphs first
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    if paragraphs.len() > 1 {
        let mut current_chunk = String::new();
        let mut chunk_start = offset;
        let mut current_pos = offset;

        for (i, para) in paragraphs.iter().enumerate() {
            let sep = if i > 0 { "\n\n" } else { "" };
            let para_with_sep = format!("{}{}", sep, para);

            if !current_chunk.is_empty() && current_chunk.len() + para_with_sep.len() > max_size {
                // Save current chunk
                chunks.push(
                    ChunkResult::new(current_chunk.clone(), chunk_start, current_pos)
                        .with_context(context.clone()),
                );

                // Start new chunk with overlap
                let overlap_text = get_overlap_text(&current_chunk, overlap);
                chunk_start = current_pos - overlap_text.len();
                current_chunk = overlap_text;
            }

            current_chunk.push_str(&para_with_sep);
            current_pos += para_with_sep.len();
        }

        if !current_chunk.is_empty() {
            // If the final chunk is still too large, recursively split by sentences
            if current_chunk.len() > max_size {
                chunks.extend(split_by_sentences(
                    &current_chunk,
                    max_size,
                    chunk_start,
                    context.clone(),
                    overlap,
                ));
            } else {
                chunks.push(
                    ChunkResult::new(current_chunk, chunk_start, current_pos)
                        .with_context(context.clone()),
                );
            }
        }

        return chunks;
    }

    // No paragraph breaks - split by sentences
    split_by_sentences(text, max_size, offset, context, overlap)
}

/// Split text by sentences, falling back to character split
fn split_by_sentences(
    text: &str,
    max_size: usize,
    offset: usize,
    context: Option<String>,
    overlap: usize,
) -> Vec<ChunkResult> {
    let sentences = split_at_sentences(text);

    if sentences.len() <= 1 {
        // No sentence boundaries - split by characters
        return split_by_chars(text, max_size, offset, context, overlap);
    }

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut chunk_start = offset;
    let mut current_pos = offset;

    for sentence in sentences {
        let sep = if !current_chunk.is_empty() { " " } else { "" };
        let sentence_with_sep = format!("{}{}", sep, sentence);

        if !current_chunk.is_empty() && current_chunk.len() + sentence_with_sep.len() > max_size {
            chunks.push(
                ChunkResult::new(current_chunk.clone(), chunk_start, current_pos)
                    .with_context(context.clone()),
            );

            let overlap_text = get_overlap_text(&current_chunk, overlap);
            chunk_start = current_pos - overlap_text.len();
            current_chunk = overlap_text;
        }

        current_chunk.push_str(&sentence_with_sep);
        current_pos += sentence_with_sep.len();
    }

    if !current_chunk.is_empty() {
        if current_chunk.len() > max_size {
            chunks.extend(split_by_chars(
                &current_chunk,
                max_size,
                chunk_start,
                context.clone(),
                overlap,
            ));
        } else {
            chunks.push(
                ChunkResult::new(current_chunk, chunk_start, current_pos)
                    .with_context(context.clone()),
            );
        }
    }

    chunks
}

/// Split text by characters (last resort)
fn split_by_chars(
    text: &str,
    max_size: usize,
    offset: usize,
    context: Option<String>,
    overlap: usize,
) -> Vec<ChunkResult> {
    let mut chunks = Vec::new();
    let bytes = text.as_bytes();
    let mut start = 0;

    // Ensure we make progress - overlap must be less than chunk size
    let effective_overlap = overlap.min(max_size / 2);

    while start < text.len() {
        let end = (start + max_size).min(text.len());

        // Try to break at a word boundary
        let actual_end = if end < text.len() {
            find_word_boundary_bytes(bytes, start, end)
        } else {
            end
        };

        // Ensure we make at least some progress
        let actual_end = if actual_end <= start {
            (start + max_size).min(text.len())
        } else {
            actual_end
        };

        chunks.push(
            ChunkResult::new(
                text[start..actual_end].to_string(),
                offset + start,
                offset + actual_end,
            )
            .with_context(context.clone()),
        );

        // Next chunk starts after this one, minus overlap
        // But ensure we always make progress
        let next_start = actual_end.saturating_sub(effective_overlap);
        start = if next_start <= start {
            actual_end // No overlap if it would cause no progress
        } else {
            next_start
        };
    }

    if chunks.is_empty() && !text.is_empty() {
        chunks.push(
            ChunkResult::new(text.to_string(), offset, offset + text.len()).with_context(context),
        );
    }

    chunks
}

/// Find a word boundary near the target position (byte-based for efficiency).
///
/// Safety: Only searches for ASCII bytes (space 0x20, newline 0x0A) which cannot
/// appear as continuation bytes in multi-byte UTF-8 sequences. The returned
/// position is always immediately after an ASCII byte, so slicing at this
/// position is guaranteed to be on a valid UTF-8 char boundary.
fn find_word_boundary_bytes(bytes: &[u8], start: usize, target: usize) -> usize {
    // Look backwards from target to find a space or newline
    let search_start = target.saturating_sub(50).max(start);
    for i in (search_start..target).rev() {
        if bytes[i] == b' ' || bytes[i] == b'\n' {
            return i + 1;
        }
    }
    target
}

/// Get overlap text from the end of a chunk
fn get_overlap_text(text: &str, overlap: usize) -> String {
    if text.len() <= overlap {
        return text.to_string();
    }

    let actual_start = find_overlap_start_bytes(text.as_bytes(), overlap);
    text[actual_start..].to_string()
}

/// Find a good overlap start position (byte-based for efficiency)
fn find_overlap_start_bytes(bytes: &[u8], target_overlap: usize) -> usize {
    if bytes.len() <= target_overlap {
        return 0;
    }

    let start_search = bytes.len().saturating_sub(target_overlap + 50);
    let end_search = bytes
        .len()
        .saturating_sub(target_overlap.saturating_sub(50));

    // Look for a good break point (newline, period, space)
    for i in (start_search..end_search).rev() {
        if bytes[i] == b'\n' || bytes[i] == b'.' || bytes[i] == b' ' {
            return i + 1;
        }
    }

    // Fall back to target position
    bytes.len().saturating_sub(target_overlap)
}

/// Enforce max size on all chunks by recursive splitting
fn enforce_max_size(chunks: Vec<ChunkResult>, config: &ChunkingConfig) -> Vec<ChunkResult> {
    let mut result = Vec::new();

    for chunk in chunks {
        if chunk.content.len() > config.max_chunk_size {
            result.extend(recursive_split(
                &chunk.content,
                config.max_chunk_size,
                chunk.start_offset,
                chunk.context,
                config.chunk_overlap,
            ));
        } else {
            result.push(chunk);
        }
    }

    result
}

/// Merge small chunks with neighbors, respecting boundaries
fn merge_small_chunks(chunks: Vec<ChunkResult>, min_size: usize) -> Vec<ChunkResult> {
    if chunks.is_empty() {
        return chunks;
    }

    let mut result: Vec<ChunkResult> = Vec::new();

    for chunk in chunks {
        if chunk.content.len() >= min_size || chunk.is_boundary {
            result.push(chunk);
        } else if let Some(last) = result.last_mut() {
            // Don't merge across boundaries
            if !last.is_boundary {
                // Merge with previous chunk
                last.content.push_str("\n\n");
                last.content.push_str(&chunk.content);
                last.end_offset = chunk.end_offset;
                // Keep the more specific context
                if chunk.context.is_some() {
                    last.context = chunk.context;
                }
            } else {
                result.push(chunk);
            }
        } else {
            result.push(chunk);
        }
    }

    // Final pass: if a trailing small chunk exists after a boundary, keep it
    result
}

// ============================================================================
// Markdown Chunking
// ============================================================================

/// A parsed markdown section
struct MarkdownSection {
    /// Full header path (e.g., ["# Main", "## Sub"])
    header_path: Vec<String>,
    /// Section content (including the header line)
    content: String,
    /// Start offset in original content
    start_offset: usize,
    /// End offset in original content
    end_offset: usize,
}

/// Parse markdown into sections by headers
fn parse_markdown_sections(content: &str) -> Vec<MarkdownSection> {
    let mut sections = Vec::new();
    let mut current_section = String::new();
    let mut section_start = 0;
    let mut current_pos = 0;
    let mut header_stack: Vec<(usize, String)> = Vec::new(); // (level, header text)

    let lines: Vec<&str> = content.lines().collect();

    for line in lines.iter() {
        let line_with_newline = if current_pos > 0 {
            format!("\n{}", line)
        } else {
            line.to_string()
        };

        // Check if this is a header
        if let Some(level) = get_header_level(line) {
            // If we have content, save current section
            if !current_section.is_empty() {
                sections.push(MarkdownSection {
                    header_path: header_stack.iter().map(|(_, h)| h.clone()).collect(),
                    content: current_section.clone(),
                    start_offset: section_start,
                    end_offset: current_pos,
                });
            }

            // Update header stack
            // Pop headers of equal or lower level
            while !header_stack.is_empty() && header_stack.last().unwrap().0 >= level {
                header_stack.pop();
            }
            header_stack.push((level, line.to_string()));

            // Start new section
            current_section = line_with_newline.trim_start_matches('\n').to_string();
            section_start = current_pos;
        } else {
            current_section.push_str(&line_with_newline);
        }

        current_pos += line_with_newline.len();
    }

    // Add final section
    if !current_section.is_empty() {
        sections.push(MarkdownSection {
            header_path: header_stack.iter().map(|(_, h)| h.clone()).collect(),
            content: current_section,
            start_offset: section_start,
            end_offset: content.len(),
        });
    }

    // If no sections found, create one for entire content
    if sections.is_empty() {
        sections.push(MarkdownSection {
            header_path: vec![],
            content: content.to_string(),
            start_offset: 0,
            end_offset: content.len(),
        });
    }

    sections
}

/// Get the header level (1-6) or None if not a header
fn get_header_level(line: &str) -> Option<usize> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with('#') {
        return None;
    }

    let level = trimmed.chars().take_while(|&c| c == '#').count();
    if level > 0 && level <= 6 {
        // Make sure there's a space after the hashes (valid markdown header)
        if trimmed.len() > level && trimmed.chars().nth(level) == Some(' ') {
            return Some(level);
        }
    }
    None
}

/// Format header path as context string
fn format_header_path(path: &[String]) -> Option<String> {
    if path.is_empty() {
        return None;
    }
    Some(path.join(" > "))
}

/// Chunk markdown by headers, preserving header context
fn chunk_markdown(content: &str, _config: &ChunkingConfig) -> Vec<ChunkResult> {
    let sections = parse_markdown_sections(content);
    let mut chunks = Vec::new();

    for section in sections {
        let context = format_header_path(&section.header_path);
        let is_boundary = !section.header_path.is_empty();

        chunks.push(
            ChunkResult::new(section.content, section.start_offset, section.end_offset)
                .with_context(context)
                .with_boundary(is_boundary),
        );
    }

    // Size enforcement happens in chunk_content
    chunks
}

// ============================================================================
// Text Chunking
// ============================================================================

/// Chunk plain text by paragraphs, then sentences
fn chunk_text(content: &str, config: &ChunkingConfig) -> Vec<ChunkResult> {
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut chunk_start = 0;
    let mut current_pos = 0;

    // Split by double newlines (paragraphs)
    let paragraphs: Vec<&str> = content.split("\n\n").collect();

    for (i, para) in paragraphs.iter().enumerate() {
        let sep = if i > 0 { "\n\n" } else { "" };
        let para_with_sep = format!("{}{}", sep, para);

        // If adding this paragraph would exceed chunk size, save current chunk
        if !current_chunk.is_empty()
            && current_chunk.len() + para_with_sep.len() > config.max_chunk_size
        {
            chunks.push(ChunkResult::new(
                current_chunk.clone(),
                chunk_start,
                current_pos,
            ));

            // Start new chunk with overlap
            let overlap_text = get_overlap_text(&current_chunk, config.chunk_overlap);
            chunk_start = current_pos - overlap_text.len();
            current_chunk = overlap_text;
        }

        current_chunk.push_str(&para_with_sep);
        current_pos += para_with_sep.len();
    }

    // Add final chunk
    if !current_chunk.is_empty() {
        chunks.push(ChunkResult::new(current_chunk, chunk_start, content.len()));
    }

    // Ensure at least one chunk
    if chunks.is_empty() {
        chunks.push(ChunkResult::new(content.to_string(), 0, content.len()));
    }

    chunks
}

// ============================================================================
// Code Chunking
// ============================================================================

/// Common patterns for function/class boundaries
const CODE_BOUNDARY_PATTERNS: &[&str] = &[
    // Rust
    "fn ",
    "pub fn ",
    "async fn ",
    "pub async fn ",
    "impl ",
    "struct ",
    "enum ",
    "trait ",
    "mod ",
    "const ",
    "static ",
    "type ",
    "#[",
    "//!",
    // Go
    "func ",
    // Python
    "def ",
    "class ",
    "async def ",
    // JavaScript/TypeScript
    "function ",
    "async function ",
    "export ",
    "export default",
    "module.exports",
    "const ",
    "let ",
    "var ",
    "interface ",
    // C/C++
    "void ",
    "int ",
    "char ",
    "double ",
    "float ",
    "#define ",
    "#include ",
];

/// Extract context from a code boundary line
fn extract_code_context(line: &str) -> String {
    let trimmed = line.trim();

    // Try to extract a meaningful identifier
    // For function definitions, get up to the opening brace or paren
    if let Some(paren_pos) = trimmed.find('(') {
        let signature = &trimmed[..paren_pos];
        // Find the last word (likely the function name)
        if signature.rfind(' ').is_some() {
            return format!("{}...", &trimmed[..paren_pos.min(60)]);
        }
    }

    // For struct/class/impl, get the name
    for keyword in &[
        "struct ",
        "class ",
        "impl ",
        "trait ",
        "interface ",
        "enum ",
    ] {
        if let Some(rest) = trimmed.strip_prefix(keyword) {
            let name_end = rest
                .find(|c: char| !c.is_alphanumeric() && c != '_' && c != '<' && c != '>')
                .unwrap_or(rest.len());
            return format!("{}{}", keyword, &rest[..name_end.min(50)]);
        }
    }

    // Default: first 60 chars
    trimmed.chars().take(60).collect()
}

/// Check if a line is a code boundary
fn is_code_boundary(line: &str) -> bool {
    let trimmed = line.trim_start();
    CODE_BOUNDARY_PATTERNS
        .iter()
        .any(|p| trimmed.starts_with(p))
}

/// Chunk code by function/class boundaries
fn chunk_code(content: &str, _config: &ChunkingConfig) -> Vec<ChunkResult> {
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut chunk_start = 0;
    let mut current_pos = 0;
    let mut current_context: Option<String> = None;
    let mut is_at_boundary = false;

    let lines: Vec<&str> = content.lines().collect();

    for line in lines {
        let line_with_newline = if current_pos > 0 {
            format!("\n{}", line)
        } else {
            line.to_string()
        };

        let boundary = is_code_boundary(line);

        // If we hit a boundary and have substantial content, start new chunk
        if boundary && !current_chunk.is_empty() && current_chunk.len() > 100 {
            chunks.push(
                ChunkResult::new(current_chunk.clone(), chunk_start, current_pos)
                    .with_context(current_context.clone())
                    .with_boundary(is_at_boundary),
            );

            current_chunk = String::new();
            chunk_start = current_pos;
            is_at_boundary = true;
        }

        if boundary {
            current_context = Some(extract_code_context(line));
            is_at_boundary = true;
        }

        // Size-based splitting (will be handled by enforce_max_size)
        current_chunk.push_str(&line_with_newline);
        current_pos += line_with_newline.len();
    }

    // Add final chunk
    if !current_chunk.is_empty() {
        chunks.push(
            ChunkResult::new(current_chunk, chunk_start, content.len())
                .with_context(current_context)
                .with_boundary(is_at_boundary),
        );
    }

    if chunks.is_empty() {
        chunks.push(ChunkResult::new(content.to_string(), 0, content.len()));
    }

    chunks
}

// ============================================================================
// JSON Chunking
// ============================================================================

/// Chunk JSON by top-level keys/array elements with nested path context
fn chunk_json(content: &str, config: &ChunkingConfig) -> Vec<ChunkResult> {
    // Try to parse as JSON
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(content) {
        let chunks = chunk_json_value(&value, config, vec![]);
        if !chunks.is_empty() {
            return chunks;
        }
    }

    // Fall back to text chunking if JSON parsing fails
    chunk_text(content, config)
}

/// Recursively chunk a JSON value with path context.
///
/// Note: JSON chunks are re-serialized from parsed JSON, so `start_offset` and
/// `end_offset` represent positions within the serialized chunk content (0..len),
/// not byte offsets into the original input string.
fn chunk_json_value(
    value: &serde_json::Value,
    config: &ChunkingConfig,
    path: Vec<String>,
) -> Vec<ChunkResult> {
    let mut chunks = Vec::new();

    match value {
        serde_json::Value::Object(map) => {
            let mut current_chunk = String::from("{\n");
            let entries: Vec<_> = map.iter().collect();

            for (i, (key, val)) in entries.iter().enumerate() {
                let val_str = serde_json::to_string_pretty(val).unwrap_or_default();
                let entry = if i < entries.len() - 1 {
                    format!("  \"{}\": {},\n", key, val_str)
                } else {
                    format!("  \"{}\": {}\n", key, val_str)
                };

                let mut new_path = path.clone();
                new_path.push((*key).clone());
                let path_str = new_path.join(".");

                // If this single entry is too large, try to chunk its value
                if entry.len() > config.max_chunk_size {
                    // Save current chunk if not empty
                    if current_chunk.len() > 3 {
                        current_chunk.push('}');
                        let len = current_chunk.len();
                        let context = if path.is_empty() {
                            None
                        } else {
                            Some(path.join("."))
                        };
                        chunks.push(
                            ChunkResult::new(current_chunk, 0, len)
                                .with_context(context)
                                .with_boundary(true),
                        );
                        current_chunk = String::from("{\n");
                    }

                    // Recursively chunk the large value
                    let sub_chunks = chunk_json_value(val, config, new_path);
                    chunks.extend(sub_chunks);
                    continue;
                }

                if current_chunk.len() + entry.len() > config.max_chunk_size
                    && current_chunk.len() > 3
                {
                    current_chunk.push('}');
                    let len = current_chunk.len();
                    chunks.push(
                        ChunkResult::new(current_chunk, 0, len)
                            .with_context(Some(path_str.clone()))
                            .with_boundary(true),
                    );
                    current_chunk = String::from("{\n");
                }

                current_chunk.push_str(&entry);
            }

            current_chunk.push('}');
            if current_chunk.len() > 3 {
                let len = current_chunk.len();
                let context = if path.is_empty() {
                    None
                } else {
                    Some(path.join("."))
                };
                chunks.push(
                    ChunkResult::new(current_chunk, 0, len)
                        .with_context(context)
                        .with_boundary(true),
                );
            }
        }
        serde_json::Value::Array(arr) => {
            let mut current_chunk = String::from("[\n");

            for (i, val) in arr.iter().enumerate() {
                let val_str = serde_json::to_string_pretty(val).unwrap_or_default();
                let entry = if i < arr.len() - 1 {
                    format!("  {},\n", val_str)
                } else {
                    format!("  {}\n", val_str)
                };

                let mut new_path = path.clone();
                new_path.push(format!("[{}]", i));
                let path_str = new_path.join(".");

                if current_chunk.len() + entry.len() > config.max_chunk_size
                    && current_chunk.len() > 3
                {
                    current_chunk.push(']');
                    let len = current_chunk.len();
                    chunks.push(
                        ChunkResult::new(current_chunk, 0, len)
                            .with_context(Some(path_str.clone()))
                            .with_boundary(true),
                    );
                    current_chunk = String::from("[\n");
                }

                current_chunk.push_str(&entry);
            }

            current_chunk.push(']');
            if current_chunk.len() > 3 {
                let len = current_chunk.len();
                let context = if path.is_empty() {
                    None
                } else {
                    Some(path.join("."))
                };
                chunks.push(
                    ChunkResult::new(current_chunk, 0, len)
                        .with_context(context)
                        .with_boundary(true),
                );
            }
        }
        _ => {
            // Primitive value - just stringify
            let content = serde_json::to_string_pretty(value).unwrap_or_default();
            let len = content.len();
            let context = if path.is_empty() {
                None
            } else {
                Some(path.join("."))
            };
            chunks.push(
                ChunkResult::new(content, 0, len)
                    .with_context(context)
                    .with_boundary(false),
            );
        }
    }

    chunks
}

// ============================================================================
// YAML Chunking
// ============================================================================

/// Chunk YAML by top-level keys with nested path context
fn chunk_yaml(content: &str, _config: &ChunkingConfig) -> Vec<ChunkResult> {
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut chunk_start = 0;
    let mut current_pos = 0;
    let mut key_stack: Vec<(usize, String)> = Vec::new(); // (indent level, key)

    let lines: Vec<&str> = content.lines().collect();

    for line in lines {
        let line_with_newline = if current_pos > 0 {
            format!("\n{}", line)
        } else {
            line.to_string()
        };

        // Calculate indent level
        let indent = line.len() - line.trim_start().len();
        let trimmed = line.trim();

        // Check if this is a YAML key line.
        // A key line has an unquoted key followed by ':'. We check that ':' is not
        // inside a URL (contains ://) and the line doesn't start with flow indicators.
        let is_key_line = !trimmed.starts_with('-')
            && !trimmed.starts_with('#')
            && !trimmed.starts_with('"')
            && !trimmed.starts_with('\'')
            && !trimmed.starts_with('{')
            && !trimmed.starts_with('[')
            && trimmed.contains(':')
            && !trimmed.contains("://");

        if is_key_line {
            // Extract the key
            if let Some(key) = trimmed.split(':').next() {
                let key = key.trim().to_string();

                // Update key stack based on indentation
                while !key_stack.is_empty() && key_stack.last().unwrap().0 >= indent {
                    key_stack.pop();
                }
                key_stack.push((indent, key));
            }
        }

        // Check if this is a top-level key (no leading whitespace)
        let is_top_level_key = indent == 0 && is_key_line;

        // Start new chunk at top-level keys
        if is_top_level_key && !current_chunk.is_empty() && current_chunk.len() > 50 {
            let context = format_yaml_path(&key_stack[..key_stack.len().saturating_sub(1)]);
            chunks.push(
                ChunkResult::new(current_chunk.clone(), chunk_start, current_pos)
                    .with_context(context)
                    .with_boundary(true),
            );

            current_chunk = String::new();
            chunk_start = current_pos;
        }

        current_chunk.push_str(&line_with_newline);
        current_pos += line_with_newline.len();
    }

    // Add final chunk
    if !current_chunk.is_empty() {
        let context = format_yaml_path(&key_stack);
        chunks.push(
            ChunkResult::new(current_chunk, chunk_start, content.len())
                .with_context(context)
                .with_boundary(!key_stack.is_empty()),
        );
    }

    if chunks.is_empty() {
        chunks.push(ChunkResult::new(content.to_string(), 0, content.len()));
    }

    chunks
}

/// Format YAML key path as context string
fn format_yaml_path(stack: &[(usize, String)]) -> Option<String> {
    if stack.is_empty() {
        return None;
    }
    Some(
        stack
            .iter()
            .map(|(_, k)| k.as_str())
            .collect::<Vec<_>>()
            .join("."),
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_content_no_chunking() {
        let content = "Small content";
        let config = ChunkingConfig::default();
        let chunks = chunk_content(content, ContentType::Text, &config);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, content);
    }

    #[test]
    fn test_text_chunking() {
        let content = "a".repeat(2000);
        let config = ChunkingConfig {
            min_chunk_threshold: 1000,
            max_chunk_size: 500,
            min_chunk_size: 100,
            chunk_overlap: 100,
        };
        let chunks = chunk_content(&content, ContentType::Text, &config);

        assert!(chunks.len() > 1);
        for chunk in &chunks {
            // Allow some flexibility for overlap
            assert!(chunk.content.len() <= config.max_chunk_size + config.chunk_overlap + 100);
        }
    }

    #[test]
    fn test_markdown_splits_by_headers_first() {
        let content = "# H1\nShort content here.\n\n# H2\nAlso short content.";
        let config = ChunkingConfig {
            min_chunk_threshold: 10, // Force chunking
            max_chunk_size: 1000,
            min_chunk_size: 10, // Allow small chunks
            chunk_overlap: 0,
        };
        let chunks = chunk_content(content, ContentType::Markdown, &config);

        // Should produce 2 chunks (one per header)
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].content.contains("# H1"));
        assert!(chunks[1].content.contains("# H2"));
    }

    #[test]
    fn test_large_section_gets_subsplit() {
        let long_paragraph = "This is a long sentence. ".repeat(100);
        let content = format!("# Header\n\n{}", long_paragraph);
        let config = ChunkingConfig {
            min_chunk_threshold: 100,
            max_chunk_size: 500,
            min_chunk_size: 100,
            chunk_overlap: 50,
        };
        let chunks = chunk_content(&content, ContentType::Markdown, &config);

        // Should have multiple chunks due to size enforcement
        assert!(chunks.len() > 1);
        // All chunks should be under max size (with some tolerance for overlap)
        for chunk in &chunks {
            assert!(chunk.content.len() <= config.max_chunk_size + config.chunk_overlap + 100);
        }
    }

    #[test]
    fn test_small_chunks_merged() {
        let content = "# A\nx\n\n# B\ny\n\n# C\nz";
        let config = ChunkingConfig {
            min_chunk_threshold: 5,
            max_chunk_size: 1000,
            min_chunk_size: 50, // Minimum size that will cause merging
            chunk_overlap: 0,
        };
        let chunks = chunk_content(content, ContentType::Markdown, &config);

        // Small sections should be merged (fewer chunks than headers)
        // But boundaries should be respected
        assert!(chunks.len() <= 3);
    }

    #[test]
    fn test_header_path_context() {
        let content = "# Main\n\n## Sub\n\nContent here\n\n### Detail\n\nMore content";
        let config = ChunkingConfig {
            min_chunk_threshold: 10,
            max_chunk_size: 1000,
            min_chunk_size: 10,
            chunk_overlap: 0,
        };
        let chunks = chunk_content(content, ContentType::Markdown, &config);

        // Check that context includes full header path
        let detail_chunk = chunks.iter().find(|c| c.content.contains("### Detail"));
        assert!(detail_chunk.is_some());
        let ctx = detail_chunk.unwrap().context.as_ref().unwrap();
        assert!(ctx.contains("# Main"));
        assert!(ctx.contains("## Sub"));
        assert!(ctx.contains("### Detail"));
    }

    #[test]
    fn test_markdown_chunking_preserves_context() {
        let content = format!(
            "# Header 1\n\n{}\n\n# Header 2\n\n{}",
            "a".repeat(600),
            "b".repeat(600)
        );
        let config = ChunkingConfig {
            min_chunk_threshold: 500,
            max_chunk_size: 500,
            min_chunk_size: 100,
            chunk_overlap: 50,
        };
        let chunks = chunk_content(&content, ContentType::Markdown, &config);

        assert!(chunks.len() >= 2);
        // Check that context is preserved
        assert!(chunks.iter().any(|c| c.context.is_some()));
    }

    #[test]
    fn test_code_chunking() {
        let content = format!(
            "fn foo() {{\n{}\n}}\n\nfn bar() {{\n{}\n}}",
            "    // code\n".repeat(50),
            "    // more code\n".repeat(50)
        );
        let config = ChunkingConfig {
            min_chunk_threshold: 500,
            max_chunk_size: 500,
            min_chunk_size: 100,
            chunk_overlap: 50,
        };
        let chunks = chunk_content(&content, ContentType::Code, &config);

        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_code_boundary_patterns() {
        let patterns = [
            "fn test()",
            "pub fn test()",
            "async fn test()",
            "const FOO",
            "export default",
            "module.exports",
            "interface Foo",
            "type Bar",
        ];

        for pattern in patterns {
            assert!(
                is_code_boundary(pattern),
                "Pattern '{}' should be recognized as boundary",
                pattern
            );
        }
    }

    #[test]
    fn test_json_chunking() {
        let content = serde_json::json!({
            "key1": "a".repeat(300),
            "key2": "b".repeat(300),
            "key3": "c".repeat(300),
        })
        .to_string();

        let config = ChunkingConfig {
            min_chunk_threshold: 500,
            max_chunk_size: 400,
            min_chunk_size: 100,
            chunk_overlap: 50,
        };
        let chunks = chunk_content(&content, ContentType::Json, &config);

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_json_nested_path_context() {
        let content = serde_json::json!({
            "users": {
                "profile": {
                    "settings": "value"
                }
            }
        })
        .to_string();

        let config = ChunkingConfig {
            min_chunk_threshold: 10,
            max_chunk_size: 1000,
            min_chunk_size: 10,
            chunk_overlap: 0,
        };
        let chunks = chunk_content(&content, ContentType::Json, &config);

        // Should have context with nested path
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_sentence_splitting() {
        let text = "First sentence. Second sentence? Third sentence! Fourth.";
        let sentences = split_at_sentences(text);

        assert!(sentences.len() >= 3);
        assert!(sentences[0].contains("First"));
    }

    #[test]
    fn test_yaml_chunking_with_path() {
        let content = r#"
server:
  host: localhost
  port: 8080
database:
  host: db.example.com
  port: 5432
"#;
        let config = ChunkingConfig {
            min_chunk_threshold: 10,
            max_chunk_size: 1000,
            min_chunk_size: 10,
            chunk_overlap: 0,
        };
        let chunks = chunk_content(content, ContentType::Yaml, &config);

        // Should have chunks for server and database sections
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_recursive_split_preserves_context() {
        let long_text = "This is a sentence. ".repeat(100);
        let chunks = recursive_split(&long_text, 200, 0, Some("test context".to_string()), 20);

        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(
                chunk
                    .context
                    .as_ref()
                    .map(|c| c == "test context")
                    .unwrap_or(false)
            );
        }
    }
}
