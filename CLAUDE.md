# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build
cargo build --release

# Run MCP server (default)
cargo run --release

# Run with verbose logging
cargo run --release -- --verbose

# Run CLI commands
cargo run --release -- init          # Initialize project
cargo run --release -- stats         # Show database statistics
cargo run --release -- list          # List stored items

# Run tests (requires model download on first run)
cargo test
cargo test -- --ignored              # Include tests that require model download

# Install locally
cargo install --path .
```

## Architecture

Sediment is a semantic memory system for AI agents, running as an MCP (Model Context Protocol) server. It combines vector search, a property graph, and access tracking into a unified memory intelligence layer.

### Two-Database Hybrid (all local, embedded, zero config)

- **LanceDB** — Vector embeddings + semantic similarity (items and chunks)
- **SQLite** (`access.db`) — Graph relationships, access tracking, decay scoring, consolidation queue

### Core Components

- **`src/main.rs`** - CLI entry point with subcommands (init, stats, list) and MCP server startup
- **`src/lib.rs`** - Library root exposing public API, project detection, scope types, and project ID migration
- **`src/db.rs`** - LanceDB wrapper handling vector storage, hybrid search (vector + FTS/BM25), and CRUD operations
- **`src/embedder.rs`** - Local embeddings using `all-MiniLM-L6-v2` via Candle (384-dim vectors)
- **`src/chunker.rs`** - Smart content chunking by type (markdown, code, JSON, YAML, text)
- **`src/document.rs`** - ContentType enum for routing content to the appropriate chunker
- **`src/item.rs`** - Unified Item, Chunk, SearchResult, StoreResult, and ConflictInfo types
- **`src/access.rs`** - SQLite-based access tracking, validation counting, and memory decay scoring
- **`src/graph.rs`** - SQLite graph store: relationship tracking (RELATED, SUPERSEDES, CO_ACCESSED, CLUSTER_SIBLING edges)
- **`src/consolidation.rs`** - Background consolidation: auto-merging near-duplicates, linking similar items
- **`src/error.rs`** - SedimentError enum with typed error variants (Database, Embedding, Arrow, etc.)
- **`src/retry.rs`** - Retry utilities with exponential backoff (3 attempts, 100ms–2s)

### MCP Server (`src/mcp/`)

- **`mod.rs`** - Module exports
- **`server.rs`** - stdio JSON-RPC server with shared embedder, graph path, consolidation semaphore
- **`tools.rs`** - 4 MCP tools: `store`, `recall`, `list`, `forget`
- **`protocol.rs`** - MCP protocol types and JSON-RPC handling

### Data Flow

1. **Store**: Content → Embedder (384-dim vector) → LanceDB storage → Graph node creation → Conflict detection → Consolidation queue
2. **Chunking**: Long content (>1000 chars) → Type-aware splitting → Individual chunk embeddings
3. **Recall**: Query → Embedder → Vector similarity search → Project boosting → Decay scoring → Trust-weighted re-ranking → Graph backfill → 1-hop graph expansion → Co-access suggestions → Cross-project flagging → Background consolidation + co-access recording
4. **Consolidation** (background): Queue candidates → >=0.95 similarity: merge (delete old, transfer edges, SUPERSEDES edge) → 0.85-0.95: link (RELATED edge)
5. **Clustering** (periodic): Triangle detection in graph → CLUSTER_SIBLING edges

### Key Design Decisions

- **Two-database hybrid**: LanceDB for vectors, SQLite for graph relationships + mutable counters
- **Single central database** at `~/.sediment/data/` stores all projects; graph + access at `~/.sediment/access.db`
- **Project scoping** via UUID stored in `.sediment/config` per project
- **Similarity boosting**: Same-project items unchanged, different projects get 0.875x penalty (12.5% spread)
- **Hybrid search**: Vector similarity combined with FTS/BM25 scoring. BM25 boost is additive (max 0.12, power-law gamma 2.0). FTS index rebuilt on each store
- **Conflict detection**: Items with >=0.85 similarity flagged on store and enqueued for consolidation
- **Fresh DB connection per tool call** with shared embedder for efficiency
- **Memory decay scoring**: Recall results re-ranked using freshness (hyperbolic decay, 0.5 at 30 days) and access frequency (log-scaled). Tracked in SQLite sidecar since LanceDB is append-oriented.
- **Trust-weighted scoring**: `final_score = similarity * freshness * frequency * trust_bonus` where `trust_bonus = 1.0 + 0.05*ln(1+validation_count) + 0.005*edge_count`
- **Non-blocking intelligence**: All background tasks (consolidation, co-access tracking, clustering) run as fire-and-forget `tokio::spawn` tasks. Tool responses return immediately. `Semaphore(1)` prevents concurrent consolidation.
- **Lazy graph backfill**: Pre-existing items get graph nodes when they appear in recall results
- **Auto-migration**: Database schema is automatically migrated on startup when upgrading from older versions

## MCP Tools Reference

The 4-tool API is defined in `src/mcp/tools.rs`:

| Tool | Purpose |
|------|---------|
| `store` | Store content with optional scope (project/global) |
| `recall` | Semantic search with decay scoring, trust weighting, graph expansion |
| `list` | List items by scope (project/global/all) |
| `forget` | Delete item by ID (removes from LanceDB and graph) |

### Store Parameters
- `content` (required) — The content to store
- `scope` (optional, default: "project") — Where to store: "project" or "global"

### Recall Parameters
- `query` (required) — Semantic search query
- `limit` (optional, default: 5) — Maximum number of results

### List Parameters
- `limit` (optional, default: 10) — Maximum number of results
- `scope` (optional, default: "project") — Which items to list: "project", "global", or "all"

### Recall Response Fields
- `results[]` — standard results with `similarity`, `related_ids`, optional `cross_project` flag
- `graph_expanded[]` — 1-hop neighbors from graph not in original results (marked `graph_expanded: true`)
- `suggested[]` — items frequently co-recalled with top results (co-access count >= 3)

## SQLite Schema (access.db)

```sql
-- Graph nodes
CREATE TABLE graph_nodes (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL
);

-- Graph edges (RELATED, SUPERSEDES, CO_ACCESSED, CLUSTER_SIBLING)
CREATE TABLE graph_edges (
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,     -- 'related', 'supersedes', 'co_accessed', 'cluster_sibling'
    strength REAL NOT NULL DEFAULT 0.0,
    rel_type TEXT NOT NULL DEFAULT '',
    count INTEGER NOT NULL DEFAULT 0,
    last_at INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    UNIQUE(from_id, to_id, edge_type)
);
CREATE INDEX idx_edges_from ON graph_edges(from_id, edge_type);
CREATE INDEX idx_edges_to ON graph_edges(to_id, edge_type);

-- Access tracking and decay scoring
CREATE TABLE access_log (
    item_id TEXT PRIMARY KEY,
    access_count INTEGER DEFAULT 0,
    last_accessed_at INTEGER,
    created_at INTEGER,
    validation_count INTEGER DEFAULT 0  -- incremented on replace
);

-- Consolidation queue (populated on store conflicts)
CREATE TABLE consolidation_queue (
    item_id_a TEXT, item_id_b TEXT, similarity REAL,
    status TEXT DEFAULT 'pending',  -- pending/merged/linked
    created_at INTEGER,
    UNIQUE(item_id_a, item_id_b)
);
```

## Testing Notes

- Embedder tests are marked `#[ignore]` because they require downloading the model (~90MB)
- Use `tempfile` crate for database tests to avoid polluting real data
- LanceDB operations are async; tests use `tokio::runtime::Runtime`
