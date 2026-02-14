[![Crates.io](https://img.shields.io/crates/v/sediment-mcp.svg)](https://crates.io/crates/sediment-mcp)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/rendro/sediment/actions/workflows/ci.yml/badge.svg)](https://github.com/rendro/sediment/actions/workflows/ci.yml)

# Sediment

Semantic memory for AI agents. Local-first, MCP-native.

Combines vector search, a relationship graph, and access tracking into a unified memory intelligence layer — all running locally as a single binary.

## Why Sediment?

- **Single binary, zero config** — no Docker, no Postgres, no Qdrant. Just `sediment`.
- **50ms store, 103ms recall** — local embeddings and vector search at 1K items, no network round-trips.
- **4-tool focused API** — `store`, `recall`, `list`, `forget`. That's it.
- **Works everywhere** — macOS (Intel + ARM), Linux x86_64. All data stays on your machine.

### Comparison

| | Sediment | ChromaDB | Mem0 | MCP Memory Svc |
|---|---|---|---|---|
| Install | Single binary | Python + pip | Docker + Postgres + Qdrant | Python + pip |
| Tools | 4 | API | 4 | 12 |
| R@1 | **50%** | 47% | 47% | 38% |
| Recency@1 | **100%** | 14% | 14% | 10% |
| Dedup | **99%** | 0% | 0% | 0% |
| Store p50 | 50ms | 692ms | 14ms | 2ms |

## Install

```bash
# Via crates.io
cargo install sediment-mcp

# Via Homebrew
brew tap rendro/tap
brew install sediment

# Via shell installer
curl -fsSL https://raw.githubusercontent.com/rendro/sediment/main/install.sh | sh

# From source
cargo install --path .
```

## Setup

Add Sediment to your MCP client configuration:

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sediment": {
      "command": "sediment"
    }
  }
}
```

### Claude Code

Run `sediment init` in your project, or add manually to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "sediment": {
      "command": "sediment"
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json` in your project:

```json
{
  "mcpServers": {
    "sediment": {
      "command": "sediment"
    }
  }
}
```

### VS Code (Copilot)

Add to `.vscode/mcp.json` in your project:

```json
{
  "servers": {
    "sediment": {
      "command": "sediment"
    }
  }
}
```

### Windsurf

Add to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "sediment": {
      "command": "sediment"
    }
  }
}
```

### JetBrains IDEs

Go to **Settings > Tools > AI Assistant > MCP Servers**, click **+**, and add:

```json
{
  "sediment": {
    "command": "sediment"
  }
}
```

## Tools

| Tool | Parameters | Description |
|------|------------|-------------|
| `store` | `content`, `scope?`, `replace_id?` | Save content to memory |
| `recall` | `query`, `limit?` | Search by semantic similarity |
| `list` | `limit?`, `scope?` | List stored items |
| `forget` | `id` | Delete an item by ID |

## CLI

```bash
sediment                          # Start MCP server
sediment init                     # Set up Claude Code integration
sediment stats                    # Show database statistics
sediment list                     # List stored items
sediment list --scope global      # List global items
sediment store "some content"     # Store content
sediment store -                  # Store content from stdin
sediment recall "search query"    # Search by semantic similarity
sediment forget <id>              # Delete an item by ID
```

All commands support `--json` for machine-readable output.

## How It Works

### Two-Database Hybrid

All local, embedded, zero config:

- **LanceDB** — Vector embeddings and semantic similarity search
- **SQLite** (`access.db`) — Relationship graph, access tracking, decay scoring, consolidation queue

### Key Features

- **Memory decay**: Results re-ranked by freshness (30-day half-life) and access frequency. Old memories rank lower but are never auto-deleted.
- **Trust-weighted scoring**: Validated and well-connected memories score higher.
- **Hybrid search**: Vector similarity combined with FTS/BM25 scoring for better retrieval quality.
- **Project scoping**: Automatic context isolation between projects. Different-project items receive a similarity penalty.
- **Relationship graph**: Items linked via RELATED, SUPERSEDES, and CO_ACCESSED edges. Recall expands results with 1-hop graph neighbors and co-access suggestions.
- **Background consolidation**: Near-duplicates (≥0.95 similarity) auto-merged; similar items (0.85–0.95) linked.
- **Type-aware chunking**: Intelligent splitting for markdown, code, JSON, YAML, and plain text.
- **Conflict detection**: Items with ≥0.85 similarity flagged on store.
- **Cross-project recall**: Results from other projects flagged.
- **Local embeddings**: all-MiniLM-L6-v2 via Candle (384-dim vectors, no API keys).
- **Model integrity**: SHA-256 verification of all model files on every load, pinned to a specific revision.
- **Auto-migration**: Database schema automatically migrated when upgrading from older versions.

### Security

- **Input bounds**: Content (1MB), queries (10KB), JSON-RPC lines (10MB).
- **Rate limiting**: 600 tool calls per minute.
- **SQL injection prevention**: Sanitized filter expressions for LanceDB; parameterized queries for SQLite.
- **Cross-project access control**: Forget enforces project isolation. Cross-project content is flagged in recall results.
- **Error sanitization**: Internal errors logged to stderr; only generic messages returned to MCP clients.
- **Retry with backoff**: Transient failures retried with exponential backoff (3 attempts, 100ms–2s).

## Performance

Benchmarked against 5 alternatives with 1,000 memories and 200 queries. See [BENCHMARKS.md](BENCHMARKS.md) for full results.

| Metric | Sediment | ChromaDB | Mem0 |
|--------|----------|----------|------|
| R@1 | **50%** | 47% | 47% |
| MRR | **62%** | 61% | 61% |
| Recency@1 | **100%** | 14% | 14% |
| Dedup | **99%** | 0% | 0% |
| Store p50 | 50ms | 692ms | 14ms |
| Recall p50 | 103ms | 694ms | 8ms |

### Data Location

- Vector store: `~/.sediment/data/`
- Graph + access tracking: `~/.sediment/access.db`

Everything runs locally. Your data never leaves your machine.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for build instructions and PR guidelines.

## License

[MIT](LICENSE)
