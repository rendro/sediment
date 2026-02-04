# Changelog

## [0.3.0] - 2026-02-03

### Changed
- **BREAKING**: Simplified API from 5 tools to 4 tools (removed `connections`)
- **BREAKING**: Simplified `store` from 9 parameters to 2 (`content`, `scope`)
- **BREAKING**: Simplified `recall` from 4 parameters to 2 (`query`, `limit`)
- **BREAKING**: Simplified `list` from 3 parameters to 2 (`limit`, `scope`)
- **BREAKING**: Removed fields from Item: `title`, `tags`, `source`, `metadata`, `expires_at`

### Added
- Auto-migration: existing databases automatically migrated to new schema on startup
- Schema versioning for future migrations

### Removed
- `connections` tool (graph is now internal infrastructure)
- Tag-based filtering (semantic search handles categorization)
- Item expiration (simplified lifecycle)
- Replace functionality (use `forget` + `store`)
- Related item linking on store (handled by auto-consolidation)

## [0.2.3] - 2026-02-03

### Added
- Statically-linked musl Linux binary in releases (fixes glibc version mismatch on older systems)
- Shell installer and Homebrew formula now default to musl binary on Linux

## [0.2.2] - 2026-02-02

### Changed
- Bumped rate limit from 60 to 600 tool calls per minute
- Fixed security audit findings: unsafe elimination, rate limiter hardening

### Added
- Security section in README

## [0.2.1] - 2025-05-21

### Added
- crates.io publishing (`cargo install sediment`)
- Shell installer (`install.sh`) for macOS and Linux
- README: badges, "Why Sediment?" section, comparison table, setup guides for 6 MCP clients
- Performance section with benchmark summary
- CONTRIBUTING.md, CHANGELOG.md, GitHub issue templates
- BENCHMARKS.md with recall latency methodology
- MIT LICENSE file

## [0.2.0] - 2025-05-20

### Changed
- Replaced Kuzu graph database with SQLite for relationship tracking
- 5-tool MCP API: `store`, `recall`, `list`, `forget`, `connections`

### Added
- SQLite-based graph store with RELATED, SUPERSEDES, CO_ACCESSED, CLUSTER_SIBLING edges
- Benchmarks for recall latency at various database sizes
- Optimized recall latency: vector index, batch fetch, cached project_id
- Memory decay scoring for recall results
- Background consolidation engine

## [0.1.0] - 2025-05-15

### Added
- Initial release
- Local semantic memory with LanceDB vector storage
- Local embeddings via all-MiniLM-L6-v2 (Candle)
- MCP server with stdio JSON-RPC transport
- Project scoping with automatic isolation
- Type-aware content chunking (markdown, code, JSON, YAML, text)
- CLI: `sediment init`, `sediment stats`, `sediment list`
- Homebrew tap installation
