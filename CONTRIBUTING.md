# Contributing to Sediment

Thanks for your interest in contributing to Sediment!

## Getting Started

```bash
git clone https://github.com/rendro/sediment.git
cd sediment
cargo build
```

The first build will download the embedding model (~90MB).

## Development

```bash
# Build
cargo build

# Run tests
cargo test

# Run ignored tests (require model download)
cargo test -- --ignored

# Lint
cargo clippy --all-targets -- -D warnings

# Format
cargo fmt --all

# Benchmarks
cargo bench
```

## Pull Requests

1. Fork the repo and create a feature branch
2. Make your changes
3. Ensure `cargo test`, `cargo clippy`, and `cargo fmt --check` all pass
4. Write a clear PR description explaining the change

## Architecture

See [CLAUDE.md](./CLAUDE.md) for a detailed architecture overview including:
- Two-database hybrid design (LanceDB + SQLite)
- MCP server structure
- Data flow and key design decisions

## Reporting Issues

Use [GitHub Issues](https://github.com/rendro/sediment/issues) with the provided templates.
