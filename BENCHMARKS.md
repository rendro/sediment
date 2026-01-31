# Benchmarks

Recall latency benchmarks at various database sizes, with and without graph features (backfill, expansion, co-access, decay scoring).

Run benchmarks yourself:

```bash
cargo bench
```

## Results

Measured on Apple M3 Max. Times are per-recall (single query, top-5 results, criterion.rs p50).

| DB Size | Graph Off | Graph On | Overhead |
|---------|-----------|----------|----------|
| 100     | ~12ms     | ~15ms    | 1.2x     |
| 1,000   | ~36ms     | ~65ms    | 1.8x     |

### What's measured

- **Graph Off**: Pure vector search + result formatting
- **Graph On**: Vector search + decay scoring + graph backfill + 1-hop expansion + co-access suggestions

### Key takeaways

- Sub-16ms recall at 100 items with full graph features
- Graph overhead scales with DB size: 1.2x at 100 items, 1.8x at 1K
- Embedding computation dominates base latency (model inference is ~5ms per query)
- All operations are local — no network round-trips

## Methodology

- Criterion.rs with 20 samples, 10s measurement time
- Databases seeded with mixed content (short text, markdown, code)
- ~30% of items have graph edges
- Queries rotate through 5 representative search terms
- Fresh DB connection per iteration (matches real-world MCP usage)
