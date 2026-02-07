# Benchmarks

Retrieval quality and latency compared against ChromaDB and Mem0, using 1,000 developer memories and 200 search queries. Full benchmark suite: [sediment-benchmark](https://github.com/rendro/sediment-benchmark).

## Results

Measured on Apple M3 Max, 36GB RAM.

| Metric | Sediment | ChromaDB | Mem0 |
|--------|----------|----------|------|
| Recall@1 | 45.0% | 47.0% | 47.0% |
| Recall@3 | **69.0%** | 69.0% | 69.0% |
| Recall@5 | 78.0% | 78.5% | 78.5% |
| Recall@10 | **90.5%** | 90.0% | 90.0% |
| MRR | 59.0% | 60.8% | 60.8% |
| Recency@1 | **100%** | 14% | 14% |
| Consolidation | **99%** | 0% | 0% |
| Store p50 | 22ms | 696ms | 16ms |
| Recall p50 | 26ms | 694ms | 8ms |

## Key takeaways

- **Retrieval quality**: Within 0.5pp of ChromaDB on Recall@5 (78.0% vs 78.5%), matching on Recall@3
- **Temporal correctness**: 100% Recency@1 — updated memories always rank first. Others: 14%
- **Deduplication**: 99% consolidation rate — near-duplicates auto-merged. Others: 0%
- **Latency**: 32x faster store than ChromaDB (22ms vs 696ms). All operations local, no network

## Methodology

- **Dataset**: 1,000 memories across 6 categories (architecture, code patterns, project facts, troubleshooting, user preferences, cross-project)
- **Queries**: 200 queries with known ground-truth expected results
- **Temporal**: 50 sequences testing whether updated information ranks above stale versions
- **Dedup**: 50 pairs of near-duplicate content testing consolidation
- **Baselines**: ChromaDB with default ONNX embeddings; Mem0 with local Qdrant + HuggingFace embeddings
- **Sediment**: Hybrid vector + BM25 search, local Candle embeddings (all-MiniLM-L6-v2)
