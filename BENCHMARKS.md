# Benchmarks

Retrieval quality and latency compared against ChromaDB and Mem0, using 1,000 developer memories and 200 search queries. Full benchmark suite: [sediment-benchmark](https://github.com/rendro/sediment-benchmark).

## Results

Measured on Apple M3 Max, 36GB RAM.

| Metric | Sediment | ChromaDB | Mem0 |
|--------|----------|----------|------|
| Recall@1 | **50.0%** | 47.0% | 47.0% |
| Recall@3 | 69.0% | 69.0% | 69.0% |
| Recall@5 | 77.5% | 78.5% | 78.5% |
| Recall@10 | 89.5% | 90.0% | 90.0% |
| MRR | **61.9%** | 60.8% | 60.8% |
| nDCG@5 | 58.7% | 59.9% | 59.9% |
| Recency@1 | **100%** | 14% | 14% |
| Consolidation | **99%** | 0% | 0% |
| Store p50 | 49ms | 696ms | 16ms |
| Recall p50 | 103ms | 694ms | 8ms |

## Key takeaways

- **Retrieval quality**: Best R@1 (50.0%) and MRR (61.9%) — top result is correct more often than alternatives
- **Temporal correctness**: 100% Recency@1 — updated memories always rank first. Others: 14%
- **Deduplication**: 99% consolidation rate — near-duplicates auto-merged. Others: 0%
- **Latency**: 14x faster store than ChromaDB (49ms vs 696ms). All operations local, no network

## Category breakdown

### Recall@5 by category

| Category | Sediment | ChromaDB | Mem0 |
|----------|----------|----------|------|
| `architecture` | **82.9%** | 71.4% | 71.4% |
| `code_patterns` | **88.6%** | **88.6%** | **88.6%** |
| `cross_project` | 65.6% | **68.8%** | **68.8%** |
| `project_facts` | 60.6% | **75.8%** | **75.8%** |
| `troubleshooting` | 78.1% | **81.2%** | **81.2%** |
| `user_preferences` | **87.9%** | 84.9% | 84.9% |

### MRR by category

| Category | Sediment | ChromaDB | Mem0 |
|----------|----------|----------|------|
| `architecture` | **66.8%** | 55.8% | 55.8% |
| `code_patterns` | 70.4% | **71.1%** | **71.1%** |
| `cross_project` | **50.7%** | 47.1% | 47.1% |
| `project_facts` | 51.6% | **59.2%** | **59.2%** |
| `troubleshooting` | **63.2%** | 62.8% | 62.8% |
| `user_preferences` | 67.6% | **67.9%** | **67.9%** |

## Temporal correctness

| Metric | Sediment | ChromaDB | Mem0 |
|--------|----------|----------|------|
| Recency@1 | **100%** | 14% | 14% |
| Recency@3 | **100%** | 94% | 94% |
| MRR | **100%** | 48.8% | 48.8% |
| Mean Rank | **1.00** | 2.38 | 2.38 |

## Latency

### Store latency

| Metric | Sediment | ChromaDB | Mem0 |
|--------|----------|----------|------|
| p50 | 49ms | 696ms | **16ms** |
| p95 | 62ms | 726ms | **19ms** |
| p99 | 88ms | 729ms | **20ms** |

### Recall latency

| Metric | Sediment | ChromaDB | Mem0 |
|--------|----------|----------|------|
| p50 | 103ms | 694ms | **8ms** |
| p95 | 109ms | 728ms | **12ms** |
| p99 | 132ms | 746ms | **12ms** |

## Methodology

- **Dataset**: 1,000 memories across 6 categories (architecture, code patterns, project facts, troubleshooting, user preferences, cross-project)
- **Queries**: 200 queries with known ground-truth expected results
- **Temporal**: 50 sequences testing whether updated information ranks above stale versions
- **Dedup**: 50 pairs of near-duplicate content testing consolidation
- **Baselines**: ChromaDB with default ONNX embeddings; Mem0 with local Qdrant + HuggingFace embeddings
- **Sediment**: Hybrid vector + BM25 search, local Candle embeddings (all-MiniLM-L6-v2)
