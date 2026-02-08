# Benchmarks

Retrieval quality and latency compared against five MCP memory systems, using 1,000 developer memories and 200 search queries. Full benchmark suite: [sediment-benchmark](https://github.com/rendro/sediment-benchmark).

## Results

Measured on Apple M3 Max, 36GB RAM.

| Metric | Sediment | ChromaDB | Mem0 | MCP Memory Service | MCP Server Memory | Basic Memory |
|--------|----------|----------|------|--------------------|--------------------|-------------|
| Recall@1 | **50.0%** | 47.0% | 47.0% | 38.0% | 1.0% | 9.0% |
| Recall@5 | 77.5% | **78.5%** | **78.5%** | 66.0% | 2.0% | 11.5% |
| MRR | **61.9%** | 60.8% | 60.8% | 49.0% | 1.5% | 10.2% |
| nDCG@5 | 58.7% | **59.9%** | **59.9%** | 47.8% | 1.6% | 10.4% |
| Recency@1 | **100%** | 14% | 14% | 10% | 0% | 0% |
| Consolidation | **99%** | 0% | 0% | 0% | 0% | 0% |
| Store p50 | 50ms | 692ms | 14ms | **2ms** | **2ms** | 49ms |
| Recall p50 | 103ms | 694ms | **8ms** | 10ms | **2ms** | 28ms |

MCP Server Memory and Basic Memory use keyword/entity matching rather than vector search, which explains their low retrieval scores on semantic queries.

## Key takeaways

- **Retrieval quality**: Best R@1 (50.0%) and MRR (61.9%) among all systems — top result is correct more often than alternatives
- **Temporal correctness**: 100% Recency@1 — updated memories always rank first. Best competitor: 14%
- **Deduplication**: 99% consolidation rate — near-duplicates auto-merged. All others: 0%
- **Latency**: 14x faster store than ChromaDB (50ms vs 692ms). All operations local, no network

## Embedding model comparison

Sediment supports multiple embedding models via the `SEDIMENT_EMBEDDING_MODEL` environment variable. All models achieve 100% temporal correctness and 99% dedup consolidation.

| Metric | all-MiniLM-L6-v2 (default) | bge-base-en-v1.5 | bge-small-en-v1.5 | e5-small-v2 |
|--------|---------------------------|-------------------|-------------------|-------------|
| Dimensions | 384 | 768 | 384 | 384 |
| Recall@1 | 50.0% | **50.5%** | 46.5% | 42.0% |
| Recall@5 | **77.5%** | 76.0% | 75.0% | 68.0% |
| MRR | 61.9% | **62.0%** | 58.4% | 53.5% |
| nDCG@5 | **58.7%** | 57.6% | 55.5% | 51.1% |
| Store p50 | **50ms** | 92ms | 64ms | 63ms |
| Recall p50 | **103ms** | 149ms | 120ms | 118ms |

all-MiniLM-L6-v2 remains the default: essentially tied with bge-base-en-v1.5 on quality but ~2x faster due to smaller dimensions (384 vs 768).

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
| p50 | 50ms | 692ms | **14ms** |
| p95 | 63ms | 726ms | **19ms** |
| p99 | 65ms | 729ms | **20ms** |

### Recall latency

| Metric | Sediment | ChromaDB | Mem0 |
|--------|----------|----------|------|
| p50 | 103ms | 694ms | **8ms** |
| p95 | 110ms | 728ms | **12ms** |
| p99 | 124ms | 746ms | **12ms** |

## Methodology

- **Dataset**: 1,000 memories across 6 categories (architecture, code patterns, project facts, troubleshooting, user preferences, cross-project)
- **Queries**: 200 queries with known ground-truth expected results
- **Temporal**: 50 sequences testing whether updated information ranks above stale versions
- **Dedup**: 50 pairs of near-duplicate content testing consolidation
- **Systems tested**: Sediment (hybrid vector + BM25, local Candle embeddings), ChromaDB (default ONNX embeddings), Mem0 (local Qdrant + HuggingFace), MCP Memory Service (ChromaDB-based), MCP Server Memory (entity graph), Basic Memory (markdown file-based)
