//! Benchmark: Recall latency with/without graph features
//!
//! Measures end-to-end recall latency at 3 DB sizes (100, 1K, 10K items)
//! with graph features enabled vs disabled.

use std::path::PathBuf;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::StdRng;
use tempfile::TempDir;

use sediment::access::AccessTracker;
use sediment::db::Database;
use sediment::embedder::Embedder;
use sediment::graph::GraphStore;
use sediment::item::{Item, ItemFilters};
use sediment::mcp::{RecallConfig, recall_pipeline};

const SHORT_TEXTS: &[&str] = &[
    "User prefers dark mode for all editors",
    "The API endpoint for user auth is /api/v2/auth",
    "Deploy pipeline uses GitHub Actions with docker build",
    "Database migrations run on startup via flyway",
    "Frontend uses React 18 with TypeScript strict mode",
    "Redis cache TTL is set to 300 seconds for sessions",
    "Error handling follows Result<T, AppError> pattern",
    "Logging uses structured JSON format with tracing crate",
    "CI runs clippy, fmt check, and tests on every PR",
    "The project uses workspace with three crates",
];

const MARKDOWN_TEXTS: &[&str] = &[
    "# Architecture\n\nThe system uses a three-tier architecture:\n- **Frontend**: React SPA\n- **Backend**: Rust Actix-Web\n- **Database**: PostgreSQL with pgvector\n\nAll services communicate via gRPC.",
    "# Deployment Guide\n\n## Prerequisites\n- Docker 24+\n- kubectl configured\n- Helm 3.x\n\n## Steps\n1. Build images: `make docker-build`\n2. Push to registry: `make docker-push`\n3. Deploy: `helm upgrade --install app ./charts/app`",
    "# API Reference\n\n## POST /api/items\nCreate a new item.\n\n### Request Body\n```json\n{\"content\": \"string\", \"tags\": [\"string\"]}\n```\n\n### Response\n- 201: Item created\n- 400: Invalid input",
];

const CODE_TEXTS: &[&str] = &[
    "fn calculate_score(similarity: f32, freshness: f32) -> f32 {\n    let base = similarity * 0.7 + freshness * 0.3;\n    base.clamp(0.0, 1.0)\n}",
    "async fn handle_request(req: Request) -> Result<Response> {\n    let body = req.json::<CreateItem>().await?;\n    let item = db.store(body).await?;\n    Ok(Response::json(&item).status(201))\n}",
    "impl Iterator for ChunkIterator {\n    type Item = Chunk;\n    fn next(&mut self) -> Option<Self::Item> {\n        if self.pos >= self.content.len() { return None; }\n        let end = (self.pos + self.size).min(self.content.len());\n        let chunk = &self.content[self.pos..end];\n        self.pos = end;\n        Some(Chunk::new(chunk))\n    }\n}",
];

const QUERIES: &[&str] = &[
    "how does authentication work",
    "database migration strategy",
    "error handling patterns",
    "deployment process",
    "frontend architecture",
];

fn generate_content(index: usize) -> String {
    match index % 3 {
        0 => format!(
            "{} (variant {})",
            SHORT_TEXTS[index % SHORT_TEXTS.len()],
            index
        ),
        1 => format!(
            "{}\n\n<!-- variant {} -->",
            MARKDOWN_TEXTS[index % MARKDOWN_TEXTS.len()],
            index
        ),
        _ => format!(
            "// variant {}\n{}",
            index,
            CODE_TEXTS[index % CODE_TEXTS.len()]
        ),
    }
}

#[allow(dead_code)] // TempDir fields kept alive for RAII cleanup
struct SeededDb {
    db_dir: TempDir,
    access_dir: TempDir,
    db_path: PathBuf,
    access_path: PathBuf,
    embedder: Arc<Embedder>,
}

async fn seed_database(n: usize, embedder: Arc<Embedder>) -> SeededDb {
    let db_dir = TempDir::new().expect("create temp dir for lancedb");
    let access_dir = TempDir::new().expect("create temp dir for access db");

    let db_path = db_dir.path().to_path_buf();
    let access_path = access_dir.path().join("access.db");

    let mut db =
        Database::open_with_embedder(&db_path, Some("bench-project".into()), embedder.clone())
            .await
            .expect("open database");

    let graph = GraphStore::open(&access_path).expect("open graph");

    let mut rng = StdRng::seed_from_u64(42);

    // Collect item IDs for graph edge creation
    let mut item_ids: Vec<String> = Vec::with_capacity(n);

    for i in 0..n {
        let content = generate_content(i);
        let item = Item::new(&content).with_project_id("bench-project");

        let result = db.store_item(item).await.expect("store item");
        let id = result.id.clone();

        // Create graph node
        let now = chrono::Utc::now().timestamp();
        let _ = graph.add_node(&id, Some("bench-project"), now);

        // Add graph edges for ~30% of items
        if i > 0 && rand::Rng::gen_bool(&mut rng, 0.3) {
            let target_idx = rand::Rng::gen_range(&mut rng, 0..i);
            let _ = graph.add_related_edge(&id, &item_ids[target_idx], 0.8, "bench");
        }

        item_ids.push(id);
    }

    SeededDb {
        db_dir,
        access_dir,
        db_path,
        access_path,
        embedder,
    }
}

fn recall_benchmarks(c: &mut Criterion) {
    let embedder = Arc::new(Embedder::new().expect("Failed to load embedder model"));
    let rt = tokio::runtime::Runtime::new().unwrap();

    for &size in &[100usize, 1_000, 10_000] {
        let mut group = c.benchmark_group(format!("recall_{}", size));
        group.sample_size(20);
        group.warm_up_time(std::time::Duration::from_secs(3));
        group.measurement_time(std::time::Duration::from_secs(10));

        let seeded = rt.block_on(seed_database(size, embedder.clone()));

        // Benchmark: graph features OFF
        {
            let db_path = seeded.db_path.clone();
            let access_path = seeded.access_path.clone();
            let emb = seeded.embedder.clone();

            group.bench_function(BenchmarkId::new("graph_off", size), |b| {
                let mut query_idx = 0usize;
                b.to_async(tokio::runtime::Runtime::new().unwrap())
                    .iter(|| {
                        let query = QUERIES[query_idx % QUERIES.len()];
                        query_idx = query_idx.wrapping_add(1);
                        let db_path = db_path.clone();
                        let access_path = access_path.clone();
                        let emb = emb.clone();
                        async move {
                            let mut db = Database::open_with_embedder(
                                &db_path,
                                Some("bench-project".into()),
                                emb,
                            )
                            .await
                            .unwrap();
                            let tracker = AccessTracker::open(&access_path).unwrap();
                            let graph = GraphStore::open(&access_path).unwrap();
                            let config = RecallConfig {
                                enable_graph_backfill: false,
                                enable_graph_expansion: false,
                                enable_co_access: false,
                                enable_decay_scoring: false,
                                enable_background_tasks: false,
                            };
                            let filters = ItemFilters::new();
                            let _ = recall_pipeline(
                                &mut db, &tracker, &graph, query, 5, filters, &config,
                            )
                            .await;
                        }
                    });
            });
        }

        // Benchmark: graph features ON
        {
            let db_path = seeded.db_path.clone();
            let access_path = seeded.access_path.clone();
            let emb = seeded.embedder.clone();

            group.bench_function(BenchmarkId::new("graph_on", size), |b| {
                let mut query_idx = 0usize;
                b.to_async(tokio::runtime::Runtime::new().unwrap())
                    .iter(|| {
                        let query = QUERIES[query_idx % QUERIES.len()];
                        query_idx = query_idx.wrapping_add(1);
                        let db_path = db_path.clone();
                        let access_path = access_path.clone();
                        let emb = emb.clone();
                        async move {
                            let mut db = Database::open_with_embedder(
                                &db_path,
                                Some("bench-project".into()),
                                emb,
                            )
                            .await
                            .unwrap();
                            let tracker = AccessTracker::open(&access_path).unwrap();
                            let graph = GraphStore::open(&access_path).unwrap();
                            let config = RecallConfig::default();
                            let filters = ItemFilters::new();
                            let _ = recall_pipeline(
                                &mut db, &tracker, &graph, query, 5, filters, &config,
                            )
                            .await;
                        }
                    });
            });
        }

        group.finish();

        // Keep seeded alive until group is done
        drop(seeded);
    }
}

criterion_group!(benches, recall_benchmarks);
criterion_main!(benches);
