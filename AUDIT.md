# Security & Code Audit Report

**Date:** 2026-01-31 (third pass)
**Scope:** Full repository audit of sediment-mcp v0.2.1
**Auditor:** Automated code review
**Based on:** Latest main branch (commit 4a11455) merged into `claude/audit-repository-DY9FC`

---

## Validation of All Previously Reported Issues

### Round 1 Issues (original 19): All resolved

| # | Issue | Status |
|---|-------|--------|
| 1 | LanceDB filter injection | **FIXED** — `sanitize_sql_string()` applied at all filter sites |
| 2 | `truncate()` UTF-8 panic | **FIXED** — Uses `char_indices()` in `tools.rs:965-970`, `consolidation.rs:242-249`, `main.rs:339` |
| 3 | CLAUDE.md schema mismatch | **FIXED** — Schema matches `graph.rs:59-68` exactly |
| 4 | Generated instructions say "4 tools" | **FIXED** — `main.rs:358` says "5 total", lists all 5 |
| 5 | Duplicate `boost_similarity` | **FIXED** — `db.rs:27` imports `use crate::boost_similarity;` |
| 6 | Similarity exceeds 1.0 | **FIXED** — `tools.rs:594` has `.min(1.0)` |
| 7 | Silent error swallowing | **FIXED** — All `let _ =` replaced with `if let Err(e)` + `tracing::warn!` |
| 8 | `timestamp_opt().unwrap()` panic | **FIXED** — Uses `.single().unwrap_or_else(Utc::now)` at `db.rs:1070-1072,1079-1081` |
| 9 | Inconsistent WAL mode | **FIXED** — `access.rs:33` and `consolidation.rs:35` both set WAL |
| 10 | Fire-and-forget tasks unmonitored | **FIXED** — `.instrument(tracing::info_span!(...))` + error logging in spawned futures |
| 11 | No rate limiting | **FIXED** — `server.rs:202-226` implements 60/min with `compare_exchange` |
| 12 | TOCTOU race in conflict detection | **FIXED** — Conflict detection moved to after store (`db.rs:354-365`) |
| 13 | Model download without integrity check | **FIXED** — Pinned revision + TOFU SHA256 for all 3 files |
| 14 | `split_at_sentences` ASCII-only | **FIXED** — `chunker.rs:119` matches Unicode sentence terminators |
| 15 | Unused `jsonrpc-core` dependency | **FIXED** — Removed from `Cargo.toml` |
| 16 | `lib.rs` docstring says "4 tools" | **FIXED** — Says "5 tools" |
| 17 | No expiration cleanup | **FIXED** — `db.rs:826-850` `cleanup_expired()`, triggered every 10th recall |
| 18 | Config file permissions | **EXCLUDED** per user request |
| 19 | `find_project_root` unbounded traversal | **FIXED** — `lib.rs:207-209` has `depth >= 100` guard |

### Round 2 Issues (10 new): All resolved

| # | Issue | Status |
|---|-------|--------|
| NEW-1 | `partial_cmp().unwrap()` NaN panic | **FIXED** — All 3 sites now use `.unwrap_or(std::cmp::Ordering::Equal)` (`db.rs:538-542`, `db.rs:605-609`, `tools.rs:597-601`) |
| NEW-2 | Rate limiter race condition | **FIXED** — Uses `SeqCst` ordering and `compare_exchange` for window reset (`server.rs:208-217`) |
| NEW-3 | TOFU only on model.safetensors | **FIXED** — `embedder.rs:217-219` now verifies all 3 files: `model.safetensors`, `tokenizer.json`, `config.json` |
| NEW-4 | `expire_item` non-atomic | **FIXED** — Now insert-before-delete (`db.rs:764-778`), comment at line 764 confirms pattern |
| NEW-5 | Unquoted int in cleanup filter | **FIXED** — Comment added at `db.rs:834`: "now is a system-generated i64 timestamp, no string sanitization needed" |
| NEW-6 | Silent error in `transfer_edges` | **FIXED** — `graph.rs:345-347` now logs: `tracing::warn!("transfer edge to {} failed: {}", neighbor, e)` |
| NEW-7 | O(n^2) co-access growth | **FIXED** — `graph.rs:228-232` limits to first 3 IDs (max 3 pairs instead of 10) |
| NEW-8 | Stale `replace` description | **FIXED** — `tools.rs:65` now says "stores new item first, then deletes old" |
| NEW-9 | Counter sharing (informational) | **ACCEPTED** — Low risk, not a bug |
| NEW-10 | install.sh silent skip | **FIXED** — `install.sh:73` now outputs `WARNING:` to stderr |

---

## Fresh Audit: Round 3 Findings

### R3-1. `expire_item` re-embeds content unnecessarily on every soft-delete

**Severity: LOW**

`db.rs:759-761`:
```rust
let embedding_text = item.embedding_text();
item.embedding = self.embedder.embed(&embedding_text)?;
```

Every call to `expire_item` (triggered during consolidation merges) regenerates the 384-dim embedding from scratch. This is computationally expensive (~5-10ms per embed) and unnecessary since the content hasn't changed. The `get_item()` method returns items with empty embeddings because the vector column isn't read back from Arrow batches (`db.rs:1088: embedding: Vec::new()`), forcing the re-embed.

**Recommendation:** Either read back the vector column in `batch_to_items`, or store expired items by modifying the filter expression rather than delete-and-reinsert.

---

### R3-2. `get_item` and `batch_to_items` discard embedding data

**Severity: LOW**

`db.rs:1088` always sets `embedding: Vec::new()` when deserializing items from Arrow batches. This means any code path that reads an item and then re-stores it (like `expire_item`) must re-embed, wasting compute. It also means `get_items_batch` returns items without embeddings, which could be surprising to callers.

---

### R3-3. `detect_clusters` SQL query may produce false triangles

**Severity: LOW**

`graph.rs:357-361`:
```sql
JOIN graph_edges e2 ON e1.to_id = e2.from_id ...
JOIN graph_edges e3 ON e2.to_id = e3.to_id AND e3.from_id = e1.from_id ...
```

This query only detects triangles where edges follow a specific directional pattern (A→B→C and A→C). Since edges are stored unidirectionally (UNIQUE on `from_id, to_id, edge_type`), many real triangles where edges were inserted in different orders (e.g., B→A, C→B, A→C) will be missed. The query would need to check both `from_id` and `to_id` for each join to detect all triangles.

---

### R3-4. `graph_nodes.project_id` has `NOT NULL DEFAULT ''` but nullable in CLAUDE.md docs

**Severity: LOW**

`graph.rs:55`: `project_id TEXT NOT NULL DEFAULT ''`
`CLAUDE.md:110`: `project_id TEXT` (implies nullable)

The actual schema uses a non-null empty string as the default for missing project IDs, while the documentation implies nullable. Functionally harmless, but the docs are slightly inaccurate.

---

### R3-5. `transfer_edges` uses `.filter_map(|r| r.ok())` silently dropping row errors

**Severity: LOW**

`graph.rs:340`:
```rust
.filter_map(|r| r.ok())
```

While individual edge creation errors are now logged (R2 fix), row-level read errors during the initial query are silently dropped. If a row in the result set fails to deserialize (e.g., schema mismatch), it's silently skipped with no indication.

---

### R3-6. Rate limiter allows first request of a new window unconditionally

**Severity: LOW**

`server.rs:211-217`: When the `compare_exchange` succeeds (new window), the count is set to 1 and the request proceeds. However, if the `compare_exchange` fails (another thread already reset), execution falls through without incrementing the counter at all — that request gets a "free" call that isn't counted.

Specifically, if `compare_exchange` fails at line 211, the `else` branch at line 218 is not reached because the outer `if` at line 209 was already entered. The request proceeds with no count increment.

---

### R3-7. `access_log` schema migration uses silent error for `ALTER TABLE`

**Severity: LOW**

`access.rs:48-50`:
```rust
let _ = conn.execute_batch(
    "ALTER TABLE access_log ADD COLUMN validation_count INTEGER NOT NULL DEFAULT 0;",
);
```

This silently ignores the error if the column already exists (expected) but also silently ignores any *other* error (unexpected). A more defensive approach would check if the error message contains "duplicate column" before ignoring it.

---

### R3-8. `MCP_VERSION` uses `2024-11-05` — may be outdated

**Severity: LOW**

`protocol.rs:10`: `pub const MCP_VERSION: &str = "2024-11-05";`

The MCP protocol version is hardcoded to `2024-11-05`. If the MCP specification has been updated since then, clients expecting a newer protocol version may have compatibility issues.

---

### R3-9. No graceful shutdown — background tasks and SQLite connections not cleaned up

**Severity: LOW**

The MCP server (`server.rs:79-96`) runs a `for line in reader.lines()` loop. When stdin is closed (client disconnects), the loop exits and the process terminates. Any in-flight `tokio::spawn` tasks (consolidation, co-access recording, clustering, cleanup) may be dropped mid-execution. SQLite connections in those tasks won't be properly closed, which could leave WAL checkpoint files behind.

---

### R3-10. `ensure_vector_index` silently degrades to brute-force search

**Severity: LOW**

`db.rs:228-237`: If index creation fails, the warning is logged but search continues with brute-force scans. This is intentional and documented in the comment, but there's no mechanism to retry index creation later. Once it fails, every subsequent search will be brute-force until the server restarts.

---

## Summary

| Severity | Count | Issues |
|----------|-------|--------|
| **Critical** | 0 | None |
| **High** | 0 | None |
| **Medium** | 0 | None |
| **Low** | 10 | Unnecessary re-embedding (#R3-1), discarded embeddings (#R3-2), false triangle detection (#R3-3), doc mismatch (#R3-4), silent row errors (#R3-5), rate limiter off-by-one (#R3-6), schema migration error handling (#R3-7), potentially stale MCP version (#R3-8), no graceful shutdown (#R3-9), degraded index (#R3-10) |

### Assessment

All critical, high, and medium severity issues from the previous two audit rounds have been successfully addressed. The remaining findings are all low severity — mostly minor robustness improvements, documentation accuracy, and edge cases that are unlikely to manifest in normal operation. The codebase is in good shape from a security perspective.

### Optional Improvements (not urgent)

1. **Read embedding vectors back from LanceDB** in `batch_to_items` to avoid unnecessary re-embedding in `expire_item` (#R3-1, #R3-2)
2. **Bidirectional triangle detection** in `detect_clusters` (#R3-3)
3. **Handle `compare_exchange` failure** in rate limiter by falling through to the count-increment path (#R3-6)
