# Sediment Security & Code Audit

**Date:** 2026-01-31
**Scope:** Full codebase review — security, concurrency, logic, error handling

---

## Critical Issues

### 1. Rate Limiter Race Condition (server.rs:228-231)

After a successful CAS on `rate_limit_window`, the code does a **non-atomic** `store(1)` on `rate_limit_count`. Between the CAS success and the store, another thread could read the stale count, pass the rate limit check via `fetch_add`, and then have its count overwritten by the `store(1)`. This creates a window where the count is reset while concurrent calls have already incremented it, effectively losing count increments and allowing burst bypasses.

```rust
// Thread A wins CAS, about to store(1)
// Thread B enters the else branch, does fetch_add on stale count, passes limit check
// Thread A stores 1, erasing Thread B's increment
ctx.rate_limit_count.store(1, Ordering::SeqCst);
```

**Impact:** Rate limit can be bypassed under concurrent load. In practice, this is mitigated because `handle_call_tool` runs synchronously on a single-threaded stdin loop via `rt.block_on`, but if the server ever moves to concurrent request handling, this becomes exploitable.

### 2. Data Loss Window in `expire_item` (db.rs:767-781)

The `expire_item` function deletes an item then re-inserts it with an updated `expires_at`. If the process crashes between delete and re-insert, the item is permanently lost. The comment at line 769 acknowledges this but says "delete-then-insert because both rows share the same id" — however this is a fundamental design limitation of the LanceDB update pattern.

**Impact:** Data loss on crash during consolidation soft-delete.

### 3. SQL Filter Injection via `sanitize_sql_string` (db.rs:11-13)

The sanitization only escapes single quotes (`'` → `''`). LanceDB's SQL filter dialect may support backslash escapes or other injection vectors depending on the underlying query engine. If LanceDB ever adopts a SQL dialect where `\'` escapes a quote, inputs containing backslashes followed by quotes could break out.

```rust
fn sanitize_sql_string(s: &str) -> String {
    s.replace('\'', "''")
}
```

All user-controlled strings (item IDs, project IDs) flow through this function into `only_if()` and `delete()` filter expressions. Item IDs are UUIDs (safe), but project IDs come from files on disk that could be tampered with.

**Impact:** Low currently (UUIDs are generated internally), but fragile if inputs ever come from untrusted sources.

---

## High Severity Issues

### 4. Unbounded Consolidation Queue Growth

The consolidation queue cleanup (`cleanup_processed` in consolidation.rs:114) only deletes entries with status != 'pending' older than 7 days. **Pending entries are never cleaned up.** If consolidation repeatedly fails for certain pairs (e.g., items deleted between enqueue and processing), pending entries accumulate indefinitely.

**Impact:** Unbounded SQLite table growth over time.

### 5. `record_validation` Called on Wrong Item (tools.rs:470)

When replacing an item, `record_validation` is called with the **old** item's ID:

```rust
if let Err(e) = tracker.record_validation(old_id, now_ts) {
```

But the old item has just been deleted (line 466). The validation count is recorded against a deleted item's ID, which will never be queried again. The validation should be recorded against `new_id` to properly track that the new item is a validated replacement.

**Impact:** Trust scoring never benefits from replace operations — `validation_count` always stays at 0 for live items that were created via replace.

### 6. Graph Node Lifecycle Inconsistency (tools.rs:459-478)

On replace:
1. Graph node created for `new_id` (line 459)
2. Old item deleted from LanceDB (line 466)
3. Supersedes edge added `new_id -> old_id` (line 473)
4. Old node **removed** from graph (line 476)

Step 4 removes the old node and **all its edges** (graph.rs:107-116), including the supersedes edge just created in step 3. The `transfer_edges` call is missing here (unlike in consolidation.rs:244), so all of the old item's graph relationships are silently lost.

**Impact:** Replace operations destroy the old item's entire relationship graph instead of transferring it.

---

## Medium Severity Issues

### 7. `get_neighbors` Returns Input IDs as Neighbors

The `get_neighbors` query (graph.rs:182-190) uses a CASE expression to pick the "other" side of an edge. But when an edge connects two IDs that are **both** in the input set, it returns one of them as a "neighbor" even though it's already in the input. The caller in `recall_pipeline` (tools.rs:623) filters against `existing_ids`, but this is a set of search result IDs, not the input `top_ids`. If an item is in `top_ids` but not in `existing_ids` (unlikely but possible with future changes), it would appear as a graph-expanded result pointing to itself.

### 8. Co-Access Edge Directionality

`record_co_access` (graph.rs:241-257) always stores edges as `(item_ids[i], item_ids[j])` where `i < j` (lexicographic order of array position, not ID value). But `get_co_accessed` (graph.rs:276-279) searches both `from_id` and `to_id`. This works but means the same logical co-access pair could end up stored as both `(A,B)` and `(B,A)` if the order of items in the result array changes between recalls. The UNIQUE constraint prevents this per direction, but swapped orderings create duplicate edges.

**Impact:** Double-counting of co-access relationships. The deduplication at line 312 mitigates this for reads, but the database accumulates redundant edges.

### 9. Consolidation Processes Stale Similarity Scores

Consolidation candidates are enqueued with a similarity score computed at store time (tools.rs:495). By the time consolidation runs (potentially much later), the items may have been modified via `replace`, making the stored similarity inaccurate. Consolidation uses the stale score to decide merge vs. link (consolidation.rs:230: `candidate.similarity >= 0.95`).

**Impact:** False merges or missed merges based on outdated similarity data.

### 10. No Input Validation on `limit` Parameter

`RecallParams.limit` is `Option<usize>` (tools.rs:188). While capped at 100 (tools.rs:699), the `usize` type on a 64-bit system allows values up to 2^64-1 before the `.min(100)` cap. The JSON deserialization of a negative number or floating point for `usize` would fail gracefully, but a very large number (e.g., `99999999999999`) would parse successfully and only be capped after allocation-related arithmetic in `search_items` (line 403: `limit * 2`, line 463: `limit * 3`), which could overflow on 32-bit systems.

**Impact:** Minimal on 64-bit (`.min(100)` prevents issues), but the `limit * 2` and `limit * 3` multiplications could theoretically overflow before the cap is applied if the cap were removed.

### 11. YAML Detection False Positives (db.rs:913-923)

The YAML content type detection triggers on any text containing `:\n` and a line with `:` that doesn't start with `http`. This matches many non-YAML formats, e.g.:

```
Dear John:
I wanted to write you about something.
Subject: important matter
```

This would be detected as YAML and chunked with the YAML chunker, which would produce poor results.

### 12. Silent PRAGMA Failures (graph.rs:50-51, access.rs:33-34, consolidation.rs:35-36)

All three SQLite stores ignore PRAGMA failures with `.ok()`:

```rust
conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")
    .ok();
```

If WAL mode fails to activate (e.g., due to filesystem restrictions), the database falls back to DELETE journal mode with no warning. This can cause lock contention issues under concurrent access from background tasks.

---

## Low Severity Issues

### 13. `sha2` Dependency Unused

`Cargo.toml` includes `sha2 = "0.10"` but no code references it. This was likely left over from the removed TOFU hash verification (embedder.rs:221-224).

### 14. Expired Items Never Cleaned from Chunks Table

`cleanup_expired` (db.rs:839-862) only deletes from the items table. Chunks belonging to expired items remain in the chunks table indefinitely, wasting storage and potentially appearing in chunk search results (the chunk search has no expiration filter).

### 15. `chunk_index` Truncation (db.rs:1147)

`chunk.chunk_index` is `usize` but cast to `i32` for Arrow storage:

```rust
let chunk_index = Int32Array::from(vec![chunk.chunk_index as i32]);
```

If an item has more than 2^31 chunks (impossible in practice given the 1MB limit), this silently overflows.

### 16. Background Task Error Swallowing

Numerous background `tokio::spawn` tasks (tools.rs:793-804, 813-836, 842-854) log warnings but never propagate errors. If a background task panics, it's silently lost. No monitoring or health checks exist for background task failures.

### 17. `ListScope` Default Inconsistency

`ListScope` defaults to `All` (lib.rs:82), but the `list` tool defaults to `"project"` scope (tools.rs:897). The schema documents `"project"` as default (tools.rs:125), matching the tool behavior, but the Rust type default doesn't match.

### 18. Project Config TOCTOU

`get_or_create_project_id` (lib.rs:150-173) checks if the config file exists, reads it, and if missing, creates it. Two concurrent processes starting in the same project directory could both see the file as missing and generate different UUIDs, with the last writer winning.

---

## Positive Observations

- SQL parameter binding used correctly in all SQLite queries (graph.rs, access.rs)
- NaN/Inf guards in `score_with_decay` prevent scoring corruption
- 1MB content limit prevents OOM during embedding
- Store-before-delete pattern in replace prevents data loss
- Semaphore(1) prevents concurrent consolidation correctly
- UUIDs for all item/chunk IDs eliminates ID collision concerns
- Proper use of `serde(skip)` to avoid leaking embeddings in JSON
- WAL mode and busy_timeout configured for SQLite concurrency
- Depth limit on `find_project_root` prevents infinite loops
