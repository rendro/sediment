# Sediment Repository Audit

**Date:** 2026-01-31
**Scope:** Full codebase — security, data integrity, logic errors, error handling

---

## Critical

### 1. Cross-database inconsistency on replace failure (data loss)

**`src/mcp/tools.rs:462-497`**

The `execute_store` replace workflow spans three databases with no transactional guarantees. Each step (transfer edges, add SUPERSEDES edge, delete old item, remove old node) catches errors with `tracing::warn!` and continues. If any step fails or the process crashes mid-sequence:

- Old item may remain in LanceDB while a SUPERSEDES edge already points to the new one
- Graph edges may be partially transferred (some on old node, some on new)
- Old graph node may persist after the LanceDB item is deleted

There is no repair or rollback mechanism.

### 2. `expire_item` delete-then-insert has a data loss window

**`src/db.rs:757-804`**

`expire_item` deletes the item from LanceDB (line 776), then re-inserts it with updated `expires_at` (line 785). If the process crashes between delete and re-insert, the item is permanently lost. The 3-retry loop only handles transient insert failures, not crashes. The comment on line 474 ("store-before-delete ensures no data loss on crash") shows this pattern was considered for replace, but `expire_item` uses the opposite order.

### 3. No shared transactions across GraphStore and AccessTracker

**`src/mcp/tools.rs:278-283`**

Each tool call opens separate `rusqlite::Connection` objects to the same `access.db` file for `AccessTracker`, `GraphStore`, and `ConsolidationQueue`. Background tasks (`spawn_logged`, `spawn_consolidation`) open additional connections concurrently. Operations like "record access + record co-access + update consolidation queue" are not atomic. Under concurrent access, partial updates can leave inconsistent state.

---

## High

### 4. Chunking threshold uses byte length, not character count

**`src/chunker.rs:87`**

```rust
if content.len() < config.min_chunk_threshold {
```

The caller in `db.rs:294` uses `.chars().count()` for the `CHUNK_THRESHOLD` check, but the chunker itself uses `.len()` (byte length). For multi-byte UTF-8 content, the two checks disagree — `store_item` may decide to chunk content that the chunker then skips (or vice versa), producing incorrect results.

### 5. `split_by_chars` can panic on multi-byte UTF-8

**`src/chunker.rs:301-320`**

When `find_word_boundary_bytes` fails to find a boundary, the fallback on line 312-313 computes `(start + max_size).min(text.len())` in bytes. On line 320, `text[start..actual_end]` will panic if `actual_end` falls in the middle of a multi-byte UTF-8 character ("byte index N is not a char boundary").

### 6. No input size validation on recall query

**`src/mcp/tools.rs:709-717`**

The `store` tool rejects content over 1MB (line 348), but `recall` has no size limit on the `query` string. A multi-GB query string would be passed to the tokenizer, which allocates memory proportional to input before truncation to 512 tokens. This is a denial-of-service vector.

### 7. Forced runtime shutdown kills in-flight consolidation

**`src/mcp/server.rs:108-111`**

```rust
drop(ctx);
rt.shutdown_timeout(std::time::Duration::from_secs(2));
```

Background consolidation tasks open new DB connections, perform vector searches, and modify multiple databases. The 2-second shutdown timeout will forcibly kill long-running tasks mid-operation, potentially leaving databases inconsistent.

### 8. Similarity scores become semantically meaningless after boosting

**`src/mcp/tools.rs:609-612`**

The `base_score` from `score_with_decay` can already exceed the original similarity (frequency > 1.0 when access_count > 0). Then it's multiplied by `trust_bonus` (always >= 1.0). While `.min(1.0)` caps the final value, items with high access counts are ranked by popularity rather than actual semantic similarity. The field name "similarity" becomes misleading.

---

## Medium

### 9. Filesystem path leakage in provenance metadata

**`src/mcp/tools.rs:414`**

```rust
"project_path": ctx.cwd.to_string_lossy()
```

The working directory (e.g., `/home/username/secret-project/`) is stored in item metadata and exposed via `recall` cross-project results (line 770). This leaks local filesystem structure to API consumers.

### 10. `delay_for_attempt` overflows on large attempt numbers

**`src/retry.rs:54`**

```rust
let delay_ms = self.initial_delay_ms * 2u64.pow(attempt);
```

`2u64.pow(64+)` panics in debug mode. While `max_attempts` defaults to 3, `RetryConfig::new` accepts arbitrary `u32` — any value >= 64 triggers a panic.

### 11. Dangling graph edges from `related` parameter

**`src/mcp/tools.rs:500-506`**

When `store` is called with `related` IDs, edges are created without verifying those IDs exist as graph nodes. The `connections` tool will then return references to non-existent items.

### 12. `Relaxed` atomic ordering on recall counter

**`src/mcp/tools.rs:822-825`**

```rust
let run_count = ctx.recall_count.fetch_add(1, Ordering::Relaxed);
if run_count % 10 == 9 {
```

`Relaxed` ordering allows two concurrent calls to see the same pre-increment value, potentially both triggering (or both skipping) periodic maintenance. The current single-threaded `block_on` architecture makes this unlikely but not impossible if the server model changes.

---

## Low

### 13. Multiple `unwrap()` on `serde_json::to_string_pretty`

**`src/mcp/tools.rs:541, 876, 941, 972, 1033`**

These would panic on serialization failure. Unlikely with constructed `json!()` values but violates the principle of not panicking in server code.

### 14. `unwrap()` on Arrow downcast

**`src/db.rs:924`**

```rust
let id_array = id_col.as_any().downcast_ref::<StringArray>().unwrap();
```

Panics if the column type changes due to schema migration.

---

## Summary

| Severity | Count | Key Theme |
|----------|-------|-----------|
| Critical | 3 | No cross-database atomicity; data loss on crash |
| High | 5 | UTF-8 panics, missing input validation, unclean shutdown |
| Medium | 4 | Info leakage, overflow, dangling references |
| Low | 2 | Unwrap in server paths |

The fundamental architectural risk is the three-database design (LanceDB + 2x SQLite connections) with no transactional guarantees across them. Every multi-step operation that spans databases can leave inconsistent state on failure. The most impactful fix would be consolidating GraphStore and AccessTracker into a single shared SQLite connection with proper transactions, and implementing compensating actions (or at minimum, logging enough to detect and repair) for cross-database operations involving LanceDB.
