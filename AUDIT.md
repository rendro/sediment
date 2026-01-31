# Sediment Security & Code Audit Report

**Date**: 2026-01-31
**Scope**: Full codebase review of all 16 Rust source files
**Commit**: 1bfcc46

---

## Critical Issues

### 1. Race condition in `expire_item` causes data duplication (db.rs:748-779)

`expire_item()` does insert-before-delete to avoid data loss, but the delete filter is `id = '<id>'` which will delete **both** the original row and the newly inserted row, since they share the same `id`. LanceDB doesn't have transactional semantics here, so if the insert succeeds but the delete uses the same filter, the item disappears entirely rather than being updated.

**Impact**: Data loss on every `expire_item` call. The "insert-before-delete" pattern is self-defeating because the delete matches both rows.

### 2. Rate limiter is trivially bypassable and has a race condition (server.rs:209-237)

The rate limiter uses non-atomic check-then-act with `compare_exchange` on the window but a separate `fetch_add` on the count. When the CAS to reset the window fails (another call reset it), the code sets `in_current_window = true` and proceeds to increment the old window's counter, but the window was already reset — so the count is stale. Multiple concurrent calls can all slip through the rate limit.

Additionally, the rate limiter returns a **success** response (`Response::success`) with an error-like `CallToolResult::error` body. This means the JSON-RPC layer reports success while the tool reports an error, which is inconsistent but technically correct for MCP.

**Impact**: Rate limiting is unreliable under concurrent load (though the server is single-threaded for stdio, this is mostly theoretical).

### 3. SQLite concurrent access without proper locking (access.rs, graph.rs, consolidation.rs)

Multiple SQLite `Connection` objects are opened to the **same file** (`access.db`) from different threads/tasks:
- `AccessTracker::open()` in the main tool handler
- `GraphStore::open()` in the main tool handler
- `ConsolidationQueue::open()` during store
- Background `spawn_consolidation` opens its own connections
- Background co-access recording opens another `GraphStore`

While WAL mode helps with concurrent reads, concurrent writers to SQLite will encounter `SQLITE_BUSY` errors. The code does not set `busy_timeout`, so any write contention will fail immediately rather than retrying.

**Impact**: Background tasks (consolidation, co-access recording, clustering) may silently fail with "database is locked" errors under load. These failures are logged as warnings but data is lost.

---

## High Severity Issues

### 4. TOFU hash verification is security theater (embedder.rs:236-259)

The trust-on-first-use hash verification stores hashes as `.sha256` sidecar files next to model files. However:
- The hash is written by the same process that downloads the file, so a compromised download writes a matching hash
- If an attacker can modify the model file, they can also modify the `.sha256` file (same directory, same permissions)
- The pinned revision (`e4ce9877abf3edfe10b0d82785e83bdcb973e22e`) provides more actual security than TOFU

**Impact**: False sense of security. The pinned git revision is the real protection; TOFU adds complexity without meaningful security.

### 5. `unwrap()` on path conversion can panic (db.rs:142)

```rust
let db = connect(path.to_str().unwrap())
```

If the database path contains non-UTF-8 characters (valid on Linux), this will panic and crash the server.

**Impact**: Server crash on non-UTF-8 filesystem paths.

### 6. Consolidation merge deletes data then fails silently (consolidation.rs:262-273)

When `expire_item` fails, the code falls back to `delete_item`, which is a hard delete. But the error from `expire_item` might indicate the item was already partially modified. The fallback to hard delete after a failed soft-delete could result in losing data that was supposed to be preserved.

**Impact**: Potential data loss during consolidation if `expire_item` fails mid-operation.

### 7. Graph `get_neighbors` SQL query has incorrect parameter binding (graph.rs:164-219)

The SQL query uses `{ph}` three times (for `from_id IN`, `to_id IN`, and the initial CASE), but only one set of parameter values is provided. The query references `?1` through `?N` for IDs plus `?{strength_idx}` for the strength threshold. However, each `{ph}` placeholder expands to the same `?1,...,?N` positional parameters, which is correct for SQLite (same parameters reused). This is actually fine, but the code structure makes it look like a bug and is fragile.

---

## Medium Severity Issues

### 8. No input validation on `limit` parameters (tools.rs:699, 869)

`limit` from user input is used directly in database queries without upper-bound clamping. A malicious or buggy client could request `limit: 999999999`, causing the server to load massive result sets into memory.

**Impact**: Potential OOM/DoS via large limit values.

### 9. `delete_item` returns `Ok(true)` even when no rows were deleted (db.rs:782-803)

The `delete_item` method always returns `Ok(true)` if the items table exists, regardless of whether any rows were actually deleted. LanceDB's delete doesn't report affected row count, so the method cannot distinguish between "deleted" and "not found".

**Impact**: `forget` tool reports success even when the item doesn't exist.

### 10. Chunking offset tracking is incorrect for JSON (chunker.rs:829, 847, 866, etc.)

JSON chunking sets `start_offset: 0` and `end_offset: 0` for all chunks because the content is re-serialized from parsed JSON rather than sliced from the original string. These offsets are meaningless.

**Impact**: Chunk offset metadata is inaccurate for JSON content; any feature relying on offsets to map back to original content will be wrong.

### 11. `find_word_boundary_bytes` operates on raw bytes, not UTF-8 characters (chunker.rs:347-356)

The function scans individual bytes looking for space/newline. If the text contains multi-byte UTF-8 characters, slicing at a byte position found by this function could split a UTF-8 character, causing a panic when creating a `&str` slice.

However, the function only looks for ASCII bytes (space `0x20`, newline `0x0A`), which cannot appear as continuation bytes in UTF-8. The actual slicing at `find_word_boundary_bytes` result is safe because it returns the position *after* an ASCII byte. So this is safe in practice but fragile — any future extension to look for non-ASCII break characters would be dangerous.

### 12. Background task error handling is fire-and-forget (tools.rs:782-855)

All background tasks (`spawn_consolidation`, co-access recording, clustering, expired cleanup) use `tokio::spawn` with no error propagation. If they panic, the panic is silently swallowed by the Tokio runtime. There's no monitoring, no health checks, and no way for the operator to know if background tasks are failing consistently.

### 13. `consolidation_run_count` serves double duty (tools.rs:807-810)

The counter `consolidation_run_count` is named for consolidation but actually controls periodic clustering and expired cleanup. The name is misleading, and the counter is incremented on every recall regardless of whether consolidation actually runs.

---

## Low Severity Issues

### 14. Code detection heuristic has false positives (db.rs:913-928)

`detect_content_type` checks for patterns like `"let "`, `"const "`, `"import "` which commonly appear in English prose. A paragraph containing "let me explain" or "import regulations" would be misidentified as code.

### 15. YAML key detection is naive (chunker.rs:960-964)

The YAML key line detection `!trimmed.starts_with('"')` doesn't handle all YAML edge cases. Quoted keys, flow mappings, and complex YAML constructs may be misidentified.

### 16. No cleanup of consolidation queue (consolidation.rs)

Processed entries in `consolidation_queue` (status = 'merged' or 'linked') are never deleted. Over time this table will grow unboundedly.

### 17. `Embedder` is not `Send` (embedder.rs)

The `Embedder` wraps `BertModel` and `Tokenizer`. If these types are not `Send`, sharing via `Arc<Embedder>` across `tokio::spawn` tasks would fail to compile. Currently this works because the embedder is only used synchronously on the main thread (via `rt.block_on`), but future refactoring could hit issues.

### 18. Missing `PRAGMA busy_timeout` on SQLite connections

Neither `AccessTracker::open`, `GraphStore::open`, nor `ConsolidationQueue::open` set `PRAGMA busy_timeout`. With WAL mode and multiple connections, writes will fail instantly instead of waiting for locks. Setting `busy_timeout = 5000` would make concurrent access more robust.

### 19. `project_id` scoping inconsistency in `list_items` (db.rs:638-641)

When scope is `Project` and `self.project_id` is `None`, no project filter is applied, so **all items** are returned instead of an empty set. This silently degrades to `All` scope.

### 20. `score_with_decay` ignores NaN/Inf edge cases (db.rs:863-878)

If `similarity` is NaN (possible from corrupted vector data), the function propagates NaN through all arithmetic without any guard.

---

## Architectural Observations

### Strengths
- SQL injection prevention via `sanitize_sql_string` is consistently applied
- Content size limit (1MB) prevents OOM during embedding
- Store-before-delete pattern in `replace` shows defensive thinking
- Semaphore-based consolidation prevents concurrent merges
- Pinned model revision prevents supply-chain drift

### Design Concerns
- **Three separate SQLite connections** to the same file (AccessTracker, GraphStore, ConsolidationQueue) is an anti-pattern. A single connection pool or shared connection would be more robust.
- **Synchronous embedding on the server thread** blocks the entire MCP server during embedding operations. Long content with many chunks will cause noticeable latency.
- **No authentication/authorization** on the MCP server. Any process that can connect to stdio can store, recall, and delete all memories. This is inherent to the MCP stdio model but worth noting.
- **No backup/export mechanism**. The central database at `~/.sediment/data/` is the single source of truth with no built-in backup.

---

## Summary

| Severity | Count |
|----------|-------|
| Critical | 3 |
| High | 4 |
| Medium | 6 |
| Low | 7 |

The most impactful issues are the `expire_item` data loss bug (#1), the lack of SQLite busy timeout (#3/#18), and the unbounded `limit` parameter (#8). The security surface is relatively small given the local-first architecture — the main risks are data integrity rather than remote exploitation.
