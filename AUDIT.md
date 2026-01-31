# Sediment Repository Audit

**Date:** 2026-01-31
**Scope:** Full source code review of all `.rs` files
**Severity scale:** Critical > High > Medium > Low > Info

---

## Critical Issues

### 1. Data loss in `expire_item` on concurrent access
**File:** `src/db.rs:756-803`
**Severity:** Critical

`expire_item()` performs a delete-then-reinsert pattern. If the process crashes or is killed between the delete and the re-insert, the item is permanently lost. The 3-retry loop mitigates transient insert failures but cannot recover from a process crash. This is the only non-idempotent destructive operation in the codebase.

**Impact:** Permanent data loss during consolidation merges (the primary caller of `expire_item`).

### 2. Race condition between `delete_item` existence check and delete
**File:** `src/db.rs:807-834`
**Severity:** High

`delete_item()` calls `get_item()` to check existence, then separately deletes. Between these two operations, another connection could delete or modify the item (TOCTOU). The existence check return value (`Ok(true)`) is therefore unreliable.

### 3. Unbounded graph expansion query in `get_neighbors`
**File:** `src/graph.rs:170-229`
**Severity:** High

`get_neighbors()` has no `LIMIT` clause on its SQL query. With a large graph, a single node with thousands of edges could return an unbounded result set, consuming excessive memory and time. The `get_co_accessed` query similarly lacks a result count limit.

### 4. `remove_node` deletes SUPERSEDES edges, breaking lineage
**File:** `src/graph.rs:107-121` and `src/mcp/tools.rs:490-496`

In the replace workflow (`execute_store`), the code creates a SUPERSEDES edge from new→old, then calls `remove_node(old_id)` which deletes *all* edges including the just-created SUPERSEDES edge. The SUPERSEDES lineage is destroyed immediately after creation.

**Impact:** The provenance chain (SUPERSEDES edges) that documents item replacement history is never preserved. `connections` tool will never show supersedes relationships for replaced items.

---

## High Issues

### 5. No authorization or access control on MCP tools
**Severity:** High

Any process that can communicate over stdio can invoke all tools including `forget` (delete) and `store` (write). There is no authentication, no per-tool permissions, and no audit log of destructive operations beyond tracing. In a multi-user or networked scenario, any connected client can delete all stored memories.

### 6. SQLite database shared between three independent connection types without coordination
**File:** `src/mcp/tools.rs:269-283`
**Severity:** High

`AccessTracker`, `GraphStore`, and `ConsolidationQueue` each open independent `rusqlite::Connection` instances to the same `access.db` file. While WAL mode permits concurrent readers, concurrent writers will serialize on SQLite's write lock with only a 5-second busy timeout. Background consolidation, co-access recording, and clustering all write concurrently from `tokio::spawn` tasks, creating contention.

**Impact:** Under moderate load, `SQLITE_BUSY` errors after the 5-second timeout, causing silent failures in background tasks.

### 7. Rate limiter is trivially bypassable by window boundary
**File:** `src/mcp/server.rs:218-238`
**Severity:** Medium

The rate limiter resets the full window on expiry, allowing a burst of 60 calls in the last second of a window followed by 60 calls in the first second of the next window (120 calls in ~2 seconds). A sliding window or token bucket would be more robust.

Additionally, the first call in a new window sets `count = 1` and skips the limit check, so the effective limit is 61 calls per window.

### 8. Background task panics are logged but may leave semaphore permits unreturned
**File:** `src/mcp/tools.rs:24-31` and `src/consolidation.rs:161-178`

`spawn_logged` wraps the future in `tokio::task::spawn` to catch panics, but `spawn_consolidation` acquires a semaphore permit *inside* the outer `tokio::spawn`. If the inner future panics after acquiring the permit, the panic is caught by `spawn_logged`'s inner `tokio::task::spawn`, and the `drop(permit)` line at `consolidation.rs:177` never runs. The semaphore permit leaks, permanently blocking all future consolidation.

Wait — actually re-reading: the `permit` is acquired in the outer spawn, and the inner `run_consolidation_batch` is called directly (not via the `spawn_logged` wrapper). So in `spawn_consolidation`, if `run_consolidation_batch` panics, the `drop(permit)` at line 177 is skipped because the panic unwinds past it. The outer `tokio::spawn` catches the panic, but the semaphore is poisoned.

**Impact:** A single panic in consolidation permanently disables all future consolidation.

---

## Medium Issues

### 9. Tag filtering applied post-query, not in the database
**File:** `src/db.rs:437-439` (search) and `src/db.rs:680-683` (list)

Tags are stored as a JSON string column. Tag filtering is done in Rust after fetching results. For `search_items`, this means the `limit * 2` vector search may return items that are all filtered out by tags, yielding fewer results than expected or none at all.

**Impact:** Recall with tag filters may return significantly fewer results than the requested limit, even when matching items exist in the database.

### 10. `sanitize_sql_string` is custom and incomplete
**File:** `src/db.rs:11-13`
**Severity:** Medium

The function only escapes backslashes and single quotes. While LanceDB's SQL dialect may only need these, the function is not tested against null bytes, unicode escape sequences, or other edge cases. Using parameterized queries would be safer, but LanceDB's filter API appears to only accept string expressions.

### 11. Consolidation can merge items across projects
**File:** `src/consolidation.rs:230-340`
**Severity:** Medium

`process_candidate` does not check whether `item_id_a` and `item_id_b` belong to the same project. Two items from different projects with >=0.95 cosine similarity will be merged, with one being soft-deleted. This violates project isolation.

### 12. `content.len()` used for chunking threshold instead of `content.chars().count()`
**File:** `src/db.rs:293`
**Severity:** Low

`item.content.len()` counts bytes, not characters. For UTF-8 content with multi-byte characters (CJK, emoji), the byte count can be 2-4x the character count, causing premature chunking. The `CHUNK_THRESHOLD` comment says "in characters" but the code checks bytes.

### 13. `chunk_index` cast from `usize` to `i32` can silently saturate
**File:** `src/db.rs:1234`

`i32::try_from(chunk.chunk_index).unwrap_or(i32::MAX)` silently saturates if there are more than 2^31 chunks. While unlikely, this means chunk ordering would be wrong for extremely large documents.

### 14. No limit validation on user-provided `limit` parameter
**File:** `src/mcp/tools.rs:717`

`recall` caps limit at 100, and `list` caps at 100. But the `limit * 2` and `limit * 3` multiplications in `search_items` (capped at 1000 internally) could still cause LanceDB to scan significant data. The internal cap at 1000 in `search_items` is good, but this means a user requesting `limit=100` causes internal queries of `limit=200` and `limit=300`.

### 15. `co_access` recording truncates to top 3 results silently
**File:** `src/graph.rs:238-242`

`record_co_access` silently truncates to the first 3 items. If a user recalls 5 items, only the first 3 get co-access edges. The truncation criteria (first 3 by position, not by similarity) may not capture the most relevant co-access pairs.

---

## Low / Informational Issues

### 16. `Relaxed` ordering on `recall_count` atomic
**File:** `src/mcp/server.rs:43` and `src/mcp/tools.rs:822-824`

`fetch_add(1, Ordering::Relaxed)` on `recall_count` means periodic maintenance (every 10th recall) may fire slightly more or less often than intended due to relaxed ordering. This is harmless for maintenance scheduling but technically imprecise.

### 17. Model pinning by git revision, not by content hash
**File:** `src/embedder.rs:203-207`

The model is pinned to a specific git revision on Hugging Face Hub. If Hugging Face allows force-pushing to that revision (which they do for some repos), the model content could change. A SHA-256 hash of the downloaded files would provide stronger integrity.

### 18. `get_or_create_project_id` race on rename
**File:** `src/lib.rs:176`

The comment acknowledges that concurrent processes may race on `fs::rename`. The "last writer wins" behavior means two concurrent `init` calls may create different project IDs, with one being silently overwritten. The re-read on line 185 mitigates this, but there's a TOCTOU window where the file could be deleted between rename and re-read.

### 19. No input validation on `id` parameters
**File:** `src/mcp/tools.rs:953-976` (forget), `src/mcp/tools.rs:984-990` (connections)

The `id` parameter in `forget` and `connections` is passed directly to `sanitize_sql_string` but there's no validation that it's a valid UUID. Malformed IDs will simply return "not found" but waste database queries.

### 20. `detect_content_type` has false positive potential
**File:** `src/db.rs:972-1043`

The content type detection uses heuristics that can misclassify content. For example, a text document starting with `let me explain` would match the `"let "` code pattern. The code mitigates this by checking line-start positions, but edge cases remain.

### 21. No database migrations or schema versioning
**Severity:** Low

There is no mechanism to handle schema changes in LanceDB tables. If the Arrow schema changes (e.g., adding a column), existing databases will fail to open. The SQLite tables use `CREATE TABLE IF NOT EXISTS` and one `ALTER TABLE ADD COLUMN` migration, but there's no version tracking.

### 22. Error messages may leak internal paths
**File:** Various

Error messages like `"Failed to open database: {}"` may include full filesystem paths in responses sent to MCP clients. This leaks information about the server's directory structure.

### 23. `NamedTempFile` in tests drops before `GraphStore::open`
**File:** `src/graph.rs:490-492`

In `open_test_graph()`, the `NamedTempFile` is created and `.path()` is extracted, but the `NamedTempFile` is dropped at the end of the function. On some platforms, the temp file may be unlinked when dropped, which would cause the SQLite database to be opened on a deleted path. This works on Linux (where open file handles keep the file alive) but is technically fragile.

### 24. Consolidation processes items that may have already been deleted
**File:** `src/consolidation.rs:253`

When `fresh_similarity` falls back to `candidate.similarity` because items are missing/have no embeddings, the code still proceeds with the old similarity value. If one item was deleted, the fallback incorrectly claims the items are similar.

### 25. `truncate` function panics on empty strings with `max_len < 3`
**File:** `src/mcp/tools.rs:1041-1052`

If `max_len` is less than 3 and the string is longer than `max_len`, `nth(max_len - 3)` wraps around due to underflow (since `max_len` is `usize`). In practice `max_len` is always 80 or 100, but the function is not defensive.

---

## Summary

| Severity | Count |
|----------|-------|
| Critical | 2 |
| High | 4 |
| Medium | 7 |
| Low/Info | 12 |

The most impactful issues are:
1. **Data loss risk** in `expire_item`'s delete-then-reinsert pattern (#1)
2. **Broken SUPERSEDES lineage** due to `remove_node` deleting the just-created edge (#4)
3. **Cross-project consolidation merging** violating project isolation (#11)
4. **Semaphore leak** on consolidation panic permanently disabling background tasks (#8)
