# Security & Code Audit Report

**Date:** 2026-01-30
**Scope:** Full repository audit of sediment-mcp v0.2.1
**Auditor:** Automated code review

---

## Critical: SQL Injection Vulnerabilities

### 1. Unparameterized SQL queries with user-controlled input

**Severity: CRITICAL**

Multiple locations construct SQL filter strings by interpolating user-supplied values directly into SQL, without parameterized queries:

- **`src/db.rs:619`** — `list_items()` interpolates `project_id` directly:
  ```rust
  filter_parts.push(format!("project_id = '{}'", pid));
  ```
  The `project_id` originates from a file on disk (`.sediment/config`), so the direct risk is lower, but it sets a dangerous pattern.

- **`src/db.rs:671`** — `get_item()` interpolates the `id` parameter:
  ```rust
  .only_if(format!("id = '{}'", id))
  ```
  The `id` is a UUID generated internally on store, but on `forget` and `connections` tools, it comes directly from user input (`ForgetParams.id`, `ConnectionsParams.id`). A malicious MCP client could inject LanceDB filter expressions via a crafted `id` value like `' OR 1=1; --`.

- **`src/db.rs:701-702`** — `get_items_batch()` interpolates multiple IDs:
  ```rust
  let quoted: Vec<String> = ids.iter().map(|id| format!("'{}'", id)).collect();
  ```

- **`src/db.rs:727-739`** — `delete_item()` interpolates `id` into delete filters:
  ```rust
  chunks_table.delete(&format!("item_id = '{}'", id))
  table.delete(&format!("id = '{}'", id))
  ```

**Impact:** While LanceDB's filter language is not full SQL, injection into filter expressions can allow unauthorized data access or deletion. An attacker controlling MCP tool input (e.g., via a compromised LLM prompt) could craft IDs to manipulate queries.

**Recommendation:** Sanitize or validate all IDs are valid UUIDs before interpolation, or use parameterized queries if LanceDB supports them.

---

### 2. SQLite queries in `graph.rs` use parameterized queries — but dynamic SQL construction

**Severity: LOW**

`graph.rs` correctly uses `params![]` for most queries, but `get_neighbors()` (line 177) and `get_co_accessed()` (line 263) build SQL with dynamic placeholder counts. The actual values are passed as parameters, so this is safe — but worth noting as a pattern that could regress.

---

## High: Denial of Service / Resource Exhaustion

### 3. No input size limits on stored content

**Severity: HIGH**

The `store` tool accepts arbitrarily large `content` strings with no size limit. A single store call with a multi-gigabyte string would:
- Consume unbounded memory during embedding (tokenization, tensor creation)
- Create an extremely large number of chunks (each embedded individually)
- Fill disk storage rapidly

**Location:** `src/mcp/tools.rs:321` (`execute_store`) and `src/db.rs:279` (`store_item`)

**Recommendation:** Add a maximum content size (e.g., 1MB) and reject inputs exceeding it.

### 4. No rate limiting on tool calls

**Severity: MEDIUM**

The MCP server processes all requests synchronously on the main thread. A flood of requests would block the server. While MCP runs over stdio (single client), a compromised or misbehaving LLM could issue rapid-fire calls.

### 5. Unbounded auto-tag inference queries

**Severity: MEDIUM**

`execute_store` (line 419-443) performs `find_similar_items` and then individually fetches each similar item (`db.get_item()`) in a loop. With 5 similar items, this is 5 additional DB queries + 5 additional tag inspections. Combined with the conflict detection search that already runs, every `store` call performs 2 vector searches + up to 10 item fetches.

---

## High: Data Integrity Issues

### 6. Race condition in `replace` operation (non-atomic)

**Severity: HIGH**

The `replace` parameter in `store` (line 359-379) deletes the old item first, then stores the new one. If the process crashes between delete and store, data is lost. This is documented as "atomic" in the tool description but is not actually atomic — there is no transaction wrapping both operations.

**Location:** `src/mcp/tools.rs:359-379`

### 7. TOCTOU race in conflict detection

**Severity: MEDIUM**

`store_item()` in `db.rs:286-294` searches for conflicts *before* storing the new item. Between the conflict check and the actual store, another concurrent store could insert a near-duplicate. Since LanceDB operations are async and the MCP server could theoretically have concurrent requests (via `tokio::spawn`), this is a real window.

### 8. Consolidation merge can delete data without preserving content

**Severity: HIGH**

In `consolidation.rs:228-229`, when two items have >=0.95 similarity, the older item is deleted entirely. Only graph edges are transferred — the actual content of the deleted item is lost forever. If the similarity score was a false positive (e.g., due to short content or embedding quirks), unique information is permanently destroyed.

**Recommendation:** Consider soft-deletion or content merging rather than hard deletion.

---

## Medium: Logic Bugs and Inconsistencies

### 9. Schema mismatch between documentation and code

The `CLAUDE.md` documents the graph_edges schema with columns `source_id, target_id, rel_type`, but the actual code in `graph.rs:59-68` uses `from_id, to_id, edge_type` with an additional `rel_type` column. The documented schema and actual schema are inconsistent:

- Doc says `UNIQUE(source_id, target_id, rel_type)`
- Code has `UNIQUE(from_id, to_id, edge_type)`
- Code has both `edge_type` AND `rel_type` columns with different semantics

### 10. Generated CLAUDE.md instructions list 4 tools, but 5 exist

**Location:** `src/main.rs:356` — The `generate_claude_md_instructions()` function says "## Tools (4 total)" and lists only `store`, `recall`, `list`, and `forget` — omitting the `connections` tool.

### 11. Duplicate `boost_similarity` function

**Location:** `src/lib.rs:181-191` and `src/db.rs:798-804` — The `boost_similarity` function is defined identically in both files. The one in `lib.rs` is `pub` and exported; the one in `db.rs` is private. `db.rs` uses its own private copy instead of the public one.

### 12. `truncate()` panics on multi-byte UTF-8

**Severity: MEDIUM**

`src/mcp/tools.rs:953-958`:
```rust
format!("{}...", &s[..max_len - 3])
```
This slices at a byte offset. If `max_len - 3` falls in the middle of a multi-byte UTF-8 character, this will **panic** at runtime. Similarly, `src/main.rs:339`:
```rust
let ellipsis = if item.content.len() > 80 { "..." } else { "" };
```
uses `.len()` (bytes) but `.chars().take(80)` (characters) — these can disagree for non-ASCII content.

### 13. `split_at_sentences` assumes ASCII

**Severity: LOW**

`src/chunker.rs:114` operates on `bytes` directly, checking for `b'.'`, `b'?'`, `b'!'`. This works for ASCII but will not correctly handle sentence boundaries in non-Latin scripts or content with multi-byte punctuation (e.g., Chinese `。`, Japanese `？`).

### 14. Similarity score can exceed 1.0 with trust bonus

**Severity: LOW**

`src/mcp/tools.rs:566`: `result.similarity = base_score * trust_bonus` where `trust_bonus >= 1.0`. Since `base_score` can already be up to 1.0 (after boosting), the final similarity can exceed 1.0. The recall results show this raw value, which violates the documented 0.0-1.0 range.

---

## Medium: Error Handling Issues

### 15. Silent error swallowing throughout graph/access operations

Many graph and access operations use `let _ = ...` to silently discard errors:
- `src/mcp/tools.rs:453` — `let _ = graph.add_node(...)`
- `src/mcp/tools.rs:458-459` — `let _ = graph.add_supersedes_edge(...)`, `let _ = graph.remove_node(...)`
- `src/mcp/tools.rs:465` — `let _ = graph.add_related_edge(...)`
- `src/consolidation.rs:223-232` — Multiple `let _ = graph.transfer_edges(...)`, etc.

If the SQLite database becomes corrupted or locked, all graph operations silently fail with no indication to the user.

### 16. `timestamp_opt().unwrap()` can panic

**Severity: MEDIUM**

`src/db.rs:993-999`:
```rust
Utc.timestamp_opt(c.value(i), 0).unwrap()
```
`timestamp_opt` returns `LocalResult` which can be `None` for invalid timestamps. If a corrupt or malicious database record contains an out-of-range timestamp, this will panic and crash the server.

---

## Medium: Concurrency Issues

### 17. Multiple SQLite connections without WAL on access.db

`GraphStore`, `AccessTracker`, and `ConsolidationQueue` each open their own `Connection` to `access.db`. `GraphStore::open()` sets `PRAGMA journal_mode=WAL` (line 50), but `AccessTracker::open()` and `ConsolidationQueue::open()` do not. This means:
- If `AccessTracker` opens the database first, it won't be in WAL mode
- Concurrent writes from different connections could block or produce SQLITE_BUSY errors

### 18. Fire-and-forget tasks may outlive the request context

`src/mcp/tools.rs:761-765` spawns a `tokio::spawn` that opens a new `GraphStore` connection and records co-access. Since the MCP server is synchronous (`block_on`), this spawn happens inside `block_on`, but the task runs after the response is sent. If the server shuts down, these tasks may be dropped without completing.

---

## Low: Dependency & Supply Chain

### 19. `hf-hub` downloads model from internet without integrity verification

`src/embedder.rs:191-211` downloads model files from HuggingFace Hub. There is no checksum verification of the downloaded files. A man-in-the-middle attack or compromised HF account could serve a malicious model. The model is loaded via `unsafe` memory-mapped safetensors (line 64).

**Recommendation:** Pin a specific model revision hash and verify checksums.

### 20. `jsonrpc-core` is declared as a dependency but never used

`Cargo.toml` includes `jsonrpc-core = "18"` but no code imports from it. The JSON-RPC protocol is implemented manually in `src/mcp/protocol.rs`. This is dead weight in the dependency tree.

### 21. Broad `tokio` feature set

`tokio = { version = "1", features = ["full"] }` pulls in all tokio features. The codebase only needs `rt-multi-thread`, `macros`, `time`, `io-util`, and `io-std`. Using `"full"` increases compile time and binary size unnecessarily.

---

## Low: Miscellaneous

### 22. `install.sh` uses `curl | bash` pattern implicitly

While `install.sh` itself is fine, it's designed to be piped from a URL (`curl ... | bash`). The script downloads a binary and installs it to `/usr/local/bin` — standard practice but worth noting that the downloaded binary has no signature verification.

### 23. `.sediment/config` written as world-readable JSON

`get_or_create_project_id()` in `lib.rs:169` writes the project config with default permissions. On multi-user systems, the project UUID would be readable by other users. While not sensitive on its own, it could leak project association information.

### 24. No expiration cleanup mechanism

Expired items are filtered out of search results (`expires_at IS NULL OR expires_at > now`), but never actually deleted. Over time, expired items accumulate on disk, consuming storage indefinitely.

### 25. `find_project_root` traverses to filesystem root

`src/lib.rs:205-221` walks up the directory tree with no depth limit. On deeply nested paths or mounted filesystems, this could traverse unexpected directories. There's also no symlink loop detection.

---

## Summary

| Severity | Count | Key Issues |
|----------|-------|------------|
| **Critical** | 1 | SQL/filter injection via unparameterized LanceDB queries |
| **High** | 3 | No input size limits; non-atomic replace; destructive consolidation merge |
| **Medium** | 8 | UTF-8 panics; schema mismatches; silent error swallowing; concurrency issues |
| **Low** | 7 | Unused deps; no model integrity checks; no expiration cleanup |

The most impactful issues to address first are the LanceDB filter injection (#1), the UTF-8 panic in `truncate()` (#12), the non-atomic replace (#6), and input size limits (#3).
