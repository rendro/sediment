# Security & Code Audit Report

**Date:** 2026-01-31 (updated)
**Scope:** Full repository audit of sediment-mcp v0.2.1
**Auditor:** Automated code review
**Based on:** Latest main branch (commit 55a9e97)

---

## Changes Since Previous Audit

The main branch addressed three findings from the initial audit:

1. **FIXED: LanceDB filter injection (#1)** — A `sanitize_sql_string()` function was added to `db.rs:11-13` that escapes single quotes. All LanceDB filter interpolation sites now use it. **Downgraded from CRITICAL to LOW** (see #1 below for residual concerns).

2. **FIXED: Content size limit (#3)** — A 1MB (`MAX_CONTENT_BYTES = 1_000_000`) limit was added at `tools.rs:337-344`. Content exceeding this is rejected before embedding.

3. **FIXED: Destructive consolidation merge (#8)** — Consolidation now uses soft-deletion via `expire_item()` (`consolidation.rs:247-253`) instead of hard-deleting merged items. An archive preview is also stored in a RELATED edge label. Falls back to hard delete only if expire fails.

4. **FIXED: Replace operation ordering** — The replace flow was changed to store-before-delete (`tools.rs:460-468`), eliminating the crash-window data loss. The old item is only deleted after the new one is successfully stored.

---

## Remaining Issues

### 1. Residual injection risk via `sanitize_sql_string`

**Severity: LOW** (downgraded from CRITICAL)

The new `sanitize_sql_string()` function at `db.rs:11-13` escapes single quotes by doubling them (`'` → `''`). This is the standard SQL escaping approach and handles the primary injection vector.

However, there is one site that was **not updated**: `expire_item()` at `db.rs:754`:
```rust
table.delete(&format!("id = '{}'", id))
```
This line in `expire_item` does not use `sanitize_sql_string`. Currently `expire_item` is only called from consolidation with internally-generated IDs, so the practical risk is minimal, but it breaks the consistent sanitization pattern.

Additionally, the `sanitize_sql_string` approach only escapes single quotes. If LanceDB's filter language supports backslash escapes or other special characters, this may be insufficient. A UUID format validation would be a more robust defense-in-depth measure.

---

### 2. `truncate()` panics on multi-byte UTF-8

**Severity: MEDIUM** — **Not fixed**

`src/mcp/tools.rs:961-967`:
```rust
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
```
This slices at a byte offset. If `max_len - 3` falls in the middle of a multi-byte UTF-8 character (e.g., emoji, CJK characters), this will **panic** at runtime, crashing the MCP server.

Similarly, the consolidation archive preview at `consolidation.rs:232-233` has the same issue:
```rust
format!("{}...", &remove.content[..497])
```
If byte 497 lands mid-character, this panics.

And `src/main.rs:339`:
```rust
let ellipsis = if item.content.len() > 80 { "..." } else { "" };
```
uses `.len()` (bytes) but `.chars().take(80)` (characters) — these can disagree for non-ASCII.

---

### 3. Schema mismatch between documentation and code

**Severity: LOW** — **Not fixed**

The `CLAUDE.md` documents the graph_edges schema with columns `source_id, target_id, rel_type`, but the actual code in `graph.rs:59-68` uses `from_id, to_id, edge_type` with an additional `rel_type` column:

- Doc: `UNIQUE(source_id, target_id, rel_type)`
- Code: `UNIQUE(from_id, to_id, edge_type)`
- Code has both `edge_type` AND `rel_type` columns with different semantics

---

### 4. Generated CLAUDE.md instructions list 4 tools, but 5 exist

**Severity: LOW** — **Not fixed**

`src/main.rs:354` — The `generate_claude_md_instructions()` function says "## Tools (4 total)" and lists only `store`, `recall`, `list`, and `forget` — omitting the `connections` tool.

---

### 5. Duplicate `boost_similarity` function

**Severity: LOW** — **Not fixed**

`src/lib.rs:181-191` and `src/db.rs:845-851` — The `boost_similarity` function is defined identically in both files. The one in `lib.rs` is `pub` and exported; the one in `db.rs` is private. `db.rs` uses its own private copy instead of the public one.

---

### 6. Similarity score can exceed 1.0 with trust bonus

**Severity: LOW** — **Not fixed**

`src/mcp/tools.rs:576`: `result.similarity = base_score * trust_bonus` where `trust_bonus >= 1.0`. Combined with the 1.15x project boost (capped at 1.0 before decay, but not after), the final similarity displayed to users can exceed 1.0.

---

### 7. Silent error swallowing throughout graph/access operations

**Severity: MEDIUM** — **Not fixed**

Many graph and access operations use `let _ = ...` to silently discard errors:
- `src/mcp/tools.rs:458` — `let _ = graph.add_node(...)`
- `src/mcp/tools.rs:463-467` — `let _ = db.delete_item(...)`, `let _ = tracker.record_validation(...)`, `let _ = graph.add_supersedes_edge(...)`, `let _ = graph.remove_node(...)`
- `src/mcp/tools.rs:473` — `let _ = graph.add_related_edge(...)`
- `src/consolidation.rs:223-252` — Multiple `let _ = graph.transfer_edges(...)`, etc.

If the SQLite database becomes corrupted or locked, all graph operations silently fail with no indication to the user.

---

### 8. `timestamp_opt().unwrap()` can panic

**Severity: MEDIUM** — **Not fixed**

`src/db.rs:1040-1045`:
```rust
Utc.timestamp_opt(c.value(i), 0).unwrap()
```
`timestamp_opt` returns `LocalResult` which can be `None` for invalid timestamps. Corrupt or malicious database records with out-of-range timestamps will panic and crash the server.

---

### 9. Multiple SQLite connections without consistent WAL mode

**Severity: MEDIUM** — **Not fixed**

`GraphStore::open()` sets `PRAGMA journal_mode=WAL` (line 50), but `AccessTracker::open()` and `ConsolidationQueue::open()` do not. The first connection to open the database determines the journal mode. If `AccessTracker` opens first, WAL is not enabled, risking `SQLITE_BUSY` errors under concurrent writes.

---

### 10. Fire-and-forget tasks may outlive the request context

**Severity: LOW** — **Not fixed**

`src/mcp/tools.rs:769-773` spawns `tokio::spawn` tasks for co-access recording and clustering. These tasks may be dropped on server shutdown without completing, potentially losing co-access data.

---

### 11. No rate limiting on tool calls

**Severity: MEDIUM** — **Not fixed**

The MCP server processes all requests synchronously. A misbehaving client could issue rapid-fire calls to exhaust resources.

---

### 12. TOCTOU race in conflict detection

**Severity: LOW** — **Not fixed**

`store_item()` in `db.rs` searches for conflicts *before* storing. A concurrent store could insert a near-duplicate in the window between check and store.

---

### 13. `hf-hub` downloads model without integrity verification

**Severity: LOW** — **Not fixed**

Model files are downloaded from HuggingFace Hub with no checksum or signature verification. A MITM or compromised HF account could serve a malicious model loaded via memory-mapped safetensors.

---

### 14. `split_at_sentences` assumes ASCII punctuation

**Severity: LOW** — **Not fixed**

`src/chunker.rs` operates on bytes for sentence detection, missing non-ASCII sentence terminators.

---

### 15. `jsonrpc-core` unused dependency

**Severity: LOW** — **Not fixed**

`Cargo.toml` includes `jsonrpc-core = "18"` but it's never imported. Dead weight in the dependency tree.

---

### 16. `lib.rs` module docstring says "4 tools" (stale)

**Severity: LOW** — **Not fixed**

`src/lib.rs:9` says "MCP-native - 4 tools for seamless LLM integration" but there are 5 tools.

---

### 17. No expiration cleanup mechanism

**Severity: LOW** — **Worsened slightly**

Expired items were already accumulating before. Now that consolidation uses soft-deletion via `expire_item()`, merged items also persist as expired records indefinitely, accelerating disk usage growth.

---

### 18. `.sediment/config` written with default (world-readable) permissions

**Severity: LOW** — **Not fixed**

---

### 19. `find_project_root` traverses to filesystem root

**Severity: LOW** — **Not fixed**

---

## Summary

| Severity | Count | Status |
|----------|-------|--------|
| **Critical** | 0 | All critical issues fixed |
| **Medium** | 5 | UTF-8 panics (#2), silent errors (#7), timestamp panics (#8), WAL inconsistency (#9), no rate limiting (#11) |
| **Low** | 14 | Various doc mismatches, unused deps, missing cleanup, residual injection gap |

### Priority Recommendations

1. **Fix UTF-8 panics (#2)** — Use `s.char_indices()` or `s.floor_char_boundary()` instead of byte slicing in `truncate()` and the consolidation archive. This is the most likely remaining crash vector.
2. **Fix `timestamp_opt().unwrap()` (#8)** — Replace with `.single()` or provide a fallback to prevent panics on corrupt data.
3. **Set WAL mode consistently (#9)** — Add `PRAGMA journal_mode=WAL` to `AccessTracker::open()` and `ConsolidationQueue::open()`.
4. **Sanitize the missed `expire_item` call (#1)** — Apply `sanitize_sql_string` to `db.rs:754` for consistency.
5. **Update stale documentation (#3, #4, #16)** — Fix tool counts and schema docs.
