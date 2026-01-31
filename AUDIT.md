# Security & Code Audit Report

**Date:** 2026-01-31
**Scope:** Full repository audit of sediment-mcp v0.2.1
**Auditor:** Automated code review
**Based on:** Latest main branch merged into `claude/audit-repository-DY9FC`

---

## Validation of Previously Reported Issues

### Fixed (17 of 18 tracked issues resolved)

| # | Issue | Status | Verification |
|---|-------|--------|-------------|
| 1 | LanceDB filter injection | **FIXED** | `sanitize_sql_string()` at `db.rs:11-13` escapes single quotes. All filter sites use it including `expire_item` (`db.rs:758`), `delete_item` (`db.rs:778,790`), `get_item` (`db.rs:687`), `get_items_batch` (`db.rs:719`), `list_items` (`db.rs:635`). |
| 2 | `truncate()` UTF-8 panic | **FIXED** | `tools.rs:1018-1029` now uses `s.chars().count()` and `s.char_indices().nth()` for safe slicing. `consolidation.rs:242-249` also uses `char_indices().nth(497)`. `main.rs:339` now uses `chars().count()` consistently. |
| 3 | CLAUDE.md schema mismatch | **FIXED** | `CLAUDE.md:106-143` now documents the actual schema: `from_id`, `to_id`, `edge_type`, `rel_type`, `count`, `last_at`, `UNIQUE(from_id, to_id, edge_type)`. Matches `graph.rs:59-68` exactly. |
| 4 | Generated instructions say "4 tools" | **FIXED** | `main.rs:358` now says "## Tools (5 total)" and lists all 5 tools including `connections`. |
| 5 | Duplicate `boost_similarity` | **FIXED** | `db.rs:27` now imports `use crate::boost_similarity;`. No private copy exists in `db.rs` anymore. |
| 6 | Similarity exceeds 1.0 | **FIXED** | `tools.rs:594` now has `.min(1.0)` cap. |
| 7 | Silent error swallowing | **FIXED** | All `let _ = ...` sites in `tools.rs` and `consolidation.rs` replaced with `if let Err(e) = ... { tracing::warn!(...) }`. |
| 8 | `timestamp_opt().unwrap()` panic | **FIXED** | `db.rs:1061-1063` and `db.rs:1070-1072` now use `.single().unwrap_or_else(Utc::now)`. |
| 9 | Inconsistent WAL mode | **FIXED** | `access.rs:33` and `consolidation.rs:35` both now set `PRAGMA journal_mode=WAL;`. |
| 10 | Fire-and-forget tasks unmonitored | **FIXED** | `tools.rs:789-800` and `tools.rs:809-832` now use `.instrument(tracing::info_span!(...))` and log errors within the spawned futures. |
| 11 | No rate limiting | **FIXED** | `server.rs:202-221` implements a 60-calls-per-minute window using `AtomicU64`. |
| 12 | TOCTOU race in conflict detection | **ACCEPTED** | Low severity with single-client MCP over stdio. Not fixed but risk is negligible. |
| 13 | Model download without integrity check | **FIXED** | `embedder.rs:198-201` pins revision `e4ce9877abf3edfe10b0d82785e83bdcb973e22e`. `embedder.rs:217` calls `verify_tofu_hash()` which implements trust-on-first-use SHA256 verification (`embedder.rs:234-257`). |
| 14 | `split_at_sentences` ASCII-only | **FIXED** | `chunker.rs:119` now matches `'.' | '?' | '!' | '。' | '？' | '！'` using char comparison instead of byte comparison. |
| 15 | Unused `jsonrpc-core` dependency | **FIXED** | Removed from `Cargo.toml`. Not present. |
| 16 | `lib.rs` docstring says "4 tools" | **FIXED** | `lib.rs:9` now says "5 tools". |
| 17 | No expiration cleanup | **FIXED** | `db.rs:818-841` implements `cleanup_expired()`. `tools.rs:814-831` triggers it every 10th recall. |
| 18 | Config file permissions | **EXCLUDED** | Per user request, not tracked. |
| 19 | `find_project_root` unbounded traversal | **FIXED** | `lib.rs:207-209` now has `depth >= 100` guard. |

---

## Fresh Audit: New Issues Found

### NEW-1. `partial_cmp().unwrap()` panics on NaN similarity scores

**Severity: MEDIUM**

Three sort operations use `partial_cmp(...).unwrap()` which panics if any similarity score is `NaN`:

- `src/db.rs:538` — `search_results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());`
- `src/db.rs:601` — `conflicts.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());`
- `src/mcp/tools.rs:597` — `results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());`

A `NaN` can occur if the embedding is all zeros (e.g., empty input after truncation, model failure returning zeros), since cosine similarity involves division. The L2 normalization in `embedder.rs:260-272` divides by the norm — if the norm is zero, the result contains `NaN` values that propagate through similarity calculations.

**Recommendation:** Replace `.unwrap()` with `.unwrap_or(std::cmp::Ordering::Equal)` or use `f32::total_cmp()`.

---

### NEW-2. Rate limiter race condition allows burst beyond limit

**Severity: LOW**

`server.rs:208-221` — The rate limiter uses `Relaxed` ordering for both the window check and the count increment. Under concurrent access (even though MCP is typically single-client), the following race exists:

1. Thread A reads `window` and sees it's stale, enters the reset branch
2. Thread B reads `window` before A writes, also enters the reset branch
3. Both threads reset the counter to 1, effectively allowing 2x the rate limit

Since MCP runs over stdio with `block_on`, true concurrency is unlikely here, but the logic is incorrect in principle. Additionally, the window-reset branch stores count=1 without checking if another thread already reset it.

---

### NEW-3. TOFU hash verification only covers `model.safetensors`, not `tokenizer.json` or `config.json`

**Severity: LOW**

`embedder.rs:217` only verifies `model.safetensors`:
```rust
verify_tofu_hash(&model_path, "model.safetensors")?;
```

The `tokenizer.json` and `config.json` files are also downloaded and loaded but not hash-verified. A compromised tokenizer could manipulate how text is split into tokens, affecting embedding quality or causing unexpected behavior.

**Recommendation:** Also call `verify_tofu_hash()` for `tokenizer_path` and `config_path`.

---

### NEW-4. `expire_item` re-embeds content unnecessarily and is non-atomic

**Severity: LOW**

`db.rs:743-770` — `expire_item()` reads an item, regenerates its embedding from scratch, then deletes and re-inserts it. This is:
1. **Wasteful** — the embedding hasn't changed, only `expires_at` has
2. **Non-atomic** — if the process crashes between delete (`db.rs:757-760`) and re-insert (`db.rs:764-768`), the item is permanently lost

The same delete-then-reinsert pattern that was fixed for `replace` (store-before-delete) was not applied here.

**Recommendation:** Re-insert before deleting, or find a way to avoid re-embedding.

---

### NEW-5. `cleanup_expired` uses unquoted integer in LanceDB filter

**Severity: LOW**

`db.rs:826`:
```rust
let filter = format!("expires_at IS NOT NULL AND expires_at < {}", now);
```

The `now` value is a system-generated `i64` (not user-controlled), so there is no injection risk. However, it breaks the pattern of sanitizing all interpolated values in LanceDB filters. If this line is ever copy-pasted with a different source value, the missing sanitization could become an issue.

---

### NEW-6. `graph.rs:339` silently ignores errors in `transfer_edges` internals

**Severity: LOW**

`graph.rs:339`:
```rust
let _ = self.add_related_edge(to_id, neighbor, *strength, rel_type);
```

While consolidation callers now log errors from `transfer_edges` itself, the internal implementation still silently discards errors when creating individual new edges on the target node. If individual edge transfers fail, there's no indication of partial failure.

---

### NEW-7. Co-access recording creates O(n^2) edges with no pruning

**Severity: MEDIUM**

`graph.rs:230-248` — `record_co_access()` creates edges between every pair of result IDs. With the default limit of 5 results, this creates up to 10 edge-upserts per recall. There is no pruning or TTL for co-access edges.

With heavy usage (100 recalls/day), this produces ~1000 new edge-upserts daily. The `get_co_accessed` query (`graph.rs:263-271`) scans with `OR` conditions across both `from_id` and `to_id`, which degrades as the table grows.

**Recommendation:** Consider limiting co-access tracking to top-3 results, or adding periodic pruning of low-count co-access edges older than 30 days.

---

### NEW-8. `store` tool description says "atomically delete before storing" but behavior is store-before-delete

**Severity: LOW**

`tools.rs:65`:
```rust
"description": "ID of an existing item to replace (atomically delete before storing)"
```

The actual behavior (since the safety fix) is store-before-delete (`tools.rs:462-478`). The tool schema description is now inaccurate and could mislead MCP clients about the operation semantics.

---

### NEW-9. `sha2` crate added but no `Cargo.lock` dependency audit

**Severity: LOW**

`Cargo.toml:53` adds `sha2 = "0.10"` for TOFU hashing. This is a well-known, widely-used crate, but it's a new cryptographic dependency that wasn't present in the initial version. Worth noting for dependency tracking purposes. The crate itself is trustworthy.

---

### NEW-10. `install.sh` checksum verification is optional and silently skipped

**Severity: LOW**

`install.sh:52-74` — If `checksums.txt` is not available at the release URL, or if neither `sha256sum` nor `shasum` are installed, the binary is installed without any integrity verification. The script prints a warning but continues. A MITM attack during the download window would succeed if checksums.txt is not published.

---

## Summary

| Severity | Count | Issues |
|----------|-------|--------|
| **Critical** | 0 | None |
| **Medium** | 2 | NaN panic in sort (#NEW-1), O(n^2) co-access growth (#NEW-7) |
| **Low** | 8 | Rate limiter race (#NEW-2), incomplete TOFU (#NEW-3), non-atomic expire (#NEW-4), unsanitized int in filter (#NEW-5), silent transfer_edges error (#NEW-6), stale description (#NEW-8), new crypto dep (#NEW-9), optional install checksum (#NEW-10) |

### Priority Recommendations

1. **Fix `partial_cmp().unwrap()` (#NEW-1)** — This is the highest-risk remaining crash vector. Replace with `.unwrap_or(std::cmp::Ordering::Equal)` or `f32::total_cmp()` at all three sites.
2. **Add TOFU verification for tokenizer and config (#NEW-3)** — Two additional `verify_tofu_hash()` calls in `embedder.rs`.
3. **Fix store tool description (#NEW-8)** — Change "atomically delete before storing" to "atomically replace (store then delete old)".
4. **Consider co-access pruning (#NEW-7)** — Add periodic cleanup of co-access edges with count < 2 and age > 30 days.
