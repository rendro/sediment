//! MCP Server implementation

use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use tokio::runtime::Runtime;
use tokio::sync::Semaphore;

use crate::Embedder;

use super::protocol::{
    CallToolParams, INVALID_PARAMS, INVALID_REQUEST, InitializeResult, ListToolsResult,
    MCP_VERSION, METHOD_NOT_FOUND, PARSE_ERROR, Request, Response, ServerCapabilities, ServerInfo,
    ToolsCapability,
};
use super::tools::{execute_tool, get_tools};

/// Rate limiter state, protected by a mutex to avoid race conditions
/// between window reset and count increment.
pub struct RateLimitState {
    pub window_start_ms: u64,
    pub count: u64,
}

/// Server context holding shared resources
pub struct ServerContext {
    /// Path to the database directory
    pub db_path: PathBuf,
    /// Path to the access tracking SQLite database
    pub access_db_path: PathBuf,
    /// Optional project ID for scoped operations
    pub project_id: Option<String>,
    /// Shared embedder instance (expensive to create, loaded once)
    pub embedder: Arc<Embedder>,
    /// Current working directory (for provenance tracking)
    pub cwd: PathBuf,
    /// Semaphore to ensure only one consolidation task runs at a time
    pub consolidation_semaphore: Arc<Semaphore>,
    /// Counter for recall invocations (triggers periodic clustering and expired cleanup)
    pub recall_count: std::sync::atomic::AtomicU64,
    /// Rate limiter state (mutex protects window+count as a unit)
    pub rate_limit: Mutex<RateLimitState>,
}

/// Run the MCP server
pub fn run(db_path: &Path, project_id: Option<String>) -> Result<()> {
    // Create tokio runtime for async operations
    let rt = Runtime::new().context("Failed to create tokio runtime")?;

    // Load embedder once (expensive operation)
    tracing::info!("Loading embedder model...");
    let embedder = Arc::new(Embedder::new().context("Failed to load embedder")?);
    tracing::info!("Embedder loaded successfully");

    // Derive access DB path as sibling to the LanceDB data directory
    let sediment_dir = db_path.parent().unwrap_or(db_path);
    let access_db_path = sediment_dir.join("access.db");

    // Get current working directory
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    // Create server context with shared resources
    let ctx = ServerContext {
        db_path: db_path.to_path_buf(),
        access_db_path,
        project_id,
        embedder,
        cwd,
        consolidation_semaphore: Arc::new(Semaphore::new(1)),
        recall_count: std::sync::atomic::AtomicU64::new(0),
        rate_limit: Mutex::new(RateLimitState {
            window_start_ms: 0,
            count: 0,
        }),
    };

    // Run the request loop in a block so borrows of `rt` end before shutdown
    {
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        let mut reader = stdin.lock();

        // Maximum line size to prevent OOM from a malicious client sending
        // a single huge line without a newline character.
        const MAX_LINE_BYTES: usize = 10 * 1024 * 1024; // 10 MB

        tracing::info!("MCP server ready, waiting for requests...");

        let mut line = String::new();
        loop {
            line.clear();
            // Use Read::take to bound memory before the newline is found.
            // Without this, read_line would buffer an unlimited amount of data
            // if the client never sends a newline character.
            let bytes_read = (&mut reader)
                .take((MAX_LINE_BYTES + 1) as u64)
                .read_line(&mut line)
                .context("Failed to read line")?;
            if bytes_read == 0 {
                break; // EOF
            }

            if line.len() > MAX_LINE_BYTES
                || (line.len() >= MAX_LINE_BYTES && !line.ends_with('\n'))
            {
                tracing::error!(
                    "Rejecting oversized request: {} bytes (max {})",
                    line.len(),
                    MAX_LINE_BYTES
                );
                // Drain remaining bytes until newline or EOF to resync the stream.
                // Use a fixed-size buffer to avoid unbounded allocation from a
                // malicious client that never sends a newline.
                if !line.ends_with('\n') {
                    let mut drain_buf = [0u8; 8192];
                    loop {
                        match reader.read(&mut drain_buf) {
                            Ok(0) => break,
                            Ok(n) => {
                                if drain_buf[..n].contains(&b'\n') {
                                    break;
                                }
                            }
                            Err(_) => break,
                        }
                    }
                }
                let response = Response::error(None, INVALID_REQUEST, "Request too large");
                let response_json = serde_json::to_string(&response)?;
                writeln!(stdout, "{}", response_json)?;
                stdout.flush()?;
                continue;
            }

            if line.trim().is_empty() {
                continue;
            }

            tracing::debug!(
                "Received: ({} bytes) {}",
                line.len(),
                if line.len() > 200 {
                    &line[..200]
                } else {
                    &line
                }
            );

            // Handle request and get optional response (notifications don't get responses)
            if let Some(response) = handle_request(&rt, &ctx, &line) {
                let response_json = serde_json::to_string(&response)?;
                tracing::debug!("Sending: {}", response_json);

                writeln!(stdout, "{}", response_json)?;
                stdout.flush()?;
            }
        }
    }

    // Graceful shutdown: wait for in-flight consolidation before dropping
    tracing::info!("Client disconnected, shutting down...");
    let sem = ctx.consolidation_semaphore.clone();
    let _ = rt.block_on(async {
        tokio::time::timeout(std::time::Duration::from_secs(10), sem.acquire()).await
    });
    drop(ctx);
    rt.shutdown_timeout(std::time::Duration::from_secs(5));

    Ok(())
}

/// Handle a single request (returns None for notifications)
fn handle_request(rt: &Runtime, ctx: &ServerContext, line: &str) -> Option<Response> {
    // Parse request
    let request: Request = match serde_json::from_str(line) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("Parse error: {}", e);
            return Some(Response::error(
                None,
                PARSE_ERROR,
                "Parse error: invalid JSON-RPC request",
            ));
        }
    };

    // Validate JSON-RPC version
    if request.jsonrpc != "2.0" {
        return Some(Response::error(
            request.id,
            INVALID_REQUEST,
            "Invalid JSON-RPC version, expected 2.0",
        ));
    }

    // Check if this is a notification (no id = notification, don't respond)
    let is_notification = request.id.is_none();

    // Route to handler
    match request.method.as_str() {
        "initialize" => Some(handle_initialize(request.id)),
        "notifications/initialized" | "initialized" => {
            tracing::info!("Client initialized");
            None // Notifications don't get responses
        }
        "tools/list" => Some(handle_list_tools(request.id)),
        "tools/call" => Some(handle_call_tool(rt, ctx, request.id, request.params)),
        "ping" => Some(handle_ping(request.id)),
        _ => {
            // For unknown notifications, just ignore them
            if is_notification {
                tracing::debug!("Ignoring unknown notification: {}", request.method);
                None
            } else {
                tracing::warn!("Unknown method: {}", request.method);
                Some(Response::error(
                    request.id,
                    METHOD_NOT_FOUND,
                    &format!("Unknown method: {}", request.method),
                ))
            }
        }
    }
}

/// Handle initialize request
fn handle_initialize(id: Option<Value>) -> Response {
    let result = InitializeResult {
        protocol_version: MCP_VERSION.to_string(),
        capabilities: ServerCapabilities {
            tools: ToolsCapability {
                list_changed: false,
            },
        },
        server_info: ServerInfo {
            name: "sediment".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        },
    };

    // InitializeResult is a simple struct; serialization is infallible.
    Response::success(
        id,
        serde_json::to_value(result).expect("InitializeResult serialization"),
    )
}

/// Handle tools/list request
fn handle_list_tools(id: Option<Value>) -> Response {
    let result = ListToolsResult { tools: get_tools() };

    // ListToolsResult is a simple struct; serialization is infallible.
    Response::success(
        id,
        serde_json::to_value(result).expect("ListToolsResult serialization"),
    )
}

/// Handle tools/call request
fn handle_call_tool(
    rt: &Runtime,
    ctx: &ServerContext,
    id: Option<Value>,
    params: Option<Value>,
) -> Response {
    let params: CallToolParams = match params {
        Some(p) => match serde_json::from_value(p) {
            Ok(p) => p,
            Err(e) => {
                tracing::debug!("Invalid tool call params: {}", e);
                return Response::error(id, INVALID_PARAMS, "Invalid params");
            }
        },
        None => {
            return Response::error(id, INVALID_PARAMS, "Missing params");
        }
    };

    tracing::info!("Calling tool: {}", params.name);

    // Rate limiting: 60 calls per 60-second window.
    // Uses a mutex to atomically check window expiry and increment count.
    {
        const MAX_CALLS_PER_MINUTE: u64 = 60;
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut state = ctx.rate_limit.lock().unwrap_or_else(|e| e.into_inner());
        if now_ms.saturating_sub(state.window_start_ms) > 60_000 {
            // Window expired, reset
            state.window_start_ms = now_ms;
            state.count = 0;
        }
        state.count += 1;
        if state.count > MAX_CALLS_PER_MINUTE {
            let result =
                super::protocol::CallToolResult::error("Rate limit exceeded, try again later");
            // CallToolResult is a simple struct; serialization is infallible.
            return Response::success(
                id,
                serde_json::to_value(result).expect("CallToolResult serialization"),
            );
        }
    }

    // Execute tool async with fresh DB connection and retry logic
    let result = rt.block_on(execute_tool(ctx, &params.name, params.arguments));

    // CallToolResult is a simple struct; serialization is infallible.
    Response::success(
        id,
        serde_json::to_value(result).expect("CallToolResult serialization"),
    )
}

/// Handle ping request
fn handle_ping(id: Option<Value>) -> Response {
    Response::success(id, json!({}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter_blocks_after_limit() {
        // Fix #1: Rate limiter should correctly count within a window
        let mut state = RateLimitState {
            window_start_ms: 0,
            count: 0,
        };
        let now_ms = 100_000u64;
        // Window 0 is expired (100000 - 0 = 100000 > 60000)
        assert!(now_ms.saturating_sub(state.window_start_ms) > 60_000);
        state.window_start_ms = now_ms;
        state.count = 1;

        // Now simulate 59 more calls in the same window
        for _ in 0..59 {
            state.count += 1;
        }
        assert_eq!(state.count, 60);

        // 61st call should exceed the limit
        state.count += 1;
        assert!(state.count > 60, "Should exceed rate limit");
    }

    #[test]
    fn test_rate_limiter_resets_after_window() {
        let mut state = RateLimitState {
            window_start_ms: 100_000,
            count: 60,
        };

        // Simulate time advancing past the window
        let now_ms = 200_000u64; // 100 seconds later
        if now_ms.saturating_sub(state.window_start_ms) > 60_000 {
            state.window_start_ms = now_ms;
            state.count = 0;
        }
        state.count += 1;

        assert_eq!(state.count, 1, "Count should reset after window expires");
        assert_eq!(state.window_start_ms, 200_000);
    }

    #[test]
    fn test_rate_limiter_exactly_at_limit() {
        // Bug #7: Effective limit should be exactly MAX_CALLS, not MAX_CALLS+1.
        const MAX_CALLS: u64 = 60;
        let mut state = RateLimitState {
            window_start_ms: 100_000,
            count: 0,
        };
        let now_ms = 100_000u64; // same window

        for _ in 0..MAX_CALLS {
            if now_ms.saturating_sub(state.window_start_ms) > 60_000 {
                state.window_start_ms = now_ms;
                state.count = 0;
            }
            state.count += 1;
            assert!(state.count <= MAX_CALLS, "Should not exceed limit");
        }
        assert_eq!(state.count, MAX_CALLS);

        // The (MAX_CALLS+1)th call should be rejected
        state.count += 1;
        assert!(state.count > MAX_CALLS, "Next call should exceed limit");
    }
}
