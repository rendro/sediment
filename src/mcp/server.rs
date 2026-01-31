//! MCP Server implementation

use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::runtime::Runtime;
use tokio::sync::Semaphore;

use crate::Embedder;

use super::protocol::{
    CallToolParams, INVALID_PARAMS, INVALID_REQUEST, InitializeResult, ListToolsResult,
    MCP_VERSION, METHOD_NOT_FOUND, PARSE_ERROR, Request, Response, ServerCapabilities, ServerInfo,
    ToolsCapability,
};
use super::tools::{execute_tool, get_tools};

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
    /// Counter for consolidation runs (for periodic clustering)
    pub consolidation_run_count: std::sync::atomic::AtomicU64,
    /// Rate limiter: start of the current 60-second window (unix timestamp in millis)
    pub rate_limit_window: AtomicU64,
    /// Rate limiter: number of tool calls in the current window
    pub rate_limit_count: AtomicU64,
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
        consolidation_run_count: std::sync::atomic::AtomicU64::new(0),
        rate_limit_window: AtomicU64::new(0),
        rate_limit_count: AtomicU64::new(0),
    };

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let reader = stdin.lock();

    tracing::info!("MCP server ready, waiting for requests...");

    for line in reader.lines() {
        let line = line.context("Failed to read line")?;

        if line.trim().is_empty() {
            continue;
        }

        tracing::debug!("Received: {}", line);

        // Handle request and get optional response (notifications don't get responses)
        if let Some(response) = handle_request(&rt, &ctx, &line) {
            let response_json = serde_json::to_string(&response)?;
            tracing::debug!("Sending: {}", response_json);

            writeln!(stdout, "{}", response_json)?;
            stdout.flush()?;
        }
    }

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
                &format!("Parse error: {}", e),
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

    Response::success(id, serde_json::to_value(result).unwrap())
}

/// Handle tools/list request
fn handle_list_tools(id: Option<Value>) -> Response {
    let result = ListToolsResult { tools: get_tools() };

    Response::success(id, serde_json::to_value(result).unwrap())
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
                return Response::error(id, INVALID_PARAMS, &format!("Invalid params: {}", e));
            }
        },
        None => {
            return Response::error(id, INVALID_PARAMS, "Missing params");
        }
    };

    tracing::info!("Calling tool: {}", params.name);

    // Rate limiting: 60 calls per 60-second window
    {
        const MAX_CALLS_PER_MINUTE: u64 = 60;
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let window = ctx.rate_limit_window.load(Ordering::Relaxed);
        if now_ms - window > 60_000 {
            // Reset window
            ctx.rate_limit_window.store(now_ms, Ordering::Relaxed);
            ctx.rate_limit_count.store(1, Ordering::Relaxed);
        } else {
            let count = ctx.rate_limit_count.fetch_add(1, Ordering::Relaxed) + 1;
            if count > MAX_CALLS_PER_MINUTE {
                let result =
                    super::protocol::CallToolResult::error("Rate limit exceeded, try again later");
                return Response::success(id, serde_json::to_value(result).unwrap());
            }
        }
    }

    // Execute tool async with fresh DB connection and retry logic
    let result = rt.block_on(execute_tool(ctx, &params.name, params.arguments));

    Response::success(id, serde_json::to_value(result).unwrap())
}

/// Handle ping request
fn handle_ping(id: Option<Value>) -> Response {
    Response::success(id, json!({}))
}
