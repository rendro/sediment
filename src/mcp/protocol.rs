//! MCP Protocol types
//!
//! Model Context Protocol uses JSON-RPC 2.0 over stdio.
//! See: https://spec.modelcontextprotocol.io/

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const JSONRPC_VERSION: &str = "2.0";
pub const MCP_VERSION: &str = "2025-03-26";

/// JSON-RPC request
#[derive(Debug, Deserialize)]
pub struct Request {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

/// JSON-RPC response
#[derive(Debug, Serialize)]
pub struct Response {
    pub jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

impl Response {
    pub fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION,
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Option<Value>, code: i32, message: &str) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION,
            id,
            result: None,
            error: Some(RpcError {
                code,
                message: message.to_string(),
                data: None,
            }),
        }
    }
}

/// JSON-RPC error
#[derive(Debug, Serialize)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

// Standard JSON-RPC error codes
pub const PARSE_ERROR: i32 = -32700;
pub const INVALID_REQUEST: i32 = -32600;
pub const METHOD_NOT_FOUND: i32 = -32601;
pub const INVALID_PARAMS: i32 = -32602;

/// Server capabilities for MCP
#[derive(Debug, Serialize)]
pub struct ServerCapabilities {
    pub tools: ToolsCapability,
}

#[derive(Debug, Serialize)]
pub struct ToolsCapability {
    #[serde(rename = "listChanged")]
    pub list_changed: bool,
}

/// Server info
#[derive(Debug, Serialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

/// Initialize result
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    pub protocol_version: String,
    pub capabilities: ServerCapabilities,
    pub server_info: ServerInfo,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

/// Tool definition
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// List tools result
#[derive(Debug, Serialize)]
pub struct ListToolsResult {
    pub tools: Vec<Tool>,
}

/// Call tool params
#[derive(Debug, Deserialize)]
pub struct CallToolParams {
    pub name: String,
    #[serde(default)]
    pub arguments: Option<Value>,
}

/// Tool result content
#[derive(Debug, Serialize)]
pub struct TextContent {
    #[serde(rename = "type")]
    pub content_type: &'static str,
    pub text: String,
}

impl TextContent {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            content_type: "text",
            text: text.into(),
        }
    }
}

/// Call tool result
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CallToolResult {
    pub content: Vec<TextContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

impl CallToolResult {
    pub fn success(text: impl Into<String>) -> Self {
        Self {
            content: vec![TextContent::new(text)],
            is_error: None,
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self {
            content: vec![TextContent::new(text)],
            is_error: Some(true),
        }
    }
}
