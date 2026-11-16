//! Parking MCP server: advertises the caller's tools over stdio JSON-RPC and never
//! answers `tools/call`, so claude waits until the bridge stops it (port of
//! `probes/fake_mcp.py` with an infinite delay).

use std::path::Path;

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};

/// Loads the tools file: a JSON array of `{name, description, inputSchema}` (also accepts
/// `parameters` instead of `inputSchema`). A missing or broken file means no tools.
pub fn load_tools(path: &Path) -> Vec<Value> {
    let Ok(text) = std::fs::read_to_string(path) else { return Vec::new() };
    let Ok(Value::Array(items)) = serde_json::from_str::<Value>(&text) else { return Vec::new() };
    items
        .into_iter()
        .filter_map(|t| {
            let name = t.get("name").and_then(Value::as_str)?.to_string();
            let description = t.get("description").and_then(Value::as_str).unwrap_or_default().to_string();
            let mut schema = t
                .get("inputSchema")
                .or_else(|| t.get("parameters"))
                .cloned()
                .filter(Value::is_object)
                .unwrap_or_else(|| json!({"type": "object"}));
            if let Some(obj) = schema.as_object_mut() {
                obj.entry("type").or_insert_with(|| json!("object"));
            }
            Some(json!({"name": name, "description": description, "inputSchema": schema}))
        })
        .collect()
}

/// Response for one JSON-RPC message, or `None` (notifications, `tools/call`, junk).
pub fn handle(msg: &Value, tools: &[Value]) -> Option<Value> {
    let id = msg.get("id").filter(|i| !i.is_null())?.clone();
    let method = msg.get("method").and_then(Value::as_str).unwrap_or_default();
    let result = match method {
        "initialize" => {
            let version = msg.pointer("/params/protocolVersion").and_then(Value::as_str).unwrap_or("2025-06-18");
            json!({
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "bridge", "version": env!("CARGO_PKG_VERSION")},
            })
        }
        "tools/list" => json!({"tools": tools}),
        "ping" => json!({}),
        // Parked: the call is never answered; the bridge stops claude instead.
        "tools/call" => return None,
        "" => return None,
        other => {
            return Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {"code": -32601, "message": format!("method not found: {other}")},
            }))
        }
    };
    Some(json!({"jsonrpc": "2.0", "id": id, "result": result}))
}

pub async fn serve<R, W>(tools: Vec<Value>, input: R, mut output: W) -> anyhow::Result<()>
where
    R: tokio::io::AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut lines = BufReader::new(input).lines();
    while let Some(line) = lines.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(msg) = serde_json::from_str::<Value>(line) else { continue };
        let batch = match msg {
            Value::Array(items) => items,
            other => vec![other],
        };
        for m in batch {
            if let Some(resp) = handle(&m, &tools) {
                let mut s = resp.to_string();
                s.push('\n');
                output.write_all(s.as_bytes()).await?;
                output.flush().await?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn methods() {
        let tools = vec![json!({"name": "read", "description": "d", "inputSchema": {"type": "object"}})];
        let init = handle(&json!({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}}), &tools).unwrap();
        assert_eq!(init["result"]["protocolVersion"], "2025-11-25");
        assert_eq!(init["result"]["capabilities"], json!({"tools": {}}));
        let list = handle(&json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}), &tools).unwrap();
        assert_eq!(list["result"]["tools"][0]["name"], "read");
        assert_eq!(handle(&json!({"jsonrpc": "2.0", "id": "p", "method": "ping"}), &tools).unwrap()["result"], json!({}));
        assert!(handle(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}), &tools).is_none());
        assert!(handle(&json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "read"}}), &tools).is_none());
        let unk = handle(&json!({"jsonrpc": "2.0", "id": 3, "method": "resources/list"}), &tools).unwrap();
        assert_eq!(unk["error"]["code"], -32601);
    }

    #[test]
    fn tools_file_formats() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.json");
        std::fs::write(&p, r#"[{"name":"a","description":"x","inputSchema":{"type":"object","properties":{}}},{"name":"b","parameters":{"properties":{}}},{"nope":1}]"#).unwrap();
        let t = load_tools(&p);
        assert_eq!(t.len(), 2);
        assert_eq!(t[1]["inputSchema"]["type"], "object");
        assert!(load_tools(&dir.path().join("missing.json")).is_empty());
    }

    #[tokio::test]
    async fn serve_over_pipes() {
        let input = b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}\n\
            garbage\n\
            {\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n\
            {\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{}}\n\
            {\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/list\"}\n";
        let mut out = Vec::new();
        serve(vec![], &input[..], &mut out).await.unwrap();
        let text = String::from_utf8(out).unwrap();
        let ids: Vec<i64> = text.lines().map(|l| serde_json::from_str::<Value>(l).unwrap()["id"].as_i64().unwrap()).collect();
        assert_eq!(ids, vec![1, 3]);
    }
}
