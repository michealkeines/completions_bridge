//! Standalone parking MCP server: `bridge-claude-park [mcp-park] --tools <file>`.
//! Used by tests; the bridge binary exposes the same server as `mcp-park`.

use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut tools: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        if a == "--tools" {
            tools = args.next().map(PathBuf::from);
        }
    }
    let tools = tools.ok_or_else(|| anyhow::anyhow!("usage: bridge-claude-park --tools <file>"))?;
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    rt.block_on(bridge_claude::run_park_server(&tools))
}
