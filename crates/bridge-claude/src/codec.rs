//! Name and id codecs (ARCHITECTURE 6.3, 6.4).
//!
//! claude sees MCP tools as `mcp__bridge__<name>`; the whole name must match
//! `^[a-zA-Z0-9_-]{1,64}$`. Tool use ids must match `^[a-zA-Z0-9_-]+$`.
//! Both codecs are deterministic so checkpoints written in one request stay valid
//! for the next one, and keep a reverse map so output can be translated back.

use std::collections::HashMap;

use sha2::{Digest, Sha256};

/// Prefix claude puts in front of tools of our MCP server (named "bridge").
/// Not configurable: claude's naming scheme plus the server name the bridge registers;
/// stored checkpoints contain it.
pub const TOOL_PREFIX: &str = "mcp__bridge__";
/// Max length of a claude tool name including the prefix.
/// Not configurable: an Anthropic API limit.
pub const MAX_TOOL_NAME: usize = 64;
/// Max length of our part of the name.
pub const MAX_ENCODED_NAME: usize = MAX_TOOL_NAME - TOOL_PREFIX.len();
/// Prefix of ids generated for caller ids Anthropic would reject.
/// Not configurable: decoding relies on it, and stored checkpoints contain encoded ids.
pub const ENCODED_ID_PREFIX: &str = "toolu_b_";
/// Not configurable: Anthropic's tool_use id length limit.
const MAX_ID_LEN: usize = 128;

fn is_safe_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

fn short_hash(s: &str, hex_chars: usize) -> String {
    let digest = Sha256::digest(s.as_bytes());
    let mut h = hex::encode(digest);
    h.truncate(hex_chars);
    h
}

/// Encodes one caller tool name (without the MCP prefix). Pure function.
pub fn encode_tool_name(name: &str) -> String {
    if !name.is_empty() && name.len() <= MAX_ENCODED_NAME && name.chars().all(is_safe_char) {
        return name.to_string();
    }
    let mut base: String = name.chars().map(|c| if is_safe_char(c) { c } else { '_' }).collect();
    // `<base>_<8 hex>` must fit.
    let max_base = MAX_ENCODED_NAME - 9;
    if base.len() > max_base {
        // All chars are ASCII now, so byte truncation is safe.
        base.truncate(max_base);
    }
    if base.is_empty() {
        base.push_str("tool");
    }
    format!("{base}_{}", short_hash(name, 8))
}

/// Per-turn tool name codec: caller name <-> encoded name.
#[derive(Debug, Clone, Default)]
pub struct ToolNameCodec {
    to_caller: HashMap<String, String>,
}

impl ToolNameCodec {
    pub fn new<'a>(names: impl IntoIterator<Item = &'a str>) -> Self {
        let mut c = ToolNameCodec::default();
        for n in names {
            c.register(n);
        }
        c
    }

    /// Registers a caller name (e.g. one appearing only in history) and returns its encoding.
    pub fn register(&mut self, caller: &str) -> String {
        let enc = encode_tool_name(caller);
        self.to_caller.entry(enc.clone()).or_insert_with(|| caller.to_string());
        enc
    }

    pub fn encode(&self, caller: &str) -> String {
        encode_tool_name(caller)
    }

    /// Full claude-side name, `mcp__bridge__<encoded>`.
    pub fn claude_name(&self, caller: &str) -> String {
        format!("{TOOL_PREFIX}{}", encode_tool_name(caller))
    }

    /// Encoded name (without prefix) back to the caller's name. Unknown names are returned
    /// unchanged (they can only come from history the codec never saw).
    pub fn decode(&self, encoded: &str) -> String {
        self.to_caller.get(encoded).cloned().unwrap_or_else(|| encoded.to_string())
    }

    /// claude-side name (with or without the prefix) back to the caller's name.
    pub fn decode_claude_name(&self, claude_name: &str) -> String {
        self.decode(claude_name.strip_prefix(TOOL_PREFIX).unwrap_or(claude_name))
    }
}

/// True if Anthropic accepts this id as a tool_use id.
pub fn is_valid_call_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= MAX_ID_LEN && id.chars().all(is_safe_char)
}

/// Encodes one caller call id. Valid ids pass through; others get a stable hash id.
pub fn encode_call_id(id: &str) -> String {
    if is_valid_call_id(id) {
        id.to_string()
    } else {
        format!("{ENCODED_ID_PREFIX}{}", short_hash(id, 32))
    }
}

/// Per-turn call id codec with a reverse map.
#[derive(Debug, Clone, Default)]
pub struct CallIdCodec {
    to_caller: HashMap<String, String>,
}

impl CallIdCodec {
    pub fn encode(&mut self, caller: &str) -> String {
        let enc = encode_call_id(caller);
        if enc != caller {
            self.to_caller.entry(enc.clone()).or_insert_with(|| caller.to_string());
        }
        enc
    }

    /// Ids produced by claude are returned unchanged; ids we encoded map back.
    pub fn decode(&self, id: &str) -> String {
        self.to_caller.get(id).cloned().unwrap_or_else(|| id.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_full(n: &str) -> bool {
        let full = format!("{TOOL_PREFIX}{n}");
        !full.is_empty() && full.len() <= MAX_TOOL_NAME && full.chars().all(is_safe_char)
    }

    #[test]
    fn identity_for_safe_short_names() {
        for n in ["read", "edit", "grep_files", "a-b", "X9"] {
            assert_eq!(encode_tool_name(n), n);
        }
    }

    #[test]
    fn unsafe_names_are_sanitised_and_hashed() {
        let e = encode_tool_name("todo.write");
        assert!(e.starts_with("todo_write_"), "{e}");
        assert_eq!(e.len(), "todo_write_".len() + 8);
        assert!(valid_full(&e));
        // Deterministic and distinct from a near-collision.
        assert_eq!(e, encode_tool_name("todo.write"));
        assert_ne!(e, encode_tool_name("todo/write"));
        // The plain sanitised form is a different (identity) name.
        assert_eq!(encode_tool_name("todo_write"), "todo_write");
    }

    #[test]
    fn long_and_odd_names_fit() {
        let long = "x".repeat(200);
        let e = encode_tool_name(&long);
        assert!(valid_full(&e), "{e}");
        let exact = "y".repeat(MAX_ENCODED_NAME);
        assert_eq!(encode_tool_name(&exact), exact);
        let over = "y".repeat(MAX_ENCODED_NAME + 1);
        assert!(valid_full(&encode_tool_name(&over)));
        for n in ["", "日本語ツール", "a b c", "ü"] {
            let e = encode_tool_name(n);
            assert!(valid_full(&e), "{n:?} -> {e}");
        }
    }

    #[test]
    fn codec_round_trip_and_history_names() {
        let mut c = ToolNameCodec::new(["read", "todo.write"]);
        assert_eq!(c.decode_claude_name(&c.claude_name("todo.write")), "todo.write");
        assert_eq!(c.decode_claude_name("mcp__bridge__read"), "read");
        let enc = c.register("legacy:tool");
        assert_eq!(c.decode(&enc), "legacy:tool");
        assert_eq!(c.decode("unknown_tool"), "unknown_tool");
    }

    #[test]
    fn call_ids() {
        assert_eq!(encode_call_id("toolu_01ABC"), "toolu_01ABC");
        assert_eq!(encode_call_id("call_abc123XYZ"), "call_abc123XYZ");
        let e = encode_call_id("call:abc/1");
        assert!(e.starts_with(ENCODED_ID_PREFIX));
        assert!(is_valid_call_id(&e));
        assert_eq!(e, encode_call_id("call:abc/1"));
        assert_ne!(e, encode_call_id("call:abc/2"));
        let mut c = CallIdCodec::default();
        let enc = c.encode("call:abc/1");
        assert_eq!(c.decode(&enc), "call:abc/1");
        assert_eq!(c.decode("toolu_01XYZ"), "toolu_01XYZ");
        assert_eq!(c.encode("toolu_ok"), "toolu_ok");
        assert!(!is_valid_call_id(&"a".repeat(200)));
    }
}
