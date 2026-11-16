//! Rewrites `mcp__bridge__<encoded>` in streamed text to the caller's tool name (P10),
//! holding back a possibly partial prefix/name across deltas.

use crate::codec::{ToolNameCodec, MAX_ENCODED_NAME, TOOL_PREFIX};

fn is_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

#[derive(Debug, Default)]
pub struct Sanitizer {
    held: String,
}

impl Sanitizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds a delta; returns the text that is safe to emit now.
    pub fn push(&mut self, text: &str, names: &ToolNameCodec) -> String {
        self.held.push_str(text);
        self.drain(names, false)
    }

    /// End of stream: emits everything still held.
    pub fn finish(&mut self, names: &ToolNameCodec) -> String {
        self.drain(names, true)
    }

    fn drain(&mut self, names: &ToolNameCodec, eof: bool) -> String {
        let mut out = String::new();
        let buf = std::mem::take(&mut self.held);
        let mut rest: &str = &buf;
        loop {
            if let Some(i) = rest.find(TOOL_PREFIX) {
                out.push_str(&rest[..i]);
                let after = &rest[i + TOOL_PREFIX.len()..];
                let name_len = after.find(|c: char| !is_name_char(c)).unwrap_or(after.len());
                let complete = name_len < after.len() || eof || name_len > MAX_ENCODED_NAME;
                if !complete {
                    self.held = rest[i..].to_string();
                    return out;
                }
                out.push_str(&names.decode(&after[..name_len]));
                rest = &after[name_len..];
                continue;
            }
            if eof {
                out.push_str(rest);
                return out;
            }
            // Hold back the longest suffix that is a proper prefix of TOOL_PREFIX.
            let mut keep = 0;
            for k in (1..TOOL_PREFIX.len()).rev() {
                if rest.len() >= k && rest.is_char_boundary(rest.len() - k) && TOOL_PREFIX.starts_with(&rest[rest.len() - k..]) {
                    keep = k;
                    break;
                }
            }
            out.push_str(&rest[..rest.len() - keep]);
            self.held = rest[rest.len() - keep..].to_string();
            return out;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(chunks: &[&str], names: &ToolNameCodec) -> (Vec<String>, String) {
        let mut s = Sanitizer::new();
        let mut parts: Vec<String> = chunks.iter().map(|c| s.push(c, names)).collect();
        parts.push(s.finish(names));
        let joined = parts.concat();
        (parts, joined)
    }

    #[test]
    fn rewrites_across_every_split() {
        let names = ToolNameCodec::new(["read", "todo.write"]);
        let enc = names.encode("todo.write");
        let text = format!("I can use mcp__bridge__read and {TOOL_PREFIX}{enc}, plus mcp__other.");
        let expected = "I can use read and todo.write, plus mcp__other.";
        for i in 0..text.len() {
            for j in i..text.len() {
                let (a, b, c) = (&text[..i], &text[i..j], &text[j..]);
                let (_, joined) = run(&[a, b, c], &names);
                assert_eq!(joined, expected, "split at {i},{j}");
            }
        }
        // One char at a time; nothing leaks in any single emitted delta.
        let chars: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        let refs: Vec<&str> = chars.iter().map(String::as_str).collect();
        let (parts, joined) = run(&refs, &names);
        assert_eq!(joined, expected);
        assert!(parts.iter().all(|p| !p.contains("mcp__bridge")));
    }

    #[test]
    fn name_at_end_of_stream_and_plain_text() {
        let names = ToolNameCodec::new(["bash"]);
        assert_eq!(run(&["call mcp__bridge__", "bash"], &names).1, "call bash");
        assert_eq!(run(&["no tools here"], &names).1, "no tools here");
        assert_eq!(run(&["ends with mcp_"], &names).1, "ends with mcp_");
        assert_eq!(run(&["multi\u{00e9}byte mcp", "__bridge__bash!"], &names).1, "multi\u{00e9}byte bash!");
    }

    #[test]
    fn emits_promptly() {
        let names = ToolNameCodec::new(["read"]);
        let mut s = Sanitizer::new();
        assert_eq!(s.push("Hello world", &names), "Hello world");
        assert_eq!(s.push(" mcp__br", &names), " ");
        assert_eq!(s.push("idge__read ok", &names), "read ok");
    }
}
