//! Message normalisation and rolling prefix hashes (ARCHITECTURE.md section 6.2).
//!
//! `norm(m)` is a canonical byte string for one message:
//! - role (`system` and `developer` are already one role in the IR);
//! - text parts: adjacent text parts merged, `\r\n` → `\n`, trailing whitespace trimmed,
//!   empty text dropped (so `content: null`, `""` and a missing text part are equal);
//! - tool calls: id, name, arguments re-serialised as canonical JSON (sorted keys, no
//!   whitespace) when they parse, otherwise the raw string;
//! - tool results: call id and normalised text;
//! - images: SHA-256 digest of the base64 data or URL, plus media type;
//! - reasoning parts are excluded.
//!
//! `prefixHash_k = SHA256(prefixHash_{k-1} ‖ norm(m_k))`, `prefixHash_0` = 32 zero bytes.
//! Hashes are hex strings.

use bridge_types::{ImageSource, Message, Part, Role};
use sha2::{Digest, Sha256};

/// Canonical bytes of one message.
pub fn norm(message: &Message) -> Vec<u8> {
    let mut items: Vec<String> = Vec::new();
    let mut pending_text: Option<String> = None;

    let flush = |pending: &mut Option<String>, items: &mut Vec<String>| {
        if let Some(t) = pending.take() {
            let t = norm_text(&t);
            if !t.is_empty() {
                items.push(format!("{{\"t\":{}}}", json_str(&t)));
            }
        }
    };

    for part in &message.parts {
        match part {
            Part::Text { text } => {
                pending_text.get_or_insert_with(String::new).push_str(text);
            }
            Part::Reasoning { .. } => {}
            Part::Image(img) => {
                flush(&mut pending_text, &mut items);
                let digest = match &img.source {
                    ImageSource::Base64 { data } => sha_hex(format!("b64:{data}").as_bytes()),
                    ImageSource::Url { url } => sha_hex(format!("url:{url}").as_bytes()),
                };
                items.push(format!("{{\"img\":{},\"mt\":{}}}", json_str(&digest), json_str(&img.media_type)));
            }
            Part::ToolCall(call) => {
                flush(&mut pending_text, &mut items);
                items.push(format!(
                    "{{\"call\":{},\"name\":{},\"args\":{}}}",
                    json_str(&call.id),
                    json_str(&call.name),
                    json_str(&canonical_args(&call.arguments))
                ));
            }
            Part::ToolResult(res) => {
                flush(&mut pending_text, &mut items);
                items.push(format!(
                    "{{\"result\":{},\"text\":{}}}",
                    json_str(&res.call_id),
                    json_str(&norm_text(&res.content))
                ));
            }
        }
    }
    flush(&mut pending_text, &mut items);

    let role = match message.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    format!("{{\"role\":\"{role}\",\"parts\":[{}]}}", items.join(",")).into_bytes()
}

/// `\r\n` → `\n`, trailing whitespace trimmed.
pub fn norm_text(text: &str) -> String {
    text.replace("\r\n", "\n").trim_end().to_string()
}

/// Arguments as canonical JSON when parseable, else the raw string.
pub fn canonical_args(args: &str) -> String {
    if args.trim().is_empty() {
        // An empty argument string means "no arguments"; treat like `{}`.
        return "{}".to_string();
    }
    match serde_json::from_str::<serde_json::Value>(args) {
        Ok(v) => canonical_json(&v),
        Err(_) => args.to_string(),
    }
}

/// Canonical JSON: object keys sorted, no insignificant whitespace. Independent of
/// serde_json's `preserve_order` feature.
pub fn canonical_json(value: &serde_json::Value) -> String {
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &serde_json::Value, out: &mut String) {
    use serde_json::Value;
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&json_str(k));
                out.push(':');
                if let Some(v) = map.get(*k) {
                    write_canonical(v, out);
                }
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(v, out);
            }
            out.push(']');
        }
        other => out.push_str(&other.to_string()),
    }
}

fn json_str(s: &str) -> String {
    serde_json::Value::String(s.to_string()).to_string()
}

fn sha_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Next rolling hash from the previous one (hex) and a message.
pub fn next_hash(prev: Option<&str>, message: &Message) -> String {
    let mut h = Sha256::new();
    match prev.and_then(|p| hex::decode(p).ok()) {
        Some(bytes) => h.update(&bytes),
        None => h.update([0u8; 32]),
    }
    h.update(norm(message));
    hex::encode(h.finalize())
}

/// `out[k-1]` = hash of the first `k` messages.
pub fn prefix_hashes(messages: &[Message]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(messages.len());
    for m in messages {
        let next = next_hash(out.last().map(String::as_str), m);
        out.push(next);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use bridge_types::{Image, ToolCall, ToolResult};

    fn user(t: &str) -> Message {
        Message::text(Role::User, t)
    }

    #[test]
    fn trailing_whitespace_and_crlf_are_ignored() {
        assert_eq!(norm(&user("hello\r\nworld  \n\n")), norm(&user("hello\nworld")));
        assert_ne!(norm(&user("hello world")), norm(&user("hello  world")));
        // Leading whitespace matters.
        assert_ne!(norm(&user(" hi")), norm(&user("hi")));
    }

    #[test]
    fn adjacent_text_parts_merge_and_empty_text_is_dropped() {
        let split = Message {
            role: Role::User,
            parts: vec![Part::Text { text: "ab".into() }, Part::Text { text: "cd".into() }],
        };
        assert_eq!(norm(&split), norm(&user("abcd")));
        let call = ToolCall { id: "c1".into(), name: "read".into(), arguments: "{}".into() };
        let with_empty = Message {
            role: Role::Assistant,
            parts: vec![Part::Text { text: "".into() }, Part::ToolCall(call.clone())],
        };
        let without = Message { role: Role::Assistant, parts: vec![Part::ToolCall(call)] };
        assert_eq!(norm(&with_empty), norm(&without));
    }

    #[test]
    fn tool_args_key_order_and_whitespace_ignored() {
        let a = Message {
            role: Role::Assistant,
            parts: vec![Part::ToolCall(ToolCall {
                id: "c1".into(),
                name: "read".into(),
                arguments: r#"{"b": 1, "a": {"y": [1, 2], "x": "s"}}"#.into(),
            })],
        };
        let b = Message {
            role: Role::Assistant,
            parts: vec![Part::ToolCall(ToolCall {
                id: "c1".into(),
                name: "read".into(),
                arguments: r#"{"a":{"x":"s","y":[1,2]},"b":1}"#.into(),
            })],
        };
        assert_eq!(norm(&a), norm(&b));
        // Different id → different.
        let mut c = b.clone();
        if let Part::ToolCall(call) = &mut c.parts[0] {
            call.id = "c2".into();
        }
        assert_ne!(norm(&b), norm(&c));
        // Unparseable args are kept raw.
        assert_eq!(canonical_args("not json"), "not json");
        assert_eq!(canonical_args(""), "{}");
    }

    #[test]
    fn reasoning_is_excluded() {
        let with = Message {
            role: Role::Assistant,
            parts: vec![Part::Reasoning { text: "thinking".into() }, Part::Text { text: "answer".into() }],
        };
        assert_eq!(norm(&with), norm(&Message::text(Role::Assistant, "answer")));
    }

    #[test]
    fn roles_matter_and_system_equals_developer() {
        // The frontend maps developer → System, so both produce the same IR role.
        assert_ne!(norm(&user("x")), norm(&Message::text(Role::Assistant, "x")));
        assert_eq!(norm(&Message::text(Role::System, "x")), norm(&Message::text(Role::System, "x ")));
    }

    #[test]
    fn tool_results_and_images() {
        let t1 = Message {
            role: Role::Tool,
            parts: vec![Part::ToolResult(ToolResult { call_id: "c1".into(), content: "ok\r\n".into() })],
        };
        let t2 = Message {
            role: Role::Tool,
            parts: vec![Part::ToolResult(ToolResult { call_id: "c1".into(), content: "ok".into() })],
        };
        assert_eq!(norm(&t1), norm(&t2));
        let img = |d: &str| Message {
            role: Role::User,
            parts: vec![Part::Image(Image {
                media_type: "image/png".into(),
                source: ImageSource::Base64 { data: d.into() },
            })],
        };
        assert_eq!(norm(&img("AAAA")), norm(&img("AAAA")));
        assert_ne!(norm(&img("AAAA")), norm(&img("BBBB")));
        // The data itself is not embedded, only its digest.
        assert!(!String::from_utf8_lossy(&norm(&img("QUJDREVGR0g="))).contains("QUJDREVGR0g="));
    }

    #[test]
    fn prefix_hashes_are_rolling() {
        let msgs = vec![Message::text(Role::System, "s"), user("a"), Message::text(Role::Assistant, "b")];
        let h = prefix_hashes(&msgs);
        assert_eq!(h.len(), 3);
        assert_eq!(h[2], next_hash(Some(&h[1]), &msgs[2]));
        let h2 = prefix_hashes(&msgs[..2]);
        assert_eq!(h2, h[..2].to_vec());
        // Different earlier message changes all later hashes.
        let mut other = msgs.clone();
        other[1] = user("A");
        let h3 = prefix_hashes(&other);
        assert_eq!(h3[0], h[0]);
        assert_ne!(h3[1], h[1]);
        assert_ne!(h3[2], h[2]);
        // Text that would collide by concatenation does not.
        let x = prefix_hashes(&[user("ab"), user("c")]);
        let y = prefix_hashes(&[user("a"), user("bc")]);
        assert_ne!(x[1], y[1]);
    }
}
