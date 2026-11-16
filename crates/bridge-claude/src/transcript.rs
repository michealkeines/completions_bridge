//! TRANSCRIPT mode (ARCHITECTURE 8.5): the whole history rendered as one labelled user
//! message, used when the session-file tricks failed the self-test.

use bridge_types::{Message, Part, Role, Trigger};
use serde_json::{json, Value};

use crate::session::image_block;

/// Built-in transcript header (`prompts.transcript_header` overrides it per backend).
pub const HEADER: &str = "The earlier part of this conversation is reproduced below as a transcript \
(between <transcript> and </transcript>). Treat it as what has already happened: messages \
labelled [User] were written by the user, [Assistant] by you, [Tool call] are tool calls you \
made and [Tool result] are their results. Do not mention the transcript format.";

fn render_message(m: &Message, out: &mut String) {
    let label = match m.role {
        Role::System => "[System]",
        Role::User => "[User]",
        Role::Assistant => "[Assistant]",
        Role::Tool => "",
    };
    let text = m.text_content();
    let images = m.parts.iter().filter(|p| matches!(p, Part::Image(_))).count();
    if m.role != Role::Tool && (!text.is_empty() || images > 0) {
        out.push_str(label);
        out.push('\n');
        if !text.is_empty() {
            out.push_str(&text);
            out.push('\n');
        }
        for _ in 0..images {
            out.push_str("(an image was attached here)\n");
        }
        out.push('\n');
    }
    for c in m.tool_calls() {
        out.push_str(&format!("[Tool call] id={} tool={} arguments={}\n\n", c.id, c.name, c.arguments));
    }
    for r in m.tool_results() {
        out.push_str(&format!("[Tool result] id={}\n{}\n\n", r.call_id, r.content));
    }
}

/// Builds the stdin `message.content` for a transcript-mode turn. `header` introduces the
/// transcript (the backend's `Texts::transcript_header`).
pub fn render(history: &[Message], trigger: &Trigger, header: &str) -> Value {
    let mut text = String::new();
    if !history.is_empty() {
        text.push_str(header);
        text.push_str("\n\n<transcript>\n");
        for m in history {
            render_message(m, &mut text);
        }
        text.push_str("</transcript>\n\n");
    }
    let mut images = Vec::new();
    match trigger {
        Trigger::User { message } => {
            if !history.is_empty() {
                text.push_str("The user's new message follows. Reply to it as the assistant.\n\n");
            }
            text.push_str(&message.text_content());
            for p in &message.parts {
                if let Part::Image(img) = p {
                    images.push(image_block(img));
                }
            }
        }
        Trigger::ContinueAfterResults => {
            text.push_str("Continue as the assistant from the end of the transcript (for example, use the latest tool results).");
        }
    }
    if images.is_empty() {
        return Value::String(text);
    }
    let mut blocks = vec![json!({"type": "text", "text": text})];
    blocks.extend(images);
    Value::Array(blocks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bridge_types::{Image, ImageSource, ToolCall, ToolResult};

    #[test]
    fn renders_labelled_history() {
        let h = vec![
            Message::text(Role::User, "read a.txt"),
            Message { role: Role::Assistant, parts: vec![Part::ToolCall(ToolCall { id: "c1".into(), name: "read".into(), arguments: "{}".into() })] },
            Message { role: Role::Tool, parts: vec![Part::ToolResult(ToolResult { call_id: "c1".into(), content: "hello".into() })] },
        ];
        let v = render(&h, &Trigger::ContinueAfterResults, HEADER);
        let s = v.as_str().unwrap();
        assert!(s.contains("<transcript>\n[User]\nread a.txt"));
        assert!(s.contains("[Tool call] id=c1 tool=read"));
        assert!(s.contains("[Tool result] id=c1\nhello"));
        assert!(s.starts_with(HEADER));
        let custom = render(&h, &Trigger::ContinueAfterResults, "HISTORY FOLLOWS");
        assert!(custom.as_str().unwrap().starts_with("HISTORY FOLLOWS\n\n<transcript>"));
        let img = Message {
            role: Role::User,
            parts: vec![
                Part::Text { text: "what is this".into() },
                Part::Image(Image { media_type: "image/png".into(), source: ImageSource::Base64 { data: "AA".into() } }),
            ],
        };
        let v = render(&[], &Trigger::User { message: img }, HEADER);
        assert_eq!(v[0]["text"], "what is this");
        assert_eq!(v[1]["type"], "image");
    }
}
