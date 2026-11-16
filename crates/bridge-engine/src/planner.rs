//! Planner (ARCHITECTURE.md sections 3.3 and 7.1): pick the deepest usable checkpoint
//! and split the rest of the request into `append` and the trigger.

use std::collections::{BTreeMap, BTreeSet};

use bridge_types::{CheckpointKind, Message, Role, Trigger};

use crate::store::CheckpointRecord;

/// Number of leading system messages (they become the system prompt, not history).
pub fn leading_system_count(messages: &[Message]) -> usize {
    messages.iter().take_while(|m| m.role == Role::System).count()
}

/// Leading system messages joined with a blank line.
pub fn leading_system_text(messages: &[Message]) -> String {
    messages[..leading_system_count(messages)]
        .iter()
        .map(Message::text_content)
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Can `suffix` be attached to a checkpoint of this kind?
pub fn compatible(kind: CheckpointKind, pending: &[String], suffix: &[Message]) -> bool {
    let Some(first) = suffix.first() else { return false };
    match kind {
        CheckpointKind::Idle => matches!(first.role, Role::User | Role::System),
        CheckpointKind::Awaiting => {
            let answered: BTreeSet<&str> = suffix
                .iter()
                .take_while(|m| m.role == Role::Tool)
                .flat_map(|m| m.tool_results())
                .map(|r| r.call_id.as_str())
                .collect();
            let pending: BTreeSet<&str> = pending.iter().map(String::as_str).collect();
            !pending.is_empty() && answered == pending
        }
    }
}

/// Deepest checkpoint whose kind accepts the rest of the request.
/// `candidates` maps prefix depth (number of messages) → newest checkpoint at that depth.
pub fn choose_base<'a>(
    messages: &[Message],
    candidates: &'a BTreeMap<usize, CheckpointRecord>,
) -> Option<(usize, &'a CheckpointRecord)> {
    candidates.iter().rev().find_map(|(&depth, rec)| {
        if depth == 0 || depth >= messages.len() {
            return None;
        }
        compatible(rec.snapshot.kind, &rec.snapshot.pending_call_ids, &messages[depth..]).then_some((depth, rec))
    })
}

/// Split everything after the base (and after the leading system messages) into the
/// messages to write into history and the trigger.
pub fn split_suffix(messages: &[Message], base_depth: usize) -> (Vec<Message>, Trigger) {
    let start = base_depth.max(leading_system_count(messages)).min(messages.len());
    let rest = &messages[start..];
    match rest.last() {
        Some(last) if last.role == Role::User => {
            (rest[..rest.len() - 1].to_vec(), Trigger::User { message: last.clone() })
        }
        Some(_) => (rest.to_vec(), Trigger::ContinueAfterResults),
        None => {
            // Only system messages (or nothing): deliver the last one as the user input so
            // the model has something to answer. Validation normally prevents this.
            let text = messages.last().map(Message::text_content).unwrap_or_default();
            (Vec::new(), Trigger::User { message: Message::text(Role::User, text) })
        }
    }
}

/// Side request (titles, summaries): no tools and at most two non-system messages.
pub fn is_side_request(messages: &[Message], tool_count: usize) -> bool {
    tool_count == 0 && messages.iter().filter(|m| m.role != Role::System).count() <= 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use bridge_types::{Part, SnapshotInfo, ToolCall, ToolResult};

    fn sys(t: &str) -> Message {
        Message::text(Role::System, t)
    }
    fn user(t: &str) -> Message {
        Message::text(Role::User, t)
    }
    fn asst(t: &str) -> Message {
        Message::text(Role::Assistant, t)
    }
    fn call(ids: &[&str]) -> Message {
        Message {
            role: Role::Assistant,
            parts: ids
                .iter()
                .map(|id| Part::ToolCall(ToolCall { id: (*id).into(), name: "read".into(), arguments: "{}".into() }))
                .collect(),
        }
    }
    fn tool(id: &str) -> Message {
        Message {
            role: Role::Tool,
            parts: vec![Part::ToolResult(ToolResult { call_id: id.into(), content: "r".into() })],
        }
    }
    fn rec(kind: CheckpointKind, pending: &[&str]) -> CheckpointRecord {
        CheckpointRecord {
            id: 1,
            backend: "b".into(),
            prefix_hash: "h".into(),
            snapshot: SnapshotInfo {
                path: "/p".into(),
                kind,
                pending_call_ids: pending.iter().map(|s| s.to_string()).collect(),
                size_bytes: 1,
                backend_version: "v".into(),
            },
            created_at: 0,
            last_used_at: 0,
        }
    }

    #[test]
    fn idle_accepts_user_or_system_only() {
        assert!(compatible(CheckpointKind::Idle, &[], &[user("x")]));
        assert!(compatible(CheckpointKind::Idle, &[], &[sys("x"), user("y")]));
        assert!(!compatible(CheckpointKind::Idle, &[], &[asst("x")]));
        assert!(!compatible(CheckpointKind::Idle, &[], &[tool("c")]));
        assert!(!compatible(CheckpointKind::Idle, &[], &[]));
    }

    #[test]
    fn awaiting_needs_exactly_the_pending_ids_in_any_order() {
        let p = vec!["c1".to_string(), "c2".to_string()];
        assert!(compatible(CheckpointKind::Awaiting, &p, &[tool("c2"), tool("c1")]));
        assert!(compatible(CheckpointKind::Awaiting, &p, &[tool("c1"), tool("c2"), user("u")]));
        assert!(!compatible(CheckpointKind::Awaiting, &p, &[tool("c1")]));
        assert!(!compatible(CheckpointKind::Awaiting, &p, &[tool("c1"), user("u"), tool("c2")]));
        assert!(!compatible(CheckpointKind::Awaiting, &p, &[user("u")]));
        assert!(!compatible(CheckpointKind::Awaiting, &p, &[tool("c1"), tool("c2"), tool("c3")]));
    }

    #[test]
    fn choose_deepest_compatible() {
        // [sys, u1, a1(call c1), t1, a2, u2]
        let msgs = vec![sys("s"), user("u1"), call(&["c1"]), tool("c1"), asst("a2"), user("u2")];
        let mut cands = BTreeMap::new();
        cands.insert(3, rec(CheckpointKind::Awaiting, &["c1"]));
        cands.insert(5, rec(CheckpointKind::Idle, &[]));
        assert_eq!(choose_base(&msgs, &cands).map(|(d, _)| d), Some(5));
        // An incompatible deeper one is skipped.
        cands.insert(5, rec(CheckpointKind::Awaiting, &["zz"]));
        assert_eq!(choose_base(&msgs, &cands).map(|(d, _)| d), Some(3));
        // Full-length match is not usable (nothing to attach).
        let mut full = BTreeMap::new();
        full.insert(6, rec(CheckpointKind::Idle, &[]));
        assert!(choose_base(&msgs, &full).is_none());
    }

    #[test]
    fn split_user_trigger_and_leading_system() {
        let msgs = vec![sys("s1"), sys("s2"), user("u1"), asst("a1"), sys("mid"), user("u2")];
        let (append, trigger) = split_suffix(&msgs, 0);
        assert_eq!(append, vec![user("u1"), asst("a1"), sys("mid")]);
        assert_eq!(trigger, Trigger::User { message: user("u2") });
        assert_eq!(leading_system_text(&msgs), "s1\n\ns2");
        let (append, trigger) = split_suffix(&msgs, 4);
        assert_eq!(append, vec![sys("mid")]);
        assert_eq!(trigger, Trigger::User { message: user("u2") });
    }

    #[test]
    fn split_tool_results_and_prefill() {
        let msgs = vec![user("u1"), call(&["c1"]), tool("c1")];
        let (append, trigger) = split_suffix(&msgs, 2);
        assert_eq!(append, vec![tool("c1")]);
        assert_eq!(trigger, Trigger::ContinueAfterResults);
        let (append, trigger) = split_suffix(&msgs, 0);
        assert_eq!(append.len(), 3);
        assert_eq!(trigger, Trigger::ContinueAfterResults);
        // tool results + user message
        let msgs2 = vec![user("u1"), call(&["c1"]), tool("c1"), user("u2")];
        let (append, trigger) = split_suffix(&msgs2, 2);
        assert_eq!(append, vec![tool("c1")]);
        assert_eq!(trigger, Trigger::User { message: user("u2") });
        // assistant prefill
        let msgs3 = vec![user("u1"), asst("Sure, ")];
        let (append, trigger) = split_suffix(&msgs3, 0);
        assert_eq!(append, msgs3);
        assert_eq!(trigger, Trigger::ContinueAfterResults);
        // only system
        let (append, trigger) = split_suffix(&[sys("hello")], 0);
        assert!(append.is_empty());
        assert_eq!(trigger, Trigger::User { message: user("hello") });
    }

    #[test]
    fn side_request_detection() {
        assert!(is_side_request(&[sys("s"), user("title this")], 0));
        assert!(is_side_request(&[sys("s"), user("a"), asst("b")], 0));
        assert!(!is_side_request(&[user("a"), asst("b"), user("c")], 0));
        assert!(!is_side_request(&[user("a")], 1));
    }
}
