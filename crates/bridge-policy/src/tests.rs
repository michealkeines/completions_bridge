use super::*;
use bridge_types::{BridgeError, ErrorClass, LimitInfo, LimitStatus};
use serde_json::json;

// ------------------------------------------------------------ helpers

fn spec() -> PolicySpec {
    PolicySpec {
        tool_choice: ToolChoice::Auto,
        parallel_tool_calls: true,
        max_tokens: None,
        stop: Vec::new(),
        response_format: ResponseFormat::Text,
        tool_names: vec!["get_weather".into(), "search".into()],
        chars_per_token: 4,
        json_max_candidates: 32,
        instructions: PolicyInstructions::default(),
    }
}

fn td(s: &str) -> TurnEvent {
    TurnEvent::TextDelta { text: s.into() }
}
fn rd(s: &str) -> TurnEvent {
    TurnEvent::ReasoningDelta { text: s.into() }
}
fn call(id: &str, name: &str) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: "{}".into(),
    }
}
fn calls(cs: Vec<ToolCall>) -> TurnEvent {
    TurnEvent::ToolCalls { calls: cs }
}
fn done(f: FinishReason) -> TurnEvent {
    TurnEvent::Done { finish: f }
}
fn usage() -> TurnEvent {
    TurnEvent::Usage {
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            cached_tokens: 0,
            reasoning_tokens: 0,
        },
    }
}
fn limit() -> TurnEvent {
    TurnEvent::Limit {
        info: LimitInfo {
            status: LimitStatus::Warning,
            utilization: Some(0.9),
            resets_at: None,
        },
    }
}
fn emit(e: TurnEvent) -> PolicyOutput {
    PolicyOutput::Emit(e)
}

/// Feed all events and `finish()`, return (outputs, policy).
fn run(
    s: PolicySpec,
    final_attempt: bool,
    events: Vec<TurnEvent>,
) -> (Vec<PolicyOutput>, OutputPolicy) {
    let mut p = OutputPolicy::new(s, final_attempt);
    let mut out = Vec::new();
    for e in events {
        out.extend(p.on_event(e));
    }
    out.extend(p.finish());
    (out, p)
}

fn text_of(out: &[PolicyOutput]) -> String {
    out.iter()
        .filter_map(|o| match o {
            PolicyOutput::Emit(TurnEvent::TextDelta { text }) => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

fn text_delta_count(out: &[PolicyOutput]) -> usize {
    out.iter()
        .filter(|o| matches!(o, PolicyOutput::Emit(TurnEvent::TextDelta { .. })))
        .count()
}

fn done_of(out: &[PolicyOutput]) -> Option<FinishReason> {
    out.iter().find_map(|o| match o {
        PolicyOutput::Emit(TurnEvent::Done { finish }) => Some(*finish),
        _ => None,
    })
}

fn calls_of(out: &[PolicyOutput]) -> Vec<ToolCall> {
    out.iter()
        .filter_map(|o| match o {
            PolicyOutput::Emit(TurnEvent::ToolCalls { calls }) => Some(calls.clone()),
            _ => None,
        })
        .flatten()
        .collect()
}

fn is_retry(out: &[PolicyOutput]) -> bool {
    matches!(out, [PolicyOutput::Retry { .. }])
}

fn has_stop_backend(out: &[PolicyOutput]) -> bool {
    out.contains(&PolicyOutput::StopBackend)
}

// ------------------------------------------------------------ pass-through

#[test]
fn passthrough_is_unaltered_and_immediate() {
    let s = spec();
    let mut p = OutputPolicy::new(s, false);
    assert!(!p.needs_buffering());
    assert_eq!(p.on_event(rd("think")), vec![emit(rd("think"))]);
    assert_eq!(p.on_event(td("Hel")), vec![emit(td("Hel"))]);
    assert_eq!(p.on_event(td("lo")), vec![emit(td("lo"))]);
    assert_eq!(p.on_event(usage()), vec![emit(usage())]);
    assert_eq!(
        p.on_event(done(FinishReason::Stop)),
        vec![emit(done(FinishReason::Stop))]
    );
    assert!(p.finish().is_empty());
    assert!(!p.altered());
    assert_eq!(
        p.assistant_message(),
        Message::text(Role::Assistant, "Hello")
    );
}

#[test]
fn passthrough_tool_calls() {
    let cs = vec![call("c1", "get_weather"), call("c2", "search")];
    let (out, p) = run(
        spec(),
        false,
        vec![td("x"), calls(cs.clone()), done(FinishReason::ToolCalls)],
    );
    assert_eq!(calls_of(&out), cs);
    assert_eq!(done_of(&out), Some(FinishReason::ToolCalls));
    assert!(!p.altered());
    let m = p.assistant_message();
    assert_eq!(m.role, Role::Assistant);
    assert_eq!(m.text_content(), "x");
    assert_eq!(m.tool_calls().len(), 2);
    assert!(matches!(m.parts[0], Part::Text { .. }));
}

#[test]
fn limit_passes_through_immediately_even_after_close() {
    let mut p = OutputPolicy::new(
        PolicySpec {
            tool_choice: ToolChoice::Required,
            ..spec()
        },
        false,
    );
    assert_eq!(p.on_event(limit()), vec![emit(limit())]);
    p.on_event(td("a"));
    p.on_event(done(FinishReason::Stop)); // -> retry, closed
    assert_eq!(p.on_event(limit()), vec![emit(limit())]);
    assert!(p.on_event(td("late")).is_empty());
}

#[test]
fn error_passes_through_and_ends_attempt() {
    let err = TurnEvent::Error {
        error: BridgeError::new(ErrorClass::Unavailable, "overloaded"),
    };
    let (out, _) = run(
        spec(),
        false,
        vec![td("a"), err.clone(), td("b"), done(FinishReason::Stop)],
    );
    assert_eq!(out, vec![emit(td("a")), emit(err)]);
}

#[test]
fn error_in_buffered_mode_drops_buffer() {
    let err = TurnEvent::Error {
        error: BridgeError::new(ErrorClass::Internal, "x"),
    };
    let s = PolicySpec {
        response_format: ResponseFormat::JsonObject,
        ..spec()
    };
    let (out, p) = run(s, false, vec![td("{}"), err.clone()]);
    assert_eq!(out, vec![emit(err)]);
    assert!(p.assistant_message().parts.is_empty());
}

#[test]
fn assistant_message_empty_text_omits_text_part() {
    let (_, p) = run(
        spec(),
        false,
        vec![
            calls(vec![call("c1", "search")]),
            done(FinishReason::ToolCalls),
        ],
    );
    let m = p.assistant_message();
    assert_eq!(m.parts, vec![Part::ToolCall(call("c1", "search"))]);
}

#[test]
fn assistant_message_excludes_reasoning() {
    let (_, p) = run(
        spec(),
        false,
        vec![rd("hmm"), td("hi"), done(FinishReason::Stop)],
    );
    assert_eq!(
        p.assistant_message().parts,
        vec![Part::Text { text: "hi".into() }]
    );
}

// ------------------------------------------------------------ max_tokens

#[test]
fn max_tokens_truncates_mid_delta() {
    let s = PolicySpec {
        max_tokens: Some(2),
        ..spec()
    }; // 8 chars
    let mut p = OutputPolicy::new(s, false);
    assert_eq!(p.on_event(td("12345")), vec![emit(td("12345"))]);
    assert_eq!(
        p.on_event(td("6789AB")),
        vec![
            emit(td("678")),
            emit(done(FinishReason::Length)),
            PolicyOutput::StopBackend
        ]
    );
    assert!(p.altered());
    assert!(p.on_event(td("more")).is_empty());
    assert!(p.on_event(usage()).is_empty());
    assert!(p.on_event(done(FinishReason::Stop)).is_empty());
    assert!(p.finish().is_empty());
    assert_eq!(p.assistant_message().text_content(), "12345678");
}

#[test]
fn max_tokens_exact_fit_is_not_length() {
    let s = PolicySpec {
        max_tokens: Some(1),
        ..spec()
    };
    let (out, p) = run(s, false, vec![td("abcd"), done(FinishReason::Stop)]);
    assert_eq!(text_of(&out), "abcd");
    assert_eq!(done_of(&out), Some(FinishReason::Stop));
    assert!(!has_stop_backend(&out));
    assert!(!p.altered());
}

#[test]
fn max_tokens_reasoning_does_not_count() {
    let s = PolicySpec {
        max_tokens: Some(1),
        ..spec()
    };
    let (out, p) = run(
        s,
        false,
        vec![
            rd("a very long reasoning text"),
            td("ok"),
            done(FinishReason::Stop),
        ],
    );
    assert_eq!(text_of(&out), "ok");
    assert_eq!(done_of(&out), Some(FinishReason::Stop));
    assert!(!p.altered());
}

#[test]
fn max_tokens_unicode_not_split() {
    let s = PolicySpec {
        max_tokens: Some(1),
        ..spec()
    }; // 4 chars
    let (out, p) = run(
        s,
        false,
        vec![td("héllo wörld 🎉"), done(FinishReason::Stop)],
    );
    assert_eq!(text_of(&out), "héll");
    assert_eq!(done_of(&out), Some(FinishReason::Length));
    assert!(p.altered());
    let s = PolicySpec {
        max_tokens: Some(1),
        ..spec()
    };
    let (out, _) = run(s, false, vec![td("🎉🎉🎉🎉🎉")]);
    assert_eq!(text_of(&out), "🎉🎉🎉🎉");
}

#[test]
fn max_tokens_drops_tool_calls_after_limit() {
    let s = PolicySpec {
        max_tokens: Some(1),
        ..spec()
    };
    let (out, p) = run(
        s,
        false,
        vec![
            td("abcdefgh"),
            calls(vec![call("c1", "search")]),
            done(FinishReason::ToolCalls),
        ],
    );
    assert!(calls_of(&out).is_empty());
    assert_eq!(done_of(&out), Some(FinishReason::Length));
    assert!(p.assistant_message().tool_calls().is_empty());
}

#[test]
fn max_tokens_zero() {
    let s = PolicySpec {
        max_tokens: Some(0),
        ..spec()
    };
    let (out, p) = run(s, false, vec![td("abc"), done(FinishReason::Stop)]);
    assert_eq!(
        out,
        vec![emit(done(FinishReason::Length)), PolicyOutput::StopBackend]
    );
    assert!(p.assistant_message().parts.is_empty());
}

#[test]
fn max_tokens_tool_calls_do_not_count() {
    let s = PolicySpec {
        max_tokens: Some(1),
        ..spec()
    };
    let big = ToolCall {
        id: "c".into(),
        name: "search".into(),
        arguments: "{\"q\":\"a long query string\"}".into(),
    };
    let (out, p) = run(
        s,
        false,
        vec![calls(vec![big.clone()]), done(FinishReason::ToolCalls)],
    );
    assert_eq!(calls_of(&out), vec![big]);
    assert!(!p.altered());
}

// ------------------------------------------------------------ stop sequences

#[test]
fn stop_within_one_delta() {
    let s = PolicySpec {
        stop: vec!["END".into()],
        ..spec()
    };
    let (out, p) = run(
        s,
        false,
        vec![td("hello END world"), done(FinishReason::Stop)],
    );
    assert_eq!(text_of(&out), "hello ");
    assert_eq!(done_of(&out), Some(FinishReason::Stop));
    assert!(has_stop_backend(&out));
    assert_eq!(out.last(), Some(&PolicyOutput::StopBackend));
    assert!(p.altered());
}

#[test]
fn stop_split_across_deltas() {
    let s = PolicySpec {
        stop: vec!["STOP".into()],
        ..spec()
    };
    let mut p = OutputPolicy::new(s, false);
    let mut out = Vec::new();
    for d in ["abc S", "T", "O", "P tail"] {
        out.extend(p.on_event(td(d)));
    }
    assert_eq!(text_of(&out), "abc ");
    assert_eq!(done_of(&out), Some(FinishReason::Stop));
    assert!(!text_of(&out).contains('S'));
    assert_eq!(p.assistant_message().text_content(), "abc ");
}

#[test]
fn stop_holdback_flushed_on_normal_end() {
    let s = PolicySpec {
        stop: vec!["STOP".into()],
        ..spec()
    };
    let mut p = OutputPolicy::new(s, false);
    let first = p.on_event(td("hello ST"));
    assert_eq!(text_of(&first), "hello"); // 3 bytes held back
    let rest = p.on_event(done(FinishReason::Stop));
    assert_eq!(rest, vec![emit(td(" ST")), emit(done(FinishReason::Stop))]);
    assert!(!p.altered());
    assert_eq!(p.assistant_message().text_content(), "hello ST");
}

#[test]
fn stop_partial_match_then_mismatch_releases() {
    let s = PolicySpec {
        stop: vec!["STOP".into()],
        ..spec()
    };
    let (out, p) = run(
        s,
        false,
        vec![td("aST"), td("OX"), td("yz"), done(FinishReason::Stop)],
    );
    assert_eq!(text_of(&out), "aSTOXyz");
    assert!(!p.altered());
}

#[test]
fn stop_earliest_of_multiple() {
    let s = PolicySpec {
        stop: vec!["zz".into(), "\n\n".into(), "###".into()],
        ..spec()
    };
    let (out, _) = run(s, false, vec![td("one\n\ntwo ### zz")]);
    assert_eq!(text_of(&out), "one");
}

#[test]
fn stop_at_very_start() {
    let s = PolicySpec {
        stop: vec!["X".into()],
        ..spec()
    };
    let (out, p) = run(s, false, vec![td("Xabc")]);
    assert_eq!(
        out,
        vec![emit(done(FinishReason::Stop)), PolicyOutput::StopBackend]
    );
    assert!(p.altered());
}

#[test]
fn stop_ignores_reasoning_and_tool_args() {
    let s = PolicySpec {
        stop: vec!["END".into()],
        ..spec()
    };
    let c = ToolCall {
        id: "c".into(),
        name: "search".into(),
        arguments: "{\"q\":\"END\"}".into(),
    };
    let (out, p) = run(
        s,
        false,
        vec![
            rd("END"),
            td("fine"),
            calls(vec![c.clone()]),
            done(FinishReason::ToolCalls),
        ],
    );
    assert_eq!(text_of(&out), "fine");
    assert_eq!(calls_of(&out), vec![c]);
    assert_eq!(done_of(&out), Some(FinishReason::ToolCalls));
    assert!(!p.altered());
}

#[test]
fn stop_holdback_flushed_before_tool_calls_and_usage() {
    let s = PolicySpec {
        stop: vec!["STOP".into()],
        ..spec()
    };
    let (out, _) = run(
        s,
        false,
        vec![
            td("ab"),
            calls(vec![call("c1", "search")]),
            usage(),
            done(FinishReason::ToolCalls),
        ],
    );
    assert_eq!(
        out,
        vec![
            emit(td("ab")),
            emit(calls(vec![call("c1", "search")])),
            emit(usage()),
            emit(done(FinishReason::ToolCalls)),
        ]
    );
}

#[test]
fn stop_unicode_holdback_respects_char_boundaries() {
    let s = PolicySpec {
        stop: vec!["ñé".into()],
        ..spec()
    }; // 4 bytes -> hold back 3
    let (out, _) = run(s, false, vec![td("aé"), td("bñ"), td("é rest")]);
    assert_eq!(text_of(&out), "aéb");
    let s = PolicySpec {
        stop: vec!["ñé".into()],
        ..spec()
    };
    let (out, _) = run(
        s,
        false,
        vec![td("日本語"), td("テキスト"), done(FinishReason::Stop)],
    );
    assert_eq!(text_of(&out), "日本語テキスト");
}

#[test]
fn stop_empty_string_ignored() {
    let s = PolicySpec {
        stop: vec![String::new()],
        ..spec()
    };
    let (out, p) = run(s, false, vec![td("abc"), done(FinishReason::Stop)]);
    assert_eq!(out, vec![emit(td("abc")), emit(done(FinishReason::Stop))]);
    assert!(!p.altered());
}

#[test]
fn finish_without_done_flushes_holdback_in_streaming() {
    let s = PolicySpec {
        stop: vec!["STOP".into()],
        ..spec()
    };
    let (out, _) = run(s, false, vec![td("abcST")]);
    assert_eq!(text_of(&out), "abcST");
    assert_eq!(done_of(&out), None);
}

#[test]
fn stop_and_max_tokens_earliest_wins() {
    // Limit (8 chars) comes before the stop string.
    let s = PolicySpec {
        stop: vec!["END".into()],
        max_tokens: Some(2),
        ..spec()
    };
    let (out, _) = run(s, false, vec![td("0123456789 END")]);
    assert_eq!(text_of(&out), "01234567");
    assert_eq!(done_of(&out), Some(FinishReason::Length));
    // Stop string comes before the limit.
    let s = PolicySpec {
        stop: vec!["END".into()],
        max_tokens: Some(2),
        ..spec()
    };
    let (out, _) = run(s, false, vec![td("01END23456789")]);
    assert_eq!(text_of(&out), "01");
    assert_eq!(done_of(&out), Some(FinishReason::Stop));
}

// ------------------------------------------------------------ tool_choice none

#[test]
fn tool_choice_none_text_only_ok() {
    let s = PolicySpec {
        tool_choice: ToolChoice::None,
        ..spec()
    };
    let mut p = OutputPolicy::new(s, false);
    assert!(p.needs_buffering());
    assert!(p.on_event(td("a")).is_empty());
    assert!(p.on_event(td("b")).is_empty());
    assert_eq!(p.on_event(rd("r")), vec![emit(rd("r"))]);
    assert!(p.on_event(usage()).is_empty());
    assert_eq!(
        p.on_event(done(FinishReason::Stop)),
        vec![
            emit(td("ab")),
            emit(usage()),
            emit(done(FinishReason::Stop))
        ]
    );
    assert!(!p.altered());
}

#[test]
fn tool_choice_none_with_calls_retries() {
    let s = PolicySpec {
        tool_choice: ToolChoice::None,
        ..spec()
    };
    let (out, p) = run(
        s,
        false,
        vec![
            rd("r"),
            td("a"),
            calls(vec![call("c1", "search")]),
            done(FinishReason::ToolCalls),
        ],
    );
    assert_eq!(out[0], emit(rd("r")));
    assert!(is_retry(&out[1..]));
    assert!(p.assistant_message().parts.is_empty());
}

#[test]
fn tool_choice_none_final_drops_calls() {
    let s = PolicySpec {
        tool_choice: ToolChoice::None,
        ..spec()
    };
    let (out, p) = run(
        s,
        true,
        vec![
            td("answer"),
            calls(vec![call("c1", "search")]),
            usage(),
            done(FinishReason::ToolCalls),
        ],
    );
    assert_eq!(
        out,
        vec![
            emit(td("answer")),
            emit(usage()),
            emit(done(FinishReason::Stop))
        ]
    );
    assert!(p.altered());
    assert_eq!(
        p.assistant_message(),
        Message::text(Role::Assistant, "answer")
    );
}

#[test]
fn tool_choice_none_with_stop() {
    let s = PolicySpec {
        tool_choice: ToolChoice::None,
        stop: vec!["##".into()],
        ..spec()
    };
    let (out, p) = run(
        s,
        false,
        vec![td("abc #"), td("# def"), done(FinishReason::Stop)],
    );
    assert_eq!(
        out,
        vec![
            emit(td("abc ")),
            emit(done(FinishReason::Stop)),
            PolicyOutput::StopBackend
        ]
    );
    assert!(p.altered());
}

// ------------------------------------------------------------ tool_choice required

#[test]
fn tool_choice_required_ok() {
    let s = PolicySpec {
        tool_choice: ToolChoice::Required,
        ..spec()
    };
    let (out, p) = run(
        s,
        false,
        vec![
            td("let me"),
            calls(vec![call("c1", "search")]),
            done(FinishReason::ToolCalls),
        ],
    );
    assert_eq!(
        out,
        vec![
            emit(td("let me")),
            emit(calls(vec![call("c1", "search")])),
            emit(done(FinishReason::ToolCalls))
        ]
    );
    assert!(!p.altered());
}

#[test]
fn tool_choice_required_missing_retries_then_final_emits() {
    let s = PolicySpec {
        tool_choice: ToolChoice::Required,
        ..spec()
    };
    let (out, _) = run(
        s.clone(),
        false,
        vec![td("no tools"), done(FinishReason::Stop)],
    );
    assert!(is_retry(&out));
    let (out, p) = run(s, true, vec![td("no tools"), done(FinishReason::Stop)]);
    assert_eq!(
        out,
        vec![emit(td("no tools")), emit(done(FinishReason::Stop))]
    );
    assert!(!p.altered());
}

#[test]
fn retry_swallows_later_events() {
    let s = PolicySpec {
        tool_choice: ToolChoice::Required,
        ..spec()
    };
    let mut p = OutputPolicy::new(s, false);
    p.on_event(td("x"));
    assert!(is_retry(&p.on_event(done(FinishReason::Stop))));
    assert!(p.on_event(td("y")).is_empty());
    assert!(p.finish().is_empty());
}

#[test]
fn buffered_without_done_emits_nothing() {
    let s = PolicySpec {
        tool_choice: ToolChoice::Required,
        ..spec()
    };
    let (out, p) = run(s, false, vec![td("abc")]);
    assert!(out.is_empty());
    assert!(p.assistant_message().parts.is_empty());
}

// ------------------------------------------------------------ tool_choice function

fn func_spec() -> PolicySpec {
    PolicySpec {
        tool_choice: ToolChoice::Function {
            name: "get_weather".into(),
        },
        ..spec()
    }
}

#[test]
fn tool_choice_function_ok() {
    let (out, p) = run(
        func_spec(),
        false,
        vec![
            calls(vec![call("c1", "get_weather")]),
            done(FinishReason::ToolCalls),
        ],
    );
    assert_eq!(calls_of(&out), vec![call("c1", "get_weather")]);
    assert!(!p.altered());
}

#[test]
fn tool_choice_function_wrong_name() {
    let (out, _) = run(
        func_spec(),
        false,
        vec![
            calls(vec![call("c1", "search")]),
            done(FinishReason::ToolCalls),
        ],
    );
    assert!(is_retry(&out));
    let (out, p) = run(
        func_spec(),
        true,
        vec![
            calls(vec![call("c1", "search")]),
            done(FinishReason::ToolCalls),
        ],
    );
    assert_eq!(calls_of(&out), vec![call("c1", "search")]);
    assert!(!p.altered());
}

#[test]
fn tool_choice_function_no_calls() {
    let (out, _) = run(func_spec(), false, vec![td("hi"), done(FinishReason::Stop)]);
    assert!(is_retry(&out));
}

#[test]
fn tool_choice_function_mixed_calls() {
    let mixed = vec![
        call("c1", "search"),
        call("c2", "get_weather"),
        call("c3", "get_weather"),
    ];
    let (out, _) = run(
        func_spec(),
        false,
        vec![calls(mixed.clone()), done(FinishReason::ToolCalls)],
    );
    assert!(is_retry(&out));
    let (out, p) = run(
        func_spec(),
        true,
        vec![calls(mixed), done(FinishReason::ToolCalls)],
    );
    assert_eq!(
        calls_of(&out),
        vec![call("c2", "get_weather"), call("c3", "get_weather")]
    );
    assert_eq!(done_of(&out), Some(FinishReason::ToolCalls));
    assert!(p.altered());
    assert_eq!(p.assistant_message().tool_calls().len(), 2);
}

#[test]
fn tool_choice_function_mixed_with_parallel_false() {
    let s = PolicySpec {
        parallel_tool_calls: false,
        ..func_spec()
    };
    let mixed = vec![
        call("c1", "search"),
        call("c2", "get_weather"),
        call("c3", "get_weather"),
    ];
    let (out, p) = run(s, true, vec![calls(mixed), done(FinishReason::ToolCalls)]);
    assert_eq!(calls_of(&out), vec![call("c2", "get_weather")]);
    assert!(p.altered());
}

// ------------------------------------------------------------ parallel_tool_calls

#[test]
fn parallel_false_keeps_first_without_buffering() {
    let s = PolicySpec {
        parallel_tool_calls: false,
        ..spec()
    };
    let mut p = OutputPolicy::new(s, false);
    assert!(!p.needs_buffering());
    assert_eq!(p.on_event(td("t")), vec![emit(td("t"))]);
    let out = p.on_event(calls(vec![call("c1", "search"), call("c2", "get_weather")]));
    assert_eq!(out, vec![emit(calls(vec![call("c1", "search")]))]);
    assert_eq!(
        p.on_event(done(FinishReason::ToolCalls)),
        vec![emit(done(FinishReason::ToolCalls))]
    );
    assert!(p.altered());
    assert_eq!(
        p.assistant_message().tool_calls(),
        vec![&call("c1", "search")]
    );
}

#[test]
fn parallel_false_single_call_unaltered() {
    let s = PolicySpec {
        parallel_tool_calls: false,
        ..spec()
    };
    let (_, p) = run(
        s,
        false,
        vec![
            calls(vec![call("c1", "search")]),
            done(FinishReason::ToolCalls),
        ],
    );
    assert!(!p.altered());
}

#[test]
fn required_with_parallel_false_buffered() {
    let s = PolicySpec {
        tool_choice: ToolChoice::Required,
        parallel_tool_calls: false,
        ..spec()
    };
    let (out, p) = run(
        s,
        false,
        vec![
            calls(vec![call("c1", "search"), call("c2", "search")]),
            done(FinishReason::ToolCalls),
        ],
    );
    assert_eq!(calls_of(&out), vec![call("c1", "search")]);
    assert!(p.altered());
}

// ------------------------------------------------------------ JSON formats

fn json_spec() -> PolicySpec {
    PolicySpec {
        response_format: ResponseFormat::JsonObject,
        ..spec()
    }
}

fn schema_spec() -> PolicySpec {
    PolicySpec {
        response_format: ResponseFormat::JsonSchema {
            name: "person".into(),
            schema: json!({
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
                "additionalProperties": false
            }),
            strict: true,
        },
        ..spec()
    }
}

#[test]
fn json_object_valid_emitted_as_single_delta() {
    let mut p = OutputPolicy::new(json_spec(), false);
    assert!(p.needs_buffering());
    assert!(p.on_event(td("{\"a\":")).is_empty());
    assert!(p.on_event(td(" 1}")).is_empty());
    let out = p.on_event(done(FinishReason::Stop));
    assert_eq!(
        out,
        vec![emit(td("{\"a\": 1}")), emit(done(FinishReason::Stop))]
    );
    assert_eq!(text_delta_count(&out), 1);
    assert!(!p.altered());
}

#[test]
fn json_object_extracted_from_fence() {
    let (out, p) = run(
        json_spec(),
        false,
        vec![
            td("Sure!\n```json\n{\"a\": 1}\n```"),
            done(FinishReason::Stop),
        ],
    );
    assert_eq!(text_of(&out), "{\"a\": 1}");
    assert!(p.altered());
    assert_eq!(p.assistant_message().text_content(), "{\"a\": 1}");
}

#[test]
fn json_object_extracted_from_prose() {
    let (out, p) = run(
        json_spec(),
        false,
        vec![
            td("The answer is {\"a\": {\"b\": 2}} as requested."),
            done(FinishReason::Stop),
        ],
    );
    assert_eq!(text_of(&out), "{\"a\": {\"b\": 2}}");
    assert!(p.altered());
}

#[test]
fn json_object_invalid_retries_then_raw() {
    let (out, _) = run(
        json_spec(),
        false,
        vec![td("I cannot do that"), done(FinishReason::Stop)],
    );
    assert!(is_retry(&out));
    let (out, p) = run(
        json_spec(),
        true,
        vec![td("I cannot do that"), done(FinishReason::Stop)],
    );
    assert_eq!(text_of(&out), "I cannot do that");
    assert_eq!(done_of(&out), Some(FinishReason::Stop));
    assert!(!p.altered());
}

#[test]
fn json_schema_valid() {
    let (out, p) = run(
        schema_spec(),
        false,
        vec![td("{\"name\":\"Al\",\"age\":3}"), done(FinishReason::Stop)],
    );
    assert_eq!(text_of(&out), "{\"name\":\"Al\",\"age\":3}");
    assert!(!p.altered());
}

#[test]
fn json_schema_nonconforming_retries() {
    let (out, _) = run(
        schema_spec(),
        false,
        vec![td("{\"name\":\"Al\"}"), done(FinishReason::Stop)],
    );
    assert!(is_retry(&out));
    let (out, p) = run(
        schema_spec(),
        true,
        vec![td("{\"name\":\"Al\"}"), done(FinishReason::Stop)],
    );
    assert_eq!(text_of(&out), "{\"name\":\"Al\"}");
    assert!(!p.altered());
}

#[test]
fn json_schema_extracts_conforming_block() {
    let t = "Example: {\"x\":1}. Final: ```json\n{\"name\":\"Bo\",\"age\":40}\n```";
    let (out, p) = run(schema_spec(), false, vec![td(t), done(FinishReason::Stop)]);
    assert_eq!(text_of(&out), "{\"name\":\"Bo\",\"age\":40}");
    assert!(p.altered());
}

#[test]
fn json_with_max_tokens_truncation_not_retried() {
    let s = PolicySpec {
        max_tokens: Some(2),
        ..json_spec()
    };
    let (out, p) = run(
        s,
        false,
        vec![td("{\"key\": \"value\"}"), done(FinishReason::Stop)],
    );
    assert_eq!(
        out,
        vec![
            emit(td("{\"key\": ")),
            emit(done(FinishReason::Length)),
            PolicyOutput::StopBackend
        ]
    );
    assert!(p.altered());
}

#[test]
fn json_with_max_tokens_fits() {
    let s = PolicySpec {
        max_tokens: Some(100),
        ..json_spec()
    };
    let (out, p) = run(s, false, vec![td("{\"k\": 1}"), done(FinishReason::Stop)]);
    assert_eq!(text_of(&out), "{\"k\": 1}");
    assert!(!p.altered());
}

#[test]
fn json_with_tool_calls_skips_validation() {
    let (out, p) = run(
        json_spec(),
        false,
        vec![
            calls(vec![call("c1", "search")]),
            done(FinishReason::ToolCalls),
        ],
    );
    assert_eq!(calls_of(&out), vec![call("c1", "search")]);
    assert!(!p.altered());
}

#[test]
fn json_reasoning_streams_while_buffering() {
    let mut p = OutputPolicy::new(json_spec(), false);
    assert_eq!(p.on_event(rd("thinking")), vec![emit(rd("thinking"))]);
    assert!(p.on_event(td("{}")).is_empty());
    assert!(p.on_event(limit()) == vec![emit(limit())]);
    assert_eq!(
        p.on_event(done(FinishReason::Stop)),
        vec![emit(td("{}")), emit(done(FinishReason::Stop))]
    );
}

#[test]
fn json_with_stop_sequence() {
    let s = PolicySpec {
        stop: vec!["<END>".into()],
        ..json_spec()
    };
    let (out, p) = run(s, false, vec![td("{\"a\":1}<E"), td("ND> trailing")]);
    assert_eq!(
        out,
        vec![
            emit(td("{\"a\":1}")),
            emit(done(FinishReason::Stop)),
            PolicyOutput::StopBackend
        ]
    );
    assert!(p.altered());
}

#[test]
fn json_and_tool_choice_none_combined() {
    let s = PolicySpec {
        tool_choice: ToolChoice::None,
        ..json_spec()
    };
    let (out, p) = run(
        s,
        true,
        vec![
            td("```json\n{\"a\":1}\n```"),
            calls(vec![call("c1", "search")]),
            done(FinishReason::ToolCalls),
        ],
    );
    assert_eq!(
        out,
        vec![emit(td("{\"a\":1}")), emit(done(FinishReason::Stop))]
    );
    assert!(p.altered());
}

#[test]
fn required_with_max_tokens_hit_retries() {
    let s = PolicySpec {
        tool_choice: ToolChoice::Required,
        max_tokens: Some(1),
        ..spec()
    };
    let (out, _) = run(
        s.clone(),
        false,
        vec![td("long text here"), calls(vec![call("c", "search")])],
    );
    assert!(is_retry(&out));
    let (out, _) = run(s, true, vec![td("long text here")]);
    assert_eq!(
        out,
        vec![
            emit(td("long")),
            emit(done(FinishReason::Length)),
            PolicyOutput::StopBackend
        ]
    );
}

// ------------------------------------------------------------ system_addendum

#[test]
fn addendum_none_for_defaults() {
    assert_eq!(system_addendum(&spec()), None);
    let s = PolicySpec {
        max_tokens: Some(5),
        stop: vec!["x".into()],
        ..spec()
    };
    assert_eq!(system_addendum(&s), None);
}

#[test]
fn addendum_tool_choices() {
    let s = PolicySpec {
        tool_choice: ToolChoice::None,
        ..spec()
    };
    assert_eq!(
        system_addendum(&s).as_deref(),
        Some("Do not call any tools in your next reply; answer in text.")
    );
    let s = PolicySpec {
        tool_choice: ToolChoice::Required,
        ..spec()
    };
    assert_eq!(
        system_addendum(&s).as_deref(),
        Some("You must call at least one tool in your next reply.")
    );
    assert_eq!(
        system_addendum(&func_spec()).as_deref(),
        Some("Your next reply must call the tool `get_weather`.")
    );
}

#[test]
fn addendum_parallel_false() {
    let s = PolicySpec {
        parallel_tool_calls: false,
        ..spec()
    };
    assert_eq!(
        system_addendum(&s).as_deref(),
        Some("Call at most one tool per reply.")
    );
    // No tools offered, or no tool calls allowed: nothing to say about parallelism.
    let s = PolicySpec {
        parallel_tool_calls: false,
        tool_names: vec![],
        ..spec()
    };
    assert_eq!(system_addendum(&s), None);
    let s = PolicySpec {
        parallel_tool_calls: false,
        tool_choice: ToolChoice::None,
        ..spec()
    };
    assert_eq!(
        system_addendum(&s).as_deref(),
        Some("Do not call any tools in your next reply; answer in text.")
    );
}

#[test]
fn addendum_json_formats_and_combination() {
    assert_eq!(
        system_addendum(&json_spec()).as_deref(),
        Some("Respond with a single valid JSON object only, no prose, no code fences.")
    );
    let a = system_addendum(&PolicySpec {
        tool_choice: ToolChoice::Required,
        parallel_tool_calls: false,
        ..schema_spec()
    })
    .unwrap_or_default();
    let lines: Vec<&str> = a.lines().collect();
    assert_eq!(
        lines[0],
        "You must call at least one tool in your next reply."
    );
    assert_eq!(lines[1], "Call at most one tool per reply.");
    assert!(lines[2].starts_with("Respond with a single valid JSON object only"));
    assert!(a.contains("JSON Schema:\n{\n  \"additionalProperties\": false"));
    assert!(a.contains("\"required\""));
}

#[test]
fn spec_from_request() {
    let mut req = ChatRequest::new("bridge/auto", vec![Message::text(Role::User, "hi")]);
    req.tools.push(bridge_types::ToolSpec {
        name: "search".into(),
        description: String::new(),
        parameters: json!({"type":"object"}),
        strict: false,
    });
    req.max_tokens = Some(7);
    req.stop = vec!["a".into()];
    let s = PolicySpec::from_request(&req);
    assert_eq!(s.tool_names, vec!["search".to_string()]);
    assert_eq!(s.max_tokens, Some(7));
    assert_eq!(s.tool_choice, ToolChoice::Auto);
    assert!(s.parallel_tool_calls);
}

#[test]
fn finish_reason_normalized_when_calls_dropped_by_backend_mismatch() {
    // Backend says tool_calls but sent none: caller sees stop.
    let (out, _) = run(spec(), false, vec![td("x"), done(FinishReason::ToolCalls)]);
    assert_eq!(done_of(&out), Some(FinishReason::Stop));
}

#[test]
fn no_tools_default_none_streams_without_buffering() {
    // A request without tools gets tool_choice `none` by default; nothing to enforce.
    let s = PolicySpec {
        tool_choice: ToolChoice::None,
        tool_names: vec![],
        ..spec()
    };
    assert_eq!(system_addendum(&s), None);
    let mut p = OutputPolicy::new(s, false);
    assert!(!p.needs_buffering());
    assert_eq!(p.on_event(td("a")), vec![emit(td("a"))]);
    assert_eq!(p.on_event(td("b")), vec![emit(td("b"))]);
    assert!(!p.altered());
}

// ------------------------------------------------------------ streamed tool calls (17.3)

fn start(index: u32, id: &str, name: &str) -> TurnEvent {
    TurnEvent::ToolCallStart {
        index,
        id: id.into(),
        name: name.into(),
    }
}
fn delta(index: u32, a: &str) -> TurnEvent {
    TurnEvent::ToolCallDelta {
        index,
        arguments: a.into(),
    }
}
fn call_args(id: &str, name: &str, args: &str) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: args.into(),
    }
}

fn is_preview(o: &PolicyOutput) -> bool {
    matches!(
        o,
        PolicyOutput::Emit(TurnEvent::ToolCallStart { .. } | TurnEvent::ToolCallDelta { .. })
    )
}

fn previews(out: &[PolicyOutput]) -> Vec<TurnEvent> {
    out.iter()
        .filter(|o| is_preview(o))
        .filter_map(|o| match o {
            PolicyOutput::Emit(e) => Some(e.clone()),
            _ => None,
        })
        .collect()
}

/// Arguments of preview `index` as an OpenAI SDK accumulates them.
fn accumulated(out: &[PolicyOutput], index: u32) -> String {
    out.iter()
        .filter_map(|o| match o {
            PolicyOutput::Emit(TurnEvent::ToolCallDelta {
                index: i,
                arguments,
            }) if *i == index => Some(arguments.as_str()),
            _ => None,
        })
        .collect()
}

/// Two calls streamed with awkward fragments (whitespace, escapes, unicode, key order).
fn two_streamed_calls() -> (Vec<TurnEvent>, Vec<ToolCall>) {
    let a0 = "{ \"path\" :\"a\\\"b.txt\",\n  \"content\": \"héllo\\nwörld\" }";
    let a1 = "{\"z\":1,\"a\":[ 1 , 2 ]}";
    let events = vec![
        td("Let me write."),
        start(0, "c1", "search"),
        delta(0, &a0[..10]),
        delta(0, &a0[10..30]),
        delta(0, &a0[30..]),
        start(1, "c2", "get_weather"),
        delta(1, &a1[..5]),
        delta(1, &a1[5..]),
    ];
    let committed = vec![
        call_args("c1", "search", a0),
        call_args("c2", "get_weather", a1),
    ];
    (events, committed)
}

fn with_commit(mut events: Vec<TurnEvent>, committed: &[ToolCall]) -> Vec<TurnEvent> {
    events.push(calls(committed.to_vec()));
    events.push(usage());
    events.push(done(FinishReason::ToolCalls));
    events
}

#[test]
fn passthrough_forwards_previews_immediately_and_verbatim() {
    let (events, committed) = two_streamed_calls();
    let mut p = OutputPolicy::new(spec(), false);
    for e in &events {
        let out = p.on_event(e.clone());
        // Every preview comes out at once, unchanged.
        if matches!(
            e,
            TurnEvent::ToolCallStart { .. } | TurnEvent::ToolCallDelta { .. }
        ) {
            assert_eq!(out, vec![emit(e.clone())]);
        }
    }
    let out = p.on_event(calls(committed.clone()));
    assert_eq!(out, vec![emit(calls(committed.clone()))]);
    p.on_event(done(FinishReason::ToolCalls));
    assert!(!p.altered());
    assert_eq!(
        p.assistant_message().tool_calls(),
        committed.iter().collect::<Vec<_>>()
    );
}

#[test]
fn passthrough_byte_identity_of_streamed_arguments() {
    let (events, committed) = two_streamed_calls();
    let (out, p) = run(spec(), false, with_commit(events, &committed));
    assert_eq!(accumulated(&out, 0), committed[0].arguments);
    assert_eq!(accumulated(&out, 1), committed[1].arguments);
    assert_eq!(calls_of(&out), committed);
    assert_eq!(done_of(&out), Some(FinishReason::ToolCalls));
    assert!(!p.altered());
}

#[test]
fn passthrough_preview_counts_as_emitted_content() {
    let mut p = OutputPolicy::new(spec(), false);
    assert!(p.can_retry());
    p.on_event(start(0, "c1", "search"));
    assert!(!p.can_retry());
}

#[test]
fn passthrough_stop_holdback_flushed_before_preview() {
    let s = PolicySpec {
        stop: vec!["STOP".into()],
        ..spec()
    };
    let mut p = OutputPolicy::new(s, false);
    let out1 = p.on_event(td("hello"));
    let out2 = p.on_event(start(0, "c1", "search"));
    assert_eq!(text_of(&out1) + &text_of(&out2), "hello");
    assert_eq!(out2.last(), Some(&emit(start(0, "c1", "search"))));
    // Arguments are not scanned for stop sequences.
    assert_eq!(
        p.on_event(delta(0, "{\"q\":\"STOP\"}")),
        vec![emit(delta(0, "{\"q\":\"STOP\"}"))]
    );
}

#[test]
fn passthrough_max_tokens_does_not_count_arguments() {
    let s = PolicySpec {
        max_tokens: Some(1),
        ..spec()
    };
    let big = "x".repeat(100);
    let (out, p) = run(
        s,
        false,
        with_commit(
            vec![start(0, "c1", "search"), delta(0, &big)],
            &[call_args("c1", "search", &big)],
        ),
    );
    assert_eq!(accumulated(&out, 0), big);
    assert_eq!(done_of(&out), Some(FinishReason::ToolCalls));
    assert!(!p.altered());
}

#[test]
fn passthrough_previews_swallowed_after_text_limit() {
    let s = PolicySpec {
        max_tokens: Some(1),
        ..spec()
    };
    let (out, p) = run(
        s,
        false,
        vec![
            td("12345678"),
            start(0, "c1", "search"),
            delta(0, "{}"),
            calls(vec![call("c1", "search")]),
            done(FinishReason::ToolCalls),
        ],
    );
    assert!(previews(&out).is_empty());
    assert_eq!(done_of(&out), Some(FinishReason::Length));
    assert!(p.altered());
    assert!(p.assistant_message().tool_calls().is_empty());
}

#[test]
fn passthrough_commit_mismatch_marks_altered() {
    let (_, p) = run(
        spec(),
        false,
        vec![
            start(0, "c1", "search"),
            delta(0, "{\"a\": 1}"),
            calls(vec![call_args("c1", "search", "{\"a\":1}")]),
            done(FinishReason::ToolCalls),
        ],
    );
    assert!(p.altered());
}

#[test]
fn passthrough_started_call_missing_from_commit_marks_altered() {
    let (_, p) = run(
        spec(),
        false,
        vec![
            start(0, "c1", "search"),
            delta(0, "{}"),
            done(FinishReason::ToolCalls),
        ],
    );
    assert!(p.altered());
}

#[test]
fn passthrough_delta_without_start_is_dropped() {
    let mut p = OutputPolicy::new(spec(), false);
    assert!(p.on_event(delta(0, "{}")).is_empty());
}

#[test]
fn passthrough_error_after_partial_preview() {
    let mut p = OutputPolicy::new(spec(), false);
    p.on_event(start(0, "c1", "search"));
    p.on_event(delta(0, "{\"q\":"));
    let err = TurnEvent::Error {
        error: BridgeError::new(ErrorClass::Unavailable, "boom"),
    };
    assert_eq!(p.on_event(err.clone()), vec![emit(err)]);
    assert!(p.on_event(delta(0, "1}")).is_empty());
    assert!(p.assistant_message().tool_calls().is_empty());
}

#[test]
fn parallel_false_forwards_only_first_call_previews() {
    let s = PolicySpec {
        parallel_tool_calls: false,
        ..spec()
    };
    let (events, committed) = two_streamed_calls();
    let (out, p) = run(s, false, with_commit(events, &committed));
    for e in previews(&out) {
        match e {
            TurnEvent::ToolCallStart { index, id, .. } => {
                assert_eq!(index, 0);
                assert_eq!(id, "c1");
            }
            TurnEvent::ToolCallDelta { index, .. } => assert_eq!(index, 0),
            _ => unreachable!(),
        }
    }
    assert_eq!(accumulated(&out, 0), committed[0].arguments);
    assert_eq!(calls_of(&out), vec![committed[0].clone()]);
    assert!(p.altered());
    assert_eq!(p.assistant_message().tool_calls(), vec![&committed[0]]);
}

#[test]
fn parallel_false_keeps_streamed_call_even_if_not_first_index() {
    let s = PolicySpec {
        parallel_tool_calls: false,
        ..spec()
    };
    let (out, p) = run(
        s,
        false,
        vec![
            start(1, "c2", "get_weather"),
            delta(1, "{}"),
            calls(vec![call("c1", "search"), call("c2", "get_weather")]),
            done(FinishReason::ToolCalls),
        ],
    );
    assert_eq!(
        previews(&out),
        vec![start(0, "c2", "get_weather"), delta(0, "{}")]
    );
    assert_eq!(calls_of(&out), vec![call("c2", "get_weather")]);
    assert!(p.altered());
}

#[test]
fn parallel_false_single_streamed_call_unaltered() {
    let s = PolicySpec {
        parallel_tool_calls: false,
        ..spec()
    };
    let (out, p) = run(
        s,
        false,
        with_commit(
            vec![start(0, "c1", "search"), delta(0, "{}")],
            &[call("c1", "search")],
        ),
    );
    assert_eq!(previews(&out).len(), 2);
    assert!(!p.altered());
}

fn buffered_specs() -> Vec<PolicySpec> {
    vec![
        PolicySpec {
            tool_choice: ToolChoice::None,
            ..spec()
        },
        PolicySpec {
            tool_choice: ToolChoice::Required,
            ..spec()
        },
        PolicySpec {
            tool_choice: ToolChoice::Required,
            parallel_tool_calls: false,
            ..spec()
        },
        func_spec(),
        json_spec(),
        schema_spec(),
    ]
}

#[test]
fn buffered_modes_never_forward_previews() {
    for s in buffered_specs() {
        for final_attempt in [false, true] {
            let (events, committed) = two_streamed_calls();
            let mut p = OutputPolicy::new(s.clone(), final_attempt);
            assert!(p.needs_buffering());
            let mut out = Vec::new();
            for e in with_commit(events, &committed) {
                let o = p.on_event(e);
                assert!(!o.iter().any(is_preview), "{s:?}");
                out.extend(o);
            }
            out.extend(p.finish());
            assert!(previews(&out).is_empty(), "{s:?}");
        }
    }
}

#[test]
fn buffered_previews_keep_retry_possible() {
    // tool_choice none, model called a tool while streaming its arguments.
    let s = PolicySpec {
        tool_choice: ToolChoice::None,
        ..spec()
    };
    let (events, committed) = two_streamed_calls();
    let mut p = OutputPolicy::new(s, false);
    for e in events {
        assert!(p.on_event(e).iter().all(|o| !is_preview(o)));
    }
    assert!(p.can_retry());
    p.on_event(calls(committed));
    let out = p.on_event(done(FinishReason::ToolCalls));
    assert!(is_retry(&out));
}

#[test]
fn buffered_required_commit_emitted_verbatim() {
    let s = PolicySpec {
        tool_choice: ToolChoice::Required,
        ..spec()
    };
    let (events, committed) = two_streamed_calls();
    let (out, p) = run(s, false, with_commit(events, &committed));
    assert_eq!(calls_of(&out), committed);
    assert_eq!(text_of(&out), "Let me write.");
    assert_eq!(done_of(&out), Some(FinishReason::ToolCalls));
    assert!(!p.altered());
}

#[test]
fn buffered_required_text_only_retries_despite_nothing_streamed() {
    let s = PolicySpec {
        tool_choice: ToolChoice::Required,
        ..spec()
    };
    let (out, _) = run(s, false, vec![td("no tools"), done(FinishReason::Stop)]);
    assert!(is_retry(&out));
}

#[test]
fn buffered_function_with_parallel_false_keeps_named_call() {
    let s = PolicySpec {
        parallel_tool_calls: false,
        ..func_spec()
    };
    let (events, committed) = two_streamed_calls();
    // final attempt: mixed calls are filtered instead of retried.
    let (out, p) = run(s.clone(), true, with_commit(events, &committed));
    assert!(previews(&out).is_empty());
    let ToolChoice::Function { name } = &s.tool_choice else {
        unreachable!()
    };
    let got = calls_of(&out);
    assert_eq!(got.len(), 1);
    assert_eq!(&got[0].name, name);
    assert!(p.altered());
}

#[test]
fn json_mode_tool_call_previews_swallowed_commit_kept() {
    let (out, p) = run(
        json_spec(),
        false,
        with_commit(
            vec![start(0, "c1", "search"), delta(0, "{\"q\":1}")],
            &[call_args("c1", "search", "{\"q\":1}")],
        ),
    );
    assert!(previews(&out).is_empty());
    assert_eq!(calls_of(&out), vec![call_args("c1", "search", "{\"q\":1}")]);
    assert!(!p.altered());
}

// ------------------------------------------------------------ configurable values

#[test]
fn custom_chars_per_token_sets_budget() {
    let s = PolicySpec {
        max_tokens: Some(2),
        chars_per_token: 3,
        ..spec()
    }; // 6 chars
    let (out, p) = run(s, false, vec![td("abcdefgh"), done(FinishReason::Stop)]);
    assert_eq!(text_of(&out), "abcdef");
    assert_eq!(done_of(&out), Some(FinishReason::Length));
    assert!(p.altered());
    // 0 is treated as 1 char per token.
    let s = PolicySpec {
        max_tokens: Some(2),
        chars_per_token: 0,
        ..spec()
    };
    let (out, _) = run(s, false, vec![td("abcdefgh"), done(FinishReason::Stop)]);
    assert_eq!(text_of(&out), "ab");
}

#[test]
fn custom_json_candidate_limit() {
    // The only valid object starts at the third `{`.
    let reply = "{ x { y {\"a\":1}";
    let s = PolicySpec {
        json_max_candidates: 2,
        ..json_spec()
    };
    let (out, _) = run(s, true, vec![td(reply), done(FinishReason::Stop)]);
    assert_ne!(text_of(&out), "{\"a\":1}");
    let s = PolicySpec {
        json_max_candidates: 3,
        ..json_spec()
    };
    let (out, p) = run(s, true, vec![td(reply), done(FinishReason::Stop)]);
    assert_eq!(text_of(&out), "{\"a\":1}");
    assert!(p.altered());
}

#[test]
fn custom_instruction_texts_with_placeholders() {
    let instructions = PolicyInstructions {
        tool_choice_none: Some("NO TOOLS".into()),
        tool_choice_required: Some("SOME TOOL".into()),
        tool_choice_function: Some("CALL {name} NOW ({name})".into()),
        parallel_off: Some("ONE AT A TIME".into()),
        json_object: Some("JSON ONLY".into()),
        json_schema: Some("MATCH {name}: {schema} {other}".into()),
    };
    let base = PolicySpec {
        instructions,
        ..spec()
    };
    let s = PolicySpec {
        tool_choice: ToolChoice::None,
        ..base.clone()
    };
    assert_eq!(system_addendum(&s).as_deref(), Some("NO TOOLS"));
    let s = PolicySpec {
        tool_choice: ToolChoice::Required,
        parallel_tool_calls: false,
        ..base.clone()
    };
    assert_eq!(
        system_addendum(&s).as_deref(),
        Some("SOME TOOL\nONE AT A TIME")
    );
    let s = PolicySpec {
        tool_choice: ToolChoice::Function {
            name: "get_weather".into(),
        },
        ..base.clone()
    };
    assert_eq!(
        system_addendum(&s).as_deref(),
        Some("CALL get_weather NOW (get_weather)")
    );
    let s = PolicySpec {
        response_format: ResponseFormat::JsonObject,
        ..base.clone()
    };
    assert_eq!(system_addendum(&s).as_deref(), Some("JSON ONLY"));
    let s = PolicySpec {
        response_format: ResponseFormat::JsonSchema {
            name: "person".into(),
            // A substituted value containing a placeholder is not substituted again.
            schema: json!({"title": "{name}"}),
            strict: true,
        },
        ..base.clone()
    };
    assert_eq!(
        system_addendum(&s).as_deref(),
        Some("MATCH person: {\n  \"title\": \"{name}\"\n} {other}")
    );
}

#[test]
fn empty_instruction_text_is_dropped_and_partial_config_keeps_defaults() {
    let s = PolicySpec {
        tool_choice: ToolChoice::Required,
        parallel_tool_calls: false,
        instructions: PolicyInstructions {
            tool_choice_required: Some(String::new()),
            ..PolicyInstructions::default()
        },
        ..spec()
    };
    assert_eq!(
        system_addendum(&s).as_deref(),
        Some("Call at most one tool per reply.")
    );
}

#[test]
fn spec_from_request_uses_builtin_defaults() {
    let req = ChatRequest::new("bridge/auto", vec![Message::text(Role::User, "hi")]);
    let s = PolicySpec::from_request(&req);
    assert_eq!(s.chars_per_token, 4);
    assert_eq!(s.json_max_candidates, 32);
    assert_eq!(s.instructions, PolicyInstructions::default());
}
