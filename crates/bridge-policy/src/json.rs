//! JSON validation and extraction for `response_format` json_object / json_schema.

use bridge_types::ResponseFormat;
use serde_json::Value;

/// Outcome of checking buffered content against the requested format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum JsonVerdict {
    /// The text is acceptable as is.
    Valid,
    /// The text is not acceptable, but this substring of it is.
    Extracted(String),
    /// Nothing acceptable found.
    Invalid,
}

/// Checks `text` against `format`. `ResponseFormat::Text` always yields `Valid`.
/// `max_candidates`: how many `{` start positions are tried when looking for an embedded
/// object (`output.json_extract_max_candidates`).
pub(crate) fn check(text: &str, format: &ResponseFormat, max_candidates: usize) -> JsonVerdict {
    let validator = match format {
        ResponseFormat::Text => return JsonVerdict::Valid,
        ResponseFormat::JsonObject => None,
        // An invalid schema degrades to "must be a JSON object".
        ResponseFormat::JsonSchema { schema, .. } => jsonschema::validator_for(schema).ok(),
    };
    let accepts = |candidate: &str| -> bool {
        let Ok(value) = serde_json::from_str::<Value>(candidate) else {
            return false;
        };
        match &validator {
            Some(v) => v.is_valid(&value),
            None => value.is_object(),
        }
    };

    if accepts(text) {
        return JsonVerdict::Valid;
    }
    for candidate in fenced_blocks(text) {
        if accepts(candidate) {
            return JsonVerdict::Extracted(candidate.to_string());
        }
    }
    for candidate in balanced_objects(text, max_candidates) {
        if accepts(candidate) {
            return JsonVerdict::Extracted(candidate.to_string());
        }
    }
    JsonVerdict::Invalid
}

/// Contents of Markdown code fences (```` ```json ... ``` ````), trimmed.
fn fenced_blocks(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find("```") {
        let after_open = &rest[open + 3..];
        // Skip the info string (e.g. "json") up to the end of the line.
        let body_start = match after_open.find('\n') {
            Some(nl) => nl + 1,
            None => break,
        };
        let body = &after_open[body_start..];
        let Some(close) = body.find("```") else {
            break;
        };
        out.push(body[..close].trim());
        rest = &body[close + 3..];
    }
    out
}

/// Balanced `{...}` blocks, one per `{` start position (string-aware), in order.
fn balanced_objects(text: &str, max_candidates: usize) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let starts = bytes
        .iter()
        .enumerate()
        .filter(|(_, b)| **b == b'{')
        .map(|(i, _)| i)
        .take(max_candidates);
    for start in starts {
        if let Some(end) = matching_brace(bytes, start) {
            // Both ends are ASCII braces, hence valid char boundaries.
            if let Some(block) = text.get(start..=end) {
                out.push(block);
            }
        }
    }
    out
}

/// Index of the `}` closing the `{` at `start`, ignoring braces inside strings.
fn matching_brace(bytes: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema_fmt() -> ResponseFormat {
        ResponseFormat::JsonSchema {
            name: "p".into(),
            schema: json!({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}),
            strict: true,
        }
    }

    #[test]
    fn valid_object() {
        assert_eq!(
            check(" {\"a\":1} ", &ResponseFormat::JsonObject, 32),
            JsonVerdict::Valid
        );
    }

    #[test]
    fn array_is_not_object() {
        assert_eq!(
            check("[1,2]", &ResponseFormat::JsonObject, 32),
            JsonVerdict::Invalid
        );
    }

    #[test]
    fn fenced_extraction() {
        let t = "Here you go:\n```json\n{\"a\": 1}\n```\nThanks";
        assert_eq!(
            check(t, &ResponseFormat::JsonObject, 32),
            JsonVerdict::Extracted("{\"a\": 1}".into())
        );
    }

    #[test]
    fn brace_extraction_with_strings() {
        let t = "Result: {\"s\": \"a } b {\", \"n\": {\"x\": 2}} done";
        assert_eq!(
            check(t, &ResponseFormat::JsonObject, 32),
            JsonVerdict::Extracted("{\"s\": \"a } b {\", \"n\": {\"x\": 2}}".into())
        );
    }

    #[test]
    fn schema_picks_matching_candidate() {
        let t = "{\"b\":1} and then {\"a\":3}";
        assert_eq!(
            check(t, &schema_fmt(), 32),
            JsonVerdict::Extracted("{\"a\":3}".into())
        );
        assert_eq!(check("{\"b\":1}", &schema_fmt(), 32), JsonVerdict::Invalid);
    }

    #[test]
    fn invalid_schema_falls_back_to_object() {
        let f = ResponseFormat::JsonSchema {
            name: "bad".into(),
            schema: json!({"type": 12}),
            strict: false,
        };
        assert_eq!(check("{}", &f, 32), JsonVerdict::Valid);
        assert_eq!(check("nope", &f, 32), JsonVerdict::Invalid);
    }

    #[test]
    fn unbalanced_is_invalid() {
        assert_eq!(
            check("{\"a\": 1", &ResponseFormat::JsonObject, 32),
            JsonVerdict::Invalid
        );
    }
}
