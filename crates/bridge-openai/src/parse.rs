//! Wire request (JSON) → validated IR `ChatRequest` (CONFORMANCE sections 3 and 4).
//!
//! Parsing works on `serde_json::Value` rather than derived structs so that every type
//! error can name the offending field (`param`) the way OpenAI does, and so unknown
//! fields are ignored at every level.

use std::collections::HashSet;

use bridge_types::{
    ChatRequest, Image, ImageSource, Message, ModelInfo, Part, ReasoningEffort, ResponseFormat,
    Role, ToolCall, ToolChoice, ToolResult, ToolSpec,
};
use serde_json::{Map, Value};

use crate::error::ApiError;

type Res<T> = Result<T, ApiError>;

/// Limits that come from configuration.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub max_n: u32,
}

// ------------------------------------------------------------------ helpers

fn describe(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(n) if n.is_i64() || n.is_u64() => "an integer",
        Value::Number(_) => "a decimal",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

fn invalid_type(param: &str, expected: &str, got: &Value) -> ApiError {
    ApiError::invalid(
        format!(
            "Invalid type for '{param}': expected {expected}, but got {} instead.",
            describe(got)
        ),
        Some(param),
        Some("invalid_type"),
    )
}

fn missing(param: &str) -> ApiError {
    ApiError::invalid(
        format!("Missing required parameter: '{param}'."),
        Some(param),
        Some("missing_required_parameter"),
    )
}

fn invalid_value(param: &str, got: &str, supported: &[&str]) -> ApiError {
    let list = match supported {
        [] => String::new(),
        [one] => format!("'{one}'"),
        [init @ .., last] => format!(
            "{}, and '{last}'",
            init.iter().map(|s| format!("'{s}'")).collect::<Vec<_>>().join(", ")
        ),
    };
    ApiError::invalid(
        format!("Invalid value: '{got}'. Supported values are: {list}."),
        Some(param),
        Some("invalid_value"),
    )
}

/// Field lookup that treats JSON `null` as absent.
fn get<'a>(obj: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    match obj.get(key) {
        None | Some(Value::Null) => None,
        Some(v) => Some(v),
    }
}

fn opt_bool(obj: &Map<String, Value>, key: &str, param: &str) -> Res<Option<bool>> {
    match get(obj, key) {
        None => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(v) => Err(invalid_type(param, "a boolean", v)),
    }
}

fn opt_str<'a>(obj: &'a Map<String, Value>, key: &str, param: &str) -> Res<Option<&'a str>> {
    match get(obj, key) {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.as_str())),
        Some(v) => Err(invalid_type(param, "a string", v)),
    }
}

fn opt_object<'a>(
    obj: &'a Map<String, Value>,
    key: &str,
    param: &str,
) -> Res<Option<&'a Map<String, Value>>> {
    match get(obj, key) {
        None => Ok(None),
        Some(Value::Object(m)) => Ok(Some(m)),
        Some(v) => Err(invalid_type(param, "an object", v)),
    }
}

fn fmt_num(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

fn opt_number_in(
    obj: &Map<String, Value>,
    key: &str,
    min: f64,
    max: f64,
) -> Res<Option<f64>> {
    let Some(v) = get(obj, key) else { return Ok(None) };
    let Some(x) = v.as_f64() else {
        return Err(invalid_type(key, "a decimal", v));
    };
    if x < min {
        return Err(ApiError::invalid(
            format!(
                "Invalid '{key}': decimal below minimum value. Expected a value >= {}, but got {} instead.",
                fmt_num(min),
                fmt_num(x)
            ),
            Some(key),
            Some("decimal_below_min_value"),
        ));
    }
    if x > max {
        return Err(ApiError::invalid(
            format!(
                "Invalid '{key}': decimal above maximum value. Expected a value <= {}, but got {} instead.",
                fmt_num(max),
                fmt_num(x)
            ),
            Some(key),
            Some("decimal_above_max_value"),
        ));
    }
    Ok(Some(x))
}

fn opt_int_in(obj: &Map<String, Value>, key: &str, min: i64, max: Option<i64>) -> Res<Option<i64>> {
    let Some(v) = get(obj, key) else { return Ok(None) };
    let x = match v {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i
            } else if n.as_u64().is_some() {
                i64::MAX
            } else {
                // Accept integral floats like 2.0; reject 1.5.
                match n.as_f64() {
                    Some(f) if f.fract() == 0.0 && f.abs() < 9.0e15 => f as i64,
                    _ => return Err(invalid_type(key, "an integer", v)),
                }
            }
        }
        _ => return Err(invalid_type(key, "an integer", v)),
    };
    if x < min {
        return Err(ApiError::invalid(
            format!(
                "Invalid '{key}': integer below minimum value. Expected a value >= {min}, but got {x} instead."
            ),
            Some(key),
            Some("integer_below_min_value"),
        ));
    }
    if let Some(max) = max {
        if x > max {
            return Err(ApiError::invalid(
                format!(
                    "Invalid '{key}': integer above maximum value. Expected a value <= {max}, but got {x} instead."
                ),
                Some(key),
                Some("integer_above_max_value"),
            ));
        }
    }
    Ok(Some(x))
}

// ------------------------------------------------------------------ top level

/// Parse and validate a request body. `models` are the ids the service offers.
pub fn parse_request(body: &[u8], models: &[ModelInfo], limits: Limits) -> Res<ChatRequest> {
    let value: Value = serde_json::from_slice(body).map_err(|_| {
        ApiError::invalid(
            "We could not parse the JSON body of your request. The API expects a JSON payload, \
             but what was sent was not valid JSON.",
            None,
            None,
        )
    })?;
    let Value::Object(obj) = &value else {
        return Err(ApiError::invalid(
            format!(
                "Invalid type for request body: expected an object, but got {} instead.",
                describe(&value)
            ),
            None,
            Some("invalid_type"),
        ));
    };
    parse_object(obj, models, limits)
}

fn parse_object(obj: &Map<String, Value>, models: &[ModelInfo], limits: Limits) -> Res<ChatRequest> {
    // model
    let model = match get(obj, "model") {
        None => {
            return Err(ApiError::invalid(
                "you must provide a model parameter",
                Some("model"),
                Some("missing_required_parameter"),
            ))
        }
        Some(Value::String(s)) => s.clone(),
        Some(v) => return Err(invalid_type("model", "a string", v)),
    };

    // messages
    let raw_messages = match get(obj, "messages") {
        None => return Err(missing("messages")),
        Some(Value::Array(a)) => a,
        Some(v) => return Err(invalid_type("messages", "an array", v)),
    };
    if raw_messages.is_empty() {
        return Err(ApiError::invalid(
            "Invalid 'messages': empty array. Expected an array with minimum length 1, but got an empty array instead.",
            Some("messages"),
            Some("empty_array"),
        ));
    }

    // Sampling parameters: validated, then ignored (CONFORMANCE 13).
    opt_number_in(obj, "temperature", 0.0, 2.0)?;
    opt_number_in(obj, "top_p", 0.0, 1.0)?;
    opt_number_in(obj, "presence_penalty", -2.0, 2.0)?;
    opt_number_in(obj, "frequency_penalty", -2.0, 2.0)?;

    let stream = opt_bool(obj, "stream", "stream")?.unwrap_or(false);
    let include_usage = match opt_object(obj, "stream_options", "stream_options")? {
        Some(so) => opt_bool(so, "include_usage", "stream_options.include_usage")?.unwrap_or(false),
        None => false,
    };

    let n = opt_int_in(obj, "n", 1, Some(i64::from(limits.max_n.max(1))))?.unwrap_or(1);
    let n = u32::try_from(n).unwrap_or(1);

    let max_completion = opt_int_in(obj, "max_completion_tokens", 1, None)?;
    let max_tokens = opt_int_in(obj, "max_tokens", 1, None)?;
    let max_tokens = max_completion
        .or(max_tokens)
        .map(|x| u32::try_from(x).unwrap_or(u32::MAX));

    let stop = parse_stop(obj)?;
    let response_format = parse_response_format(obj)?;
    let reasoning_effort = parse_reasoning_effort(obj)?;
    let parallel_tool_calls = opt_bool(obj, "parallel_tool_calls", "parallel_tool_calls")?.unwrap_or(true);

    // Audio is not supported (CONFORMANCE 13).
    if let Some(m) = get(obj, "modalities") {
        let Value::Array(items) = m else {
            return Err(invalid_type("modalities", "an array", m));
        };
        for (i, it) in items.iter().enumerate() {
            match it.as_str() {
                Some("text") => {}
                Some("audio") => {
                    return Err(ApiError::invalid(
                        "This model does not support audio output. Remove 'audio' from 'modalities'.",
                        Some("modalities"),
                        Some("unsupported_value"),
                    ))
                }
                Some(other) => {
                    return Err(invalid_value(&format!("modalities[{i}]"), other, &["text", "audio"]))
                }
                None => return Err(invalid_type(&format!("modalities[{i}]"), "a string", it)),
            }
        }
    }
    if get(obj, "audio").is_some() {
        return Err(ApiError::invalid(
            "This model does not support audio output.",
            Some("audio"),
            Some("unsupported_parameter"),
        ));
    }

    // Tools (and legacy functions).
    let mut tools = parse_tools(obj)?;
    let legacy_functions = if tools.is_empty() { parse_functions(obj)? } else { Vec::new() };
    let used_legacy = !legacy_functions.is_empty();
    if used_legacy {
        tools = legacy_functions;
    }
    let tool_choice = parse_tool_choice(obj, &tools, used_legacy)?;

    let messages = parse_messages(raw_messages)?;
    check_tool_structure(&messages)?;

    // Model check last, after the body is known to be well formed.
    if !models.iter().any(|m| m.id == model) {
        return Err(ApiError::model_not_found(&model));
    }

    Ok(ChatRequest {
        model,
        messages,
        tools,
        tool_choice,
        parallel_tool_calls,
        stream,
        include_usage: stream && include_usage,
        max_tokens,
        stop,
        n,
        response_format,
        reasoning_effort,
    })
}

fn parse_stop(obj: &Map<String, Value>) -> Res<Vec<String>> {
    match get(obj, "stop") {
        None => Ok(Vec::new()),
        Some(Value::String(s)) => Ok(if s.is_empty() { Vec::new() } else { vec![s.clone()] }),
        Some(Value::Array(items)) => {
            if items.len() > 4 {
                return Err(ApiError::invalid(
                    format!(
                        "Invalid 'stop': array too long. Expected an array with maximum length 4, but got an array with length {} instead.",
                        items.len()
                    ),
                    Some("stop"),
                    Some("array_above_max_length"),
                ));
            }
            let mut out = Vec::new();
            for (i, it) in items.iter().enumerate() {
                match it {
                    Value::String(s) => {
                        if !s.is_empty() {
                            out.push(s.clone());
                        }
                    }
                    other => return Err(invalid_type(&format!("stop[{i}]"), "a string", other)),
                }
            }
            Ok(out)
        }
        Some(v) => Err(invalid_type("stop", "a string or array of strings", v)),
    }
}

fn parse_response_format(obj: &Map<String, Value>) -> Res<ResponseFormat> {
    let Some(rf) = opt_object(obj, "response_format", "response_format")? else {
        return Ok(ResponseFormat::Text);
    };
    let ty = opt_str(rf, "type", "response_format.type")?.ok_or_else(|| missing("response_format.type"))?;
    match ty {
        "text" => Ok(ResponseFormat::Text),
        "json_object" => Ok(ResponseFormat::JsonObject),
        "json_schema" => {
            let js = opt_object(rf, "json_schema", "response_format.json_schema")?
                .ok_or_else(|| missing("response_format.json_schema"))?;
            let name = opt_str(js, "name", "response_format.json_schema.name")?
                .ok_or_else(|| missing("response_format.json_schema.name"))?;
            let schema = match get(js, "schema") {
                None => serde_json::json!({}),
                Some(v @ Value::Object(_)) => v.clone(),
                Some(v) => return Err(invalid_type("response_format.json_schema.schema", "an object", v)),
            };
            let strict = opt_bool(js, "strict", "response_format.json_schema.strict")?.unwrap_or(false);
            Ok(ResponseFormat::JsonSchema { name: name.to_string(), schema, strict })
        }
        other => Err(invalid_value("response_format.type", other, &["json_object", "json_schema", "text"])),
    }
}

fn parse_reasoning_effort(obj: &Map<String, Value>) -> Res<Option<ReasoningEffort>> {
    match opt_str(obj, "reasoning_effort", "reasoning_effort")? {
        None => Ok(None),
        Some("minimal") => Ok(Some(ReasoningEffort::Minimal)),
        Some("low") => Ok(Some(ReasoningEffort::Low)),
        Some("medium") => Ok(Some(ReasoningEffort::Medium)),
        Some("high") => Ok(Some(ReasoningEffort::High)),
        // Newer OpenAI values, mapped to the nearest supported level.
        Some("none") => Ok(Some(ReasoningEffort::Minimal)),
        Some("xhigh") => Ok(Some(ReasoningEffort::High)),
        Some(other) => Err(invalid_value("reasoning_effort", other, &["minimal", "low", "medium", "high"])),
    }
}

fn parse_function_def(f: &Map<String, Value>, base: &str) -> Res<ToolSpec> {
    let name = opt_str(f, "name", &format!("{base}.name"))?.ok_or_else(|| missing(&format!("{base}.name")))?;
    if name.is_empty() {
        return Err(ApiError::invalid(
            format!("Invalid '{base}.name': empty string. Expected a string with minimum length 1, but got an empty string instead."),
            Some(&format!("{base}.name")),
            Some("empty_string"),
        ));
    }
    let description = opt_str(f, "description", &format!("{base}.description"))?.unwrap_or("").to_string();
    let parameters = match get(f, "parameters") {
        None => serde_json::json!({"type": "object", "properties": {}}),
        Some(v @ Value::Object(_)) => v.clone(),
        Some(v) => return Err(invalid_type(&format!("{base}.parameters"), "an object", v)),
    };
    let strict = opt_bool(f, "strict", &format!("{base}.strict"))?.unwrap_or(false);
    Ok(ToolSpec { name: name.to_string(), description, parameters, strict })
}

fn parse_tools(obj: &Map<String, Value>) -> Res<Vec<ToolSpec>> {
    let Some(v) = get(obj, "tools") else { return Ok(Vec::new()) };
    let Value::Array(items) = v else {
        return Err(invalid_type("tools", "an array", v));
    };
    let mut out = Vec::with_capacity(items.len());
    let mut seen = HashSet::new();
    for (i, it) in items.iter().enumerate() {
        let p = format!("tools[{i}]");
        let Value::Object(t) = it else {
            return Err(invalid_type(&p, "an object", it));
        };
        let ty = opt_str(t, "type", &format!("{p}.type"))?.unwrap_or("function");
        if ty != "function" {
            return Err(invalid_value(&format!("{p}.type"), ty, &["function"]));
        }
        let f = opt_object(t, "function", &format!("{p}.function"))?
            .ok_or_else(|| missing(&format!("{p}.function")))?;
        let spec = parse_function_def(f, &format!("{p}.function"))?;
        if !seen.insert(spec.name.clone()) {
            return Err(ApiError::invalid(
                format!("Invalid 'tools': duplicate function name '{}'.", spec.name),
                Some("tools"),
                Some("invalid_value"),
            ));
        }
        out.push(spec);
    }
    Ok(out)
}

/// Legacy `functions` → tools.
fn parse_functions(obj: &Map<String, Value>) -> Res<Vec<ToolSpec>> {
    let Some(v) = get(obj, "functions") else { return Ok(Vec::new()) };
    let Value::Array(items) = v else {
        return Err(invalid_type("functions", "an array", v));
    };
    let mut out = Vec::with_capacity(items.len());
    for (i, it) in items.iter().enumerate() {
        let p = format!("functions[{i}]");
        let Value::Object(f) = it else {
            return Err(invalid_type(&p, "an object", it));
        };
        out.push(parse_function_def(f, &p)?);
    }
    Ok(out)
}

fn parse_tool_choice(obj: &Map<String, Value>, tools: &[ToolSpec], legacy: bool) -> Res<ToolChoice> {
    let (key, raw) = match (get(obj, "tool_choice"), get(obj, "function_call")) {
        (Some(v), _) => ("tool_choice", Some(v)),
        (None, Some(v)) if legacy => ("function_call", Some(v)),
        _ => ("tool_choice", None),
    };
    let default = if tools.is_empty() { ToolChoice::None } else { ToolChoice::Auto };
    let Some(raw) = raw else { return Ok(default) };

    let only_with_tools = || {
        ApiError::invalid(
            format!("Invalid value for '{key}': '{key}' is only allowed when 'tools' are specified."),
            Some(key),
            None,
        )
    };

    let choice = match raw {
        Value::String(s) => match s.as_str() {
            "none" => ToolChoice::None,
            "auto" => default.clone(),
            "required" if key == "tool_choice" => ToolChoice::Required,
            other => {
                let supported: &[&str] =
                    if key == "tool_choice" { &["none", "auto", "required"] } else { &["none", "auto"] };
                return Err(invalid_value(key, other, supported));
            }
        },
        Value::Object(o) => {
            let name = if key == "function_call" {
                opt_str(o, "name", "function_call.name")?.ok_or_else(|| missing("function_call.name"))?
            } else {
                let ty = opt_str(o, "type", "tool_choice.type")?.ok_or_else(|| missing("tool_choice.type"))?;
                if ty != "function" {
                    return Err(invalid_value("tool_choice.type", ty, &["function"]));
                }
                let f = opt_object(o, "function", "tool_choice.function")?
                    .ok_or_else(|| missing("tool_choice.function"))?;
                opt_str(f, "name", "tool_choice.function.name")?
                    .ok_or_else(|| missing("tool_choice.function.name"))?
            };
            ToolChoice::Function { name: name.to_string() }
        }
        other => return Err(invalid_type(key, "a string or object", other)),
    };

    match &choice {
        ToolChoice::Required if tools.is_empty() => Err(only_with_tools()),
        ToolChoice::Function { .. } if tools.is_empty() => Err(only_with_tools()),
        ToolChoice::Function { name } if !tools.iter().any(|t| &t.name == name) => Err(ApiError::invalid(
            format!("Invalid value for '{key}': no function named '{name}' was specified in the 'tools' parameter."),
            Some(key),
            Some("invalid_value"),
        )),
        _ => Ok(choice),
    }
}

// ------------------------------------------------------------------ messages

/// Message roles defined by the OpenAI Chat Completions spec; fixed on purpose (protocol fact,
/// not a tuning knob).
const ROLES: &[&str] = &["system", "assistant", "user", "function", "tool", "developer"];

fn parse_messages(raw: &[Value]) -> Res<Vec<Message>> {
    let mut out = Vec::with_capacity(raw.len());
    // Legacy `function_call` ids, for translating `function` role messages.
    let mut legacy_pending: Vec<(String, String)> = Vec::new();
    for (i, v) in raw.iter().enumerate() {
        let p = format!("messages[{i}]");
        let Value::Object(m) = v else {
            return Err(invalid_type(&p, "an object", v));
        };
        let role = match get(m, "role") {
            None => return Err(missing(&format!("{p}.role"))),
            Some(Value::String(s)) => s.as_str(),
            Some(other) => return Err(invalid_type(&format!("{p}.role"), "a string", other)),
        };
        let msg = match role {
            "system" | "developer" => Message { role: Role::System, parts: text_only_content(m, &p, false)? },
            "user" => Message { role: Role::User, parts: user_content(m, &p)? },
            "assistant" => assistant_message(m, &p, i, &mut legacy_pending)?,
            "tool" => {
                let call_id = match get(m, "tool_call_id") {
                    None => return Err(missing(&format!("{p}.tool_call_id"))),
                    Some(Value::String(s)) => s.clone(),
                    Some(other) => return Err(invalid_type(&format!("{p}.tool_call_id"), "a string", other)),
                };
                let content = text_only_content(m, &p, false)?
                    .into_iter()
                    .filter_map(|part| match part {
                        Part::Text { text } => Some(text),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                Message { role: Role::Tool, parts: vec![Part::ToolResult(ToolResult { call_id, content })] }
            }
            "function" => {
                let name = opt_str(m, "name", &format!("{p}.name"))?.ok_or_else(|| missing(&format!("{p}.name")))?;
                let content = match get(m, "content") {
                    None => String::new(),
                    Some(Value::String(s)) => s.clone(),
                    Some(other) => return Err(invalid_type(&format!("{p}.content"), "a string", other)),
                };
                let pos = legacy_pending.iter().position(|(_, n)| n == name);
                let call_id = match pos {
                    Some(pos) => legacy_pending.remove(pos).0,
                    None => {
                        return Err(ApiError::invalid(
                            "Invalid parameter: messages with role 'function' must be a response to a preceeding message with 'function_call'.",
                            Some(&format!("messages.[{i}].role")),
                            None,
                        ))
                    }
                };
                Message { role: Role::Tool, parts: vec![Part::ToolResult(ToolResult { call_id, content })] }
            }
            other => return Err(invalid_value(&format!("{p}.role"), other, ROLES)),
        };
        out.push(msg);
    }
    Ok(out)
}

/// Content that may only hold text (system, developer, tool). Array parts are kept as
/// separate text parts in order. `allow_refusal` admits `refusal` parts (assistant).
fn text_only_content(m: &Map<String, Value>, p: &str, allow_refusal: bool) -> Res<Vec<Part>> {
    let cp = format!("{p}.content");
    match get(m, "content") {
        None => Err(missing(&cp)),
        Some(Value::String(s)) => Ok(vec![Part::Text { text: s.clone() }]),
        Some(Value::Array(items)) => {
            let mut out = Vec::new();
            for (j, it) in items.iter().enumerate() {
                let pp = format!("{cp}[{j}]");
                let Value::Object(part) = it else {
                    return Err(invalid_type(&pp, "an object", it));
                };
                let ty = opt_str(part, "type", &format!("{pp}.type"))?.ok_or_else(|| missing(&format!("{pp}.type")))?;
                match ty {
                    "text" => {
                        let t = opt_str(part, "text", &format!("{pp}.text"))?
                            .ok_or_else(|| missing(&format!("{pp}.text")))?;
                        out.push(Part::Text { text: t.to_string() });
                    }
                    "refusal" if allow_refusal => {}
                    other => {
                        let supported: &[&str] = if allow_refusal { &["text", "refusal"] } else { &["text"] };
                        return Err(invalid_value(&format!("{pp}.type"), other, supported));
                    }
                }
            }
            Ok(out)
        }
        Some(other) => Err(invalid_type(&cp, "one of a string or array of objects", other)),
    }
}

fn user_content(m: &Map<String, Value>, p: &str) -> Res<Vec<Part>> {
    let cp = format!("{p}.content");
    match get(m, "content") {
        None => Err(missing(&cp)),
        Some(Value::String(s)) => Ok(vec![Part::Text { text: s.clone() }]),
        Some(Value::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for (j, it) in items.iter().enumerate() {
                let pp = format!("{cp}[{j}]");
                let Value::Object(part) = it else {
                    return Err(invalid_type(&pp, "an object", it));
                };
                let ty = opt_str(part, "type", &format!("{pp}.type"))?.ok_or_else(|| missing(&format!("{pp}.type")))?;
                match ty {
                    "text" => {
                        let t = opt_str(part, "text", &format!("{pp}.text"))?
                            .ok_or_else(|| missing(&format!("{pp}.text")))?;
                        out.push(Part::Text { text: t.to_string() });
                    }
                    "image_url" => {
                        let ip = format!("{pp}.image_url");
                        let url = match get(part, "image_url") {
                            None => return Err(missing(&ip)),
                            Some(Value::String(s)) => s.as_str(),
                            Some(Value::Object(o)) => {
                                opt_str(o, "url", &format!("{ip}.url"))?.ok_or_else(|| missing(&format!("{ip}.url")))?
                            }
                            Some(other) => return Err(invalid_type(&ip, "an object", other)),
                        };
                        out.push(Part::Image(parse_image_url(url, &format!("{ip}.url"))?));
                    }
                    "input_audio" => {
                        return Err(ApiError::invalid(
                            "Audio input is not supported by this model.",
                            Some(&format!("{pp}.type")),
                            Some("unsupported_value"),
                        ))
                    }
                    "file" => {
                        return Err(ApiError::invalid(
                            "File content parts are not supported by this model.",
                            Some(&format!("{pp}.type")),
                            Some("unsupported_value"),
                        ))
                    }
                    other => {
                        return Err(invalid_value(
                            &format!("{pp}.type"),
                            other,
                            &["text", "image_url", "input_audio", "file"],
                        ))
                    }
                }
            }
            Ok(out)
        }
        Some(other) => Err(invalid_type(&cp, "one of a string or array of objects", other)),
    }
}

fn parse_image_url(url: &str, param: &str) -> Res<Image> {
    if let Some(rest) = url.strip_prefix("data:") {
        let bad = |why: &str| {
            ApiError::invalid(
                format!(
                    "Invalid '{param}'. Expected a base64-encoded data URL with an image MIME type \
                     (e.g. 'data:image/png;base64,aW1nIGJ5dGVzIGhlcmU='), but {why}."
                ),
                Some(param),
                Some("invalid_image_url"),
            )
        };
        let Some((meta, data)) = rest.split_once(',') else {
            return Err(bad("got a malformed data URL"));
        };
        let mut meta_parts = meta.split(';');
        let media_type = meta_parts.next().unwrap_or("").trim().to_ascii_lowercase();
        let is_b64 = meta_parts.any(|s| s.trim().eq_ignore_ascii_case("base64"));
        if !is_b64 {
            return Err(bad("the data URL is not base64-encoded"));
        }
        if !media_type.starts_with("image/") {
            return Err(bad(&format!("got unsupported MIME type '{media_type}'")));
        }
        let data: String = data.chars().filter(|c| !c.is_ascii_whitespace()).collect();
        let valid = !data.is_empty()
            && data.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'=' | b'-' | b'_'));
        if !valid {
            return Err(bad("the base64 data is invalid"));
        }
        return Ok(Image { media_type, source: ImageSource::Base64 { data } });
    }
    let lower = url.to_ascii_lowercase();
    if lower.starts_with("https://") || lower.starts_with("http://") {
        return Ok(Image { media_type: String::new(), source: ImageSource::Url { url: url.to_string() } });
    }
    Err(ApiError::invalid(
        format!("Invalid '{param}'. Expected a valid URL or a base64-encoded data URL."),
        Some(param),
        Some("invalid_image_url"),
    ))
}

fn assistant_message(
    m: &Map<String, Value>,
    p: &str,
    i: usize,
    legacy_pending: &mut Vec<(String, String)>,
) -> Res<Message> {
    let mut parts = Vec::new();

    // De-facto extension: some clients echo reasoning back.
    if let Some(Value::String(r)) = get(m, "reasoning_content") {
        if !r.is_empty() {
            parts.push(Part::Reasoning { text: r.clone() });
        }
    }

    let has_content = get(m, "content").is_some();
    if has_content {
        parts.extend(text_only_content(m, p, true)?);
    }

    let mut calls = Vec::new();
    if let Some(v) = get(m, "tool_calls") {
        let Value::Array(items) = v else {
            return Err(invalid_type(&format!("{p}.tool_calls"), "an array", v));
        };
        for (k, it) in items.iter().enumerate() {
            let tp = format!("{p}.tool_calls[{k}]");
            let Value::Object(tc) = it else {
                return Err(invalid_type(&tp, "an object", it));
            };
            let id = opt_str(tc, "id", &format!("{tp}.id"))?.ok_or_else(|| missing(&format!("{tp}.id")))?;
            let ty = opt_str(tc, "type", &format!("{tp}.type"))?.unwrap_or("function");
            if ty != "function" {
                return Err(invalid_value(&format!("{tp}.type"), ty, &["function"]));
            }
            let f = opt_object(tc, "function", &format!("{tp}.function"))?
                .ok_or_else(|| missing(&format!("{tp}.function")))?;
            calls.push(parse_call(f, id, &format!("{tp}.function"))?);
        }
    }

    // Legacy single function_call.
    if let Some(v) = get(m, "function_call") {
        let Value::Object(f) = v else {
            return Err(invalid_type(&format!("{p}.function_call"), "an object", v));
        };
        let id = format!("call_legacy_{i}");
        let call = parse_call(f, &id, &format!("{p}.function_call"))?;
        legacy_pending.push((id, call.name.clone()));
        calls.push(call);
    }

    if !has_content && calls.is_empty() {
        return Err(ApiError::invalid(
            format!("Invalid value for '{p}.content': expected a string, got null."),
            Some(&format!("{p}.content")),
            None,
        ));
    }

    parts.extend(calls.into_iter().map(Part::ToolCall));
    Ok(Message { role: Role::Assistant, parts })
}

fn parse_call(f: &Map<String, Value>, id: &str, base: &str) -> Res<ToolCall> {
    let name = opt_str(f, "name", &format!("{base}.name"))?.ok_or_else(|| missing(&format!("{base}.name")))?;
    let arguments = match get(f, "arguments") {
        None => "{}".to_string(),
        Some(Value::String(s)) => s.clone(),
        // Lenient: some clients send an object; keep it as its JSON text.
        Some(v @ Value::Object(_)) => v.to_string(),
        Some(v) => return Err(invalid_type(&format!("{base}.arguments"), "a string", v)),
    };
    Ok(ToolCall { id: id.to_string(), name: name.to_string(), arguments })
}

/// CONV-11 / CONV-12.
fn check_tool_structure(messages: &[Message]) -> Res<()> {
    // (assistant index, call ids, answered flags)
    let mut pending: Option<(usize, Vec<String>, Vec<bool>)> = None;

    let unanswered_err = |idx: usize, ids: &[String], answered: &[bool]| -> Option<ApiError> {
        let missing_ids: Vec<&str> = ids
            .iter()
            .zip(answered)
            .filter(|(_, a)| !**a)
            .map(|(id, _)| id.as_str())
            .collect();
        if missing_ids.is_empty() {
            return None;
        }
        Some(ApiError::invalid(
            format!(
                "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'. The following tool_call_ids did not have response messages: {}",
                missing_ids.join(", ")
            ),
            Some(&format!("messages.[{idx}].role")),
            None,
        ))
    };

    for (i, m) in messages.iter().enumerate() {
        if m.role == Role::Tool {
            let results = m.tool_results();
            let ok = match &mut pending {
                Some((_, ids, answered)) => results.iter().all(|r| {
                    match ids.iter().position(|id| *id == r.call_id) {
                        Some(pos) => {
                            answered[pos] = true;
                            true
                        }
                        None => false,
                    }
                }),
                None => false,
            };
            if !ok {
                return Err(ApiError::invalid(
                    "Invalid parameter: messages with role 'tool' must be a response to a preceeding message with 'tool_calls'.",
                    Some(&format!("messages.[{i}].role")),
                    None,
                ));
            }
            continue;
        }
        if let Some((idx, ids, answered)) = pending.take() {
            if let Some(e) = unanswered_err(idx, &ids, &answered) {
                return Err(e);
            }
        }
        if m.role == Role::Assistant {
            let ids: Vec<String> = m.tool_calls().iter().map(|c| c.id.clone()).collect();
            if !ids.is_empty() {
                let answered = vec![false; ids.len()];
                pending = Some((i, ids, answered));
            }
        }
    }
    if let Some((idx, ids, answered)) = pending {
        if let Some(e) = unanswered_err(idx, &ids, &answered) {
            return Err(e);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn models() -> Vec<ModelInfo> {
        vec![ModelInfo { id: "bridge/auto".into(), created: 0, owned_by: "bridge".into(), context_limit: None }]
    }

    fn parse(v: Value) -> Res<ChatRequest> {
        parse_request(v.to_string().as_bytes(), &models(), Limits { max_n: 4 })
    }

    #[test]
    fn minimal() {
        let r = parse(json!({"model":"bridge/auto","messages":[{"role":"user","content":"hi"}]})).unwrap();
        assert_eq!(r.messages, vec![Message::text(Role::User, "hi")]);
        assert_eq!(r.tool_choice, ToolChoice::None);
        assert_eq!(r.n, 1);
    }

    #[test]
    fn max_completion_wins() {
        let r = parse(json!({"model":"bridge/auto","messages":[{"role":"user","content":"hi"}],
            "max_tokens": 10, "max_completion_tokens": 20})).unwrap();
        assert_eq!(r.max_tokens, Some(20));
    }

    #[test]
    fn ranges() {
        let base = json!({"model":"bridge/auto","messages":[{"role":"user","content":"hi"}]});
        for (k, v, param) in [
            ("temperature", json!(2.5), "temperature"),
            ("top_p", json!(-0.1), "top_p"),
            ("presence_penalty", json!(3), "presence_penalty"),
            ("frequency_penalty", json!(-3), "frequency_penalty"),
            ("n", json!(0), "n"),
            ("n", json!(5), "n"),
            ("max_tokens", json!(0), "max_tokens"),
            ("stop", json!(["a","b","c","d","e"]), "stop"),
            ("temperature", json!("hot"), "temperature"),
        ] {
            let mut b = base.clone();
            b[k] = v;
            let e = parse(b).unwrap_err();
            assert_eq!(e.status, 400);
            assert_eq!(e.param.as_deref(), Some(param), "{k}");
        }
    }

    #[test]
    fn legacy_functions() {
        let r = parse(json!({"model":"bridge/auto",
            "functions":[{"name":"get","parameters":{"type":"object"}}],
            "function_call":{"name":"get"},
            "messages":[
                {"role":"user","content":"hi"},
                {"role":"assistant","content":null,"function_call":{"name":"get","arguments":"{}"}},
                {"role":"function","name":"get","content":"42"}
            ]})).unwrap();
        assert_eq!(r.tools.len(), 1);
        assert_eq!(r.tool_choice, ToolChoice::Function { name: "get".into() });
        assert_eq!(r.messages[2].tool_results()[0].call_id, r.messages[1].tool_calls()[0].id);
    }

    #[test]
    fn tool_results_out_of_order_ok() {
        parse(json!({"model":"bridge/auto","messages":[
            {"role":"user","content":"hi"},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"a","type":"function","function":{"name":"x","arguments":"{}"}},
                {"id":"b","type":"function","function":{"name":"x","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"b","content":"1"},
            {"role":"tool","tool_call_id":"a","content":"2"},
            {"role":"user","content":"next"}
        ]})).unwrap();
    }
}
