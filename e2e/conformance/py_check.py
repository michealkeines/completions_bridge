"""OpenAI Python SDK judge. Run with the SDK venv (target/sdk/py). Prints `RESULT {json}` lines."""
import base64
import json
import os
import struct
import sys
import time
import traceback
import zlib

import openai
from openai import OpenAI

REAL = os.environ.get("BRIDGE_REAL", "")
FAKE = os.environ["BRIDGE_FAKE"]
FOVR = os.environ["BRIDGE_FOVR"]
TOKEN = os.environ["BRIDGE_TOKEN"]
MODEL = os.environ.get("BRIDGE_MODEL", "bridge/claude-haiku")
SKIP_REAL = bool(os.environ.get("BRIDGE_SKIP_REAL"))
CLIENT = "python"


def rec(case, ok, note=""):
    print("RESULT " + json.dumps({"case": case, "ok": bool(ok), "note": str(note)[:300]}), flush=True)


def check(case):
    def deco(f):
        def w():
            try:
                f()
            except Exception as e:  # noqa: BLE001
                rec(case, False, f"{type(e).__name__}: {e}"[:300])
                traceback.print_exc(limit=2)
        w.__name__ = f.__name__
        return w
    return deco


real = OpenAI(base_url=REAL + "/v1", api_key=TOKEN, max_retries=0, timeout=300) if REAL else None
real_slash = OpenAI(base_url=REAL + "/v1/", api_key=TOKEN, max_retries=0, timeout=60) if REAL else None
fake = OpenAI(base_url=FAKE + "/v1", api_key=TOKEN, max_retries=0, timeout=60)
fovr = OpenAI(base_url=FOVR + "/v1", api_key=TOKEN, max_retries=0, timeout=60)

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
                                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
DOTTED = {"type": "function", "function": {"name": "weather.get-v2", "description": "Get the current weather for a city.",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
TIME = {"type": "function", "function": {"name": "get_time", "description": "Get the current time.", "parameters": {"type": "object", "properties": {}}}}


def shape(c, n=1, model=MODEL):
    p = []
    if not c.id.startswith("chatcmpl-"):
        p.append(f"id {c.id}")
    if c.object != "chat.completion":
        p.append("object")
    if c.model != model:
        p.append(f"model {c.model}")
    if len(c.choices) != n:
        p.append(f"{len(c.choices)} choices")
    for i, ch in enumerate(c.choices):
        if ch.index != i or ch.message.role != "assistant" or ch.finish_reason not in ("stop", "length", "tool_calls", "content_filter"):
            p.append(f"choice {i}")
    if not c.usage or c.usage.total_tokens != c.usage.prompt_tokens + c.usage.completion_tokens:
        p.append("usage")
    return p


def collect_stream(stream):
    text, finish, usage, ids, created, role = "", None, None, set(), set(), None
    tcs = {}
    reasoning = ""
    for ch in stream:
        ids.add(ch.id)
        created.add(ch.created)
        if ch.usage:
            usage = ch.usage
        for c in ch.choices:
            d = c.delta
            if d.role and role is None:
                role = d.role
            if d.content:
                text += d.content
            r = getattr(d, "reasoning_content", None) or (d.model_extra or {}).get("reasoning_content")
            if r:
                reasoning += r
            for t in d.tool_calls or []:
                slot = tcs.setdefault(t.index, {"id": None, "name": None, "arguments": ""})
                if t.id:
                    slot["id"] = t.id
                if t.function and t.function.name:
                    slot["name"] = t.function.name
                if t.function and t.function.arguments:
                    slot["arguments"] += t.function.arguments
            if c.finish_reason:
                finish = c.finish_reason
    return {"text": text, "finish": finish, "usage": usage, "ids": ids, "created": created, "role": role,
            "tool_calls": [tcs[k] for k in sorted(tcs)], "reasoning": reasoning}


# ------------------------------------------------------------------ no-claude checks

@check("EP-2")
def models_list():
    ms = list(fake.models.list())
    rec("EP-2", ms and all(m.object == "model" and isinstance(m.created, int) and m.owned_by for m in ms), f"{len(ms)} models")


@check("EP-3")
def models_retrieve():
    m = fake.models.retrieve("fake/echo")
    ok = m.id == "fake/echo" and m.object == "model"
    try:
        fake.models.retrieve("no-such-model")
        ok, note = False, "no error"
    except openai.NotFoundError as e:
        ok = ok and e.status_code == 404 and e.code == "model_not_found"
        note = f"{e.status_code} {e.code}"
    rec("EP-3", ok, note)


@check("EP-4")
def base_url_slash():
    c = OpenAI(base_url=FAKE + "/v1/", api_key=TOKEN, max_retries=0)
    r = c.chat.completions.create(model="fake/echo", messages=[{"role": "user", "content": "hi"}])
    rec("EP-4", r.choices[0].message.content == "echo: hi", "base_url with trailing slash")


@check("AUTH-3")
def auth():
    for case, key in (("AUTH-3", "wrong"),):
        c = OpenAI(base_url=FAKE + "/v1", api_key=key, max_retries=0)
        try:
            c.chat.completions.create(model="fake/echo", messages=[{"role": "user", "content": "hi"}])
            rec(case, False, "no error")
        except openai.AuthenticationError as e:
            rec(case, e.status_code == 401 and e.code == "invalid_api_key" and e.type == "invalid_request_error", f"{e.status_code} {e.code}")
            rec("ERR-3", e.status_code == 401 and e.code == "invalid_api_key", "AuthenticationError")


@check("ERR-1")
def bad_request():
    try:
        fake.chat.completions.create(model="fake/echo", messages=[{"role": "user", "content": "hi"}], temperature=5)
        rec("ERR-1", False, "no error")
    except openai.BadRequestError as e:
        rec("ERR-1", e.status_code == 400 and e.param == "temperature" and e.type == "invalid_request_error", f"{e.status_code} param={e.param}")
    try:
        fake.chat.completions.create(model="fake/echo", messages=[])
        rec("REQ-2", False, "no error")
    except openai.BadRequestError as e:
        rec("REQ-2", e.status_code == 400 and e.param == "messages", f"param={e.param}")


@check("ERR-4")
def unknown_model():
    try:
        fake.chat.completions.create(model="gpt-nope", messages=[{"role": "user", "content": "hi"}])
        rec("ERR-4", False, "no error")
    except openai.NotFoundError as e:
        rec("ERR-4", e.status_code == 404 and e.code == "model_not_found", f"{e.code}")


@check("ERR-6")
def quota():
    for stream in (False, True):
        try:
            r = fake.chat.completions.create(model="fake/quota", messages=[{"role": "user", "content": "hi"}], stream=stream)
            if stream:
                list(r)
            rec("ERR-6", False, "no error")
        except openai.RateLimitError as e:
            ra = e.response.headers.get("retry-after")
            rec("ERR-6", e.status_code == 429 and ra and ra.isdigit() and e.code in ("insufficient_quota", "rate_limit_exceeded"),
                f"stream={stream} {e.code} retry-after={ra}")
            if stream:
                rec("STR-12", True, "RateLimitError raised before any chunk")


@check("ERR-8")
def server_errors():
    for model, case, status in (("fake/internal", "ERR-8", 500), ("fake/unavail", "ERR-9", 503), ("fake/timeout", "ERR-10", 504)):
        try:
            fake.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}])
            rec(case, False, "no error")
        except openai.APIStatusError as e:
            leaky = any(w in str(e.message).lower() for w in ("fake", "/secret", "claude"))
            rec(case, e.status_code == status and isinstance(e, openai.InternalServerError) and not leaky,
                f"{type(e).__name__} {e.status_code} {e.type}/{e.code}")
            if leaky:
                rec("ERR-11", False, e.message)
    try:
        fake.chat.completions.create(model="fake/context", messages=[{"role": "user", "content": "hi"}])
        rec("ERR-2.fake", False, "no error")
    except openai.BadRequestError as e:
        rec("ERR-2.fake", e.code == "context_length_exceeded", e.code)


@check("ERR-7")
def busy():
    from concurrent.futures import ThreadPoolExecutor

    def one(i):
        try:
            fake.chat.completions.create(model="fake/busy", messages=[{"role": "user", "content": f"x{i}"}])
            return "ok"
        except openai.RateLimitError as e:
            return f"429:{e.code}:{e.response.headers.get('retry-after')}"
    with ThreadPoolExecutor(2) as ex:
        out = sorted(ex.map(one, range(2)))
    rl = [o for o in out if o != "ok"]
    rec("ERR-7", out.count("ok") == 1 and len(rl) == 1 and rl[0].startswith("429:rate_limit_exceeded:") and not rl[0].endswith("None"), str(out))


@check("ERR-12")
def flaky():
    try:
        fake.chat.completions.create(model="fake/flaky", messages=[{"role": "user", "content": "FLAKY-python"}])
        first = "ok"
    except openai.APIStatusError as e:
        first = e.status_code
    # the SDK's own retry machinery: should now succeed
    c = OpenAI(base_url=FAKE + "/v1", api_key=TOKEN, max_retries=2)
    r = c.chat.completions.create(model="fake/flaky", messages=[{"role": "user", "content": "FLAKY-python"}])
    rec("ERR-12", first == 503 and r.choices[0].message.content == "echo: FLAKY-python", f"first={first}")


@check("STR-13")
def midstream():
    s = fake.chat.completions.create(model="fake/echo", messages=[{"role": "user", "content": "MIDSTREAM"}], stream=True)
    got = ""
    try:
        for ch in s:
            for c in ch.choices:
                got += c.delta.content or ""
        rec("STR-13", False, f"no exception; got {got!r}")
    except openai.APIError as e:
        rec("STR-13", bool(got), f"{type(e).__name__} after {got!r}: {e.message[:80]}")


@check("STR-11")
def reasoning():
    r = collect_stream(fake.chat.completions.create(model="fake/echo", messages=[{"role": "user", "content": "REASONME"}], stream=True))
    rec("STR-11", r["text"] == "final answer" and r["reasoning"] == "thinking hard", f"{r['text']!r} / {r['reasoning']!r}")


@check("STR-10")
def slow_start():
    t0 = time.time()
    c = OpenAI(base_url=FAKE + "/v1", api_key=TOKEN, max_retries=0, timeout=openai.Timeout(10, read=2.0))
    r = collect_stream(c.chat.completions.create(model="fake/echo", messages=[{"role": "user", "content": "SLOWSTART"}], stream=True))
    rec("STR-10", r["text"] == "slow ok", f"read timeout 2s, first token after 3.5s, took {time.time() - t0:.1f}s")


@check("STR-7")
def two_calls_fake():
    tools = [{"type": "function", "function": {"name": "a.b-c", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}}}}, WEATHER]
    r = fake.chat.completions.create(model="fake/echo", tools=tools, messages=[{"role": "user", "content": "TWOCALLS"}])
    names = [t.function.name for t in r.choices[0].message.tool_calls]
    s = collect_stream(fake.chat.completions.create(model="fake/echo", tools=tools, messages=[{"role": "user", "content": "TWOCALLS"}], stream=True))
    rec("STR-7.fake", names == ["a.b-c", "get_weather"] and [t["name"] for t in s["tool_calls"]] == names and s["finish"] == "tool_calls",
        f"{names} / {[t['name'] for t in s['tool_calls']]}")


@check("LEAK-7")
def failover():
    r = fovr.chat.completions.create(model="fake/auto", messages=[{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
                                                                  {"role": "user", "content": "u2"}])
    rec("LEAK-7", r.choices[0].message.content == "history: 3 messages; users: u1 | u2; last user: u2", r.choices[0].message.content)


# ------------------------------------------------------------------ real claude

@check("EP-1")
def basic():
    raw = real.chat.completions.with_raw_response.create(model=MODEL, messages=[{"role": "user", "content": "Reply with exactly the word PONG and nothing else."}],
                                                         temperature=0.2, extra_body={"vendor_field": {"x": 1}})
    c = raw.parse()
    p = shape(c)
    rec("EP-1", not p and "PONG" in (c.choices[0].message.content or ""), "; ".join(p))
    rec("RESP-1", not p, "; ".join(p))
    rec("RESP-6", raw.headers.get("content-type", "").startswith("application/json"), raw.headers.get("content-type"))
    rec("RESP-7", bool(raw.headers.get("x-request-id")), raw.headers.get("x-request-id"))
    rec("REQ-1", c.model == MODEL, c.model)


@check("STR-1")
def stream_usage():
    raw = real.chat.completions.with_raw_response.create(model=MODEL, stream=True, stream_options={"include_usage": True},
                                                         messages=[{"role": "user", "content": "Reply with exactly the word PONG and nothing else."}])
    rec("STR-1", raw.headers.get("content-type", "").startswith("text/event-stream"), raw.headers.get("content-type"))
    r = collect_stream(raw.parse())
    ok = "PONG" in r["text"] and r["finish"] == "stop" and r["usage"] and len(r["ids"]) == 1 and len(r["created"]) == 1 and r["role"] == "assistant"
    rec("STR-5", ok, f"{r['text']!r} finish={r['finish']}")
    rec("REQ-4", r["usage"] is not None and r["usage"].total_tokens > 0, str(r["usage"])[:100])
    rec("STR-9", r["usage"] is not None, "")
    rec("STR-15.text", ok, "")


@check("TOOL-2")
def tools_round_trip():
    msgs = [{"role": "user", "content": "What's the weather in Paris? Use the tool."}]
    c = real.chat.completions.create(model=MODEL, tools=[DOTTED], messages=msgs)
    tc = c.choices[0].message.tool_calls or []
    ok = c.choices[0].finish_reason == "tool_calls" and tc and tc[0].function.name == "weather.get-v2" and \
        "paris" in json.loads(tc[0].function.arguments).get("city", "").lower()
    rec("TOOL-2", ok, f"{[(t.function.name, t.function.arguments) for t in tc]}")
    rec("TOOL-5", ok, "dotted name exact")
    msgs.append(c.choices[0].message.model_dump(exclude_none=True))
    for t in tc:
        msgs.append({"role": "tool", "tool_call_id": t.id, "content": "Paris: 23C, light rain"})
    c2 = real.chat.completions.create(model=MODEL, tools=[DOTTED], messages=msgs)
    t = c2.choices[0].message.content or ""
    rec("MSG-13", "23" in t and c2.choices[0].finish_reason == "stop", t[:80])
    # stream with tools (STR-6) and compare with non-stream (STR-15)
    s = collect_stream(real.chat.completions.create(model=MODEL, tools=[DOTTED], messages=msgs[:1], stream=True))
    ok2 = s["finish"] == "tool_calls" and s["tool_calls"] and s["tool_calls"][0]["name"] == "weather.get-v2" and s["tool_calls"][0]["id"] and \
        "paris" in json.loads(s["tool_calls"][0]["arguments"]).get("city", "").lower()
    rec("STR-6", ok2, str(s["tool_calls"])[:120])
    rec("STR-15.tools", ok and ok2, "")


@check("TOOL-13")
def tool_choice():
    c = real.chat.completions.create(model=MODEL, tools=[WEATHER, TIME], tool_choice="required", messages=[{"role": "user", "content": "Hi there!"}])
    rec("TOOL-13", bool(c.choices[0].message.tool_calls), c.choices[0].finish_reason)
    c = real.chat.completions.create(model=MODEL, tools=[WEATHER, TIME], tool_choice={"type": "function", "function": {"name": "get_time"}},
                                     messages=[{"role": "user", "content": "What's the weather in Oslo?"}])
    tc = c.choices[0].message.tool_calls or []
    rec("TOOL-14", tc and all(t.function.name == "get_time" for t in tc), str([t.function.name for t in tc]))
    c = real.chat.completions.create(model=MODEL, tools=[WEATHER], parallel_tool_calls=False,
                                     messages=[{"role": "user", "content": "Get the weather for Paris, Tokyo and Lima. Make all calls at once."}])
    rec("REQ-7", len(c.choices[0].message.tool_calls or []) == 1, f"{len(c.choices[0].message.tool_calls or [])} calls")


@check("REQ-11")
def n2():
    c = real.chat.completions.create(model=MODEL, n=2, messages=[{"role": "user", "content": "Name a random animal. One word."}])
    rec("REQ-11", len(c.choices) == 2 and [x.index for x in c.choices] == [0, 1] and all(x.message.content for x in c.choices),
        str([x.message.content for x in c.choices]))


@check("REQ-13")
def structured():
    try:
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int
            tags: list[str]
        parse = getattr(real.chat.completions, "parse", None) or real.beta.chat.completions.parse
        c = parse(model=MODEL, response_format=Person, messages=[{"role": "user", "content": "Invent a person: Ada, 36 years old, tags math and poetry."}])
        p = c.choices[0].message.parsed
        rec("REQ-13", p is not None and p.age == 36 and p.name.lower().startswith("ada"), str(p))
    except ImportError:
        rec("REQ-13", False, "pydantic missing")


@check("REQ-8")
def limits():
    c = real.chat.completions.create(model=MODEL, max_tokens=8, messages=[{"role": "user", "content": "Count from 1 to 200, separated by spaces."}])
    rec("REQ-8", c.choices[0].finish_reason == "length" and len(c.choices[0].message.content or "") < 80, f"{c.choices[0].message.content!r}")
    c = real.chat.completions.create(model=MODEL, stop=["5"], messages=[{"role": "user", "content": "Count from 1 to 10, separated by commas. Output only the numbers."}])
    t = c.choices[0].message.content or ""
    rec("REQ-10", c.choices[0].finish_reason == "stop" and "5" not in t and "4" in t, repr(t))


@check("MSG-2")
def developer_and_image():
    c = real.chat.completions.create(model=MODEL, messages=[{"role": "developer", "content": "Whatever the user says, reply with only the single word MANGO."},
                                                            {"role": "user", "content": "How are you?"}])
    rec("MSG-2", "mango" in (c.choices[0].message.content or "").lower(), c.choices[0].message.content)
    raw = b"".join(b"\x00" + bytes((20, 40, 220)) * 32 for _ in range(32))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    c = real.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": [
        {"type": "text", "text": "What single colour fills this image? One word."}, {"type": "image_url", "image_url": {"url": url}}]}])
    rec("MSG-7", "blue" in (c.choices[0].message.content or "").lower(), c.choices[0].message.content)


@check("CONV-4")
def edited():
    def hist(name):
        return [{"role": "user", "content": f"Hi, my name is {name}."}, {"role": "assistant", "content": "Nice to meet you!"},
                {"role": "user", "content": "I like coffee."}, {"role": "assistant", "content": "Coffee is great."},
                {"role": "user", "content": "What is my name? Reply with just the name."}]
    a = real.chat.completions.create(model=MODEL, messages=hist("Bartholomew")).choices[0].message.content or ""
    b = real.chat.completions.create(model=MODEL, messages=hist("Philippa")).choices[0].message.content or ""
    rec("CONV-4", "bartholomew" in a.lower() and "philippa" in b.lower() and "bartholomew" not in b.lower(), f"{a!r} / {b!r}")


@check("LEAK-1")
def leak_tools():
    tools = [{"type": "function", "function": {"name": "fs.read-file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
    c = real.chat.completions.create(model=MODEL, tools=tools, messages=[{"role": "user", "content": "What tools do you have? List the exact names of all tools you can call, comma-separated, nothing else."}])
    t = c.choices[0].message.content or ""
    bad = [w for w in ("mcp__", "Bash", "Glob", "Grep", "WebFetch", "TodoWrite") if w in t]
    rec("LEAK-1", "fs.read-file" in t and not bad, f"{t[:100]!r} bad={bad}")


def main():
    for f in (models_list, models_retrieve, base_url_slash, auth, bad_request, unknown_model, quota, server_errors, busy,
              flaky, midstream, reasoning, slow_start, two_calls_fake, failover):
        f()
    if not SKIP_REAL:
        for f in (basic, stream_usage, tools_round_trip, tool_choice, n2, structured, limits, developer_and_image, edited, leak_tools):
            f()
    print(f"openai python sdk {openai.__version__}")


if __name__ == "__main__":
    main()
    sys.exit(0)
