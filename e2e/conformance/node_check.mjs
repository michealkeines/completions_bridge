// OpenAI Node SDK judge. Prints `RESULT {json}` lines. Loads `openai` from target/sdk/node.
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.join(here, "..", "..", "target", "sdk", "node", "package.json"));
const OpenAIMod = require("openai");
const OpenAI = OpenAIMod.OpenAI || OpenAIMod.default;

const REAL = process.env.BRIDGE_REAL || "";
const FAKE = process.env.BRIDGE_FAKE;
const FOVR = process.env.BRIDGE_FOVR;
const TOKEN = process.env.BRIDGE_TOKEN;
const MODEL = process.env.BRIDGE_MODEL || "bridge/claude-haiku";
const SKIP_REAL = !!process.env.BRIDGE_SKIP_REAL;

function rec(caseId, ok, note = "") {
  console.log("RESULT " + JSON.stringify({ case: caseId, ok: !!ok, note: String(note).slice(0, 300) }));
}

async function check(caseId, f) {
  try {
    await f();
  } catch (e) {
    rec(caseId, false, `${e?.constructor?.name}: ${e?.message}`);
  }
}

const mk = (base, opts = {}) => new OpenAI({ baseURL: base + "/v1", apiKey: TOKEN, maxRetries: 0, timeout: 300_000, ...opts });
const real = REAL ? mk(REAL) : null;
const fake = mk(FAKE);
const fovr = mk(FOVR);

const WEATHER = { type: "function", function: { name: "get_weather", description: "Get the current weather for a city.",
  parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } } };
const DOTTED = { type: "function", function: { name: "weather.get-v2", description: "Get the current weather for a city.",
  parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } } };

async function collect(stream) {
  const out = { text: "", finish: null, usage: null, ids: new Set(), created: new Set(), role: null, tools: {}, reasoning: "", chunks: 0 };
  for await (const ch of stream) {
    out.chunks++;
    out.ids.add(ch.id);
    out.created.add(ch.created);
    if (ch.usage) out.usage = ch.usage;
    for (const c of ch.choices) {
      const d = c.delta || {};
      if (d.role && !out.role) out.role = d.role;
      if (d.content) out.text += d.content;
      if (d.reasoning_content) out.reasoning += d.reasoning_content;
      for (const t of d.tool_calls || []) {
        const s = (out.tools[t.index] ||= { id: null, name: null, arguments: "" });
        if (t.id) s.id = t.id;
        if (t.function?.name) s.name = t.function.name;
        if (t.function?.arguments) s.arguments += t.function.arguments;
      }
      if (c.finish_reason) out.finish = c.finish_reason;
    }
  }
  out.toolList = Object.keys(out.tools).map(Number).sort((a, b) => a - b).map((k) => out.tools[k]);
  return out;
}

async function expectError(caseId, fn, cls, status, extra = () => true) {
  try {
    await fn();
    rec(caseId, false, "no error");
  } catch (e) {
    const ok = e instanceof cls && e.status === status && extra(e);
    rec(caseId, ok, `${e.constructor.name} ${e.status} ${e.code ?? ""} ${e.type ?? ""}`);
  }
}

// ------------------------------------------------------------------ no-claude
const u = (t) => [{ role: "user", content: t }];

await check("EP-2", async () => {
  const list = [];
  for await (const m of fake.models.list()) list.push(m);
  rec("EP-2", list.length > 0 && list.every((m) => m.object === "model" && Number.isInteger(m.created)), `${list.length} models`);
  const m = await fake.models.retrieve("fake/echo");
  rec("EP-3", m.id === "fake/echo", m.id);
});
await expectError("EP-3", () => fake.models.retrieve("no-such-model"), OpenAIMod.NotFoundError, 404, (e) => e.code === "model_not_found");
await expectError("AUTH-3", () => new OpenAI({ baseURL: FAKE + "/v1", apiKey: "wrong", maxRetries: 0 }).chat.completions.create({ model: "fake/echo", messages: u("hi") }),
  OpenAIMod.AuthenticationError, 401, (e) => e.code === "invalid_api_key");
await expectError("ERR-1", () => fake.chat.completions.create({ model: "fake/echo", messages: u("hi"), top_p: 7 }),
  OpenAIMod.BadRequestError, 400, (e) => e.param === "top_p");
await expectError("ERR-4", () => fake.chat.completions.create({ model: "gpt-nope", messages: u("hi") }),
  OpenAIMod.NotFoundError, 404, (e) => e.code === "model_not_found");
await expectError("ERR-6", () => fake.chat.completions.create({ model: "fake/quota", messages: u("hi") }),
  OpenAIMod.RateLimitError, 429, (e) => /^\d+$/.test(e.headers?.get?.("retry-after") ?? e.headers?.["retry-after"] ?? ""));
await expectError("STR-12", async () => { for await (const _ of await fake.chat.completions.create({ model: "fake/quota", messages: u("hi"), stream: true })); },
  OpenAIMod.RateLimitError, 429);
await expectError("ERR-8", () => fake.chat.completions.create({ model: "fake/internal", messages: u("hi") }), OpenAIMod.InternalServerError, 500);
await expectError("ERR-9", () => fake.chat.completions.create({ model: "fake/unavail", messages: u("hi") }), OpenAIMod.InternalServerError, 503);
await expectError("ERR-10", () => fake.chat.completions.create({ model: "fake/timeout", messages: u("hi") }), OpenAIMod.InternalServerError, 504);
await expectError("ERR-2.fake", () => fake.chat.completions.create({ model: "fake/context", messages: u("hi") }),
  OpenAIMod.BadRequestError, 400, (e) => e.code === "context_length_exceeded");

await check("ERR-12", async () => {
  let first = "ok";
  try { await fake.chat.completions.create({ model: "fake/flaky", messages: u("FLAKY-node") }); } catch (e) { first = e.status; }
  const c = await fake.chat.completions.create({ model: "fake/flaky", messages: u("FLAKY-node") });
  rec("ERR-12", first === 503 && c.choices[0].message.content === "echo: FLAKY-node", `first=${first}`);
});

await check("STR-13", async () => {
  let got = "";
  try {
    const s = await fake.chat.completions.create({ model: "fake/echo", messages: u("MIDSTREAM"), stream: true });
    for await (const ch of s) for (const c of ch.choices) got += c.delta?.content || "";
    rec("STR-13", false, `no error, got ${JSON.stringify(got)}`);
  } catch (e) {
    rec("STR-13", got.length > 0 && e instanceof OpenAIMod.APIError, `${e.constructor.name} after ${JSON.stringify(got)}`);
  }
});

await check("STR-11", async () => {
  const r = await collect(await fake.chat.completions.create({ model: "fake/echo", messages: u("REASONME"), stream: true }));
  rec("STR-11", r.text === "final answer" && r.reasoning === "thinking hard", `${r.text} / ${r.reasoning}`);
});

await check("STR-10", async () => {
  const c = mk(FAKE, { timeout: 2500 });
  const t0 = Date.now();
  const r = await collect(await c.chat.completions.create({ model: "fake/echo", messages: u("SLOWSTART"), stream: true }));
  rec("STR-10", r.text === "slow ok", `client timeout 2.5s, first token at 3.5s, took ${(Date.now() - t0) / 1000}s`);
});

await check("STR-7.fake", async () => {
  const tools = [{ type: "function", function: { name: "a.b-c", parameters: { type: "object", properties: { x: { type: "integer" } } } } }, WEATHER];
  const r = await collect(await fake.chat.completions.create({ model: "fake/echo", tools, messages: u("TWOCALLS"), stream: true }));
  rec("STR-7.fake", JSON.stringify(r.toolList.map((t) => t.name)) === '["a.b-c","get_weather"]' && r.finish === "tool_calls", JSON.stringify(r.toolList));
});

await check("LEAK-7", async () => {
  const c = await fovr.chat.completions.create({ model: "fake/auto", messages: [...u("u1"), { role: "assistant", content: "a1" }, ...u("u2")] });
  rec("LEAK-7", c.choices[0].message.content === "history: 3 messages; users: u1 | u2; last user: u2", c.choices[0].message.content);
});

// ------------------------------------------------------------------ real claude
if (!SKIP_REAL) {
  await check("EP-1", async () => {
    const { data: c, response } = await real.chat.completions
      .create({ model: MODEL, messages: u("Reply with exactly the word PONG and nothing else."), some_vendor_field: true })
      .withResponse();
    const ok = c.id.startsWith("chatcmpl-") && c.object === "chat.completion" && c.model === MODEL && c.choices.length === 1 &&
      c.choices[0].message.role === "assistant" && (c.choices[0].message.content || "").includes("PONG") &&
      c.usage.total_tokens === c.usage.prompt_tokens + c.usage.completion_tokens;
    rec("EP-1", ok, c.choices[0].message.content);
    rec("RESP-1", ok, "");
    rec("RESP-7", !!response.headers.get("x-request-id"), response.headers.get("x-request-id"));
  });

  await check("STR-5", async () => {
    const r = await collect(await real.chat.completions.create({ model: MODEL, stream: true, stream_options: { include_usage: true },
      messages: u("Reply with exactly the word PONG and nothing else.") }));
    rec("STR-5", r.text.includes("PONG") && r.finish === "stop" && r.ids.size === 1 && r.created.size === 1 && r.role === "assistant", `${r.text} ${r.finish}`);
    rec("REQ-4", r.usage && r.usage.total_tokens > 0, JSON.stringify(r.usage)?.slice(0, 100));
  });

  await check("TOOL-2", async () => {
    const msgs = u("What's the weather in Paris? Use the tool.");
    const c = await real.chat.completions.create({ model: MODEL, tools: [DOTTED], messages: msgs });
    const tc = c.choices[0].message.tool_calls || [];
    const ok = c.choices[0].finish_reason === "tool_calls" && tc[0]?.function.name === "weather.get-v2" &&
      /paris/i.test(JSON.parse(tc[0].function.arguments).city || "");
    rec("TOOL-2", ok, JSON.stringify(tc.map((t) => t.function)));
    msgs.push(c.choices[0].message);
    for (const t of tc) msgs.push({ role: "tool", tool_call_id: t.id, content: "Paris: 27C, clear sky" });
    const c2 = await real.chat.completions.create({ model: MODEL, tools: [DOTTED], messages: msgs });
    rec("MSG-13", (c2.choices[0].message.content || "").includes("27"), c2.choices[0].message.content?.slice(0, 80));
    const s = await collect(await real.chat.completions.create({ model: MODEL, tools: [DOTTED], messages: msgs.slice(0, 1), stream: true }));
    const ok2 = s.finish === "tool_calls" && s.toolList[0]?.name === "weather.get-v2" && !!s.toolList[0]?.id && /paris/i.test(JSON.parse(s.toolList[0].arguments).city || "");
    rec("STR-6", ok2, JSON.stringify(s.toolList));
    rec("STR-15.tools", ok && ok2, "");
  });

  await check("REQ-12", async () => {
    const c = await real.chat.completions.create({ model: MODEL, response_format: { type: "json_object" },
      messages: u("Give me a JSON object with keys city and country for the capital of Japan.") });
    let ok = false;
    try { ok = typeof JSON.parse(c.choices[0].message.content) === "object"; } catch {}
    rec("REQ-12", ok, c.choices[0].message.content?.slice(0, 100));
  });

  await check("REQ-11", async () => {
    const c = await real.chat.completions.create({ model: MODEL, n: 2, messages: u("Name a random fruit. One word.") });
    rec("REQ-11", c.choices.length === 2 && c.choices[1].index === 1, JSON.stringify(c.choices.map((x) => x.message.content)));
  });

  await check("REQ-8", async () => {
    const r = await collect(await real.chat.completions.create({ model: MODEL, stream: true, max_completion_tokens: 8,
      messages: u("Count from 1 to 200, separated by spaces.") }));
    rec("REQ-8", r.finish === "length" && r.text.length < 80, `${JSON.stringify(r.text)} ${r.finish}`);
    const c = await real.chat.completions.create({ model: MODEL, stop: ["5"], messages: u("Count from 1 to 10, separated by commas. Output only the numbers.") });
    const t = c.choices[0].message.content || "";
    rec("REQ-10", c.choices[0].finish_reason === "stop" && !t.includes("5") && t.includes("4"), JSON.stringify(t));
  });

  await check("CONV-5", async () => {
    const long = [...u("Remember the number 63."), { role: "assistant", content: "OK, 63." }, ...u("Actually, remember 18 instead."),
      { role: "assistant", content: "OK, 18." }, ...u("What number should you remember? Just the number.")];
    const a = (await real.chat.completions.create({ model: MODEL, messages: long })).choices[0].message.content || "";
    const b = (await real.chat.completions.create({ model: MODEL, messages: [...long.slice(0, 2), long[4]] })).choices[0].message.content || "";
    rec("CONV-5", a.includes("18") && b.includes("63") && !b.includes("18"), `${a} / ${b}`);
  });
}
