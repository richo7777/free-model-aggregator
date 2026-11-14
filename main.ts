// Free-model aggregator for make.fun: free-model-aggregator-01.
//
// One Node 22 server that speaks both client wires (OpenAI Chat Completions
// and Anthropic Messages, JSON and server-sent events) and fans each request
// out to the best free model across the ten providers the trial's brief
// names. It registers its port and catalog with the env, serves the twenty
// requests the env's client sends, and exits 0 when the env says the
// workload is done.
//
// Routing, in one line: every request goes to the highest-rated free model
// that meets its requirements; a refusal (402, 429, 5xx, timeout) fails over
// to the next candidate, and a provider that refuses twice in a row is
// demoted for the rest of the trial. Same-wire calls pass the request and
// the answer through untouched; cross-wire calls go through a canonical
// request/answer form, so the provider's echo of the system prompt, user
// text, tools, max_tokens, stop sequences, image and JSON mode always
// matches what the client sent.
//
// Determinism: nothing here reads the wall clock or unseeded randomness.
// Choices depend only on the brief, the manifest order and the providers'
// answers, so a replay leaves the same state.
//
// Run it: `node main.ts` inside a trial (MAKEFUN_PROMPT_FILE and
// MAKEFUN_ENV_URL set). `node main.ts --list` prints the catalog document
// and exits 0; outside a trial it prints {"providers":[]}.

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { readFileSync } from "node:fs";

// ---------------------------------------------------------------------------
// Small JSON helpers
// ---------------------------------------------------------------------------

type Json = Record<string, unknown>;

const isObj = (v: unknown): v is Json => v !== null && typeof v === "object" && !Array.isArray(v);
const isArr = (v: unknown): v is unknown[] => Array.isArray(v);
const asStr = (v: unknown): string | null => (typeof v === "string" ? v : null);
const asNum = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;
const asBool = (v: unknown): boolean | null => (typeof v === "boolean" ? v : null);
const asStrList = (v: unknown): string[] => {
  if (typeof v === "string") return [v];
  if (!isArr(v)) return [];
  return v.filter((x): x is string => typeof x === "string");
};

/** Text of a content value: a plain string, or its text blocks joined with \n. */
function contentText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!isArr(content)) return "";
  const parts: string[] = [];
  for (const b of content) {
    if (isObj(b) && b.type === "text" && typeof b.text === "string") parts.push(b.text);
  }
  return parts.join("\n");
}

// ---------------------------------------------------------------------------
// Brief: the three JSON documents inside MAKEFUN_PROMPT_FILE
// ---------------------------------------------------------------------------

type Wire = "openai" | "anthropic" | "gemini";
type Auth = "bearer" | "x-api-key" | "x-goog-api-key" | "none";

interface ProviderDir {
  id: string;
  name: string;
  wire: Wire;
  auth: Auth;
  keyEnv: string;
  key: string;
  path: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating: number;
  context: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requires {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

interface ManifestReq {
  id: string;
  api: "openai" | "anthropic";
  stream: boolean;
  weight: number;
  requires: Requires;
}

interface Brief {
  providers: ProviderDir[];
  ratings: RatingRow[];
  best: Map<string, number>;
  manifest: ManifestReq[];
}

/** Find the JSON document with the given schema marker and parse it. */
function extractDoc(text: string, schema: string): Json {
  const marker = `"schema":"${schema}"`;
  const at = text.indexOf(marker);
  if (at < 0) throw new Error(`brief holds no ${schema} document`);
  const start = text.lastIndexOf("{", at);
  if (start < 0) throw new Error(`brief holds no object for ${schema}`);
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i]!;
    if (inStr) {
      if (esc) esc = false;
      else if (ch === "\\") esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') inStr = true;
    else if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) return JSON.parse(text.slice(start, i + 1)) as Json;
    }
  }
  throw new Error(`brief's ${schema} document never closes`);
}

function readBrief(path: string): Brief {
  const text = readFileSync(path, "utf8");
  const dirDoc = extractDoc(text, "free-model-aggregator-01/providers/v1");
  const ratingsDoc = extractDoc(text, "free-model-aggregator-01/ratings/v1");
  const workloadDoc = extractDoc(text, "free-model-aggregator-01/workload/v1");

  const providers: ProviderDir[] = [];
  for (const raw of isArr(dirDoc.providers) ? dirDoc.providers : []) {
    if (!isObj(raw)) continue;
    const id = asStr(raw.id);
    const wire = asStr(raw.wire);
    const auth = asStr(raw.auth);
    if (id === null || (wire !== "openai" && wire !== "anthropic" && wire !== "gemini")) continue;
    providers.push({
      id,
      name: asStr(raw.name) ?? id,
      wire,
      auth: auth === "bearer" || auth === "x-api-key" || auth === "x-goog-api-key" ? auth : "none",
      keyEnv: asStr(raw.key_env) ?? "",
      key: asStr(raw.key) ?? "",
      path: asStr(raw.path) ?? `/p/${id}`,
    });
  }

  const ratings: RatingRow[] = [];
  for (const raw of isArr(ratingsDoc.ratings) ? ratingsDoc.ratings : []) {
    if (!isObj(raw)) continue;
    const model = asStr(raw.model);
    const provider = asStr(raw.provider);
    const rating = asNum(raw.rating_milli);
    if (model === null || provider === null || rating === null) continue;
    ratings.push({
      model,
      provider,
      rating,
      context: asNum(raw.context_length),
      tools: asBool(raw.tools) ?? false,
      vision: asBool(raw.vision) ?? false,
      json: asBool(raw.json) ?? false,
      free: asBool(raw.free) ?? false,
    });
  }

  const best = new Map<string, number>();
  for (const raw of isArr(ratingsDoc.best) ? ratingsDoc.best : []) {
    if (!isObj(raw)) continue;
    const id = asStr(raw.id);
    const rating = asNum(raw.rating_milli);
    if (id !== null && rating !== null) best.set(id, rating);
  }

  const manifest: ManifestReq[] = [];
  for (const raw of isArr(workloadDoc.manifest) ? workloadDoc.manifest : []) {
    if (!isObj(raw)) continue;
    const id = asStr(raw.id);
    const api = asStr(raw.api);
    if (id === null || (api !== "openai" && api !== "anthropic")) continue;
    const req: Requires = {};
    if (isObj(raw.requires)) {
      if (asBool(raw.requires.tools) === true) req.tools = true;
      if (asBool(raw.requires.vision) === true) req.vision = true;
      if (asBool(raw.requires.json) === true) req.json = true;
      const mc = asNum(raw.requires.min_context);
      if (mc !== null) req.minContext = mc;
    }
    manifest.push({
      id,
      api,
      stream: asBool(raw.stream) ?? false,
      weight: asNum(raw.weight) ?? 1,
      requires: req,
    });
  }
  return { providers, ratings, best, manifest };
}

// ---------------------------------------------------------------------------
// Catalog: probe every provider's model list, build the one document
// ---------------------------------------------------------------------------

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface Catalog {
  providers: CatalogProvider[];
}

function authHeaders(auth: Auth, key: string): Record<string, string> {
  if (auth === "bearer") return { authorization: `Bearer ${key}` };
  if (auth === "x-api-key") return { "x-api-key": key, "anthropic-version": "2023-06-01" };
  if (auth === "x-goog-api-key") return { "x-goog-api-key": key };
  return {};
}

function catalogPath(p: ProviderDir): string {
  return p.wire === "gemini" ? `${p.path}/v1beta/models` : `${p.path}/v1/models`;
}

/** Parse one provider's model list into catalog rows. */
function modelsFromList(wire: Wire, body: unknown): CatalogModel[] {
  if (!isObj(body)) return [];
  const rows: CatalogModel[] = [];
  if (wire === "gemini") {
    const list = isArr(body.models) ? body.models : [];
    for (const m of list) {
      if (!isObj(m)) continue;
      const name = asStr(m.name);
      if (name === null) continue;
      const id = name.startsWith("models/") ? name.slice("models/".length) : name;
      const ctx = asNum(m.context_length) ?? asNum(m.inputTokenLimit);
      rows.push({ id, free: asBool(m.free) ?? false, context_length: ctx });
    }
    return rows;
  }
  const list = isArr(body.data) ? body.data : [];
  for (const m of list) {
    if (!isObj(m)) continue;
    const id = asStr(m.id);
    if (id === null) continue;
    rows.push({ id, free: asBool(m.free) ?? false, context_length: asNum(m.context_length) });
  }
  return rows;
}

async function fetchJson(url: string, headers: Record<string, string>, timeoutMs: number): Promise<unknown | null> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { headers, signal: ctrl.signal });
    if (!res.ok) return null;
    return (await res.json()) as unknown;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/** Probe one provider; null when its list cannot be read. */
async function probeOne(envUrl: string, p: ProviderDir): Promise<CatalogModel[] | null> {
  const body = await fetchJson(`${envUrl}${catalogPath(p)}`, authHeaders(p.auth, p.key), 5000);
  if (body === null) return null;
  const rows = modelsFromList(p.wire, body);
  return rows.length > 0 ? rows : null;
}

/** Build the catalog: the probe's rows, with the brief's ratings table filling
 *  any gap (a listing that hides a model's free flag, or a probe that fails). */
async function buildCatalog(envUrl: string, brief: Brief): Promise<Catalog> {
  const out: CatalogProvider[] = [];
  for (const p of brief.providers) {
    const probed = await probeOne(envUrl, p);
    const rows = brief.ratings.filter((r) => r.provider === p.id);
    const models: CatalogModel[] = [];
    if (probed !== null) {
      for (const m of probed) {
        const row = rows.find((r) => r.model === m.id);
        // The brief's table is the sealed truth for free; the probe's own
        // flag agrees with it. Trust the table when the probe is silent.
        const resolvedFree = row !== undefined ? row.free : m.free;
        models.push({ id: m.id, free: resolvedFree, context_length: m.context_length ?? row?.context ?? null });
      }
      // A free model the listing omitted still belongs in the catalog.
      for (const r of rows) {
        if (r.free && !models.some((m) => m.id === r.model)) {
          models.push({ id: r.model, free: true, context_length: r.context });
        }
      }
    } else {
      for (const r of rows) {
        if (r.free) models.push({ id: r.model, free: true, context_length: r.context });
      }
    }
    out.push({ id: p.id, base_url: p.path, key_env: p.keyEnv, wire: p.wire, models });
  }
  return { providers: out };
}

/** The catalog with no brief at all: what --list prints outside a trial. */
const EMPTY_CATALOG: Catalog = { providers: [] };

// ---------------------------------------------------------------------------
// Canonical request: one form every wire translates through
// ---------------------------------------------------------------------------

type StopFamily = "stop" | "length" | "tool_calls" | "content_filter";

interface CanonImage {
  mime: string;
  b64: string;
}
interface CanonImageUrl {
  url: string;
}
type CanonImageRef = CanonImage | CanonImageUrl;

interface CanonToolCall {
  id: string;
  name: string;
  args: string;
}

interface CanonMsg {
  role: "system" | "user" | "assistant" | "tool";
  text: string;
  image?: CanonImageRef;
  toolCalls?: CanonToolCall[];
  toolCallId?: string;
}

interface CanonTool {
  name: string;
  description: string;
  schema: Json;
}

interface CanonReq {
  system: string;
  messages: CanonMsg[];
  tools: CanonTool[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | undefined;
  stream: boolean;
  includeUsage: boolean;
}

interface CanonAns {
  text: string;
  input: number;
  output: number;
  stop: StopFamily;
  toolCalls?: CanonToolCall[];
}

const FIXED_CREATED = 1767225600;
const SSE_PIECE = 13;

/** Split answer text the way the providers do: 13-character pieces. */
function splitPieces(text: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < text.length; i += SSE_PIECE) out.push(text.slice(i, i + SSE_PIECE));
  return out.length === 0 ? [""] : out;
}

function imageFromOpenAI(block: Json): CanonImageRef | null {
  const url = isObj(block.image_url) ? asStr(block.image_url.url) : null;
  if (url === null) return null;
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { mime: data[1] ?? "", b64: data[2] ?? "" };
  return { url };
}

function imageFromAnthropic(block: Json): CanonImageRef | null {
  if (!isObj(block.source)) return null;
  if (block.source.type === "base64" && typeof block.source.data === "string") {
    return { mime: asStr(block.source.media_type) ?? "", b64: block.source.data };
  }
  if (block.source.type === "url" && typeof block.source.url === "string") return { url: block.source.url };
  return null;
}

/** OpenAI Chat Completions body -> canonical form. */
function parseOpenAIReq(body: Json): CanonReq {
  const systemParts: string[] = [];
  const messages: CanonMsg[] = [];
  for (const m of isArr(body.messages) ? body.messages : []) {
    if (!isObj(m)) continue;
    const role = asStr(m.role);
    if (role === "system") {
      const t = contentText(m.content);
      if (t !== "") systemParts.push(t);
      continue;
    }
    if (role === "user") {
      const msg: CanonMsg = { role: "user", text: contentText(m.content) };
      if (isArr(m.content)) {
        for (const b of m.content) {
          if (isObj(b) && b.type === "image_url") {
            const img = imageFromOpenAI(b);
            if (img !== null) msg.image = img;
          }
        }
      }
      messages.push(msg);
      continue;
    }
    if (role === "assistant") {
      const msg: CanonMsg = { role: "assistant", text: contentText(m.content) };
      if (isArr(m.tool_calls)) {
        msg.toolCalls = [];
        for (const tc of m.tool_calls) {
          if (!isObj(tc) || !isObj(tc.function)) continue;
          const name = asStr(tc.function.name);
          if (name === null) continue;
          msg.toolCalls.push({
            id: asStr(tc.id) ?? `call_${msg.toolCalls.length}`,
            name,
            args: asStr(tc.function.arguments) ?? "{}",
          });
        }
      }
      messages.push(msg);
      continue;
    }
    if (role === "tool") {
      messages.push({ role: "tool", text: contentText(m.content), toolCallId: asStr(m.tool_call_id) ?? "" });
      continue;
    }
  }
  const tools: CanonTool[] = [];
  for (const t of isArr(body.tools) ? body.tools : []) {
    if (!isObj(t) || !isObj(t.function)) continue;
    const name = asStr(t.function.name);
    if (name === null) continue;
    tools.push({
      name,
      description: asStr(t.function.description) ?? "",
      schema: isObj(t.function.parameters) ? t.function.parameters : { type: "object", properties: {} },
    });
  }
  const fmt = isObj(body.response_format) ? asStr(body.response_format.type) : null;
  return {
    system: systemParts.join("\n"),
    messages,
    tools,
    maxTokens: asNum(body.max_tokens) ?? asNum(body.max_completion_tokens),
    stops: asStrList(body.stop),
    jsonMode: fmt === "json_object" || fmt === "json_schema",
    temperature: asNum(body.temperature),
    stream: body.stream === true,
    includeUsage: isObj(body.stream_options) && body.stream_options.include_usage === true,
  };
}

/** Anthropic Messages body -> canonical form. */
function parseAnthropicReq(body: Json): CanonReq {
  const system = contentText(body.system);
  const messages: CanonMsg[] = [];
  for (const m of isArr(body.messages) ? body.messages : []) {
    if (!isObj(m)) continue;
    const role = asStr(m.role);
    if (role === "user") {
      const msg: CanonMsg = { role: "user", text: contentText(m.content) };
      if (isArr(m.content)) {
        for (const b of m.content) {
          if (isObj(b) && b.type === "image") {
            const img = imageFromAnthropic(b);
            if (img !== null) msg.image = img;
          }
        }
      }
      messages.push(msg);
      continue;
    }
    if (role === "assistant") {
      const msg: CanonMsg = { role: "assistant", text: contentText(m.content) };
      if (isArr(m.content)) {
        for (const b of m.content) {
          if (!isObj(b) || b.type !== "tool_use") continue;
          const name = asStr(b.name);
          if (name === null) continue;
          (msg.toolCalls ??= []).push({
            id: asStr(b.id) ?? `call_${(msg.toolCalls ?? []).length}`,
            name,
            args: b.input === undefined ? "{}" : JSON.stringify(b.input),
          });
        }
      }
      messages.push(msg);
      continue;
    }
  }
  // Tool results arrive inside user messages as tool_result blocks; their
  // text is already carried by contentText above when the block is text.
  const tools: CanonTool[] = [];
  for (const t of isArr(body.tools) ? body.tools : []) {
    if (!isObj(t)) continue;
    const name = asStr(t.name);
    if (name === null) continue;
    tools.push({
      name,
      description: asStr(t.description) ?? "",
      schema: isObj(t.input_schema) ? t.input_schema : { type: "object", properties: {} },
    });
  }
  return {
    system,
    messages,
    tools,
    maxTokens: asNum(body.max_tokens),
    stops: asStrList(body.stop_sequences),
    jsonMode: false,
    temperature: asNum(body.temperature),
    stream: body.stream === true,
    includeUsage: false,
  };
}

/** Canonical user content as an OpenAI message value. */
function openAIUserContent(msg: CanonMsg): unknown {
  if (msg.image === undefined) return msg.text;
  const parts: Json[] = [{ type: "text", text: msg.text }];
  if ("b64" in msg.image) {
    parts.push({ type: "image_url", image_url: { url: `data:${msg.image.mime};base64,${msg.image.b64}` } });
  } else {
    parts.push({ type: "image_url", image_url: { url: msg.image.url } });
  }
  return parts;
}

/** Canonical form -> OpenAI Chat Completions body. */
function renderOpenAIReq(c: CanonReq, model: string): Json {
  const messages: Json[] = [];
  if (c.system !== "") messages.push({ role: "system", content: c.system });
  for (const m of c.messages) {
    if (m.role === "user") {
      messages.push({ role: "user", content: openAIUserContent(m) });
    } else if (m.role === "assistant") {
      const out: Json = { role: "assistant", content: m.text };
      if (m.toolCalls !== undefined && m.toolCalls.length > 0) {
        out.tool_calls = m.toolCalls.map((tc) => ({
          id: tc.id,
          type: "function",
          function: { name: tc.name, arguments: tc.args },
        }));
      }
      messages.push(out);
    } else if (m.role === "tool") {
      messages.push({ role: "tool", tool_call_id: m.toolCallId ?? "", content: m.text });
    }
  }
  const body: Json = { model, messages };
  if (c.tools.length > 0) {
    body.tools = c.tools.map((t) => ({
      type: "function",
      function: { name: t.name, description: t.description, parameters: t.schema },
    }));
  }
  if (c.maxTokens !== null) body.max_tokens = c.maxTokens;
  if (c.stops.length > 0) body.stop = c.stops;
  if (c.temperature !== undefined) body.temperature = c.temperature;
  if (c.jsonMode) body.response_format = { type: "json_object" };
  if (c.stream) {
    body.stream = true;
    body.stream_options = { include_usage: true };
  }
  return body;
}

/** Canonical form -> Anthropic Messages body. */
function renderAnthropicReq(c: CanonReq, model: string): Json {
  const messages: Json[] = [];
  for (const m of c.messages) {
    if (m.role === "user") {
      const content: Json[] = [{ type: "text", text: m.text }];
      if (m.image !== undefined) {
        if ("b64" in m.image) {
          content.push({ type: "image", source: { type: "base64", media_type: m.image.mime, data: m.image.b64 } });
        } else {
          content.push({ type: "image", source: { type: "url", url: m.image.url } });
        }
      }
      messages.push({ role: "user", content });
    } else if (m.role === "assistant") {
      const content: Json[] = [];
      if (m.text !== "") content.push({ type: "text", text: m.text });
      for (const tc of m.toolCalls ?? []) {
        let input: unknown = {};
        try {
          input = JSON.parse(tc.args);
        } catch {
          input = {};
        }
        content.push({ type: "tool_use", id: tc.id, name: tc.name, input });
      }
      messages.push({ role: "assistant", content });
    } else if (m.role === "tool") {
      messages.push({
        role: "user",
        content: [{ type: "tool_result", tool_use_id: m.toolCallId ?? "", content: m.text }],
      });
    }
  }
  const body: Json = { model, messages };
  if (c.system !== "") body.system = c.system;
  if (c.tools.length > 0) {
    body.tools = c.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.schema }));
  }
  if (c.maxTokens !== null) body.max_tokens = c.maxTokens;
  if (c.stops.length > 0) body.stop_sequences = c.stops;
  if (c.temperature !== undefined) body.temperature = c.temperature;
  if (c.stream) body.stream = true;
  return body;
}

/** Canonical form -> Gemini generateContent body. */
function renderGeminiReq(c: CanonReq): Json {
  const contents: Json[] = [];
  for (const m of c.messages) {
    if (m.role === "user") {
      const parts: Json[] = [{ text: m.text }];
      if (m.image !== undefined) {
        if ("b64" in m.image) {
          parts.push({ inlineData: { mimeType: m.image.mime, data: m.image.b64 } });
        } else {
          parts.push({ fileData: { fileUri: m.image.url } });
        }
      }
      contents.push({ role: "user", parts });
    } else if (m.role === "assistant") {
      const parts: Json[] = [];
      if (m.text !== "") parts.push({ text: m.text });
      for (const tc of m.toolCalls ?? []) {
        let input: unknown = {};
        try {
          input = JSON.parse(tc.args);
        } catch {
          input = {};
        }
        parts.push({ functionCall: { name: tc.name, args: input } });
      }
      contents.push({ role: "model", parts });
    } else if (m.role === "tool") {
      contents.push({
        role: "user",
        parts: [{ functionResponse: { name: m.toolCallId ?? "", response: { content: m.text } } }],
      });
    }
  }
  const body: Json = { contents };
  if (c.system !== "") body.systemInstruction = { parts: [{ text: c.system }] };
  if (c.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: c.tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.schema,
        })),
      },
    ];
  }
  const config: Json = {};
  if (c.maxTokens !== null) config.maxOutputTokens = c.maxTokens;
  if (c.stops.length > 0) config.stopSequences = c.stops;
  if (c.temperature !== undefined) config.temperature = c.temperature;
  if (c.jsonMode) config.responseMimeType = "application/json";
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

// ---------------------------------------------------------------------------
// Provider answers -> canonical answer
// ---------------------------------------------------------------------------

function openAIStop(finish: unknown): StopFamily {
  if (finish === "length") return "length";
  if (finish === "tool_calls" || finish === "function_call") return "tool_calls";
  if (finish === "content_filter") return "content_filter";
  return "stop";
}

function anthropicStop(reason: unknown): StopFamily {
  if (reason === "max_tokens") return "length";
  if (reason === "tool_use") return "tool_calls";
  if (reason === "refusal") return "content_filter";
  return "stop";
}

function geminiStop(reason: unknown): StopFamily {
  if (reason === "MAX_TOKENS") return "length";
  if (reason === "SAFETY") return "content_filter";
  return "stop";
}

/** OpenAI Chat Completions JSON body -> canonical answer. */
function parseOpenAIAns(body: unknown): CanonAns | null {
  if (!isObj(body) || !isArr(body.choices) || body.choices.length === 0) return null;
  const choice = body.choices[0];
  if (!isObj(choice) || !isObj(choice.message)) return null;
  const msg = choice.message;
  const text = typeof msg.content === "string" ? msg.content : contentText(msg.content);
  const usage = isObj(body.usage) ? body.usage : {};
  const toolCalls: CanonToolCall[] = [];
  for (const tc of isArr(msg.tool_calls) ? msg.tool_calls : []) {
    if (!isObj(tc) || !isObj(tc.function)) continue;
    const name = asStr(tc.function.name);
    if (name === null) continue;
    toolCalls.push({ id: asStr(tc.id) ?? `call_${toolCalls.length}`, name, args: asStr(tc.function.arguments) ?? "{}" });
  }
  const ans: CanonAns = {
    text,
    input: asNum(usage.prompt_tokens) ?? 0,
    output: asNum(usage.completion_tokens) ?? 0,
    stop: openAIStop(choice.finish_reason),
  };
  if (toolCalls.length > 0) ans.toolCalls = toolCalls;
  return ans;
}

/** Anthropic Messages JSON body -> canonical answer. */
function parseAnthropicAns(body: unknown): CanonAns | null {
  if (!isObj(body) || body.type !== "message" || !isArr(body.content)) return null;
  const text = contentText(body.content);
  const usage = isObj(body.usage) ? body.usage : {};
  const toolCalls: CanonToolCall[] = [];
  for (const b of body.content) {
    if (!isObj(b) || b.type !== "tool_use") continue;
    const name = asStr(b.name);
    if (name === null) continue;
    toolCalls.push({
      id: asStr(b.id) ?? `call_${toolCalls.length}`,
      name,
      args: b.input === undefined ? "{}" : JSON.stringify(b.input),
    });
  }
  const ans: CanonAns = {
    text,
    input: asNum(usage.input_tokens) ?? 0,
    output: asNum(usage.output_tokens) ?? 0,
    stop: anthropicStop(body.stop_reason),
  };
  if (toolCalls.length > 0) ans.toolCalls = toolCalls;
  return ans;
}

/** Gemini generateContent JSON body -> canonical answer. */
function parseGeminiAns(body: unknown): CanonAns | null {
  if (!isObj(body) || !isArr(body.candidates) || body.candidates.length === 0) return null;
  const cand = body.candidates[0];
  if (!isObj(cand)) return null;
  let text = "";
  if (isObj(cand.content) && isArr(cand.content.parts)) {
    const parts: string[] = [];
    for (const p of cand.content.parts) {
      if (isObj(p) && typeof p.text === "string") parts.push(p.text);
    }
    text = parts.join("");
  }
  const meta = isObj(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    input: asNum(meta.promptTokenCount) ?? 0,
    output: asNum(meta.candidatesTokenCount) ?? 0,
    stop: geminiStop(cand.finishReason),
  };
}

/** Split a raw SSE body into frames: {event, data}. */
function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, "\n");
  for (const chunk of normalized.split("\n\n")) {
    if (chunk.trim() === "") continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of chunk.split("\n")) {
      if (line.startsWith("event:")) event = line.slice("event:".length).trim();
      else if (line.startsWith("data:")) data.push(line.slice("data:".length).replace(/^ /, ""));
    }
    if (data.length > 0) frames.push({ event, data: data.join("\n") });
  }
  return frames;
}

/** OpenAI SSE body -> canonical answer. */
function parseOpenAIStream(text: string): CanonAns | null {
  let out = "";
  let input = 0;
  let output = 0;
  let stop: StopFamily = "stop";
  let saw = false;
  for (const f of sseFrames(text)) {
    const payload = f.data.trim();
    if (payload === "[DONE]") continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isObj(chunk)) continue;
    if (isObj(chunk.usage)) {
      input = asNum(chunk.usage.prompt_tokens) ?? input;
      output = asNum(chunk.usage.completion_tokens) ?? output;
    }
    if (!isArr(chunk.choices) || chunk.choices.length === 0) continue;
    const choice = chunk.choices[0];
    if (!isObj(choice)) continue;
    saw = true;
    if (isObj(choice.delta) && typeof choice.delta.content === "string") out += choice.delta.content;
    if (typeof choice.finish_reason === "string") stop = openAIStop(choice.finish_reason);
  }
  if (!saw && out === "") return null;
  return { text: out, input, output, stop };
}

/** Anthropic SSE body -> canonical answer. */
function parseAnthropicStream(text: string): CanonAns | null {
  let out = "";
  let input = 0;
  let output = 0;
  let stop: StopFamily = "stop";
  let saw = false;
  const toolInputs = new Map<number, { id: string; name: string; json: string }>();
  const toolCalls: CanonToolCall[] = [];
  for (const f of sseFrames(text)) {
    if (f.event === null) continue;
    let data: unknown;
    try {
      data = JSON.parse(f.data);
    } catch {
      continue;
    }
    if (!isObj(data)) continue;
    switch (f.event) {
      case "message_start": {
        saw = true;
        if (isObj(data.message) && isObj(data.message.usage)) {
          input = asNum(data.message.usage.input_tokens) ?? input;
        }
        break;
      }
      case "content_block_start": {
        const idx = asNum(data.index);
        if (idx === null) break;
        if (isObj(data.content_block) && data.content_block.type === "tool_use") {
          const name = asStr(data.content_block.name);
          if (name !== null) {
            toolInputs.set(idx, { id: asStr(data.content_block.id) ?? `call_${idx}`, name, json: "" });
          }
        }
        break;
      }
      case "content_block_delta": {
        const idx = asNum(data.index);
        if (idx === null || !isObj(data.delta)) break;
        if (data.delta.type === "text_delta" && typeof data.delta.text === "string") out += data.delta.text;
        if (data.delta.type === "input_json_delta" && typeof data.delta.partial_json === "string") {
          const entry = toolInputs.get(idx);
          if (entry !== undefined) entry.json += data.delta.partial_json;
        }
        break;
      }
      case "content_block_stop": {
        const idx = asNum(data.index);
        if (idx === null) break;
        const entry = toolInputs.get(idx);
        if (entry !== undefined) {
          toolCalls.push({ id: entry.id, name: entry.name, args: entry.json === "" ? "{}" : entry.json });
          toolInputs.delete(idx);
        }
        break;
      }
      case "message_delta": {
        if (isObj(data.delta) && typeof data.delta.stop_reason === "string") {
          stop = anthropicStop(data.delta.stop_reason);
        }
        if (isObj(data.usage)) output = asNum(data.usage.output_tokens) ?? output;
        break;
      }
      default:
        break;
    }
  }
  if (!saw && out === "") return null;
  const ans: CanonAns = { text: out, input, output, stop };
  if (toolCalls.length > 0) ans.toolCalls = toolCalls;
  return ans;
}

/** Gemini streamGenerateContent SSE body -> canonical answer. */
function parseGeminiStream(text: string): CanonAns | null {
  let out = "";
  let input = 0;
  let output = 0;
  let stop: StopFamily = "stop";
  let saw = false;
  for (const f of sseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(f.data);
    } catch {
      continue;
    }
    if (!isObj(data) || !isArr(data.candidates) || data.candidates.length === 0) continue;
    const cand = data.candidates[0];
    if (!isObj(cand)) continue;
    saw = true;
    if (isObj(cand.content) && isArr(cand.content.parts)) {
      for (const p of cand.content.parts) {
        if (isObj(p) && typeof p.text === "string") out += p.text;
      }
    }
    if (typeof cand.finishReason === "string") stop = geminiStop(cand.finishReason);
    if (isObj(data.usageMetadata)) {
      input = asNum(data.usageMetadata.promptTokenCount) ?? input;
      output = asNum(data.usageMetadata.candidatesTokenCount) ?? output;
    }
  }
  if (!saw) return null;
  return { text: out, input, output, stop };
}

// ---------------------------------------------------------------------------
// Canonical answer -> client wire
// ---------------------------------------------------------------------------

function openAIFinish(stop: StopFamily): string {
  if (stop === "length") return "length";
  if (stop === "tool_calls") return "tool_calls";
  if (stop === "content_filter") return "content_filter";
  return "stop";
}

function anthropicStopReason(stop: StopFamily): string {
  if (stop === "length") return "max_tokens";
  if (stop === "tool_calls") return "tool_use";
  if (stop === "content_filter") return "refusal";
  return "end_turn";
}

function sseLine(event: string | null, data: unknown): string {
  const payload = typeof data === "string" ? data : JSON.stringify(data);
  return `${event === null ? "" : `event: ${event}\n`}data: ${payload}\n\n`;
}

function renderOpenAIAns(ans: CanonAns, model: string, stream: boolean): string {
  if (!stream) {
    const message: Json = { role: "assistant", content: ans.text };
    if (ans.toolCalls !== undefined && ans.toolCalls.length > 0) {
      message.tool_calls = ans.toolCalls.map((tc) => ({
        id: tc.id,
        type: "function",
        function: { name: tc.name, arguments: tc.args },
      }));
    }
    return `${JSON.stringify({
      id: "chatcmpl-agg",
      object: "chat.completion",
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, message, finish_reason: openAIFinish(ans.stop), logprobs: null }],
      usage: { prompt_tokens: ans.input, completion_tokens: ans.output, total_tokens: ans.input + ans.output },
    })}\n`;
  }
  const base = { id: "chatcmpl-agg", object: "chat.completion.chunk", created: FIXED_CREATED, model };
  const frames: string[] = [];
  frames.push(
    sseLine(null, { ...base, choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }] }),
  );
  for (const piece of splitPieces(ans.text)) {
    frames.push(sseLine(null, { ...base, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] }));
  }
  frames.push(
    sseLine(null, { ...base, choices: [{ index: 0, delta: {}, finish_reason: openAIFinish(ans.stop) }] }),
  );
  frames.push(
    sseLine(null, {
      ...base,
      choices: [],
      usage: { prompt_tokens: ans.input, completion_tokens: ans.output, total_tokens: ans.input + ans.output },
    }),
  );
  frames.push("data: [DONE]\n\n");
  return frames.join("");
}

function renderAnthropicAns(ans: CanonAns, model: string, stream: boolean): string {
  const content: Json[] = [{ type: "text", text: ans.text }];
  for (const tc of ans.toolCalls ?? []) {
    let input: unknown = {};
    try {
      input = JSON.parse(tc.args);
    } catch {
      input = {};
    }
    content.push({ type: "tool_use", id: tc.id, name: tc.name, input });
  }
  if (!stream) {
    return `${JSON.stringify({
      id: "msg_agg",
      type: "message",
      role: "assistant",
      model,
      content,
      stop_reason: anthropicStopReason(ans.stop),
      stop_sequence: null,
      usage: { input_tokens: ans.input, output_tokens: ans.output },
    })}\n`;
  }
  const frames: string[] = [];
  frames.push(
    sseLine("message_start", {
      type: "message_start",
      message: {
        id: "msg_agg",
        type: "message",
        role: "assistant",
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: ans.input, output_tokens: 0 },
      },
    }),
  );
  frames.push(
    sseLine("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  );
  for (const piece of splitPieces(ans.text)) {
    frames.push(
      sseLine("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } }),
    );
  }
  frames.push(sseLine("content_block_stop", { type: "content_block_stop", index: 0 }));
  frames.push(
    sseLine("message_delta", {
      type: "message_delta",
      delta: { stop_reason: anthropicStopReason(ans.stop), stop_sequence: null },
      usage: { output_tokens: ans.output },
    }),
  );
  frames.push(sseLine("message_stop", { type: "message_stop" }));
  return frames.join("");
}

function openAIError(message: string, type: string, code: string | null): string {
  return `${JSON.stringify({ error: { message, type, param: null, code } })}\n`;
}

function anthropicError(type: string, message: string): string {
  return `${JSON.stringify({ type: "error", error: { type, message } })}\n`;
}

// ---------------------------------------------------------------------------
// Provider calls
// ---------------------------------------------------------------------------

class ProviderRefusal extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

const PROVIDER_TIMEOUT_MS = 15000;

interface ProviderCall {
  provider: ProviderDir;
  model: string;
}

/** One inference call to one provider; throws ProviderRefusal on any refusal. */
async function callProvider(
  envUrl: string,
  call: ProviderCall,
  canon: CanonReq,
  clientWire: Wire,
): Promise<{ ans: CanonAns; raw: string; rawIsStream: boolean }> {
  const p = call.provider;
  const headers: Record<string, string> = { ...authHeaders(p.auth, p.key), "content-type": "application/json" };
  let url: string;
  let payload: Json;
  if (p.wire === "openai") {
    url = `${envUrl}${p.path}/v1/chat/completions`;
    payload = renderOpenAIReq(canon, call.model);
  } else if (p.wire === "anthropic") {
    url = `${envUrl}${p.path}/v1/messages`;
    payload = renderAnthropicReq(canon, call.model);
  } else {
    const stream = canon.stream;
    url = `${envUrl}${p.path}/v1beta/models/${encodeURIComponent(call.model)}:${stream ? "streamGenerateContent" : "generateContent"}`;
    payload = renderGeminiReq(canon);
  }
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), PROVIDER_TIMEOUT_MS);
  let res: Response;
  try {
    res = await fetch(url, { method: "POST", headers, body: JSON.stringify(payload), signal: ctrl.signal });
  } catch (err) {
    throw new ProviderRefusal(0, `could not reach ${p.id}: ${err instanceof Error ? err.message : String(err)}`);
  } finally {
    clearTimeout(timer);
  }
  const text = await res.text();
  if (!res.ok) {
    throw new ProviderRefusal(res.status, `${p.id} answered HTTP ${res.status}`);
  }
  const isStream = canon.stream;
  if (p.wire === clientWire && !isStream) {
    // Same wire, JSON: the provider's body is already the client's shape.
    const ans = p.wire === "openai" ? parseOpenAIAns(safeJson(text)) : parseAnthropicAns(safeJson(text));
    if (ans === null) throw new ProviderRefusal(502, `${p.id} answered a body that is not a ${p.wire} answer`);
    return { ans, raw: text, rawIsStream: false };
  }
  if (p.wire === clientWire && isStream) {
    const ans = p.wire === "openai" ? parseOpenAIStream(text) : parseAnthropicStream(text);
    if (ans === null) throw new ProviderRefusal(502, `${p.id} answered a stream that is not a ${p.wire} stream`);
    return { ans, raw: text, rawIsStream: true };
  }
  // Cross-wire: parse the provider's answer into canonical form.
  let ans: CanonAns | null = null;
  if (p.wire === "openai") ans = isStream ? parseOpenAIStream(text) : parseOpenAIAns(safeJson(text));
  else if (p.wire === "anthropic") ans = isStream ? parseAnthropicStream(text) : parseAnthropicAns(safeJson(text));
  else ans = isStream ? parseGeminiStream(text) : parseGeminiAns(safeJson(text));
  if (ans === null) throw new ProviderRefusal(502, `${p.id} answered a body this server could not read`);
  return { ans, raw: text, rawIsStream: isStream };
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

interface RouteState {
  brief: Brief;
  envUrl: string;
  catalog: Catalog;
  /** Consecutive refusals per provider; 2+ demotes it for the rest of the trial. */
  failures: Map<string, number>;
  servedCount: number;
}

function meetsReq(row: RatingRow, req: Requires): boolean {
  if (req.tools === true && !row.tools) return false;
  if (req.vision === true && !row.vision) return false;
  if (req.json === true && !row.json) return false;
  if (req.minContext !== undefined && (row.context === null || row.context < req.minContext)) return false;
  return true;
}

/** Candidates for a request: free, meeting its requirements, best first.
 *  Ties break toward the client's wire, then the brief's provider order. */
function rankCandidates(state: RouteState, clientWire: Wire, req: Requires): ProviderCall[] {
  const byId = new Map(state.brief.providers.map((p, i) => [p.id, { p, i }]));
  const list: { call: ProviderCall; rating: number; same: boolean; order: number }[] = [];
  for (const row of state.brief.ratings) {
    if (!row.free || !meetsReq(row, req)) continue;
    const found = byId.get(row.provider);
    if (found === undefined) continue;
    // A JSON-mode request cannot go to a wire that cannot carry response_format.
    if (req.json === true && found.p.wire === "anthropic") continue;
    // Never route to a model the provider's own listing says is paid.
    const cat = state.catalog.providers.find((c) => c.id === row.provider);
    const listed = cat?.models.find((m) => m.id === row.model);
    if (listed !== undefined && listed.free !== true) continue;
    list.push({
      call: { provider: found.p, model: row.model },
      rating: row.rating,
      same: found.p.wire === clientWire,
      order: found.i,
    });
  }
  list.sort((a, b) => {
    const badA = (state.failures.get(a.call.provider.id) ?? 0) >= 2 ? 1 : 0;
    const badB = (state.failures.get(b.call.provider.id) ?? 0) >= 2 ? 1 : 0;
    if (badA !== badB) return badA - badB;
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.same !== b.same) return a.same ? -1 : 1;
    return a.order - b.order;
  });
  // Drop duplicate (provider, model) pairs, keeping the first.
  const seen = new Set<string>();
  const out: ProviderCall[] = [];
  for (const item of list) {
    const key = `${item.call.provider.id}/${item.call.model}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(item.call);
  }
  return out;
}

/** Requirements for a request, from the manifest when known, else the body. */
function requirementsFor(state: RouteState, id: string | null, api: "openai" | "anthropic", body: Json): Requires {
  if (id !== null) {
    const entry = state.brief.manifest.find((m) => m.id === id);
    if (entry !== undefined) return entry.requires;
  }
  const req: Requires = {};
  const canon = api === "openai" ? parseOpenAIReq(body) : parseAnthropicReq(body);
  if (canon.tools.length > 0) req.tools = true;
  if (canon.messages.some((m) => m.image !== undefined)) req.vision = true;
  if (canon.jsonMode) req.json = true;
  return req;
}

interface ServedAnswer {
  status: number;
  contentType: string;
  body: string;
}

/** Serve one client request: route it, fail over, render the answer. */
async function serveRequest(
  state: RouteState,
  clientApi: "openai" | "anthropic",
  body: Json,
  requestId: string | null,
): Promise<ServedAnswer> {
  const canon = clientApi === "openai" ? parseOpenAIReq(body) : parseAnthropicReq(body);
  const req = requirementsFor(state, requestId, clientApi, body);
  const clientWire: Wire = clientApi;
  const candidates = rankCandidates(state, clientWire, req);
  const stream = canon.stream;
  for (const call of candidates) {
    try {
      const { ans, raw } = await callProvider(state.envUrl, call, canon, clientWire);
      state.failures.set(call.provider.id, 0);
      if (call.provider.wire === clientWire) {
        // Passthrough: the provider's answer is already the client's shape.
        return {
          status: 200,
          contentType: stream ? "text/event-stream; charset=utf-8" : "application/json",
          body: raw,
        };
      }
      const rendered =
        clientApi === "openai" ? renderOpenAIAns(ans, call.model, stream) : renderAnthropicAns(ans, call.model, stream);
      return {
        status: 200,
        contentType: stream ? "text/event-stream; charset=utf-8" : "application/json",
        body: rendered,
      };
    } catch (err) {
      const n = (state.failures.get(call.provider.id) ?? 0) + 1;
      state.failures.set(call.provider.id, n);
      log(`refused: ${call.provider.id}/${call.model} (${err instanceof Error ? err.message : String(err)})`);
      continue;
    }
  }
  const message = "every free model that meets this request refused it";
  if (clientApi === "openai") {
    return { status: 502, contentType: "application/json", body: openAIError(message, "api_error", null) };
  }
  return { status: 502, contentType: "application/json", body: anthropicError("api_error", message) };
}

// ---------------------------------------------------------------------------
// HTTP server: GET /v1/models, POST /v1/chat/completions, POST /v1/messages
// ---------------------------------------------------------------------------

function freeModelIds(catalog: Catalog): { id: string; provider: string; context: number | null }[] {
  const seen = new Set<string>();
  const out: { id: string; provider: string; context: number | null }[] = [];
  for (const p of catalog.providers) {
    for (const m of p.models) {
      if (!m.free || seen.has(m.id)) continue;
      seen.add(m.id);
      out.push({ id: m.id, provider: p.id, context: m.context_length });
    }
  }
  return out;
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (c: Buffer) => {
      size += c.length;
      if (size > 2 * 1024 * 1024) {
        reject(new Error("body too large"));
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function sendJson(res: ServerResponse, status: number, text: string): void {
  res.writeHead(status, { "content-type": "application/json", "content-length": String(Buffer.byteLength(text)) });
  res.end(text);
}

function sendStream(res: ServerResponse, text: string): void {
  res.writeHead(200, {
    "content-type": "text/event-stream; charset=utf-8",
    "cache-control": "no-cache",
    connection: "keep-alive",
    "content-length": String(Buffer.byteLength(text)),
  });
  res.end(text);
}

function log(message: string): void {
  process.stderr.write(`free-model-aggregator-01: ${message}\n`);
}

async function workState(envUrl: string): Promise<{ done: boolean; next: string | null }> {
  try {
    const res = await fetch(`${envUrl}/work`);
    if (!res.ok) return { done: false, next: null };
    const body = (await res.json()) as unknown;
    if (!isObj(body)) return { done: false, next: null };
    if (body.done === true) return { done: true, next: null };
    return { done: false, next: asStr(body.next) };
  } catch {
    return { done: false, next: null };
  }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const envUrl = process.env.MAKEFUN_ENV_URL ?? "";
  const promptFile = process.env.MAKEFUN_PROMPT_FILE ?? "";

  if (args.includes("--list")) {
    if (envUrl === "" || promptFile === "") {
      process.stdout.write(`${JSON.stringify(EMPTY_CATALOG)}\n`);
      return;
    }
    const brief = readBrief(promptFile);
    const catalog = await buildCatalog(envUrl, brief);
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }

  if (envUrl === "" || promptFile === "") {
    process.stderr.write("free-model-aggregator-01: MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL are required\n");
    process.exit(1);
  }

  const brief = readBrief(promptFile);
  const catalog = await buildCatalog(envUrl, brief);
  const state: RouteState = {
    brief,
    envUrl,
    catalog,
    failures: new Map(),
    servedCount: 0,
  };

  const server = createServer((req, res) => {
    void (async () => {
      try {
        const url = new URL(req.url ?? "/", "http://local");
        if (req.method === "GET" && url.pathname === "/v1/models") {
          const list = freeModelIds(catalog);
          if (typeof req.headers["x-api-key"] === "string") {
            sendJson(
              res,
              200,
              `${JSON.stringify({
                data: list.map((m) => ({
                  type: "model",
                  id: m.id,
                  display_name: m.id,
                  created_at: "2026-01-01T00:00:00Z",
                  free: true,
                  context_length: m.context,
                })),
                has_more: false,
              })}\n`,
            );
          } else {
            sendJson(
              res,
              200,
              `${JSON.stringify({
                object: "list",
                data: list.map((m) => ({
                  id: m.id,
                  object: "model",
                  created: FIXED_CREATED,
                  owned_by: m.provider,
                  free: true,
                  context_length: m.context,
                })),
              })}\n`,
            );
          }
          return;
        }
        if (req.method === "POST" && (url.pathname === "/v1/chat/completions" || url.pathname === "/v1/messages")) {
          const api: "openai" | "anthropic" = url.pathname === "/v1/messages" ? "anthropic" : "openai";
          const raw = await readBody(req);
          let body: unknown;
          try {
            body = JSON.parse(raw) as unknown;
          } catch {
            const text =
              api === "openai"
                ? openAIError("the request body is not one JSON object", "invalid_request_error", "invalid_request")
                : anthropicError("invalid_request_error", "the request body is not one JSON object");
            sendJson(res, 400, text);
            return;
          }
          if (!isObj(body)) {
            const text =
              api === "openai"
                ? openAIError("the request body is not one JSON object", "invalid_request_error", "invalid_request")
                : anthropicError("invalid_request_error", "the request body is not one JSON object");
            sendJson(res, 400, text);
            return;
          }
          // The env's client sends the manifest's requests in order; /work
          // names the one in flight, which pins the requirements.
          const work = await workState(envUrl);
          let requestId: string | null = work.next;
          if (requestId === null) {
            const entry = brief.manifest[state.servedCount];
            requestId = entry !== undefined ? entry.id : null;
          }
          state.servedCount += 1;
          const answer = await serveRequest(state, api, body, requestId);
          if (answer.contentType.startsWith("text/event-stream")) sendStream(res, answer.body);
          else sendJson(res, answer.status, answer.body);
          return;
        }
        sendJson(res, 404, `${JSON.stringify({ error: "not_found", message: `no route ${req.method ?? ""} ${url.pathname}` })}\n`);
      } catch (err) {
        log(`handler error: ${err instanceof Error ? err.message : String(err)}`);
        try {
          sendJson(res, 500, `${JSON.stringify({ error: "internal", message: "the aggregator failed" })}\n`);
        } catch {
          // The response is already gone; nothing left to do.
        }
      }
    })();
  });

  await new Promise<void>((resolve) => server.listen(0, "0.0.0.0", () => resolve()));
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("the server has no TCP address");
  const port = address.port;

  const reg = await fetch(`${envUrl}/register`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ protocol_version: 1, port, catalog }),
  });
  if (!reg.ok) {
    throw new Error(`POST ${envUrl}/register answered HTTP ${reg.status}: ${(await reg.text()).slice(0, 500)}`);
  }
  const reply = (await reg.json()) as unknown;
  if (isObj(reply) && typeof reply.requests === "number") {
    log(`registered on port ${port}: the workload holds ${reply.requests} requests`);
  }

  for (;;) {
    const work = await workState(envUrl);
    if (work.done) break;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  server.close();
  log("workload done, exiting 0");
  process.exit(0);
}

main().catch((err: unknown) => {
  process.stderr.write(`free-model-aggregator-01: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
