/**
 * Cursor Provider Extension for pi
 *
 * Provides access to Cursor models (Claude, GPT, Gemini, etc.) via:
 * 1. Browser-based PKCE OAuth login to Cursor
 * 2. Local proxy translating OpenAI format → Cursor gRPC protocol
 *
 * Usage:
 *   /login cursor    — authenticate via browser
 *   /model           — select any Cursor model
 *
 * Based on https://github.com/ephraimduncan/opencode-cursor by Ephraim Duncan.
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import type { OAuthCredentials, OAuthLoginCallbacks } from "@mariozechner/pi-ai";
import {
  generateCursorAuthParams,
  getTokenExpiry,
  pollCursorAuth,
  refreshCursorToken,
} from "./auth.js";
import { getCursorModels, startProxy, type CursorModel } from "./proxy.js";

// ── Cost estimation ──

interface ModelCost {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
}

const MODEL_COST_TABLE: Record<string, ModelCost> = {
  "claude-4-sonnet":         { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
  "claude-4.5-haiku":        { input: 1, output: 5, cacheRead: 0.1, cacheWrite: 1.25 },
  "claude-4.5-opus":         { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
  "claude-4.5-sonnet":       { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
  "claude-4.6-opus":         { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
  "claude-4.6-sonnet":       { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
  "composer-1":              { input: 1.25, output: 10, cacheRead: 0.125, cacheWrite: 0 },
  "composer-1.5":            { input: 3.5, output: 17.5, cacheRead: 0.35, cacheWrite: 0 },
  "composer-2":              { input: 0.5, output: 2.5, cacheRead: 0.2, cacheWrite: 0 },
  "gemini-2.5-flash":        { input: 0.3, output: 2.5, cacheRead: 0.03, cacheWrite: 0 },
  "gemini-3-flash":          { input: 0.5, output: 3, cacheRead: 0.05, cacheWrite: 0 },
  "gemini-3-pro":            { input: 2, output: 12, cacheRead: 0.2, cacheWrite: 0 },
  "gemini-3.1-pro":          { input: 2, output: 12, cacheRead: 0.2, cacheWrite: 0 },
  "gpt-5":                   { input: 1.25, output: 10, cacheRead: 0.125, cacheWrite: 0 },
  "gpt-5-mini":              { input: 0.25, output: 2, cacheRead: 0.025, cacheWrite: 0 },
  "gpt-5.2":                 { input: 1.75, output: 14, cacheRead: 0.175, cacheWrite: 0 },
  "gpt-5.2-codex":           { input: 1.75, output: 14, cacheRead: 0.175, cacheWrite: 0 },
  "gpt-5.3-codex":           { input: 1.75, output: 14, cacheRead: 0.175, cacheWrite: 0 },
  "gpt-5.4":                 { input: 2.5, output: 15, cacheRead: 0.25, cacheWrite: 0 },
  "gpt-5.4-mini":            { input: 0.75, output: 4.5, cacheRead: 0.075, cacheWrite: 0 },
  "grok-4.20":               { input: 2, output: 6, cacheRead: 0.2, cacheWrite: 0 },
  "kimi-k2.5":               { input: 0.6, output: 3, cacheRead: 0.1, cacheWrite: 0 },
};

const MODEL_COST_PATTERNS: Array<{ match: (id: string) => boolean; cost: ModelCost }> = [
  { match: (id) => /claude.*opus.*fast/i.test(id),   cost: { input: 30, output: 150, cacheRead: 3, cacheWrite: 37.5 } },
  { match: (id) => /claude.*opus/i.test(id),         cost: MODEL_COST_TABLE["claude-4.6-opus"]! },
  { match: (id) => /claude.*haiku/i.test(id),        cost: MODEL_COST_TABLE["claude-4.5-haiku"]! },
  { match: (id) => /claude.*sonnet/i.test(id),       cost: MODEL_COST_TABLE["claude-4.6-sonnet"]! },
  { match: (id) => /composer/i.test(id),             cost: MODEL_COST_TABLE["composer-1"]! },
  { match: (id) => /gpt-5\.4.*mini/i.test(id),      cost: MODEL_COST_TABLE["gpt-5.4-mini"]! },
  { match: (id) => /gpt-5\.4/i.test(id),            cost: MODEL_COST_TABLE["gpt-5.4"]! },
  { match: (id) => /gpt-5\.3/i.test(id),            cost: MODEL_COST_TABLE["gpt-5.3-codex"]! },
  { match: (id) => /gpt-5\.2/i.test(id),            cost: MODEL_COST_TABLE["gpt-5.2"]! },
  { match: (id) => /gpt-5.*mini/i.test(id),          cost: MODEL_COST_TABLE["gpt-5-mini"]! },
  { match: (id) => /gpt-5/i.test(id),                cost: MODEL_COST_TABLE["gpt-5"]! },
  { match: (id) => /gemini.*3\.1/i.test(id),        cost: MODEL_COST_TABLE["gemini-3.1-pro"]! },
  { match: (id) => /gemini.*flash/i.test(id),        cost: MODEL_COST_TABLE["gemini-2.5-flash"]! },
  { match: (id) => /gemini/i.test(id),               cost: MODEL_COST_TABLE["gemini-3-pro"]! },
  { match: (id) => /grok/i.test(id),                 cost: MODEL_COST_TABLE["grok-4.20"]! },
  { match: (id) => /kimi/i.test(id),                 cost: MODEL_COST_TABLE["kimi-k2.5"]! },
];

const DEFAULT_COST: ModelCost = { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 0 };

function estimateModelCost(modelId: string): ModelCost {
  const normalized = modelId.toLowerCase();
  const exact = MODEL_COST_TABLE[normalized];
  if (exact) return exact;
  const stripped = normalized.replace(/-(high|medium|low|preview|thinking|spark-preview|fast)$/g, "");
  const strippedMatch = MODEL_COST_TABLE[stripped];
  if (strippedMatch) return strippedMatch;
  return MODEL_COST_PATTERNS.find((p) => p.match(normalized))?.cost ?? DEFAULT_COST;
}


// ── Effort-level dedup ──

const EFFORT_LEVELS = new Set(["low", "medium", "high", "xhigh", "max", "none"]);

interface ParsedModelId {
  base: string;       // model ID with effort stripped
  effort: string;     // effort level, or "" if no effort suffix
  fast: boolean;      // has -fast suffix
  thinking: boolean;  // has -thinking suffix
}

export function parseModelId(id: string): ParsedModelId {
  let remaining = id;
  let fast = false;
  let thinking = false;

  if (remaining.endsWith("-fast")) {
    fast = true;
    remaining = remaining.slice(0, -5);
  }
  if (remaining.endsWith("-thinking")) {
    thinking = true;
    remaining = remaining.slice(0, -9);
  }

  const lastDash = remaining.lastIndexOf("-");
  if (lastDash >= 0) {
    const suffix = remaining.slice(lastDash + 1);
    if (EFFORT_LEVELS.has(suffix)) {
      return { base: remaining.slice(0, lastDash), effort: suffix, fast, thinking };
    }
  }

  return { base: remaining, effort: "", fast, thinking };
}

interface ProcessedModel extends CursorModel {
  supportsEffort: boolean;
  effortMap?: Record<string, string>;
}

/** Dedup raw models: collapse effort variants into one entry with supportsReasoningEffort. */
export function processModels(raw: CursorModel[]): ProcessedModel[] {
  // Group by (base, fast, thinking)
  const groups = new Map<string, {
    base: string; fast: boolean; thinking: boolean;
    efforts: Map<string, CursorModel>;
  }>();

  for (const model of raw) {
    const p = parseModelId(model.id);
    const key = `${p.base}|${p.fast}|${p.thinking}`;
    let g = groups.get(key);
    if (!g) {
      g = { base: p.base, fast: p.fast, thinking: p.thinking, efforts: new Map() };
      groups.set(key, g);
    }
    g.efforts.set(p.effort, model);
  }

  const result: ProcessedModel[] = [];

  for (const g of groups.values()) {
    const hasDefault = g.efforts.has("");
    const hasMedium = g.efforts.has("medium");
    const shouldDedup = g.efforts.size >= 2 && (hasDefault || hasMedium);

    if (shouldDedup) {
      // Pick representative: prefer "medium" or default ("") for name
      const rep = g.efforts.get("medium") ?? g.efforts.get("") ?? [...g.efforts.values()][0]!;

      // Build deduped model ID: base + thinking/fast suffix (no effort)
      let id = g.base;
      if (g.thinking) id += "-thinking";
      if (g.fast) id += "-fast";

      const efforts = new Set(g.efforts.keys());
      const effortMap: Record<string, string> = {
        minimal: efforts.has("none") ? "none" : "low",
        low: "low",
        medium: hasMedium ? "medium" : "",
        high: "high",
        xhigh: efforts.has("max") ? "max" : "xhigh",
      };

      result.push({ ...rep, id, supportsEffort: true, effortMap });
    } else {
      // Keep all entries as-is
      for (const model of g.efforts.values()) {
        result.push({ ...model, supportsEffort: false });
      }
    }
  }

  return result.sort((a, b) => a.id.localeCompare(b.id));
}

function modelConfig(m: ProcessedModel) {
  return {
    id: m.id,
    name: m.name,
    reasoning: m.reasoning,
    input: ["text"] as ("text" | "image")[],
    cost: estimateModelCost(m.id),
    contextWindow: m.contextWindow,
    maxTokens: m.maxTokens,
    compat: {
      supportsDeveloperRole: false,
      supportsReasoningEffort: m.supportsEffort,
      ...(m.supportsEffort && m.effortMap && {
        reasoningEffortMap: m.effortMap,
      }),
      maxTokensField: "max_tokens" as const,
    },
  };
}


// ── Fallback models (from Cursor API, updated on login/refresh) ──

const FALLBACK_MODELS: CursorModel[] = [
  { id: "claude-4-sonnet", name: "Sonnet 4", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "claude-4-sonnet-1m", name: "Sonnet 4 1M", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "claude-4-sonnet-1m-thinking", name: "Sonnet 4 1M Thinking", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "claude-4-sonnet-thinking", name: "Sonnet 4 Thinking", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "claude-4.5-opus-high", name: "Opus 4.5", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "claude-4.5-opus-high-thinking", name: "Opus 4.5 Thinking", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "claude-4.5-sonnet", name: "Sonnet 4.5 1M", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "claude-4.5-sonnet-thinking", name: "Sonnet 4.5 1M Thinking", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "claude-4.6-opus-high", name: "Opus 4.6 1M", reasoning: true, contextWindow: 200_000, maxTokens: 128_000 },
  { id: "claude-4.6-opus-high-thinking", name: "Opus 4.6 1M Thinking", reasoning: true, contextWindow: 200_000, maxTokens: 128_000 },
  { id: "claude-4.6-opus-max", name: "Opus 4.6 1M Max", reasoning: true, contextWindow: 200_000, maxTokens: 128_000 },
  { id: "claude-4.6-opus-max-thinking", name: "Opus 4.6 1M Max Thinking", reasoning: true, contextWindow: 200_000, maxTokens: 128_000 },
  { id: "claude-4.6-sonnet-medium", name: "Sonnet 4.6 1M", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "claude-4.6-sonnet-medium-thinking", name: "Sonnet 4.6 1M Thinking", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "composer-1.5", name: "Composer 1.5", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "composer-2", name: "Composer 2", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "composer-2-fast", name: "Composer 2 Fast", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "default", name: "Auto", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "gemini-3-flash", name: "Gemini 3 Flash", reasoning: true, contextWindow: 1_000_000, maxTokens: 64_000 },
  { id: "gemini-3.1-pro", name: "Gemini 3.1 Pro", reasoning: true, contextWindow: 1_000_000, maxTokens: 64_000 },
  { id: "gpt-5-mini", name: "GPT-5 Mini", reasoning: true, contextWindow: 128_000, maxTokens: 64_000 },
  { id: "gpt-5.1", name: "GPT-5.1", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "gpt-5.1-codex-max-high", name: "GPT-5.1 Codex Max High", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "gpt-5.1-codex-max-high-fast", name: "GPT-5.1 Codex Max High Fast", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "gpt-5.1-codex-max-low", name: "GPT-5.1 Codex Max Low", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "gpt-5.1-codex-max-low-fast", name: "GPT-5.1 Codex Max Low Fast", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "gpt-5.1-codex-max-medium", name: "GPT-5.1 Codex Max", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "gpt-5.1-codex-max-medium-fast", name: "GPT-5.1 Codex Max Medium Fast", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "gpt-5.1-codex-max-xhigh", name: "GPT-5.1 Codex Max Extra High", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "gpt-5.1-codex-max-xhigh-fast", name: "GPT-5.1 Codex Max Extra High Fast", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "gpt-5.1-codex-mini", name: "GPT-5.1 Codex Mini", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "gpt-5.1-codex-mini-high", name: "GPT-5.1 Codex Mini High", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "gpt-5.1-codex-mini-low", name: "GPT-5.1 Codex Mini Low", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "gpt-5.1-high", name: "GPT-5.1 High", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "gpt-5.1-low", name: "GPT-5.1 Low", reasoning: true, contextWindow: 200_000, maxTokens: 64_000 },
  { id: "gpt-5.2", name: "GPT-5.2", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.2-codex", name: "GPT-5.2 Codex", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.2-codex-fast", name: "GPT-5.2 Codex Fast", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.2-codex-high", name: "GPT-5.2 Codex High", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.2-codex-high-fast", name: "GPT-5.2 Codex High Fast", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.2-codex-low", name: "GPT-5.2 Codex Low", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.2-codex-low-fast", name: "GPT-5.2 Codex Low Fast", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.2-codex-xhigh", name: "GPT-5.2 Codex Extra High", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.2-codex-xhigh-fast", name: "GPT-5.2 Codex Extra High Fast", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.2-fast", name: "GPT-5.2 Fast", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.2-high", name: "GPT-5.2 High", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.2-high-fast", name: "GPT-5.2 High Fast", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.2-low", name: "GPT-5.2 Low", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.2-low-fast", name: "GPT-5.2 Low Fast", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.2-xhigh", name: "GPT-5.2 Extra High", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.2-xhigh-fast", name: "GPT-5.2 Extra High Fast", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.3-codex", name: "GPT-5.3 Codex", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.3-codex-fast", name: "GPT-5.3 Codex Fast", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.3-codex-high", name: "GPT-5.3 Codex High", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.3-codex-high-fast", name: "GPT-5.3 Codex High Fast", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.3-codex-low", name: "GPT-5.3 Codex Low", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.3-codex-low-fast", name: "GPT-5.3 Codex Low Fast", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.3-codex-spark-preview", name: "GPT-5.3 Codex Spark", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.3-codex-spark-preview-high", name: "GPT-5.3 Codex Spark High", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.3-codex-spark-preview-low", name: "GPT-5.3 Codex Spark Low", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.3-codex-spark-preview-xhigh", name: "GPT-5.3 Codex Spark Extra High", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.3-codex-xhigh", name: "GPT-5.3 Codex Extra High", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.3-codex-xhigh-fast", name: "GPT-5.3 Codex Extra High Fast", reasoning: true, contextWindow: 400_000, maxTokens: 128_000 },
  { id: "gpt-5.4-high", name: "GPT-5.4 1M High", reasoning: true, contextWindow: 272_000, maxTokens: 128_000 },
  { id: "gpt-5.4-high-fast", name: "GPT-5.4 High Fast", reasoning: true, contextWindow: 272_000, maxTokens: 128_000 },
  { id: "gpt-5.4-low", name: "GPT-5.4 1M Low", reasoning: true, contextWindow: 272_000, maxTokens: 128_000 },
  { id: "gpt-5.4-medium", name: "GPT-5.4 1M", reasoning: true, contextWindow: 272_000, maxTokens: 128_000 },
  { id: "gpt-5.4-medium-fast", name: "GPT-5.4 Fast", reasoning: true, contextWindow: 272_000, maxTokens: 128_000 },
  { id: "gpt-5.4-mini-high", name: "GPT-5.4 Mini High", reasoning: true, contextWindow: 128_000, maxTokens: 64_000 },
  { id: "gpt-5.4-mini-low", name: "GPT-5.4 Mini Low", reasoning: true, contextWindow: 128_000, maxTokens: 64_000 },
  { id: "gpt-5.4-mini-medium", name: "GPT-5.4 Mini", reasoning: true, contextWindow: 128_000, maxTokens: 64_000 },
  { id: "gpt-5.4-mini-none", name: "GPT-5.4 Mini None", reasoning: true, contextWindow: 128_000, maxTokens: 64_000 },
  { id: "gpt-5.4-mini-xhigh", name: "GPT-5.4 Mini Extra High", reasoning: true, contextWindow: 128_000, maxTokens: 64_000 },
  { id: "gpt-5.4-nano-high", name: "GPT-5.4 Nano High", reasoning: true, contextWindow: 128_000, maxTokens: 64_000 },
  { id: "gpt-5.4-nano-low", name: "GPT-5.4 Nano Low", reasoning: true, contextWindow: 128_000, maxTokens: 64_000 },
  { id: "gpt-5.4-nano-medium", name: "GPT-5.4 Nano", reasoning: true, contextWindow: 128_000, maxTokens: 64_000 },
  { id: "gpt-5.4-nano-none", name: "GPT-5.4 Nano None", reasoning: true, contextWindow: 128_000, maxTokens: 64_000 },
  { id: "gpt-5.4-nano-xhigh", name: "GPT-5.4 Nano Extra High", reasoning: true, contextWindow: 128_000, maxTokens: 64_000 },
  { id: "gpt-5.4-xhigh", name: "GPT-5.4 1M Extra High", reasoning: true, contextWindow: 272_000, maxTokens: 128_000 },
  { id: "gpt-5.4-xhigh-fast", name: "GPT-5.4 Extra High Fast", reasoning: true, contextWindow: 272_000, maxTokens: 128_000 },
  { id: "grok-4-20", name: "Grok 4.20", reasoning: true, contextWindow: 128_000, maxTokens: 64_000 },
  { id: "grok-4-20-thinking", name: "Grok 4.20 Thinking", reasoning: true, contextWindow: 128_000, maxTokens: 64_000 },
  { id: "kimi-k2.5", name: "Kimi K2.5", reasoning: true, contextWindow: 128_000, maxTokens: 64_000 },
];

// ── Extension ──

export default function (pi: ExtensionAPI) {
  // Current access token, updated by login/refresh/getApiKey
  let currentToken = "";

  // Start proxy eagerly — it just binds a port, no auth needed until a request arrives.
  // The getAccessToken callback reads currentToken at request time.
  const proxyReady = startProxy(async () => {
    if (!currentToken) throw new Error("Not logged in to Cursor. Run /login cursor");
    return currentToken;
  });

  // Register with fallback models immediately so they appear in /model
  register(pi, 0, FALLBACK_MODELS);
  proxyReady.then((port) => {
    register(pi, port, FALLBACK_MODELS);
  }).catch(() => {});

  function register(pi: ExtensionAPI, port: number, rawModels: CursorModel[]) {
    const baseUrl = port > 0 ? `http://127.0.0.1:${port}/v1` : "http://localhost:1";
    const processed = processModels(rawModels);

    pi.registerProvider("cursor", {
      baseUrl,
      apiKey: "cursor-proxy",
      api: "openai-completions",
      models: processed.map(modelConfig),
      oauth: {
        name: "Cursor",

        async login(callbacks: OAuthLoginCallbacks): Promise<OAuthCredentials> {
          const { verifier, uuid, loginUrl } = await generateCursorAuthParams();
          callbacks.onAuth({ url: loginUrl });
          const { accessToken, refreshToken } = await pollCursorAuth(uuid, verifier);
          currentToken = accessToken;

          // Discover real models and re-register
          const realPort = await proxyReady;
          const discovered = await getCursorModels(accessToken);
          if (discovered.length > 0) register(pi, realPort, discovered);

          return {
            refresh: refreshToken,
            access: accessToken,
            expires: getTokenExpiry(accessToken),
          };
        },

        async refreshToken(credentials: OAuthCredentials): Promise<OAuthCredentials> {
          const refreshed = await refreshCursorToken(credentials.refresh);
          currentToken = refreshed.access;

          // Discover real models on refresh too
          const realPort = await proxyReady;
          const discovered = await getCursorModels(refreshed.access);
          if (discovered.length > 0) register(pi, realPort, discovered);

          return refreshed;
        },

        getApiKey(credentials: OAuthCredentials): string {
          currentToken = credentials.access;
          return "cursor-proxy";
        },
      },
    });
  }


}
