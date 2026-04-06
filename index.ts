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
import { getCursorModels, startProxy, stopProxy, type CursorModel } from "./proxy.js";

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

// ── Extension ──

let proxyStarted = false;

export default function (pi: ExtensionAPI) {
  pi.registerProvider("cursor", {
    baseUrl: "http://localhost:1", // placeholder, updated after proxy starts
    apiKey: "cursor-proxy",
    api: "openai-completions",
    models: [], // populated dynamically after login
    oauth: {
      name: "Cursor",

      async login(callbacks: OAuthLoginCallbacks): Promise<OAuthCredentials> {
        const { verifier, uuid, loginUrl } = await generateCursorAuthParams();
        callbacks.onAuth({ url: loginUrl });

        const { accessToken, refreshToken } = await pollCursorAuth(uuid, verifier);

        // Discover models and start proxy
        await initProxy(pi, accessToken);

        return {
          refresh: refreshToken,
          access: accessToken,
          expires: getTokenExpiry(accessToken),
        };
      },

      async refreshToken(credentials: OAuthCredentials): Promise<OAuthCredentials> {
        const refreshed = await refreshCursorToken(credentials.refresh);

        // Ensure proxy is running after token refresh
        if (!proxyStarted) {
          await initProxy(pi, refreshed.access);
        }

        return refreshed;
      },

      getApiKey(credentials: OAuthCredentials): string {
        // Ensure proxy is initialized if not yet
        if (!proxyStarted) {
          initProxy(pi, credentials.access).catch(() => {});
        }
        return credentials.access;
      },
    },
  });
}

async function initProxy(pi: ExtensionAPI, accessToken: string): Promise<void> {
  if (proxyStarted) return;

  // Discover available models
  const cursorModels = await getCursorModels(accessToken);

  // Start the local proxy
  let currentToken = accessToken;
  const port = await startProxy(async () => currentToken);

  proxyStarted = true;

  // Re-register provider with actual models and proxy URL
  pi.registerProvider("cursor", {
    baseUrl: `http://127.0.0.1:${port}/v1`,
    apiKey: "cursor-proxy",
    api: "openai-completions",
    models: cursorModels.map((m) => ({
      id: m.id,
      name: m.name,
      reasoning: m.reasoning,
      input: ["text"] as ("text" | "image")[],
      cost: estimateModelCost(m.id),
      contextWindow: m.contextWindow,
      maxTokens: m.maxTokens,
      compat: {
        supportsDeveloperRole: false,
        supportsReasoningEffort: false,
        maxTokensField: "max_tokens" as const,
      },
    })),
    oauth: {
      name: "Cursor",

      async login(callbacks: OAuthLoginCallbacks): Promise<OAuthCredentials> {
        const { verifier, uuid, loginUrl } = await generateCursorAuthParams();
        callbacks.onAuth({ url: loginUrl });
        const { accessToken: newToken, refreshToken } = await pollCursorAuth(uuid, verifier);
        currentToken = newToken;
        return {
          refresh: refreshToken,
          access: newToken,
          expires: getTokenExpiry(newToken),
        };
      },

      async refreshToken(credentials: OAuthCredentials): Promise<OAuthCredentials> {
        const refreshed = await refreshCursorToken(credentials.refresh);
        currentToken = refreshed.access;
        return refreshed;
      },

      getApiKey(credentials: OAuthCredentials): string {
        currentToken = credentials.access;
        return "cursor-proxy"; // proxy handles auth internally
      },
    },
  });
}
