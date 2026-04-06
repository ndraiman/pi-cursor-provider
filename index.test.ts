import { describe, expect, test } from "bun:test";
import { parseModelId, processModels } from "./index.ts";
import { resolveModelId } from "./proxy.ts";
import type { CursorModel } from "./proxy.ts";

// ── Helper ──

function m(id: string, name?: string): CursorModel {
  return { id, name: name ?? id, reasoning: true, contextWindow: 200_000, maxTokens: 64_000 };
}

// ── parseModelId ──

describe("parseModelId", () => {
  test("plain model — no effort, no variant", () => {
    expect(parseModelId("composer-2")).toEqual({ base: "composer-2", effort: "", fast: false, thinking: false });
  });

  test("plain model with -fast suffix", () => {
    expect(parseModelId("composer-2-fast")).toEqual({ base: "composer-2", effort: "", fast: true, thinking: false });
  });

  test("model with effort suffix", () => {
    expect(parseModelId("gpt-5.4-medium")).toEqual({ base: "gpt-5.4", effort: "medium", fast: false, thinking: false });
  });

  test("model with effort + fast", () => {
    expect(parseModelId("gpt-5.4-high-fast")).toEqual({ base: "gpt-5.4", effort: "high", fast: true, thinking: false });
  });

  test("model with effort + thinking", () => {
    expect(parseModelId("claude-4.6-opus-high-thinking")).toEqual({ base: "claude-4.6-opus", effort: "high", fast: false, thinking: true });
  });

  test("max effort level", () => {
    expect(parseModelId("claude-4.6-opus-max")).toEqual({ base: "claude-4.6-opus", effort: "max", fast: false, thinking: false });
  });

  test("max effort + thinking", () => {
    expect(parseModelId("claude-4.6-opus-max-thinking")).toEqual({ base: "claude-4.6-opus", effort: "max", fast: false, thinking: true });
  });

  test("none effort level", () => {
    expect(parseModelId("gpt-5.4-mini-none")).toEqual({ base: "gpt-5.4-mini", effort: "none", fast: false, thinking: false });
  });

  test("xhigh effort", () => {
    expect(parseModelId("gpt-5.2-xhigh")).toEqual({ base: "gpt-5.2", effort: "xhigh", fast: false, thinking: false });
  });

  test("xhigh effort + fast", () => {
    expect(parseModelId("gpt-5.2-xhigh-fast")).toEqual({ base: "gpt-5.2", effort: "xhigh", fast: true, thinking: false });
  });

  test("codex-max model — max is part of base, not effort", () => {
    expect(parseModelId("gpt-5.1-codex-max-high")).toEqual({ base: "gpt-5.1-codex-max", effort: "high", fast: false, thinking: false });
  });

  test("codex-max + fast", () => {
    expect(parseModelId("gpt-5.1-codex-max-medium-fast")).toEqual({ base: "gpt-5.1-codex-max", effort: "medium", fast: true, thinking: false });
  });

  test("codex-mini model", () => {
    expect(parseModelId("gpt-5.1-codex-mini-high")).toEqual({ base: "gpt-5.1-codex-mini", effort: "high", fast: false, thinking: false });
  });

  test("spark-preview model", () => {
    expect(parseModelId("gpt-5.3-codex-spark-preview-high")).toEqual({ base: "gpt-5.3-codex-spark-preview", effort: "high", fast: false, thinking: false });
  });

  test("plain thinking model — no effort", () => {
    expect(parseModelId("grok-4-20-thinking")).toEqual({ base: "grok-4-20", effort: "", fast: false, thinking: true });
  });

  test("model without any suffix", () => {
    expect(parseModelId("kimi-k2.5")).toEqual({ base: "kimi-k2.5", effort: "", fast: false, thinking: false });
  });

  test("default model", () => {
    expect(parseModelId("default")).toEqual({ base: "default", effort: "", fast: false, thinking: false });
  });

  test("claude-4.6-sonnet-medium — effort is medium", () => {
    expect(parseModelId("claude-4.6-sonnet-medium")).toEqual({ base: "claude-4.6-sonnet", effort: "medium", fast: false, thinking: false });
  });

  test("claude-4.6-sonnet-medium-thinking", () => {
    expect(parseModelId("claude-4.6-sonnet-medium-thinking")).toEqual({ base: "claude-4.6-sonnet", effort: "medium", fast: false, thinking: true });
  });
});

// ── processModels ──

describe("processModels", () => {
  test("composer-2 — no effort variants, kept as-is", () => {
    const result = processModels([m("composer-2"), m("composer-2-fast")]);
    const c2 = result.find(r => r.id === "composer-2");
    const c2f = result.find(r => r.id === "composer-2-fast");
    expect(c2).toBeDefined();
    expect(c2!.supportsEffort).toBe(false);
    expect(c2f).toBeDefined();
    expect(c2f!.supportsEffort).toBe(false);
  });

  test("gpt-5.4 — deduped from low/medium/high/xhigh", () => {
    const result = processModels([
      m("gpt-5.4-low"), m("gpt-5.4-medium"), m("gpt-5.4-high"), m("gpt-5.4-xhigh"),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("gpt-5.4");
    expect(result[0].supportsEffort).toBe(true);
    expect(result[0].effortMap!.medium).toBe("medium");
    expect(result[0].effortMap!.xhigh).toBe("xhigh");
  });

  test("gpt-5.4-fast — deduped from effort+fast variants", () => {
    const result = processModels([
      m("gpt-5.4-high-fast"), m("gpt-5.4-medium-fast"), m("gpt-5.4-xhigh-fast"),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("gpt-5.4-fast");
    expect(result[0].supportsEffort).toBe(true);
  });

  test("gpt-5.2 — deduped from default + effort variants", () => {
    const result = processModels([
      m("gpt-5.2"), m("gpt-5.2-high"), m("gpt-5.2-low"), m("gpt-5.2-xhigh"),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("gpt-5.2");
    expect(result[0].supportsEffort).toBe(true);
    expect(result[0].effortMap!.medium).toBe(""); // no-suffix = default
    expect(result[0].effortMap!.high).toBe("high");
  });

  test("gpt-5.4-mini — has none effort", () => {
    const result = processModels([
      m("gpt-5.4-mini-low"), m("gpt-5.4-mini-medium"), m("gpt-5.4-mini-high"),
      m("gpt-5.4-mini-xhigh"), m("gpt-5.4-mini-none"),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("gpt-5.4-mini");
    expect(result[0].supportsEffort).toBe(true);
    expect(result[0].effortMap!.minimal).toBe("none");
  });

  test("claude-4.6-opus — only high+max, no default/medium → NOT deduped", () => {
    const result = processModels([
      m("claude-4.6-opus-high"), m("claude-4.6-opus-max"),
    ]);
    expect(result).toHaveLength(2);
    expect(result.every(r => r.supportsEffort === false)).toBe(true);
  });

  test("claude-4.6-opus-thinking — only high+max thinking, NOT deduped", () => {
    const result = processModels([
      m("claude-4.6-opus-high-thinking"), m("claude-4.6-opus-max-thinking"),
    ]);
    expect(result).toHaveLength(2);
    expect(result.every(r => r.supportsEffort === false)).toBe(true);
  });

  test("claude-4.5-opus-high — single variant, NOT deduped", () => {
    const result = processModels([m("claude-4.5-opus-high")]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("claude-4.5-opus-high");
    expect(result[0].supportsEffort).toBe(false);
  });

  test("claude-4.6-sonnet-medium — single variant, NOT deduped", () => {
    const result = processModels([m("claude-4.6-sonnet-medium")]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("claude-4.6-sonnet-medium");
    expect(result[0].supportsEffort).toBe(false);
  });

  test("gpt-5.1-codex-max — deduped, max stays in base name", () => {
    const result = processModels([
      m("gpt-5.1-codex-max-low"), m("gpt-5.1-codex-max-medium"),
      m("gpt-5.1-codex-max-high"), m("gpt-5.1-codex-max-xhigh"),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("gpt-5.1-codex-max");
    expect(result[0].supportsEffort).toBe(true);
  });

  test("gpt-5.3-codex-spark-preview — deduped", () => {
    const result = processModels([
      m("gpt-5.3-codex-spark-preview"), m("gpt-5.3-codex-spark-preview-high"),
      m("gpt-5.3-codex-spark-preview-low"), m("gpt-5.3-codex-spark-preview-xhigh"),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("gpt-5.3-codex-spark-preview");
    expect(result[0].supportsEffort).toBe(true);
  });

  test("standalone models pass through", () => {
    const result = processModels([
      m("default"), m("gemini-3-flash"), m("kimi-k2.5"), m("grok-4-20"), m("grok-4-20-thinking"),
    ]);
    expect(result).toHaveLength(5);
    expect(result.every(r => r.supportsEffort === false)).toBe(true);
  });

  test("uses representative name from medium variant", () => {
    const result = processModels([
      m("gpt-5.4-low", "GPT-5.4 1M Low"),
      m("gpt-5.4-medium", "GPT-5.4 1M"),
      m("gpt-5.4-high", "GPT-5.4 1M High"),
    ]);
    expect(result[0].name).toBe("GPT-5.4 1M");
  });

  test("uses representative name from default (no-suffix) variant", () => {
    const result = processModels([
      m("gpt-5.2", "GPT-5.2"),
      m("gpt-5.2-high", "GPT-5.2 High"),
      m("gpt-5.2-low", "GPT-5.2 Low"),
    ]);
    expect(result[0].name).toBe("GPT-5.2");
  });

  test("full raw model list dedup count", () => {
    const raw: CursorModel[] = require("./cursor-models-raw.json");
    const result = processModels(raw);
    // Should be significantly fewer than 83
    expect(result.length).toBeLessThan(50);
    expect(result.length).toBeGreaterThan(30);

    // Spot checks
    const composer2 = result.find(r => r.id === "composer-2");
    expect(composer2).toBeDefined();
    expect(composer2!.supportsEffort).toBe(false);

    const gpt54 = result.find(r => r.id === "gpt-5.4");
    expect(gpt54).toBeDefined();
    expect(gpt54!.supportsEffort).toBe(true);

    // No raw effort IDs should leak through for deduped models
    expect(result.find(r => r.id === "gpt-5.4-medium")).toBeUndefined();
    expect(result.find(r => r.id === "gpt-5.4-high")).toBeUndefined();
    expect(result.find(r => r.id === "gpt-5.2-low")).toBeUndefined();
  });
});

// ── resolveModelId ──

describe("resolveModelId", () => {
  test("no effort — returns model as-is", () => {
    expect(resolveModelId("composer-2")).toBe("composer-2");
    expect(resolveModelId("composer-2", undefined)).toBe("composer-2");
    expect(resolveModelId("composer-2", "")).toBe("composer-2");
  });

  test("plain model + effort", () => {
    expect(resolveModelId("gpt-5.4", "medium")).toBe("gpt-5.4-medium");
    expect(resolveModelId("gpt-5.4", "high")).toBe("gpt-5.4-high");
    expect(resolveModelId("gpt-5.4", "xhigh")).toBe("gpt-5.4-xhigh");
  });

  test("fast model + effort — inserts before -fast", () => {
    expect(resolveModelId("gpt-5.4-fast", "medium")).toBe("gpt-5.4-medium-fast");
    expect(resolveModelId("gpt-5.4-fast", "high")).toBe("gpt-5.4-high-fast");
  });

  test("thinking model + effort — inserts before -thinking", () => {
    expect(resolveModelId("claude-4.6-opus-thinking", "high")).toBe("claude-4.6-opus-high-thinking");
    expect(resolveModelId("claude-4.6-opus-thinking", "max")).toBe("claude-4.6-opus-max-thinking");
  });

  test("codex-max model + effort", () => {
    expect(resolveModelId("gpt-5.1-codex-max", "high")).toBe("gpt-5.1-codex-max-high");
    expect(resolveModelId("gpt-5.1-codex-max", "medium")).toBe("gpt-5.1-codex-max-medium");
  });

  test("codex-max-fast model + effort", () => {
    expect(resolveModelId("gpt-5.1-codex-max-fast", "high")).toBe("gpt-5.1-codex-max-high-fast");
  });

  test("spark-preview model + effort", () => {
    expect(resolveModelId("gpt-5.3-codex-spark-preview", "xhigh")).toBe("gpt-5.3-codex-spark-preview-xhigh");
  });
});
