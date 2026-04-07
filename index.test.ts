import rawModels from "./cursor-models-raw.json" with { type: "json" };
import { describe, expect, test } from "bun:test";
import { buildEffortMap, FALLBACK_MODELS, parseModelId, processModels, supportsReasoningModelId } from "./index.ts";
import { resolveModelId, deriveBridgeKey, deriveConversationKey, deterministicConversationId, buildCursorRequest, parseMessages } from "./proxy.ts";
import type { CursorModel, StoredConversation } from "./proxy.ts";
import { fromBinary, toBinary } from "@bufbuild/protobuf";
import {
  AgentClientMessageSchema,
  AgentRunRequestSchema,
  ConversationStateStructureSchema,
  ConversationTurnStructureSchema,
  AgentConversationTurnStructureSchema,
  ConversationStepSchema,
  UserMessageSchema,
  AssistantMessageSchema,
} from "./proto/agent_pb.ts";

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

// ── buildEffortMap ──

describe("buildEffortMap", () => {
  test("full range: none/low/medium/high/xhigh", () => {
    const map = buildEffortMap(new Set(["none", "low", "medium", "high", "xhigh"]));
    expect(map).toEqual({ minimal: "none", low: "low", medium: "medium", high: "high", xhigh: "xhigh" });
  });

  test("with default (empty) and medium", () => {
    const map = buildEffortMap(new Set(["", "low", "medium", "high"]));
    expect(map).toEqual({ minimal: "low", low: "low", medium: "medium", high: "high", xhigh: "high" });
  });

  test("default without medium — medium maps to empty", () => {
    const map = buildEffortMap(new Set(["", "low", "high", "xhigh"]));
    expect(map.medium).toBe("");
  });

  test("high+max only — all lower levels clamp to high", () => {
    const map = buildEffortMap(new Set(["high", "max"]));
    expect(map).toEqual({ minimal: "high", low: "high", medium: "high", high: "high", xhigh: "max" });
  });

  test("none+low+medium+high+max", () => {
    const map = buildEffortMap(new Set(["none", "low", "medium", "high", "max"]));
    expect(map).toEqual({ minimal: "none", low: "low", medium: "medium", high: "high", xhigh: "max" });
  });

  test("low+high — medium falls back to low", () => {
    const map = buildEffortMap(new Set(["low", "high"]));
    expect(map).toEqual({ minimal: "low", low: "low", medium: "low", high: "high", xhigh: "high" });
  });
});

// ── processModels ──

describe("reasoning support", () => {
  test("derives reasoning from model ids", () => {
    expect(supportsReasoningModelId("gpt-5.4")).toBe(true);
    expect(supportsReasoningModelId("gpt-5.4-fast")).toBe(true);
    expect(supportsReasoningModelId("composer-2")).toBe(true);
    expect(supportsReasoningModelId("default")).toBe(true);
    expect(supportsReasoningModelId("totally-unknown-model")).toBe(false);
  });

  test("fallback models keep derived reasoning enabled", () => {
    expect(FALLBACK_MODELS.length).toBeGreaterThan(0);
    expect(FALLBACK_MODELS.find((model) => model.id === "gpt-5.4-medium")?.reasoning).toBe(true);
    expect(FALLBACK_MODELS.find((model) => model.id === "composer-2")?.reasoning).toBe(true);
  });
});

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

  test("claude-4.6-opus — high+max deduped, effort clamped to lowest", () => {
    const result = processModels([
      m("claude-4.6-opus-high"), m("claude-4.6-opus-max"),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("claude-4.6-opus");
    expect(result[0].supportsEffort).toBe(true);
    expect(result[0].effortMap!.minimal).toBe("high");
    expect(result[0].effortMap!.low).toBe("high");
    expect(result[0].effortMap!.medium).toBe("high");
    expect(result[0].effortMap!.high).toBe("high");
    expect(result[0].effortMap!.xhigh).toBe("max");
  });

  test("claude-4.6-opus-thinking — high+max thinking deduped", () => {
    const result = processModels([
      m("claude-4.6-opus-high-thinking"), m("claude-4.6-opus-max-thinking"),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("claude-4.6-opus-thinking");
    expect(result[0].supportsEffort).toBe(true);
    expect(result[0].effortMap!.high).toBe("high");
    expect(result[0].effortMap!.xhigh).toBe("max");
  });

  test("claude-4.5-opus-high — single effort variant, deduped to base", () => {
    const result = processModels([m("claude-4.5-opus-high")]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("claude-4.5-opus");
    expect(result[0].supportsEffort).toBe(true);
    expect(result[0].effortMap!.high).toBe("high");
    expect(result[0].effortMap!.minimal).toBe("high");
  });

  test("claude-4.6-sonnet-medium — single effort variant, deduped to base", () => {
    const result = processModels([m("claude-4.6-sonnet-medium")]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("claude-4.6-sonnet");
    expect(result[0].supportsEffort).toBe(true);
    expect(result[0].effortMap!.medium).toBe("medium");
  });

  test("composer-2 — single model without effort, NOT deduped", () => {
    const result = processModels([m("composer-2")]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("composer-2");
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
    const result = processModels(rawModels as CursorModel[]);
    // Should be significantly fewer than 83
    expect(result.length).toBeLessThan(50);
    expect(result.length).toBeGreaterThan(20);

    // Spot checks
    const composer2 = result.find(r => r.id === "composer-2");
    expect(composer2).toBeDefined();
    expect(composer2!.supportsEffort).toBe(false);

    const gpt54 = result.find(r => r.id === "gpt-5.4");
    expect(gpt54).toBeDefined();
    expect(gpt54!.supportsEffort).toBe(true);

    // Opus should be deduped too
    const opus46 = result.find(r => r.id === "claude-4.6-opus");
    expect(opus46).toBeDefined();
    expect(opus46!.supportsEffort).toBe(true);
    expect(result.find(r => r.id === "claude-4.6-opus-high")).toBeUndefined();
    expect(result.find(r => r.id === "claude-4.6-opus-max")).toBeUndefined();

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

// ── Session key derivation ──

const msg = (role: "user" | "assistant" | "system", content: string) => ({ role, content });

describe("deriveBridgeKey", () => {
  test("uses sessionId when provided", () => {
    const msgs = [msg("user", "hello")];
    const a = deriveBridgeKey("gpt-5", msgs, "session-abc");
    const b = deriveBridgeKey("gpt-5", msgs, "session-abc");
    expect(a).toBe(b);
  });

  test("different sessionIds produce different keys", () => {
    const msgs = [msg("user", "hello")];
    const a = deriveBridgeKey("gpt-5", msgs, "session-1");
    const b = deriveBridgeKey("gpt-5", msgs, "session-2");
    expect(a).not.toBe(b);
  });

  test("different models produce different keys", () => {
    const msgs = [msg("user", "hello")];
    const a = deriveBridgeKey("gpt-5", msgs, "session-1");
    const b = deriveBridgeKey("claude-4", msgs, "session-1");
    expect(a).not.toBe(b);
  });

  test("falls back to first user message hash without sessionId", () => {
    const msgs1 = [msg("user", "hello")];
    const msgs2 = [msg("user", "hello"), msg("assistant", "hi"), msg("user", "bye")];
    // Same first user message → same key
    expect(deriveBridgeKey("gpt-5", msgs1)).toBe(deriveBridgeKey("gpt-5", msgs2));
  });

  test("fallback differs by first user message", () => {
    const a = deriveBridgeKey("gpt-5", [msg("user", "hello")]);
    const b = deriveBridgeKey("gpt-5", [msg("user", "goodbye")]);
    expect(a).not.toBe(b);
  });
});

describe("deriveConversationKey", () => {
  test("same sessionId → same key regardless of messages", () => {
    const a = deriveConversationKey([msg("user", "hello")], "session-x");
    const b = deriveConversationKey([msg("user", "totally different")], "session-x");
    expect(a).toBe(b);
  });

  test("different sessionIds → different keys", () => {
    const a = deriveConversationKey([msg("user", "hello")], "session-1");
    const b = deriveConversationKey([msg("user", "hello")], "session-2");
    expect(a).not.toBe(b);
  });

  test("falls back to first user message hash without sessionId", () => {
    const a = deriveConversationKey([msg("user", "hello")]);
    const b = deriveConversationKey([msg("user", "hello"), msg("assistant", "hi")]);
    expect(a).toBe(b);
  });
});

// ── Turn reconstruction ──

function decodeRunRequest(payload: ReturnType<typeof buildCursorRequest>) {
  const clientMsg = fromBinary(AgentClientMessageSchema, payload.requestBytes);
  expect(clientMsg.message.case).toBe("runRequest");
  return clientMsg.message.value as InstanceType<typeof AgentRunRequestSchema["$typeName"]> & any;
}

function decodeTurns(state: any) {
  return (state.turns as Uint8Array[]).map((turnBytes: Uint8Array) => {
    const turnStruct = fromBinary(ConversationTurnStructureSchema, turnBytes);
    expect(turnStruct.turn.case).toBe("agentConversationTurn");
    const agentTurn = turnStruct.turn.value as any;
    const userMsg = fromBinary(UserMessageSchema, agentTurn.userMessage);
    const steps = (agentTurn.steps as Uint8Array[]).map((s: Uint8Array) => fromBinary(ConversationStepSchema, s));
    return { userMsg, steps };
  });
}

describe("buildCursorRequest — turn reconstruction", () => {
  test("no checkpoint, no turns — empty turns array", () => {
    const payload = buildCursorRequest("gpt-5", "system", "hello", [], "conv-1", null);
    const req = decodeRunRequest(payload);
    expect(req.conversationState.turns).toHaveLength(0);
    // User message is the action
    const userAction = req.action.action.value as any;
    expect(userAction.userMessage.text).toBe("hello");
  });

  test("no checkpoint, with turns — reconstructs protobuf turns", () => {
    const turns = [
      { userText: "first question", assistantText: "first answer" },
      { userText: "second question", assistantText: "second answer" },
    ];
    const payload = buildCursorRequest("gpt-5", "system", "third question", turns, "conv-1", null);
    const req = decodeRunRequest(payload);

    // 2 reconstructed turns
    const decoded = decodeTurns(req.conversationState);
    expect(decoded).toHaveLength(2);

    expect(decoded[0].userMsg.text).toBe("first question");
    expect(decoded[0].steps).toHaveLength(1);
    expect(decoded[0].steps[0].message.case).toBe("assistantMessage");
    expect((decoded[0].steps[0].message.value as any).text).toBe("first answer");

    expect(decoded[1].userMsg.text).toBe("second question");
    expect(decoded[1].steps[0].message.case).toBe("assistantMessage");
    expect((decoded[1].steps[0].message.value as any).text).toBe("second answer");

    // Current message includes inlined history as text fallback
    const userAction = req.action.action.value as any;
    expect(userAction.userMessage.text).toContain("<conversation_history>");
    expect(userAction.userMessage.text).toContain("first question");
    expect(userAction.userMessage.text).toContain("first answer");
    expect(userAction.userMessage.text).toEndWith("third question");
  });

  test("no checkpoint, turn with empty assistant — no steps", () => {
    const turns = [{ userText: "hello", assistantText: "" }];
    const payload = buildCursorRequest("gpt-5", "system", "follow up", turns, "conv-1", null);
    const req = decodeRunRequest(payload);
    const decoded = decodeTurns(req.conversationState);
    expect(decoded).toHaveLength(1);
    expect(decoded[0].userMsg.text).toBe("hello");
    expect(decoded[0].steps).toHaveLength(0);
  });

  test("with checkpoint — uses checkpoint, ignores turns", () => {
    // Build a checkpoint from a known conversation
    const priorPayload = buildCursorRequest("gpt-5", "system", "hello", [], "conv-1", null);
    const priorReq = decodeRunRequest(priorPayload);
    const checkpoint = toBinary(ConversationStateStructureSchema, priorReq.conversationState);

    // Now pass turns that differ — checkpoint should win
    const turns = [{ userText: "SHOULD NOT APPEAR", assistantText: "SHOULD NOT APPEAR" }];
    const payload = buildCursorRequest("gpt-5", "system", "next", turns, "conv-1", checkpoint);
    const req = decodeRunRequest(payload);

    // Should have the checkpoint's turns (0), not the passed-in turns (1)
    expect(req.conversationState.turns).toHaveLength(0);
  });

  test("system prompt stored in blobStore", () => {
    const payload = buildCursorRequest("gpt-5", "You are helpful", "hi", [], "conv-1", null);
    // rootPromptMessagesJson should have one blob ID
    const req = decodeRunRequest(payload);
    expect(req.conversationState.rootPromptMessagesJson).toHaveLength(1);
    // The blob should be in the blobStore
    const blobId = Buffer.from(req.conversationState.rootPromptMessagesJson[0]).toString("hex");
    expect(payload.blobStore.has(blobId)).toBe(true);
    const blobData = JSON.parse(new TextDecoder().decode(payload.blobStore.get(blobId)!));
    expect(blobData.role).toBe("system");
    expect(blobData.content).toBe("You are helpful");
  });

  test("each reconstructed turn has a unique messageId", () => {
    const turns = [
      { userText: "a", assistantText: "b" },
      { userText: "a", assistantText: "b" },
    ];
    const payload = buildCursorRequest("gpt-5", "system", "c", turns, "conv-1", null);
    const req = decodeRunRequest(payload);
    const decoded = decodeTurns(req.conversationState);
    expect(decoded[0].userMsg.messageId).not.toBe(decoded[1].userMsg.messageId);
  });
});

// ── Fork via checkpoint discard + reconstruction ──

describe("fork discards checkpoint, reconstruction takes over", () => {
  test("fork scenario — checkpoint discarded, turns reconstructed from messages", () => {
    // Simulate: 2-turn conversation, fork back to 1 turn
    // After fork, checkpoint is null → buildCursorRequest reconstructs from turns
    const turns = [{ userText: "first", assistantText: "response1" }];
    const payload = buildCursorRequest("gpt-5", "system", "forked question", turns, "conv-1", null);
    const req = decodeRunRequest(payload);

    const decoded = decodeTurns(req.conversationState);
    expect(decoded).toHaveLength(1);
    expect(decoded[0].userMsg.text).toBe("first");
    expect((decoded[0].steps[0].message.value as any).text).toBe("response1");

    // Current message includes inlined history as text fallback
    const userAction = req.action.action.value as any;
    expect(userAction.userMessage.text).toContain("<conversation_history>");
    expect(userAction.userMessage.text).toContain("first");
    expect(userAction.userMessage.text).toContain("response1");
    expect(userAction.userMessage.text).toEndWith("forked question");
  });

  test("fork to beginning — no turns, no reconstruction", () => {
    const payload = buildCursorRequest("gpt-5", "system", "start over", [], "conv-1", null);
    const req = decodeRunRequest(payload);
    expect(req.conversationState.turns).toHaveLength(0);
    const userAction = req.action.action.value as any;
    expect(userAction.userMessage.text).toBe("start over");
  });
});

// ── Tool call turnCount inflation ──

describe("tool call checkpoint turnCount", () => {
  /**
   * Simulates a full turn with tool calls and verifies the checkpoint's
   * turnCount stays consistent so the next normal request matches.
   *
   * Flow:
   *   1. Initial request: [system, user1] → turnCount=0
   *      → checkpoint stores checkpointTurnCount = 0+1 = 1
   *   2. Tool result request: [system, user1, assistant(tool_calls), tool(result)]
   *      → parseMessages sees {user1, ""} as a pair → turnCount=1
   *      → handleToolResultResume passes turnCount-1=0 to writeSSEStream
   *      → checkpoint stores 0+1 = 1 (unchanged)
   *   3. Next request: [system, user1, assistant(final), user2]
   *      → parseMessages sees {user1, final} as a pair → turnCount=1
   *      → 1 == 1 → checkpoint reused ✓
   *
   * BUG (before fix): step 2 passed raw turnCount=1 → stored 1+1=2
   *   → step 3: turnCount=1 ≠ 2 → false fork → checkpoint discarded every turn
   */

  test("parseMessages: tool result request inflates turnCount vs initial request", () => {
    // Step 1: initial request
    const initialMsgs = [
      { role: "system" as const, content: "system" },
      { role: "user" as const, content: "read file X" },
    ];
    const initial = parseMessages(initialMsgs);
    expect(initial.turns).toHaveLength(0);
    expect(initial.userText).toBe("read file X");
    const initialTurnCount = initial.turns.length; // 0

    // Step 2: tool result request — pi adds assistant(tool_calls) + tool(result)
    const toolResultMsgs = [
      { role: "system" as const, content: "system" },
      { role: "user" as const, content: "read file X" },
      { role: "assistant" as const, content: null, tool_calls: [{ id: "tc1", type: "function" as const, function: { name: "read", arguments: '{"path":"X"}' } }] },
      { role: "tool" as const, content: "file contents here", tool_call_id: "tc1" },
    ];
    const toolResult = parseMessages(toolResultMsgs);
    const toolResultTurnCount = toolResult.turns.length; // 1 — inflated!

    // Prove the inflation: tool result request sees 1 more turn than initial
    expect(toolResultTurnCount).toBe(initialTurnCount + 1);

    // Step 3: next normal request — pi includes the final assistant response
    const nextMsgs = [
      { role: "system" as const, content: "system" },
      { role: "user" as const, content: "read file X" },
      { role: "assistant" as const, content: "Here is file X..." },
      { role: "user" as const, content: "now do Y" },
    ];
    const next = parseMessages(nextMsgs);
    const nextTurnCount = next.turns.length; // 1

    // The next request's turnCount matches the initial's checkpointTurnCount (initial + 1)
    expect(nextTurnCount).toBe(initialTurnCount + 1);

    // BUG: if we stored checkpointTurnCount = toolResultTurnCount + 1 = 2,
    // then nextTurnCount (1) ≠ 2 → false fork detection
    const buggyCheckpointTurnCount = toolResultTurnCount + 1; // 2
    expect(nextTurnCount).not.toBe(buggyCheckpointTurnCount); // MISMATCH — proves the bug

    // FIX: subtract 1 in tool result path → checkpointTurnCount = (toolResultTurnCount - 1) + 1 = 1
    const fixedCheckpointTurnCount = (toolResultTurnCount - 1) + 1; // 1
    expect(nextTurnCount).toBe(fixedCheckpointTurnCount); // MATCH — proves the fix
  });

  test("multi-turn: tool call on 3rd turn doesn't break 4th turn", () => {
    // After 2 completed turns, 3rd turn has tool calls
    const initialMsgs = [
      { role: "system" as const, content: "sys" },
      { role: "user" as const, content: "u1" },
      { role: "assistant" as const, content: "a1" },
      { role: "user" as const, content: "u2" },
      { role: "assistant" as const, content: "a2" },
      { role: "user" as const, content: "u3" },
    ];
    const initial = parseMessages(initialMsgs);
    expect(initial.turns.length).toBe(2); // turnCount=2

    // Tool result request for turn 3
    const toolResultMsgs = [
      ...initialMsgs.slice(0, -1), // up to u2/a2
      { role: "user" as const, content: "u3" },
      { role: "assistant" as const, content: null, tool_calls: [{ id: "t1", type: "function" as const, function: { name: "bash", arguments: '{}' } }] },
      { role: "tool" as const, content: "output", tool_call_id: "t1" },
    ];
    const toolResult = parseMessages(toolResultMsgs);
    expect(toolResult.turns.length).toBe(3); // inflated

    // 4th turn normal request
    const nextMsgs = [
      { role: "system" as const, content: "sys" },
      { role: "user" as const, content: "u1" },
      { role: "assistant" as const, content: "a1" },
      { role: "user" as const, content: "u2" },
      { role: "assistant" as const, content: "a2" },
      { role: "user" as const, content: "u3" },
      { role: "assistant" as const, content: "a3 with tool results" },
      { role: "user" as const, content: "u4" },
    ];
    const next = parseMessages(nextMsgs);
    expect(next.turns.length).toBe(3); // turnCount=3

    // Bug: toolResult turnCount + 1 = 4, next turnCount = 3 → mismatch
    expect(next.turns.length).not.toBe(toolResult.turns.length + 1);

    // Fix: (toolResult turnCount - 1) + 1 = 3 → matches
    expect(next.turns.length).toBe((toolResult.turns.length - 1) + 1);
  });
});


