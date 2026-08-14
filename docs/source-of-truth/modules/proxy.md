# Module — proxy

**Path:** `proxy.ts`

**Purpose:** Translate OpenAI-compatible requests into Cursor protobuf/HTTP2 calls and maintain in-flight bridge and conversation state.

**Public surface:** `startProxy`, `stopProxy`, `getCursorModels`, provider-scoped request/key derivation helpers, request builders, test seams, and cleanup functions.

**Depends on:** Node HTTP, child-process bridge, Cursor protobuf schemas, `@bufbuild/protobuf`.

**Invariants:** Conversation checkpoints commit only after successful upstream completion; active tool bridges remain available until their tool results arrive; provider IDs isolate tokens, model caches, and session state; unscoped `/v1` routes remain backward compatible.

**Related functions:** [functions/proxy.md](../functions/proxy.md)

**Related types:** [types/proxy.md](../types/proxy.md)
