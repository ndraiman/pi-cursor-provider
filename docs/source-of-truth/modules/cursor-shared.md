# Module — cursor-shared

**Path:** `cursor-shared.ts`

**Purpose:** Provide the provider-aware Cursor adapter loaded by `pi-multi-account`.

**Public surface:** `ensureCursorProxy`, `registerCursorProvider`, and `FALLBACK_MODELS`.

**Depends on:** `auth.ts`, `index.ts`, `proxy.ts`, and Pi extension OAuth types.

**Invariants:** Every registered slot uses a provider-scoped proxy base URL and resolves its own OAuth credentials. The adapter does not replace the default `cursor` extension entrypoint.

**Related functions:** [functions/cursor-shared.md](../functions/cursor-shared.md)

**Related types:** [types/cursor-shared.md](../types/cursor-shared.md)
