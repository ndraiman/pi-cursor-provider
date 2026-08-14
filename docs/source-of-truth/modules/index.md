# Module — index

**Path:** `index.ts`

**Purpose:** Register the default Cursor provider with Pi, normalize discovered models, and connect OAuth and lifecycle events to the local proxy. Numbered provider registrations use `cursor-shared.ts`.

**Public surface:** Pi default extension, model processing helpers, fallback models, and lifecycle cleanup registration.

**Depends on:** `auth.ts`, `proxy.ts`, Pi extension and OAuth types.

**Invariants:** The default provider ID is `cursor`; `cursor-account-N` is recognized as a Cursor slot; the proxy reads the current token at request time; session cleanup must run before switching, forking, changing trees, or shutting down.

**Related functions:** [functions/index.md](../functions/index.md)

**Related types:** [types/index.md](../types/index.md)
