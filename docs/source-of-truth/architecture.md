# Architecture

`index.ts` is the Pi extension entrypoint. It registers the default Cursor provider, owns OAuth callbacks, discovers models, injects the Pi session ID into provider requests, and registers lifecycle/debug hooks. `cursor-shared.ts` exposes the provider-aware adapter consumed by `pi-multi-account` for numbered Cursor slots.

`auth.ts` implements Cursor's PKCE login and refresh requests. `proxy.ts` owns the local OpenAI-compatible HTTP server, Cursor model discovery, protobuf request construction, HTTP/2 bridge lifecycle, conversation checkpoints, and tool continuations.

## Request flow

1. Pi loads `index.ts` or `pi-multi-account` loads `cursor-shared.ts`; the extension starts `proxy.ts` on a loopback port.
2. A Cursor slot completes PKCE OAuth and stores credentials under its provider ID.
3. Pi sends `/v1/chat/completions` for the base provider or `/v1/<provider-id>/chat/completions` for a numbered slot.
4. The proxy resolves the matching access token, namespaces session state by provider ID, reconstructs the Cursor request, and starts `h2-bridge.mjs`.
5. Stream events update the OpenAI response and commit checkpoints only after a successful turn.

## State boundaries

- OAuth credentials are owned by Pi's auth store; this package resolves the current access token by provider ID at request time.
- Active bridges and conversation checkpoints are process-local maps in `proxy.ts`.
- Model discovery is cached per access token so accounts cannot reuse another account's catalog.
- Session cleanup is driven by Pi lifecycle events from `index.ts` and clears every provider slot that has been used.
