# Cross-cutting behavior

## Authentication

Cursor OAuth uses PKCE. Access tokens are stored by Pi under provider IDs and passed to the local bridge only for the active request. Refresh failures surface as provider errors.

## Conversation state

`proxy.ts` keeps active bridges and completed-turn checkpoints in memory. Session IDs are preferred for state keys; anonymous requests fall back to a hash of the first user message. Every key includes the provider ID, so two Cursor accounts cannot share a bridge or checkpoint. Stale anonymous state is TTL-evicted.

## Errors and cancellation

Invalid requests return OpenAI-shaped 400 errors. Internal proxy failures return 500 errors. Upstream stream failures return an error chunk or a 502 response. Client disconnects cancel the active bridge without committing its pending checkpoint.

## Testing

`npm test` runs the Vitest suite in `index.test.ts`. Tests replace the bridge factory and use the exported proxy seams to verify parsing, session cleanup, model handling, tool continuations, and interruption behavior.
