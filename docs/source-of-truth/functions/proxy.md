# Functions — proxy

| Symbol | File | Signature | Purpose | Side effects / errors |
| --- | --- | --- | --- | --- |
| `getCursorModels` | `proxy.ts` | `(apiKey: string) => Promise<CursorModel[]>` | Discover usable Cursor models through the unary bridge. | Network/child-process calls; returns an empty list on discovery failure. |
| `startProxy` | `proxy.ts` | `(getAccessToken: (providerId: string) => Promise<string>) => Promise<number>` | Start the loopback HTTP server and resolve tokens by provider ID. | Binds a local port and handles base or scoped chat requests. |
| `stopProxy` | `proxy.ts` | `() => void` | Stop the server and clean active state. | Cancels active bridges. |
| `buildCursorRequest` | `proxy.ts` | `(...) => CursorRequestPayload` | Encode Pi messages, tools, and checkpoint state for Cursor. | Pure protobuf construction. |
| `derivePiSessionId` | `proxy.ts` | `(body) => string \| undefined` | Select the session identity from request fields. | Pure. |
| `deriveBridgeKey` | `proxy.ts` | `(messages, sessionId?, providerId?) => string` | Derive an active bridge key scoped to a provider. | Pure hash. |
| `deriveConversationKey` | `proxy.ts` | `(messages, sessionId?, providerId?) => string` | Derive conversation checkpoint storage key scoped to a provider. | Pure hash. |
| `cleanupSessionState` | `proxy.ts` | `(sessionId?: string, providerId?: string) => void` | Cancel and remove state for one or every used provider slot. | Cancels active bridges and deletes checkpoints. |
