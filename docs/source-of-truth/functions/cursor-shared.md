# Functions — cursor-shared

| Symbol | File | Signature | Purpose | Side effects / errors |
| --- | --- | --- | --- | --- |
| `ensureCursorProxy` | `cursor-shared.ts` | `(resolveAccessToken: (providerId: string) => Promise<string>) => Promise<number>` | Start or reuse the provider-aware proxy. | Throws when a slot has no access token. |
| `registerCursorProvider` | `cursor-shared.ts` | `(pi, providerId, port, rawModels?, options?) => void` | Register one base or numbered Cursor provider with Pi. | Adds an OAuth provider and provider-scoped base URL. |
