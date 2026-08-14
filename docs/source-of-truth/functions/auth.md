# Functions — auth

| Symbol | File | Signature | Purpose | Side effects / errors |
| --- | --- | --- | --- | --- |
| `generateCursorAuthParams` | `auth.ts` | `() => Promise<CursorAuthParams>` | Generate PKCE values and a Cursor login URL. | Uses Web Crypto. |
| `pollCursorAuth` | `auth.ts` | `(uuid: string, verifier: string) => Promise<Credentials>` | Poll for completed browser authentication. | Network calls; throws on timeout or repeated errors. |
| `refreshCursorToken` | `auth.ts` | `(refreshToken: string) => Promise<CursorCredentials>` | Exchange a refresh token for current credentials. | Network call; throws on non-OK response. |
| `getTokenExpiry` | `auth.ts` | `(token: string) => number` | Read JWT expiry with a safety margin. | Pure; falls back to one hour when unreadable. |
