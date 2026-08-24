# Module — auth

**Path:** `auth.ts`

**Purpose:** Complete Cursor PKCE authentication and refresh access tokens.

**Public surface:** `generateCursorAuthParams`, `pollCursorAuth`, `refreshCursorToken`, and `getTokenExpiry`.

**Depends on:** Cursor login and API endpoints, Web Crypto, `fetch`.

**Invariants:** Polling tolerates pending 404 responses with bounded backoff; refresh preserves the prior refresh token when the server omits a replacement.

**Related functions:** [functions/auth.md](../functions/auth.md)

**Related types:** [types/auth.md](../types/auth.md)
