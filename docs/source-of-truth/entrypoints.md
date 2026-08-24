# Entrypoints

| Surface | Path or command | Owner |
| --- | --- | --- |
| Pi extension | `index.ts` default export | Provider registration and lifecycle hooks |
| Shared provider adapter | `cursor-shared.ts` | `pi-multi-account` provider-slot registration |
| OAuth login | `/login cursor` | `index.ts` and `auth.ts` |
| Model list | `GET /v1/models` | `proxy.ts` |
| Chat completion | `POST /v1/chat/completions` | `proxy.ts` |
| Scoped model list | `GET /v1/<provider-id>/models` | `proxy.ts` |
| Scoped chat completion | `POST /v1/<provider-id>/chat/completions` | `proxy.ts` |
| Debug logging | `PI_CURSOR_PROVIDER_DEBUG=1` | `index.ts` and `proxy.ts` |
| Raw model mode | `PI_CURSOR_RAW_MODELS=1` | `index.ts` |
