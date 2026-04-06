# pi-cursor-provider

[![npm version](https://img.shields.io/npm/v/pi-cursor-provider.svg)](https://www.npmjs.com/package/pi-cursor-provider)

[Pi](https://github.com/badlogic/pi-mono) extension that provides access to [Cursor](https://cursor.com) models via OAuth authentication and a local OpenAI-compatible proxy.

## How it works

```
pi  →  openai-completions  →  localhost:PORT/v1/chat/completions
                                      ↓
                              proxy.ts (HTTP server)
                                      ↓
                              h2-bridge.mjs (Node HTTP/2)
                                      ↓
                              api2.cursor.sh gRPC
```

1. **PKCE OAuth** — browser-based login to Cursor, no client secret needed
2. **Model discovery** — queries Cursor's `GetUsableModels` gRPC endpoint
3. **Local proxy** — translates OpenAI `/v1/chat/completions` to Cursor's protobuf/HTTP2 Connect protocol
4. **Tool routing** — rejects Cursor's native tools, exposes pi's tools via MCP

## Install

```bash
# Via pi install
pi install npm:pi-cursor-provider

# Or manually
git clone https://github.com/ndraiman/pi-cursor-provider ~/.pi/agent/extensions/cursor-provider
cd ~/.pi/agent/extensions/cursor-provider
npm install
```

## Usage

```
/login cursor     # authenticate via browser
/model            # select a Cursor model
```

## Model Mapping

Cursor exposes many model variants that encode **effort level** (`low`, `medium`, `high`, `xhigh`, `max`, `none`) and **speed** (`-fast`) or **thinking** (`-thinking`) in the model ID. This extension deduplicates them so pi's reasoning effort setting controls the effort level.

### How it works

Each raw Cursor model ID is parsed into components:

```
{base}-{effort}[-fast|-thinking]
```

Examples:

| Raw Cursor ID | Base | Effort | Variant |
|---|---|---|---|
| `gpt-5.4-medium` | `gpt-5.4` | `medium` | — |
| `gpt-5.4-high-fast` | `gpt-5.4` | `high` | `-fast` |
| `claude-4.6-opus-max-thinking` | `claude-4.6-opus` | `max` | `-thinking` |
| `gpt-5.1-codex-max-high` | `gpt-5.1-codex-max` | `high` | — |
| `composer-2` | `composer-2` | — | — |

Models sharing the same `(base, variant)` with **≥2 effort levels** and a sensible default (`medium` or no-suffix) are collapsed into a single entry with `supportsReasoningEffort: true`. Pi's thinking level maps to the effort suffix:

| Pi Level | Cursor Suffix |
|---|---|
| `minimal` | `none` (if available) or `low` |
| `low` | `low` |
| `medium` | `medium` or no suffix (default) |
| `high` | `high` |
| `xhigh` | `max` (Claude) or `xhigh` (GPT) |

The proxy inserts the effort before `-fast`/`-thinking`:

```
pi selects: gpt-5.4-fast  +  effort: high  →  Cursor receives: gpt-5.4-high-fast
pi selects: gpt-5.4       +  effort: medium  →  Cursor receives: gpt-5.4-medium
pi selects: composer-2     +  (no effort)     →  Cursor receives: composer-2
```

Models **not deduped** (kept as-is with effort baked into the ID):
- Single effort variant (e.g., `claude-4.6-sonnet-medium`, `claude-4.5-opus-high`)
- No default/medium variant (e.g., `claude-4.6-opus-high` / `claude-4.6-opus-max`)
- No effort at all (e.g., `composer-2`, `gemini-3.1-pro`, `kimi-k2.5`)

## Requirements

- [Pi](https://github.com/badlogic/pi-mono)
- [Node.js](https://nodejs.org) >= 18 (for the HTTP/2 bridge)
- Active [Cursor](https://cursor.com) subscription

## Credits

OAuth flow and gRPC proxy adapted from [opencode-cursor](https://github.com/ephraimduncan/opencode-cursor) by Ephraim Duncan.
