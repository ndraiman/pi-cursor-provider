// Shared Cursor provider surface for pi-multi-account.
//
// pi-multi-account loads this module when it discovers the provider checkout.
// Keep the default single-account extension in index.ts, while exposing the
// provider-aware proxy and registration seams needed for numbered slots.

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import type { OAuthCredentials, OAuthLoginCallbacks } from "@mariozechner/pi-ai";
import {
  generateCursorAuthParams,
  getTokenExpiry,
  pollCursorAuth,
  refreshCursorToken,
} from "./auth.js";
import {
  getCursorModels,
  startProxy,
  type CursorModel,
} from "./proxy.js";
import {
  FALLBACK_MODELS,
  modelConfig,
  processModels,
} from "./index.js";

export { FALLBACK_MODELS };

export const CURSOR_BASE = "cursor";

type AuthCredentials = OAuthCredentials & {
  type?: string;
};

type RegisterOptions = {
  rejectDuplicateLogin?: (slot: string, credentials: AuthCredentials) => AuthCredentials;
  onModelsDiscovered?: (models: CursorModel[]) => void;
};

let proxyPortPromise: Promise<number> | undefined;

export async function ensureCursorProxy(
  resolveAccessToken: (providerId: string) => Promise<string>,
): Promise<number> {
  if (!proxyPortPromise) {
    proxyPortPromise = startProxy(async (providerId) => {
      const accessToken = await resolveAccessToken(providerId || CURSOR_BASE);
      if (!accessToken) {
        throw new Error(
          `Not logged in to Cursor provider ${providerId || CURSOR_BASE}. Run /login ${providerId || CURSOR_BASE}`,
        );
      }
      return accessToken;
    });
  }
  return proxyPortPromise;
}

function modelList(rawModels: CursorModel[]) {
  const processed = process.env.PI_CURSOR_RAW_MODELS
    ? rawModels.map((model) => ({ ...model, supportsEffort: false }))
    : processModels(rawModels);
  return processed.map(modelConfig);
}

function register(
  pi: ExtensionAPI,
  providerId: string,
  port: number,
  rawModels: CursorModel[],
  options: RegisterOptions,
): void {
  const baseUrl = `http://127.0.0.1:${port}/v1/${encodeURIComponent(providerId)}`;
  pi.registerProvider(providerId, {
    baseUrl,
    api: "openai-completions",
    models: modelList(rawModels),
    oauth: {
      name: `Cursor (${providerId})`,

      async login(callbacks: OAuthLoginCallbacks): Promise<AuthCredentials> {
        const { verifier, uuid, loginUrl } = await generateCursorAuthParams();
        callbacks.onAuth({ url: loginUrl });
        const { accessToken, refreshToken } = await pollCursorAuth(uuid, verifier);
        const credentials: AuthCredentials = {
          type: "oauth",
          refresh: refreshToken,
          access: accessToken,
          expires: getTokenExpiry(accessToken),
        };
        const accepted = options.rejectDuplicateLogin
          ? options.rejectDuplicateLogin(providerId, credentials)
          : credentials;
        const discovered = await getCursorModels(accessToken);
        if (discovered.length > 0) options.onModelsDiscovered?.(discovered);
        return accepted;
      },

      async refreshToken(credentials: AuthCredentials): Promise<AuthCredentials> {
        const refreshed = await refreshCursorToken(credentials.refresh);
        const next = { ...credentials, ...refreshed, type: "oauth" };
        const discovered = await getCursorModels(next.access);
        if (discovered.length > 0) options.onModelsDiscovered?.(discovered);
        return next;
      },

      getApiKey(): string {
        return "cursor-proxy";
      },
    },
  });
}

export function registerCursorProvider(
  pi: ExtensionAPI,
  providerId: string,
  port: number,
  rawModels: CursorModel[] = FALLBACK_MODELS,
  options: RegisterOptions = {},
): void {
  register(pi, providerId, port, rawModels, options);
}
