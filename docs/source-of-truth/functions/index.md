# Functions — index

| Symbol | File | Signature | Purpose | Side effects / errors |
| --- | --- | --- | --- | --- |
| `parseModelId` | `index.ts` | `(id: string) => ParsedModelId` | Split effort and transport suffixes from a Cursor model ID. | Pure. |
| `processModels` | `index.ts` | `(rawModels: CursorModel[]) => ProcessedModel[]` | Collapse Cursor effort variants into Pi model entries. | Pure. |
| `modelConfig` | `index.ts` | `(model: ProcessedModel) => object` | Convert a processed model to Pi provider metadata. | Pure. |
| `registerSessionLifecycleCleanup` | `index.ts` | `(pi: ExtensionAPI) => void` | Bind Pi lifecycle events to proxy state cleanup. | Registers event listeners. |
| `default` | `index.ts` | `(pi: ExtensionAPI) => Promise<void>` | Start the proxy and register the Cursor provider. | Starts a loopback server and registers OAuth. |
