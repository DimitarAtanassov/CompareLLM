// lib/utils.ts
"use client";
import type { ProviderInfo, ProviderBrand, ProviderWire, SearchResult } from "./types";

const BRAND_TO_DEFAULT_WIRE: Partial<Record<ProviderBrand, ProviderWire>> = {
  deepseek: "openai",
  voyage: "openai",
  // NEW: Cerebras uses OpenAI-compatible API
  cerebras: "openai",
  google: "gemini"
};

export const isProviderWire = (x?: string | null): x is ProviderWire =>
  x === "anthropic" || x === "openai" || x === "gemini" || x === "ollama" || x === "cohere" || x === "unknown";

export const isProviderBrand = (x?: string | null): x is ProviderBrand =>
  x === "anthropic" || x === "openai" || x === "gemini" || x === "ollama" ||
  x === "deepseek" || x === "voyage" || x === "cerebras" || x === "google" ||
  x === "cohere" || x === "unknown";

export const coerceBrand = (t?: string): ProviderBrand => {
  const s = (t || "").toLowerCase();
  if (s.includes("gemini") || s.includes("google")) return "google";
  if (s.includes("cerebras") || s.includes("cerberus")) return "cerebras";
  if (s.includes("cohere")) return "cohere";     // <-- NEW
  if (s.includes("voyage")) return "voyage";
  // keep your old fallback:
  // if (isProviderBrand(s as ProviderBrand)) return s as ProviderBrand;
  return (["openai","anthropic","gemini","ollama","deepseek","voyage","cerebras","google","cohere", "unknown"] as const)
    .includes(s as ProviderBrand) ? (s as ProviderBrand) : "unknown";
};

export const coerceWire = (p: ProviderInfo): ProviderWire => {
  if (p.wire && ["anthropic","openai","gemini","ollama","cohere","unknown"].includes(p.wire)) return p.wire as ProviderWire;
  const brand = coerceBrand(p.type);
  return BRAND_TO_DEFAULT_WIRE[brand] ?? (["anthropic","openai","gemini","ollama","cohere"].includes(p.type) ? (p.type as ProviderWire) : "unknown");
};

export function redactResult(row: SearchResult): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(row)) {
    if (k === "similarity_score") continue;
    if (k.startsWith("_")) continue;
    out[k] = v;
  }
  return out;
}

export function primarySnippet(row: SearchResult, maxLen = 160): string {
  const textVal = row["text"];
  if (typeof textVal === "string") {
    return textVal.length > maxLen ? textVal.slice(0, maxLen) + "…" : textVal;
  }
  for (const [key, value] of Object.entries(row)) {
    if (key === "similarity_score" || key.startsWith("_")) continue;
    if (typeof value === "string" && value.trim().length > 0) {
      return value.length > maxLen ? value.slice(0, maxLen) + "…" : value;
    }
  }
  const s = JSON.stringify(row);
  return s.length > maxLen ? s.slice(0, maxLen) + "…" : s;
}

export function pruneUndefined<T extends Record<string, unknown>>(obj: T): Partial<T> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) if (v !== undefined) out[k] = v;
  return out as Partial<T>;
}

// utils.ts
import { API_BASE } from "./config";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

// ---- Embeddings API ----
export async function listEmbeddingModels(): Promise<{ embedding_models: string[] }> {
  return json(await fetch(`${API_BASE}/embeddings/models`));
}

export async function listStores(): Promise<{ stores: Record<string, string> }> {
  // { [store_id]: embedding_key }
  return json(await fetch(`${API_BASE}/embeddings/stores`));
}

export async function createStore(store_id: string, embedding_key: string) {
  return json(
    await fetch(`${API_BASE}/embeddings/stores`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ store_id, embedding_key }),
    })
  );
}

export async function deleteStore(store_id: string) {
  return json(await fetch(`${API_BASE}/embeddings/stores/${store_id}`, { method: "DELETE" }));
}

export type IndexDoc = { page_content: string; metadata?: Record<string, unknown> };

export async function indexDocs(store_id: string, docs: IndexDoc[]) {
  return json(
    await fetch(`${API_BASE}/embeddings/index/docs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ store_id, docs }),
    })
  );
}

export type QueryOptions = {
  k?: number;
  with_scores?: boolean;
  search_type?: "similarity" | "mmr" | "similarity_score_threshold";
  fetch_k?: number;
  lambda_mult?: number;
  score_threshold?: number | null;
};

export async function queryStore(
  store_id: string,
  query: string,
  opts: QueryOptions = {}
): Promise<{ matches: Array<{ page_content: string; metadata: Record<string, unknown>; score?: number }> }> {
  const payload = {
    store_id,
    query,
    k: opts.k ?? 5,
    with_scores: opts.with_scores ?? true,
    search_type: opts.search_type ?? "similarity",
    fetch_k: opts.fetch_k ?? 20,
    lambda_mult: opts.lambda_mult ?? 0.5,
    score_threshold: opts.score_threshold ?? null,
  };
  return json(
    await fetch(`${API_BASE}/embeddings/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
  );
}

// ---- Helpers for your UX ----

// Build docs from a user JSON array + the chosen text field
export function toDocsFromJsonArray(items: unknown[], textField: string): IndexDoc[] {
  const docs: IndexDoc[] = [];
  for (const obj of items) {
    if (!obj || typeof obj !== "object") continue;
    const rec = obj as Record<string, unknown>;
    const text = rec[textField];
    if (typeof text !== "string" || !text.trim()) continue;
    const { [textField]: _omit, ...metadata } = rec;
    docs.push({ page_content: text, metadata });
  }
  return docs;
}

// Naming convention you already rely on in backend conversations:
export const makeStoreId = (datasetId: string, embeddingKey: string) =>
  `${datasetId}::${embeddingKey}`;

// Little grouping util for “dataset” lists reconstructed from stores
export function groupStoresByDataset(stores: Record<string, string>) {
  const byDataset: Record<string, string[]> = {};
  for (const storeId of Object.keys(stores)) {
    const ds = storeId.split("::", 1)[0] ?? storeId;
    if (!byDataset[ds]) byDataset[ds] = [];
    byDataset[ds].push(storeId);
  }
  return byDataset;
}

// For “compare across models”: find all stores for a given model key
export function storesForModel(
  stores: Record<string, string>,
  embeddingKey: string
) {
  return Object.entries(stores)
    .filter(([, key]) => key === embeddingKey)
    .map(([storeId]) => storeId);
}
