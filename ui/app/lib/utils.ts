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
