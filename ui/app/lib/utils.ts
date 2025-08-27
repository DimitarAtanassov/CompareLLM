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
  x === "anthropic" || x === "openai" || x === "gemini" || x === "ollama" || x === "unknown";

export const isProviderBrand = (x?: string | null): x is ProviderBrand =>
  x === "anthropic" ||
  x === "openai" ||
  x === "gemini" ||
  x === "ollama" ||
  x === "deepseek" ||
  x === "voyage" ||
  x === "cerebras" ||      // <-- add
  x === "unknown";

/** Normalize any provider/type string to a canonical brand. */
export const coerceBrand = (t?: string): ProviderBrand => {
  const s = (t || "").toLowerCase();
  // handle common typos / aliases first
  if (s.includes("cerebras") || s.includes("cerberus")) return "cerebras";
  if (s.includes("voyage")) return "voyage";
  if (s.includes("gemini") || s.includes("google")) return "google";
  if (isProviderBrand(s as ProviderBrand)) return s as ProviderBrand;
  return "unknown";
};

export const coerceWire = (p: ProviderInfo): ProviderWire => {
  if (isProviderWire(p.wire)) return p.wire!;
  const brand = coerceBrand(p.type);
  const inferred = BRAND_TO_DEFAULT_WIRE[brand];
  if (inferred) return inferred;
  if (isProviderWire(p.type)) return p.type as ProviderWire;
  return "unknown";
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
