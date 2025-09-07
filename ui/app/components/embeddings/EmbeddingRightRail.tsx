// components/embeddings/EmbeddingRightRail.tsx
"use client";

import { MultiSearchResponse, ProviderBrand, SearchResult } from "@/app/lib/types";
import React, { JSX } from "react";
import Spinner from "../ui/Spinner";
import LoadingBar from "../ui/LoadingBar";
import { PROVIDER_BADGE_BG } from "@/app/lib/colors";
import { primarySnippet, redactResult } from "@/app/lib/utils";

type EmbeddingResult = {
  id?: string;
  text?: string;
  score?: number;
  similarity_score?: number;
  page_content?: string;
  metadata?: Record<string, unknown>;
};

type EmbeddingRightRailProps = {
  embedView: "single" | "compare";
  setEmbedView: (v: "single" | "compare") => void;

  // single
  isSearchingSingle: boolean;
  searchContext: { model: string; dataset: string; query: string; startedAt: number } | null;
  searchResults: SearchResult[];

  // multi-compare
  isComparing: boolean;
  multiSearchResults: MultiSearchResponse | null;

  // branding helper
  getProviderType: (m: string) => ProviderBrand;
};

// Prefer similarity_score; fall back to score; else 0
function scoreOf(row: unknown): number {
  const r = (row ?? {}) as Partial<EmbeddingResult>;
  if (typeof r.similarity_score === "number") return r.similarity_score;
  if (typeof r.score === "number") return r.score;
  return 0;
}

// Show a short, friendly metadata preview (avoid [object Object])
function metaPreview(row: Record<string, unknown>): string {
  const parts: string[] = [];

  // prefer common fields
  if (typeof row.title === "string" && row.title) parts.push(`title: ${row.title.slice(0, 40)}`);
  if (typeof row.id === "string" && row.id) parts.push(`id: ${row.id.slice(0, 40)}`);

  // look into metadata if present
  const meta = row.metadata as Record<string, unknown> | undefined;
  if (meta) {
    if (typeof meta.title === "string" && meta.title) parts.push(`title: ${meta.title.slice(0, 40)}`);
    if (typeof meta.id === "string" && meta.id) parts.push(`id: ${meta.id.slice(0, 40)}`);
  }

  // fallback: first few primitive fields (excluding embeddings/scores/internal)
  if (parts.length === 0) {
    const extras = Object.entries(row)
      .filter(
        ([k, v]) =>
          !["similarity_score", "score", "embedding"].includes(k) &&
          !k.startsWith("_") &&
          (typeof v === "string" || typeof v === "number")
      )
      .slice(0, 3)
      .map(([k, v]) => `${k}: ${String(v).slice(0, 40)}`);
    parts.push(...extras);
  }

  return parts.slice(0, 3).join(" • ");
}

export default function EmbeddingRightRail({
  embedView,
  setEmbedView,
  isSearchingSingle,
  searchContext,
  searchResults,
  isComparing,
  multiSearchResults,
  getProviderType,
}: EmbeddingRightRailProps): JSX.Element {
  return (
    <section className="space-y-4">
      {/* Toggle control */}
      <div className="inline-flex rounded-xl border border-zinc-200 dark:border-zinc-800 p-0.5 bg-white dark:bg-zinc-950">
        <button
          onClick={() => setEmbedView("single")}
          className={`px-3 py-1.5 text-sm rounded-lg transition ${
            embedView === "single"
              ? "bg-orange-600 text-white"
              : "text-zinc-600 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-900"
          }`}
        >
          Single model
        </button>
        <button
          onClick={() => setEmbedView("compare")}
          className={`px-3 py-1.5 text-sm rounded-lg transition ${
            embedView === "compare"
              ? "bg-orange-600 text-white"
              : "text-zinc-600 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-900"
          }`}
        >
          Multi-model comparison
        </button>
      </div>

      {/* Single-model results */}
      {embedView === "single" && (
        <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 bg-white dark:bg-zinc-950 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold">Single-model results</h3>
            {isSearchingSingle && <Spinner />}
          </div>
          {isSearchingSingle && <LoadingBar />}

          {!searchContext && !isSearchingSingle && (
            <div className="rounded-lg border border-dashed border-zinc-200 dark:border-zinc-800 p-4 text-sm text-zinc-500 dark:text-zinc-400">
              Run a similarity search to see results here.
            </div>
          )}

          {searchContext && (
            <div className="space-y-3">
              {(() => {
                const brand = getProviderType(searchContext.model);
                const providerBadge = PROVIDER_BADGE_BG[brand];
                return (
                  <div className="flex items-center justify-between">
                    <div className="text-sm">
                      <span className="text-zinc-500">Results for</span>{" "}
                      <span className="font-mono">{searchContext.query}</span>{" "}
                      <span className="text-zinc-500">on</span>{" "}
                      <span className="font-mono">{searchContext.dataset}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="font-mono text-xs">{searchContext.model}</div>
                      <span className={`text-[11px] px-2 py-0.5 rounded ${providerBadge}`}>{brand}</span>
                    </div>
                  </div>
                );
              })()}

              <div className="space-y-3">
                {searchResults.length === 0 && !isSearchingSingle && (
                  <div className="rounded-lg border border-dashed border-zinc-200 dark:border-zinc-800 p-4 text-sm text-zinc-500 dark:text-zinc-400">
                    No results yet.
                  </div>
                )}

                {searchResults.map((row, i) => {
                  const brand = getProviderType(searchContext.model);
                  const providerBadge = PROVIDER_BADGE_BG[brand];

                  const pct = (scoreOf(row) * 100).toFixed(1);
                  const metaLine = metaPreview(row as unknown as Record<string, unknown>);

                  return (
                    <div key={i} className="rounded-lg border border-zinc-200 dark:border-zinc-800 p-3 bg-white dark:bg-zinc-950">
                      <div className="grid grid-cols-[auto_1fr_auto] items-start gap-2">
                        <div className="shrink-0 w-6 h-6 rounded-md bg-zinc-100 dark:bg-zinc-800 text-[11px] flex items-center justify-center text-zinc-700 dark:text-zinc-300">
                          {i + 1}
                        </div>
                        <div className="min-w-0" />
                        <span
                          className={`shrink-0 inline-block px-2 py-1 rounded-full text-xs font-medium ${providerBadge}`}
                          title="cosine similarity"
                        >
                          {pct}%
                        </span>
                      </div>

                      <div className="mt-2">
                        <div className="text-[13px] leading-snug">{primarySnippet(row, 220)}</div>
                        {metaLine && (
                          <div className="mt-1 text-[11px] text-zinc-500 dark:text-zinc-400 font-mono">{metaLine}</div>
                        )}
                        <details className="mt-2">
                          <summary className="text-[11px] cursor-pointer text-zinc-500 dark:text-zinc-400">raw</summary>
                          <pre className="mt-1 whitespace-pre-wrap break-words text-[12px] leading-snug max-w-[80ch]">
                            {JSON.stringify(redactResult(row), null, 2)}
                          </pre>
                        </details>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Multi-model comparison */}
      {embedView === "compare" && (
        <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 bg-white dark:bg-zinc-950 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold">Multi-model comparison</h3>
            {isComparing && <Spinner />}
          </div>
          {isComparing && <LoadingBar />}

          {!multiSearchResults && !isComparing && (
            <div className="rounded-lg border border-dashed border-zinc-200 dark:border-zinc-800 p-4 text-sm text-zinc-500 dark:text-zinc-400">
              Run a comparison to see side-by-side results here.
            </div>
          )}

          {multiSearchResults && (
            <div className="space-y-4">
              <div className="text-sm">
                <span className="text-zinc-500">Compare query</span>{" "}
                <span className="font-mono">{multiSearchResults.query}</span>{" "}
                {typeof multiSearchResults.duration_ms === "number" && (
                  <span className="text-zinc-500"> • {(multiSearchResults.duration_ms / 1000).toFixed(1)}s</span>
                )}
              </div>

              {(() => {
                const entries = Object.entries(multiSearchResults.results);
                const modelKeys = entries.map(([k]) => k);
                const maxRows = Math.max(0, ...entries.map(([, b]) => (b?.items?.length ?? 0)));
                return (
                  <div className="overflow-x-auto">
                    <div
                      className="grid gap-3"
                      style={{ gridTemplateColumns: `repeat(${modelKeys.length}, minmax(260px, 1fr))` }}
                    >
                      {modelKeys.map((modelKey) => {
                        const brand = getProviderType(modelKey);
                        return (
                          <div
                            key={`hdr-${modelKey}`}
                            className="rounded-lg border border-zinc-200 dark:border-zinc-800 px-3 py-2 bg-zinc-50 dark:bg-zinc-900/40 flex items-center justify-between"
                          >
                            <div className="font-mono text-xs truncate">{modelKey}</div>
                            <span className={`text-[11px] px-2 py-0.5 rounded ${PROVIDER_BADGE_BG[brand]}`}>{brand}</span>
                          </div>
                        );
                      })}

                      {Array.from({ length: maxRows }).map((_, i) =>
                        modelKeys.map((modelKey) => {
                          const bucket = multiSearchResults.results[modelKey];
                          const brand = getProviderType(modelKey);
                          const providerBadge = PROVIDER_BADGE_BG[brand];

                          if (bucket?.error) {
                            return (
                              <div
                                key={`${modelKey}-err`}
                                className="rounded-lg border border-red-200 dark:border-red-900/40 p-3 text-xs text-red-600 dark:text-red-400 bg-red-50/40 dark:bg-red-900/10"
                              >
                                Error: {bucket.error}
                              </div>
                            );
                          }

                          const item = bucket?.items?.[i];
                          if (!item) {
                            return (
                              <div
                                key={`${modelKey}-${i}-empty`}
                                className="rounded-lg border border-dashed border-zinc-200 dark:border-zinc-800 p-3 text-xs text-zinc-500 dark:text-zinc-400"
                              >
                                No result for rank {i + 1}.
                              </div>
                            );
                          }

                          return (
                            <div key={`${modelKey}-${i}`} className="rounded-lg border border-zinc-200 dark:border-zinc-800 p-3">
                              <div className="grid grid-cols-[auto_1fr_auto] items-start gap-2">
                                <div className="shrink-0 w-6 h-6 rounded-md bg-zinc-100 dark:bg-zinc-800 text-[11px] flex items-center justify-center text-zinc-700 dark:text-zinc-300">
                                  {i + 1}
                                </div>
                                <div className="min-w-0" />
                                <span
                                  className={`shrink-0 inline-block px-2 py-1 rounded-full text-xs font-medium ${providerBadge}`}
                                  title="cosine similarity"
                                >
                                  {(scoreOf(item) * 100).toFixed(1)}%
                                </span>
                              </div>

                              <div className="mt-2">
                                <div className="text-[13px] leading-snug">{primarySnippet(item, 220)}</div>
                                <div className="mt-1 text-[11px] text-zinc-500 dark:text-zinc-400 font-mono">
                                  {metaPreview(item as unknown as Record<string, unknown>)}
                                </div>
                              </div>
                            </div>
                          );
                        })
                      )}
                    </div>
                  </div>
                );
              })()}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
