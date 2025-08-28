// components/chat/ChatResults.tsx
"use client";
import { AskAnswers, ProviderBrand } from "@/app/lib/types";
import React, { useMemo, useState } from "react";
import ReactMarkdown, { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import ModelBadge from "../ui/ModelBadge";
import CodeBlock from "../md/CodeBlock";

type CodeRendererProps = React.ComponentPropsWithoutRef<"code"> & {
  inline?: boolean;
  node?: unknown;
};

export default function ChatResults({
  answers,
  isRunning,
  brandOf,
  onOpenModel,
  onRemoveModel,
  onRetryModel,   // ⬅ NEW
  onReprompt,     // ⬅ NEW
}: {
  answers: AskAnswers;
  isRunning: boolean;
  brandOf: (m: string) => ProviderBrand;
  onOpenModel: (m: string) => void;
  onRemoveModel: (m: string) => void;
  onRetryModel: (m: string) => void;   // ⬅ NEW
  onReprompt: () => void;              // ⬅ NEW
}) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const toggleExpanded = (model: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(model) ? next.delete(model) : next.add(model);
      return next;
    });

  const components = useMemo(
    () =>
      ({
        code: ({ inline, className, children, ...rest }: CodeRendererProps) => (
          <CodeBlock inline={!!inline} className={className} {...rest}>
            {children as React.ReactNode}
          </CodeBlock>
        ),
      }) satisfies Components,
    []
  );

  return (
    <div className="mx-auto w-full max-w-3xl flex flex-col gap-4">
      {Object.entries(answers).map(([model, { answer, error, latency_ms }]) => {
        const brand = brandOf(model);
        const hasErr = Boolean(error);
        const isOpen = expanded.has(model);

        return (
          <div
            key={model}
            className={[
              "relative rounded-2xl border p-0 shadow-sm bg-white dark:bg-zinc-950",
              "border-zinc-200 dark:border-zinc-800",
            ].join(" ")}
          >
            {/* Header */}
            <div
              className="flex items-center justify-between px-4 py-2 rounded-t-2xl cursor-pointer hover:bg-zinc-50 dark:hover:bg-zinc-900 transition"
              onClick={() => onOpenModel(model)}
              title="Click to continue chatting with this model"
            >
              <div className="flex items-center gap-2 min-w-0">
                <h2 className="text-sm font-semibold font-mono truncate max-w-[16rem]" title={model}>
                  {model}
                </h2>
                <ModelBadge brand={brand} />
              </div>

              <div className="flex items-center gap-2">
                <span className="text-xs text-zinc-500 dark:text-zinc-400">
                  {hasErr
                    ? "⚠ Error"
                    : latency_ms
                    ? `${(latency_ms / 1000).toFixed(1)}s`
                    : isRunning
                    ? "running…"
                    : ""}
                </span>

                {/* Remove (X) */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onRemoveModel(model);
                  }}
                  className="ml-1 rounded-md p-1 text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800"
                  aria-label={`Remove ${model} result`}
                  title="Remove this result"
                >
                  ✕
                </button>
              </div>
            </div>

            {/* Body */}
            <div className="px-4 pb-3">
              <div
                className={[
                  "rounded-xl border bg-zinc-50/40 dark:bg-zinc-900/50",
                  "border-zinc-100 dark:border-zinc-800",
                  isOpen ? "min-h-[18rem] max-h-none" : "min-h-[18rem] max-h-[18rem] overflow-y-auto",
                ].join(" ")}
              >
                <div className="prose dark:prose-invert max-w-none prose-sm md:prose-base px-4 py-3 prose-pre:overflow-x-auto prose-pre:whitespace-pre [&_pre]:rounded-lg [&_pre]:p-4">
                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
                    {hasErr ? error || "" : answer || (isRunning ? "…" : "")}
                  </ReactMarkdown>
                </div>
              </div>

              {/* Footer actions */}
              <div className="mt-2 flex items-center justify-between gap-2">
                <span className="text-xs text-zinc-500 dark:text-zinc-400">Click header to open a focused chat</span>

                <div className="flex items-center gap-2">
                  {/* Reprompt uses the last run's prompt and focuses the input */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onReprompt();
                    }}
                    className="text-xs px-2 py-1 rounded-md border border-zinc-200 dark:border-zinc-700 hover:bg-zinc-50 dark:hover:bg-zinc-800 transition"
                    title="Load the last prompt back into the editor"
                  >
                    Reprompt
                  </button>

                  {/* Retry (single model) */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onRetryModel(model);
                    }}
                    className={[
                      "text-xs px-2 py-1 rounded-md transition",
                      hasErr
                        ? "border border-red-200 text-red-700 bg-red-50 hover:bg-red-100 dark:border-red-800 dark:text-red-400 dark:bg-red-900/20 dark:hover:bg-red-900/40"
                        : "border border-zinc-200 dark:border-zinc-700 hover:bg-zinc-50 dark:hover:bg-zinc-800",
                    ].join(" ")}
                    title="Retry this model with the last prompt"
                  >
                    Retry
                  </button>

                  <button
                    onClick={() => toggleExpanded(model)}
                    className="text-xs px-2 py-1 rounded-md border border-zinc-200 dark:border-zinc-700 hover:bg-zinc-50 dark:hover:bg-zinc-800 transition"
                    aria-expanded={isOpen}
                  >
                    {isOpen ? "Collapse" : "Expand"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
