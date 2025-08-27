// components/chat/ChatResults.tsx
"use client";
import { AskAnswers, ProviderBrand } from "@/app/lib/types";
import React from "react";
import ReactMarkdown, { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import ModelBadge from "../ui/ModelBadge";
import CodeBlock from "../md/CodeBlock";

// Props the `code` renderer actually sees
type CodeRendererProps = React.ComponentPropsWithoutRef<"code"> & {
  inline?: boolean;
  node?: unknown; // react-markdown adds this
};

export default function ChatResults({
  answers, isRunning, brandOf, onOpenModel,
}: {
  answers: AskAnswers;
  isRunning: boolean;
  brandOf: (m: string) => ProviderBrand;
  onOpenModel: (m: string) => void;
}) {
  const components = {
    code: ({ inline, className, children, ...rest }: CodeRendererProps) => (
      <CodeBlock inline={!!inline} className={className} {...rest}>
        {children as React.ReactNode}
      </CodeBlock>
    ),
  } satisfies Components;

  return (
    <>
      {Object.entries(answers).map(([model, { answer, error, latency_ms }]) => {
        const brand = brandOf(model);
        const hasErr = Boolean(error);
        return (
          <div
            key={model}
            className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 bg-white dark:bg-zinc-950 shadow-sm cursor-pointer hover:border-orange-300 dark:hover:border-orange-600 transition-colors group"
            onClick={() => onOpenModel(model)}
            title="Click to continue chatting with this model"
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-semibold font-mono">{model}</h2>
                <ModelBadge brand={brand} />
              </div>
              <span className="text-xs text-zinc-500 dark:text-zinc-400">
                {hasErr ? "⚠ Error" : latency_ms ? `${(latency_ms / 1000).toFixed(1)}s` : isRunning ? "running…" : ""}
              </span>
            </div>

            <div className="prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
                {hasErr ? error || "" : (answer || (isRunning ? "…" : ""))}
              </ReactMarkdown>
            </div>

            <div className="mt-2 text-xs text-zinc-500 dark:text-zinc-400 opacity-0 group-hover:opacity-100 transition-opacity">
              Click to continue chatting with this model
            </div>
          </div>
        );
      })}
    </>
  );
}
