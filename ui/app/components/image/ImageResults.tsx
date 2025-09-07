// app/components/image/ImageResults.tsx
"use client";

import React, { JSX } from "react";
import Spinner from "../ui/Spinner";
import LoadingBar from "../ui/LoadingBar";
import { ProviderBrand } from "@/app/lib/types";
import { PROVIDER_BADGE_BG } from "@/app/lib/colors";

import ReactMarkdown, { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import CodeBlock from "../md/CodeBlock";

export type ImageResponse = {
  text?: string;
  image_base64?: string;
  image_mime?: string;
  image_url?: string;
};

type ModelOutputs = Record<
  string,
  { response: ImageResponse | null; error: string | null }
>;

type Props = {
  isProcessing: boolean;
  error: string | null;
  outputs: ModelOutputs;
  brandOf: (model: string) => ProviderBrand;
};

function toDataUrl(resp: ImageResponse | null): string | null {
  if (!resp?.image_base64) return null;
  const isAlready = resp.image_base64.startsWith("data:");
  if (isAlready) return resp.image_base64;
  const mime = resp.image_mime || "image/png";
  return `data:${mime};base64,${resp.image_base64}`;
}

type CodeRendererProps = React.ComponentPropsWithoutRef<"code"> & {
  inline?: boolean;
  node?: unknown;
};
const mdComponents: Components = {
  code: ({ inline, className, children, ...rest }: CodeRendererProps) => (
    <CodeBlock inline={!!inline} className={className} {...rest}>
      {children as React.ReactNode}
    </CodeBlock>
  ),
};

export default function ImageResults({
  isProcessing,
  error,
  outputs,
  brandOf,
}: Props): JSX.Element {
  const hasAny = Object.keys(outputs || {}).length > 0;

  return (
    <section className="space-y-4">
      <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 bg-white dark:bg-zinc-950 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold">Result</h3>
          {isProcessing && <Spinner />}
        </div>

        {isProcessing && <LoadingBar />}

        {!isProcessing && !hasAny && !error && (
          <div className="rounded-lg border border-dashed border-zinc-200 dark:border-zinc-800 p-4 text-sm text-zinc-500 dark:text-zinc-400">
            Upload an image, select at least one model, then click{" "}
            <span className="font-medium">Run</span> to see results here.
          </div>
        )}

        {error && (
          <div className="rounded-lg border border-red-200 dark:border-red-900/40 p-3 text-sm text-red-700 dark:text-red-400 bg-red-50/40 dark:bg-red-900/10">
            {error}
          </div>
        )}

        {hasAny && (
          <div className="space-y-4">
            {Object.entries(outputs).map(([model, { response, error }]) => {
              const brand = brandOf(model);
              const dataUrl = toDataUrl(response);
              const hasImage = Boolean(response?.image_url || response?.image_base64);

              return (
                <div
                  key={model}
                  className="rounded-xl border border-zinc-200 dark:border-zinc-800 overflow-hidden"
                >
                  {/* Header */}
                  <div className="px-3 py-2 flex items-center justify-between bg-zinc-50 dark:bg-zinc-900/60 border-b border-zinc-200 dark:border-zinc-800">
                    <div className="min-w-0 flex items-center gap-2">
                      <span className="font-mono text-xs truncate">{model}</span>
                      <span className={`text-[10px] px-1.5 py-0.5 rounded ${PROVIDER_BADGE_BG[brand]}`}>
                        {brand}
                      </span>
                    </div>
                  </div>

                  {/* Body */}
                  <div className="p-3 space-y-3">
                    {error && (
                      <div className="rounded-lg border border-red-200 dark:border-red-900/40 p-2 text-sm text-red-700 dark:text-red-400 bg-red-50/40 dark:bg-red-900/10">
                        {error}
                      </div>
                    )}

                    {!error && hasImage && (
                      <div className="rounded-lg overflow-hidden border border-zinc-200 dark:border-zinc-800">
                        <img
                          src={response?.image_url || dataUrl || ""}
                          alt={`result from ${model}`}
                          className="w-full h-auto object-contain"
                        />
                      </div>
                    )}

                    {!error && response?.text && (
                      <div className="prose dark:prose-invert max-w-none prose-sm md:prose-base px-2 py-1 prose-pre:overflow-x-auto prose-pre:whitespace-pre [&_pre]:rounded-lg [&_pre]:p-4 border border-zinc-200 dark:border-zinc-800 rounded-lg">
                        <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>
                          {response.text}
                        </ReactMarkdown>
                      </div>
                    )}

                    {!error && !hasImage && !response?.text && (
                      <div className="text-sm text-zinc-500 dark:text-zinc-400">
                        No content returned.
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
