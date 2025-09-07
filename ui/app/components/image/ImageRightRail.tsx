// app/components/image/ImageRightRail.tsx
"use client";

import React, { JSX } from "react";
import Spinner from "../ui/Spinner";
import LoadingBar from "../ui/LoadingBar";

import ReactMarkdown, { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import CodeBlock from "../md/CodeBlock";

type Props = {
  isProcessing: boolean;
  error: string | null;
  response: {
    text?: string;
    image_base64?: string;
    image_mime?: string;
    image_url?: string;
  } | null;
};

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

export default function ImageRightRail({
  isProcessing,
  error,
  response,
}: Props): JSX.Element {
  const hasImage = !!response?.image_url || !!response?.image_base64;

  const dataUrl = (() => {
    if (!response?.image_base64) return null;
    const isAlreadyDataUrl = response.image_base64.startsWith("data:");
    if (isAlreadyDataUrl) return response.image_base64;
    const mime = response.image_mime || "image/png";
    return `data:${mime};base64,${response.image_base64}`;
  })();

  return (
    <section className="space-y-4">
      <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 bg-white dark:bg-zinc-950 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold">Result</h3>
          {isProcessing && <Spinner />}
        </div>
        {isProcessing && <LoadingBar />}

        {!isProcessing && !response && !error && (
          <div className="rounded-lg border border-dashed border-zinc-200 dark:border-zinc-800 p-4 text-sm text-zinc-500 dark:text-zinc-400">
            Upload an image and click <span className="font-medium">Run</span> to see results here.
          </div>
        )}

        {error && (
          <div className="rounded-lg border border-red-200 dark:border-red-900/40 p-3 text-sm text-red-700 dark:text-red-400 bg-red-50/40 dark:bg-red-900/10">
            {error}
          </div>
        )}

        {response && (
          <div className="space-y-4">
            {hasImage && (
              <div className="rounded-xl overflow-hidden border border-zinc-200 dark:border-zinc-800">
                <img
                  src={response.image_url || dataUrl || ""}
                  alt="processed"
                  className="w-full h-auto object-contain"
                />
              </div>
            )}

            {response.text && (
              <div className="prose dark:prose-invert max-w-none prose-sm md:prose-base px-2 py-1 prose-pre:overflow-x-auto prose-pre:whitespace-pre [&_pre]:rounded-lg [&_pre]:p-4 border border-zinc-200 dark:border-zinc-800 rounded-lg">
                <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>
                  {response.text}
                </ReactMarkdown>
              </div>
            )}

            {!hasImage && !response.text && (
              <div className="text-sm text-zinc-500 dark:text-zinc-400">
                No content returned.
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
