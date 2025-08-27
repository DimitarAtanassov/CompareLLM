// components/md/CodeBlock.tsx
"use client";
import React from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

type CodeBlockProps = React.HTMLAttributes<HTMLElement> & {
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
};

export default function CodeBlock({ inline, className, children, ...props }: CodeBlockProps) {
  const code = String(children ?? "");
  const match = /language-(\w+)/.exec(className || "");
  const lang = match?.[1] || "bash";

  if (inline) {
    return <code className={`px-1 py-0.5 rounded bg-zinc-100 dark:bg-zinc-800 ${className || ""}`} {...props}>{children}</code>;
  }

  return (
    <div className="rounded-xl overflow-hidden border border-zinc-200 dark:border-zinc-800 my-3">
      <div className="flex items-center justify-between px-3 py-2 text-xs bg-zinc-100 dark:bg-zinc-900">
        <span className="font-mono lowercase opacity-70">{lang}</span>
      </div>
      <SyntaxHighlighter language={lang} style={oneDark} PreTag="div" customStyle={{ margin: 0, padding: "12px 16px" }}>
        {code.replace(/\n$/, "")}
      </SyntaxHighlighter>
    </div>
  );
}
