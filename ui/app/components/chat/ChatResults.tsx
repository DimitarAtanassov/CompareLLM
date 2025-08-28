// components/chat/ChatResults.tsx
"use client";

import { AskAnswers, ProviderBrand } from "@/app/lib/types";
import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown, { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import ModelBadge from "../ui/ModelBadge";
import CodeBlock from "../md/CodeBlock";

type CodeRendererProps = React.ComponentPropsWithoutRef<"code"> & {
  inline?: boolean;
  node?: unknown;
};

const DEFAULT_HEIGHT = 288; // ~18rem
const MIN_HEIGHT = 160;
const MAX_HEIGHT = 900;

export default function ChatResults({
  answers,
  isRunning,
  brandOf,
  onOpenModel,
  onRemoveModel,
  onRetryModel,
}: {
  answers: AskAnswers;
  isRunning: boolean;
  brandOf: (m: string) => ProviderBrand;
  onOpenModel: (m: string) => void;
  onRemoveModel: (m: string) => void;
  onRetryModel: (m: string) => void;
}) {
  // Expand / collapse per-model
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const toggleExpanded = (model: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(model) ? next.delete(model) : next.add(model);
      return next;
    });

  // Flash + keep green border after a run
  const prevIsRunning = useRef<boolean>(isRunning);
  const [flashModels, setFlashModels] = useState<Set<string>>(new Set());
  const [completedModels, setCompletedModels] = useState<Set<string>>(new Set());

  useEffect(() => {
    const wasRunning = prevIsRunning.current;

    // New run started: clear prior visual state so we can retrigger animation later
    if (!wasRunning && isRunning) {
      setFlashModels(new Set());
      setCompletedModels(new Set());
    }

    // Run finished: mark all cards as completed, then trigger a pulse on next frame
    let clearPulseTimer: number | undefined;
    if (wasRunning && !isRunning) {
      const keys = Object.keys(answers);
      const toSet = new Set(keys);

      // 1) set completed (keeps the green border)
      setCompletedModels(toSet);

      // 2) trigger pulse after layout has caught up (avoids missed animations)
      requestAnimationFrame(() => {
        setFlashModels(toSet);
        clearPulseTimer = window.setTimeout(() => {
          setFlashModels(new Set()); // remove pulse but keep green border
        }, 900);
      });
    }

    prevIsRunning.current = isRunning;

    return () => {
      if (clearPulseTimer) clearTimeout(clearPulseTimer);
    };
    // Keep `answers` in deps so keys are accurate at finish time
  }, [isRunning, answers]);

  // Per-model resizable heights
  const [heights, setHeights] = useState<Record<string, number>>({});
  const [isResizing, setIsResizing] = useState(false);
  const resizeRef = useRef<{ model: string; startY: number; startH: number } | null>(null);

  const startResize = (model: string, e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    const h = heights[model] ?? DEFAULT_HEIGHT;
    resizeRef.current = { model, startY: e.clientY, startH: h };
    setIsResizing(true);

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
  };

  const onMouseMove = (e: MouseEvent) => {
    const st = resizeRef.current;
    if (!st) return;
    const dy = e.clientY - st.startY;
    const raw = st.startH + dy;
    const clamped = Math.max(MIN_HEIGHT, Math.min(MAX_HEIGHT, raw));
    setHeights((prev) => ({ ...prev, [st.model]: clamped }));
  };

  const onMouseUp = () => {
    resizeRef.current = null;
    setIsResizing(false);
    window.removeEventListener("mousemove", onMouseMove);
    window.removeEventListener("mouseup", onMouseUp);
  };

  // Markdown code renderer
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
    <div
      className={[
        "mx-auto w-full max-w-3xl flex flex-col gap-4",
        isResizing ? "select-none cursor-ns-resize" : "",
      ].join(" ")}
    >
      {/* Local styles for pulse + resize grip */}
      <style jsx>{`
        @keyframes greenPulse {
          0% {
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.0);
            outline-color: rgba(16, 185, 129, 0.0);
          }
          25% {
            box-shadow: 0 0 0 6px rgba(16, 185, 129, 0.15);
            outline-color: rgba(16, 185, 129, 0.3);
          }
          50% {
            box-shadow: 0 0 0 10px rgba(16, 185, 129, 0.18);
            outline-color: rgba(16, 185, 129, 0.45);
          }
          75% {
            box-shadow: 0 0 0 6px rgba(16, 185, 129, 0.12);
            outline-color: rgba(16, 185, 129, 0.3);
          }
          100% {
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.0);
            outline-color: rgba(16, 185, 129, 0.0);
          }
        }
        .flash-green {
          animation: greenPulse 0.9s ease-in-out;
          will-change: box-shadow, outline-color;
        }
        .resize-grip {
          position: absolute;
          right: 10px;
          bottom: 8px;
          width: 42px;
          height: 10px;
          border-radius: 9999px;
          background: linear-gradient(
            to right,
            rgba(161, 161, 170, 0.6),
            rgba(161, 161, 170, 0.5)
          );
          opacity: 0.9;
          transition: background 0.2s ease, opacity 0.2s ease, transform 0.05s ease;
        }
        .resize-grip:hover {
          background: linear-gradient(
            to right,
            rgba(16, 185, 129, 0.9),
            rgba(16, 185, 129, 0.75)
          );
          opacity: 1;
        }
        .resize-grip:active {
          transform: scale(0.98);
        }
        @media (prefers-color-scheme: dark) {
          .resize-grip {
            background: linear-gradient(
              to right,
              rgba(82, 82, 91, 0.8),
              rgba(82, 82, 91, 0.7)
            );
          }
        }
      `}</style>

      {Object.entries(answers).map(([model, { answer, error, latency_ms }]) => {
        const brand = brandOf(model);
        const hasErr = Boolean(error);
        const isOpen = expanded.has(model);
        const isFlashing = flashModels.has(model);
        const isCompleted = completedModels.has(model);
        const h = heights[model] ?? DEFAULT_HEIGHT;

        return (
          <div
            key={model}
            className={[
              "relative rounded-2xl border p-0 shadow-sm bg-white dark:bg-zinc-950 transition-colors overflow-visible", // prevent pulse from being clipped
              isCompleted
                ? "border-emerald-500/70 dark:border-emerald-500/70"
                : "border-zinc-200 dark:border-zinc-800",
              isCompleted ? "shadow-[0_8px_24px_-8px_rgba(16,185,129,0.25)]" : "shadow-sm",
              isFlashing ? "flash-green" : "",
            ].join(" ")}
            style={{
              // give the animation an outline to show even when shadows get clipped by parent layouts
              outline: isFlashing ? "2px solid rgba(16,185,129,0.0)" : "none",
              outlineOffset: isFlashing ? "2px" : undefined,
            }}
          >
            {/* Header — clicking opens interactive modal */}
            <div
              className={[
                "flex items-center justify-between px-4 py-2 rounded-t-2xl cursor-pointer transition",
                "hover:bg-zinc-50 dark:hover:bg-zinc-900",
                isCompleted ? "bg-emerald-50/30 dark:bg-emerald-900/10" : "",
              ].join(" ")}
              onClick={() => onOpenModel(model)}
              title="Click to continue chatting with this model"
            >
              <div className="flex items-center gap-2 min-w-0">
                <h2
                  className="text-sm font-semibold font-mono truncate max-w-[16rem]"
                  title={model}
                >
                  {model}
                </h2>
                <ModelBadge brand={brand} />
              </div>

              <div className="flex items-center gap-2">
                <span
                  className={[
                    "text-xs",
                    hasErr
                      ? "text-red-600 dark:text-red-400"
                      : isCompleted
                      ? "text-emerald-700 dark:text-emerald-400"
                      : "text-zinc-500 dark:text-zinc-400",
                  ].join(" ")}
                >
                  {hasErr
                    ? "⚠ Error"
                    : typeof latency_ms === "number"
                    ? `${(latency_ms / 1000).toFixed(1)}s`
                    : isRunning
                    ? "running…"
                    : "done"}
                </span>

                {/* Remove */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onRemoveModel(model);
                  }}
                  className="ml-1 rounded-md p-1 text-zinc-400 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-300 hover:bg-zinc-100"
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
                id={`answer-${model}`}
                className={[
                  "relative rounded-xl border bg-zinc-50/40 dark:bg-zinc-900/50 transition-colors",
                  isCompleted
                    ? "border-emerald-200/70 dark:border-emerald-800/60"
                    : "border-zinc-100 dark:border-zinc-800",
                ].join(" ")}
                style={
                  isOpen
                    ? {
                        height: "auto",
                        maxHeight: "none",
                        overflowY: "visible",
                      }
                    : {
                        height: Math.max(h, MIN_HEIGHT),
                        maxHeight: Math.max(h, MIN_HEIGHT),
                        overflowY: "auto",
                      }
                }
              >
                <div className="prose dark:prose-invert max-w-none prose-sm md:prose-base px-4 py-3 prose-pre:overflow-x-auto prose-pre:whitespace-pre [&_pre]:rounded-lg [&_pre]:p-4">
                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
                    {hasErr ? error || "" : answer || (isRunning ? "…" : "")}
                  </ReactMarkdown>
                </div>

                {/* Drag handle — only show when collapsed (clamped) */}
                {!isOpen && (
                  <div
                    className="resize-grip cursor-ns-resize"
                    title={`Drag to resize (min ${MIN_HEIGHT}px, max ${MAX_HEIGHT}px)`}
                    onMouseDown={(e) => startResize(model, e)}
                    role="separator"
                    aria-orientation="vertical"
                    aria-label="Resize result panel"
                  />
                )}
              </div>

              {/* Footer actions */}
              <div className="mt-2 flex items-center justify-between gap-2">
                <span className="text-xs text-zinc-500 dark:text-zinc-400">
                  Click header to open a focused chat
                </span>

                <div className="flex items-center gap-2">
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
                        : isCompleted
                        ? "border-emerald-300/60 text-emerald-700 hover:bg-emerald-50 dark:border-emerald-800 dark:text-emerald-300 dark:hover:bg-emerald-900/20"
                        : "border border-zinc-200 dark:border-zinc-700 hover:bg-zinc-50 dark:hover:bg-zinc-800",
                    ].join(" ")}
                    title="Retry this model with the last prompt"
                  >
                    Retry
                  </button>

                  {/* Expand / Collapse toggles height only */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation(); // don't trigger header click
                      toggleExpanded(model);
                    }}
                    className={[
                      "text-xs px-2 py-1 rounded-md border transition",
                      isCompleted
                        ? "border-emerald-300/60 hover:bg-emerald-50 dark:border-emerald-800 dark:hover:bg-emerald-900/20"
                        : "border-zinc-200 dark:border-zinc-700 hover:bg-zinc-50 dark:hover:bg-zinc-800",
                    ].join(" ")}
                    aria-expanded={isOpen}
                    aria-controls={`answer-${model}`}
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
