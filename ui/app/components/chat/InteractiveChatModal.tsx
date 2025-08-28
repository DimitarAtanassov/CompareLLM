// components/chat/InteractiveChatModal.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import { ChatMessage } from "@/app/lib/types";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import CodeBlock from "../md/CodeBlock";
import { Maximize2, Minimize2 } from "lucide-react";

export default function InteractiveChatModal({
  activeModel,
  messages,
  isStreaming,
  currentResponse,
  onClose,
  prompt,
  setPrompt,
  onSend,
}: {
  activeModel: string;
  messages: ChatMessage[];
  isStreaming: boolean;
  currentResponse: string;
  prompt: string;
  setPrompt: (v: string) => void;
  onSend: () => void;
  onClose: () => void;
}) {
  const [isFullscreen, setIsFullscreen] = useState(false);

  // ---- Resizable panel state ----
  const MIN_W = 520;  // px
  const MIN_H = 420;  // px
  const MAX_W = 1200; // px (only applies when not fullscreen)
  const MAX_H = 900;  // px

  const [size, setSize] = useState<{ w: number; h: number }>({ w: 768, h: 640 });
  const resizingRef = useRef(false);
  const originRef = useRef<{ mx: number; my: number; w0: number; h0: number } | null>(null);

  const onResizeStart = (e: React.MouseEvent | React.TouchEvent) => {
    if (isFullscreen) return;
    resizingRef.current = true;

    const isTouch = "touches" in e;
    const mx = isTouch ? e.touches[0].clientX : (e as React.MouseEvent).clientX;
    const my = isTouch ? e.touches[0].clientY : (e as React.MouseEvent).clientY;
    originRef.current = { mx, my, w0: size.w, h0: size.h };

    // Prevent text selection while dragging
    document.body.style.userSelect = "none";
  };

  useEffect(() => {
    const onMove = (e: MouseEvent | TouchEvent) => {
      if (!resizingRef.current || !originRef.current) return;
      const isTouch = "touches" in e && (e as TouchEvent).touches.length > 0;

      const mx = isTouch ? (e as TouchEvent).touches[0].clientX : (e as MouseEvent).clientX;
      const my = isTouch ? (e as TouchEvent).touches[0].clientY : (e as MouseEvent).clientY;

      const dx = mx - originRef.current.mx;
      const dy = my - originRef.current.my;

      const nextW = Math.min(Math.max(originRef.current.w0 + dx, MIN_W), MAX_W);
      const nextH = Math.min(Math.max(originRef.current.h0 + dy, MIN_H), MAX_H);
      setSize({ w: nextW, h: nextH });
    };

    const onUp = () => {
      if (!resizingRef.current) return;
      resizingRef.current = false;
      originRef.current = null;
      document.body.style.userSelect = "";
    };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    window.addEventListener("touchmove", onMove, { passive: false });
    window.addEventListener("touchend", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("touchmove", onMove);
      window.removeEventListener("touchend", onUp);
    };
  }, []);

  // Reset size constraints when toggling fullscreen
  useEffect(() => {
    if (isFullscreen) return;
    // Clamp back into bounds if needed
    setSize((s) => ({
      w: Math.min(Math.max(s.w, MIN_W), MAX_W),
      h: Math.min(Math.max(s.h, MIN_H), MAX_H),
    }));
  }, [isFullscreen]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div
        className={[
          "relative bg-white dark:bg-zinc-900 rounded-2xl shadow-2xl flex flex-col border transition-all duration-200",
          isFullscreen
            ? "w-[96vw] h-[96vh] max-w-none max-h-none border-emerald-300/70 dark:border-emerald-700/60 ring-2 ring-emerald-400/40"
            : "border-zinc-200 dark:border-zinc-700",
        ].join(" ")}
        style={
          isFullscreen
            ? undefined
            : {
                width: `${size.w}px`,
                height: `${size.h}px`,
              }
        }
      >
        {/* Header */}
        <div
          className={[
            "flex items-center justify-between p-4 border-b transition-colors rounded-t-2xl",
            isFullscreen
              ? "border-emerald-200/70 dark:border-emerald-800/60 bg-emerald-50/20 dark:bg-emerald-900/10"
              : "border-zinc-200 dark:border-zinc-700",
          ].join(" ")}
        >
          <h2 className="text-lg font-semibold">
            Chat with{" "}
            <span className="font-mono text-orange-600 dark:text-orange-400">
              {activeModel}
            </span>
          </h2>

          <div className="flex items-center gap-2">
            {/* Fullscreen toggle */}
            <button
              onClick={() => setIsFullscreen((v) => !v)}
              className={[
                "group relative inline-flex h-9 w-9 items-center justify-center rounded-lg",
                "border bg-white text-zinc-700 shadow-sm transition",
                "hover:border-emerald-300 hover:bg-emerald-50 hover:text-emerald-700",
                "active:scale-[0.98]",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400/60",
                "dark:bg-zinc-800 dark:text-zinc-200 dark:border-zinc-700",
                "dark:hover:border-emerald-700 dark:hover:bg-emerald-900/20 dark:hover:text-emerald-300",
              ].join(" ")}
              aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
              title={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
            >
              {isFullscreen ? <Minimize2 className="h-4.5 w-4.5" /> : <Maximize2 className="h-4.5 w-4.5" />}
            </button>

            {/* Close */}
            <button
              onClick={onClose}
              className={[
                "inline-flex h-9 w-9 items-center justify-center rounded-lg",
                "border bg-white text-zinc-700 shadow-sm transition",
                "hover:bg-zinc-50",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-300",
                "dark:bg-zinc-800 dark:text-zinc-200 dark:border-zinc-700 dark:hover:bg-zinc-700/60",
              ].join(" ")}
              aria-label="Close"
              title="Close"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
              <div className="w-full sm:max-w-[80%] space-y-1">
                <div
                  className={[
                    "rounded-2xl px-4 py-2",
                    m.role === "user" ? "bg-orange-600 text-white" : "bg-zinc-100 dark:bg-zinc-800",
                  ].join(" ")}
                >
                  <div className="prose prose-sm dark:prose-invert max-w-[75ch]">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        code: CodeBlock,
                        pre: ({ children, ...props }) => (
                          <pre className="overflow-x-auto" {...props}>
                            {children}
                          </pre>
                        ),
                      }}
                    >
                      {m.content}
                    </ReactMarkdown>
                  </div>

                  <div className="text-xs opacity-70 mt-1">{new Date(m.timestamp).toLocaleTimeString()}</div>
                </div>
              </div>
            </div>
          ))}

          {isStreaming && currentResponse && (
            <div className="flex justify-start">
              <div className="w-full sm:max-w-[80%] rounded-2xl px-4 py-2 bg-zinc-100 dark:bg-zinc-800">
                <div className="prose prose-sm dark:prose-invert max-w-[75ch]">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      code: CodeBlock,
                      pre: ({ children, ...props }) => (
                        <pre className="overflow-x-auto" {...props}>
                          {children}
                        </pre>
                      ),
                    }}
                  >
                    {currentResponse}
                  </ReactMarkdown>
                </div>
                <div className="text-xs opacity-70 mt-1">Typing...</div>
              </div>
            </div>
          )}
        </div>

        {/* Input */}
        <div
          className={[
            "p-4 border-t",
            isFullscreen ? "border-emerald-200/70 dark:border-emerald-800/60" : "border-zinc-200 dark:border-zinc-700",
          ].join(" ")}
        >
          <div className="flex gap-3">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Type your message..."
              rows={isFullscreen ? 3 : 2}
              className="flex-1 rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 outline-none focus:ring-2 focus:ring-orange-300/60 bg-white dark:bg-zinc-800 resize-none"
              onKeyDown={(e) => {
                if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                  e.preventDefault();
                  onSend();
                }
                if (e.key === "Escape") {
                  e.preventDefault();
                  onClose();
                }
              }}
            />
            <button
              onClick={onSend}
              disabled={!prompt.trim() || isStreaming}
              className="px-6 py-2 rounded-xl font-medium text-white bg-orange-600 hover:bg-orange-700 dark:bg-orange-500 dark:hover:bg-orange-600 transition disabled:opacity-50"
            >
              Send
            </button>
          </div>
          <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-2">Press Cmd/Ctrl + Enter to send • Esc to close</div>
        </div>

        {/* ===== Corner Resize Handle (outside content, on the border) ===== */}
        {!isFullscreen && (
          <button
            aria-label="Resize"
            title="Resize"
            className={[
              "absolute -bottom-2 -right-2 h-5 w-5 rounded-md",
              "border border-emerald-300 bg-white shadow-sm",
              "dark:bg-zinc-900 dark:border-emerald-700",
              "hover:bg-emerald-50 dark:hover:bg-emerald-900/20",
              "cursor-se-resize",
            ].join(" ")}
            onMouseDown={onResizeStart}
            onTouchStart={onResizeStart}
          >
            {/* Visual corner chevrons */}
            <svg viewBox="0 0 24 24" className="h-4 w-4 mx-auto my-auto text-emerald-600 dark:text-emerald-400">
              <path d="M6 18h12M10 14h8M14 10h6" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
}
