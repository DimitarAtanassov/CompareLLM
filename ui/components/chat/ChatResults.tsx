// components/chat/ChatResults.tsx
import React, { forwardRef } from 'react';
import { AskAnswers } from '../../types';

interface ChatResultsProps {
  answers: AskAnswers;
  isRunning: boolean;
  onOpenModelChat: (model: string) => void;
}

export const ChatResults = forwardRef<HTMLDivElement, ChatResultsProps>(({
  answers,
  isRunning,
  onOpenModelChat
}, ref) => {
  return (
    <section className="space-y-4">
      {Object.entries(answers).map(([model, { answer, error, latency_ms }]) => (
        <div
          key={model}
          className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 bg-white dark:bg-zinc-950 shadow-sm cursor-pointer hover:border-orange-300 dark:hover:border-orange-600 transition-colors group"
          onClick={() => onOpenModelChat(model)}
        >
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-sm font-semibold font-mono">{model}</h2>
            <div className="flex items-center gap-2">
              <span className="text-xs text-zinc-500 dark:text-zinc-400">
                {error ? "⚠ Error" : latency_ms ? `${(latency_ms / 1000).toFixed(1)}s` : isRunning ? "running…" : ""}
              </span>
              <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                <svg className="w-4 h-4 text-orange-600 dark:text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-3.582 8-8 8a8.001 8.001 0 01-7.93-6.94c-.04-.24-.04-.46-.04-.68l.01-.08c.05-4.345 3.578-7.88 7.93-7.93.24 0 .46.04.68.04.08 0 .16-.01.24-.01" />
                </svg>
              </div>
            </div>
          </div>
          <pre className="whitespace-pre-wrap text-sm">{error ? error : answer}</pre>
          <div className="mt-2 text-xs text-zinc-500 dark:text-zinc-400 opacity-0 group-hover:opacity-100 transition-opacity">
            Click to continue chatting with this model
          </div>
        </div>
      ))}
      <div ref={ref} />
    </section>
  );
});

ChatResults.displayName = 'ChatResults';