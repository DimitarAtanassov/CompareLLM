// components/chat/ModelList.tsx
"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PROVIDER_TEXT_COLOR } from "@/app/lib/colors";
import { ProviderBrand } from "@/app/lib/types";

export default function ModelList({
  models, selected, onToggle, brandOf, initialHeightPx = 200, minHeightPx = 140, maxHeightPx = 520,
}: {
  models: string[];
  selected: string[];
  onToggle: (m: string) => void;
  brandOf: (m: string) => ProviderBrand;
  /** Optional: starting height for the scroll area (px) */
  initialHeightPx?: number;
  /** Optional: minimum height (px) */
  minHeightPx?: number;
  /** Optional: maximum height (px) */
  maxHeightPx?: number;
}) {
  const [height, setHeight] = useState<number>(initialHeightPx);
  const dragRef = useRef<{ startY: number; startH: number } | null>(null);

  const beginDrag = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    dragRef.current = { startY: e.clientY, startH: height };
    const onMove = (ev: MouseEvent) => {
      if (!dragRef.current) return;
      const dy = ev.clientY - dragRef.current.startY;
      const next = Math.min(maxHeightPx, Math.max(minHeightPx, dragRef.current.startH + dy));
      setHeight(next);
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      dragRef.current = null;
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, [height, minHeightPx, maxHeightPx]);

  if (models.length === 0) {
    return <div className="text-sm text-zinc-500 dark:text-zinc-400">No models discovered yet.</div>;
  }

  return (
    <div className="rounded-xl border border-zinc-200 dark:border-zinc-800">
      {/* Scrollable, resizable list */}
      <div
        className="overflow-auto p-2 grid grid-cols-1 gap-1"
        style={{ height }}
        aria-label="Model selection list"
      >
        {models.map((m) => {
          const brand = brandOf(m);
          const checked = selected.includes(m);
          return (
            <label
              key={m}
              className={[
                "flex items-center gap-2 px-2 py-1 rounded-lg cursor-pointer",
                "hover:bg-orange-50 dark:hover:bg-orange-400/10",
                checked ? "ring-1 ring-orange-300/60 dark:ring-orange-500/40" : "",
              ].join(" ")}
              title={m}
            >
              <input
                type="checkbox"
                className="accent-orange-600 dark:accent-orange-500 cursor-pointer"
                checked={checked}
                onChange={() => onToggle(m)}
              />
              <span className="text-sm font-mono flex-1 truncate">{m}</span>
              <span className={`text-xs px-1.5 py-0.5 rounded ${PROVIDER_TEXT_COLOR[brand]} bg-current/10`}>{brand}</span>
            </label>
          );
        })}
      </div>
    </div>
  );
}
