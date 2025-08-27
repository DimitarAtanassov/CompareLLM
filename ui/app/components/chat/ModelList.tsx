// components/chat/ModelList.tsx
"use client";

import { PROVIDER_TEXT_COLOR } from "@/app/lib/colors";
import { ProviderBrand } from "@/app/lib/types";


export default function ModelList({
  models, selected, onToggle, brandOf,
}: {
  models: string[];
  selected: string[];
  onToggle: (m: string) => void;
  brandOf: (m: string) => ProviderBrand;
}) {
  if (models.length === 0) {
    return <div className="text-sm text-zinc-500 dark:text-zinc-400">No models discovered yet.</div>;
  }
  return (
    <div className="max-h-[200px] overflow-auto rounded-xl border border-zinc-200 dark:border-zinc-800 p-2 grid grid-cols-1 gap-1">
      {models.map((m) => {
        const brand = brandOf(m);
        return (
          <label key={m} className="flex items-center gap-2 px-2 py-1 rounded-lg cursor-pointer hover:bg-orange-50 dark:hover:bg-orange-400/10">
            <input
              type="checkbox"
              className="accent-orange-600 dark:accent-orange-500 cursor-pointer"
              checked={selected.includes(m)}
              onChange={() => onToggle(m)}
            />
            <span className="text-sm font-mono flex-1">{m}</span>
            <span className={`text-xs px-1.5 py-0.5 rounded ${PROVIDER_TEXT_COLOR[brand]} bg-current/10`}>{brand}</span>
          </label>
        );
      })}
    </div>
  );
}
