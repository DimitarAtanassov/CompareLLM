// components/chat/ModelSelector.tsx
import React from 'react';

interface ModelSelectorProps {
  allModels: string[];
  selected: string[];
  onToggleModel: (model: string) => void;
  onSelectAll: () => void;
  onClearAll: () => void;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  allModels,
  selected,
  onToggleModel,
  onSelectAll,
  onClearAll
}) => {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium">Models</label>
        <div className="flex gap-2 text-xs">
          <button
            onClick={onSelectAll}
            className="px-2 py-1 rounded-lg border border-orange-200 text-zinc-800 dark:text-zinc-100 bg-orange-50 hover:bg-orange-100 dark:bg-orange-400/10 dark:hover:bg-orange-400/20 transition"
          >
            Select all
          </button>
          <button
            onClick={onClearAll}
            className="px-2 py-1 rounded-lg border border-orange-200 text-zinc-800 dark:text-zinc-100 bg-orange-50 hover:bg-orange-100 dark:bg-orange-400/10 dark:hover:bg-orange-400/20 transition"
          >
            Clear
          </button>
        </div>
      </div>

      <div className="max-h-[280px] overflow-auto rounded-xl border border-zinc-200 dark:border-zinc-800 p-2 grid grid-cols-1 sm:grid-cols-2 gap-1">
        {allModels.length === 0 ? (
          <div className="text-sm text-zinc-500 dark:text-zinc-400">No models discovered yet.</div>
        ) : (
          allModels.map((model) => (
            <label
              key={model}
              className="flex items-center gap-2 px-2 py-1 rounded-lg cursor-pointer hover:bg-orange-50 dark:hover:bg-orange-400/10"
            >
              <input
                type="checkbox"
                className="accent-orange-600 dark:accent-orange-500 cursor-pointer"
                checked={selected.includes(model)}
                onChange={() => onToggleModel(model)}
              />
              <span className="text-sm font-mono">{model}</span>
            </label>
          ))
        )}
      </div>
    </div>
  );
};
