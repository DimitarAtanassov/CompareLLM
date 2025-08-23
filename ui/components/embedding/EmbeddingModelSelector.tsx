// components/embedding/EmbeddingModelSelector.tsx
import React from 'react';

interface EmbeddingModelSelectorProps {
  allEmbeddingModels: string[];
  selectedEmbeddingModels: string[];
  onToggleEmbeddingModel: (model: string) => void;
  onSelectAllEmbedding: () => void;
  onClearAllEmbedding: () => void;
}

export const EmbeddingModelSelector: React.FC<EmbeddingModelSelectorProps> = ({
  allEmbeddingModels,
  selectedEmbeddingModels,
  onToggleEmbeddingModel,
  onSelectAllEmbedding,
  onClearAllEmbedding
}) => {
  return (
    <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 sm:p-5 bg-white dark:bg-zinc-950 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-orange-600 dark:text-orange-400">Select Embedding Models</h3>
        <div className="flex gap-2 text-xs">
          <button
            onClick={onSelectAllEmbedding}
            className="px-2 py-1 rounded-lg border border-orange-200 text-zinc-800 dark:text-zinc-100 bg-orange-50 hover:bg-orange-100 dark:bg-orange-400/10 dark:hover:bg-orange-400/20 transition"
          >
            Select all
          </button>
          <button
            onClick={onClearAllEmbedding}
            className="px-2 py-1 rounded-lg border border-orange-200 text-zinc-800 dark:text-zinc-100 bg-orange-50 hover:bg-orange-100 dark:bg-orange-400/10 dark:hover:bg-orange-400/20 transition"
          >
            Clear
          </button>
        </div>
      </div>
      
      <div className="max-h-[200px] overflow-auto rounded-xl border border-zinc-200 dark:border-zinc-800 p-2 grid grid-cols-1 gap-1">
        {allEmbeddingModels.length === 0 ? (
          <div className="text-sm text-zinc-500 dark:text-zinc-400">No embedding models discovered yet.</div>
        ) : (
          allEmbeddingModels.map((model) => (
            <label
              key={model}
              className="flex items-center gap-2 px-2 py-1 rounded-lg cursor-pointer hover:bg-orange-50 dark:hover:bg-orange-400/10"
            >
              <input
                type="checkbox"
                className="accent-orange-600 dark:accent-orange-500 cursor-pointer"
                checked={selectedEmbeddingModels.includes(model)}
                onChange={() => onToggleEmbeddingModel(model)}
              />
              <span className="text-sm font-mono">{model}</span>
            </label>
          ))
        )}
      </div>
    </div>
  );
};