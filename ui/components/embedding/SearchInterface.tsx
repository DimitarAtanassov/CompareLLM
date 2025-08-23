// components/embedding/SearchInterface.tsx
import React from 'react';

interface SearchInterfaceProps {
  selectedSearchModel: string;
  onSearchModelChange: (model: string) => void;
  allEmbeddingModels: string[];
  selectedDataset: string;
  onSelectedDatasetChange: (dataset: string) => void;
  datasets: Array<{ dataset_id: string; document_count: number }>;
  searchQuery: string;
  onSearchQueryChange: (query: string) => void;
  isSearching: boolean;
  onPerformSearch: () => void;
}

export const SearchInterface: React.FC<SearchInterfaceProps> = ({
  selectedSearchModel,
  onSearchModelChange,
  allEmbeddingModels,
  selectedDataset,
  onSelectedDatasetChange,
  datasets,
  searchQuery,
  onSearchQueryChange,
  isSearching,
  onPerformSearch
}) => {
  return (
    <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 sm:p-5 bg-white dark:bg-zinc-950 shadow-sm">
      <h3 className="text-lg font-semibold mb-4 text-orange-600 dark:text-orange-400">Semantic Search</h3>
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Search Model</label>
          <select
            value={selectedSearchModel}
            onChange={(e) => onSearchModelChange(e.target.value)}
            className="w-full rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 outline-none focus:ring-2 focus:ring-orange-300/60 bg-white dark:bg-zinc-900"
          >
            {allEmbeddingModels.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Dataset</label>
          <select
            value={selectedDataset}
            onChange={(e) => onSelectedDatasetChange(e.target.value)}
            className="w-full rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 outline-none focus:ring-2 focus:ring-orange-300/60 bg-white dark:bg-zinc-900"
          >
            <option value="">Select a dataset</option>
            {datasets.map((dataset) => (
              <option key={dataset.dataset_id} value={dataset.dataset_id}>
                {dataset.dataset_id} ({dataset.document_count} docs)
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Search Query</label>
          <textarea
            value={searchQuery}
            onChange={(e) => onSearchQueryChange(e.target.value)}
            placeholder="What are you looking for?"
            rows={3}
            className="w-full rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 outline-none focus:ring-2 focus:ring-orange-300/60 bg-white dark:bg-zinc-900"
          />
        </div>

        <button
          onClick={onPerformSearch}
          disabled={isSearching || !searchQuery.trim() || !selectedDataset || !selectedSearchModel}
          className="w-full rounded-xl py-2 px-4 font-medium text-white bg-orange-600 hover:bg-orange-700 dark:bg-orange-500 dark:hover:bg-orange-600 transition disabled:opacity-50"
        >
          {isSearching ? "Searching…" : "Search"}
        </button>
      </div>
    </div>
  );
};