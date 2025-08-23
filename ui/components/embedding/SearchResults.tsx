// components/embedding/SearchResults.tsx
import React from 'react';
import { SearchResult } from '../../types';

interface SearchResultsProps {
  searchResults: SearchResult[];
  searchQuery: string;
  isSearching: boolean;
  searchContext: {
    model: string;
    dataset: string;
    query: string;
    startedAt: number;
  } | null;
  selectedSearchModel: string;
}

export const SearchResults: React.FC<SearchResultsProps> = ({
  searchResults,
  searchQuery,
  isSearching,
  searchContext,
  selectedSearchModel
}) => {
  if (searchResults.length > 0) {
    return (
      <section className="space-y-4">
        <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 bg-white dark:bg-zinc-950 shadow-sm">
          <h3 className="text-lg font-semibold mb-4 text-orange-600 dark:text-orange-400">
            Search Results ({searchResults.length})
          </h3>
          <div className="space-y-4">
            {searchResults.map((result, index) => (
              <div
                key={index}
                className="p-4 rounded-xl border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800/50"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold">Result #{index + 1}</span>
                    <span className="text-xs px-2 py-1 rounded-md bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 font-mono">
                      {searchContext?.model ?? selectedSearchModel}
                    </span>
                  </div>
                  <span className="text-xs px-2 py-1 rounded-full bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400">
                    {(result.similarity_score * 100).toFixed(1)}% match
                  </span>
                </div>
                <div className="space-y-2 text-sm">
                  {Object.entries(result).map(([key, value]) => {
                    if (key === "similarity_score" || key === "embedding" || key.startsWith("_")) return null;
                    return (
                      <div key={key}>
                        <span className="font-medium text-zinc-600 dark:text-zinc-400">{key}:</span>{" "}
                        <span className="text-zinc-900 dark:text-zinc-100">
                          {typeof value === "string" && value.length > 200
                            ? value.substring(0, 200) + "..."
                            : String(value)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    );
  }

  if (searchResults.length === 0 && searchQuery && !isSearching) {
    return (
      <section className="space-y-4">
        <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-8 bg-white dark:bg-zinc-950 shadow-sm text-center">
          <p className="text-zinc-500 dark:text-zinc-400">No results found for your search query.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="space-y-4">
      <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-8 bg-white dark:bg-zinc-950 shadow-sm text-center">
        <p className="text-zinc-500 dark:text-zinc-400">
          Upload a dataset and perform a search to see results here.
        </p>
      </div>
    </section>
  );
};