// components/Footer.tsx
import React from 'react';

interface FooterProps {
  activeTab: "chat" | "embedding";
  selected: string[];
  anyErrors: boolean;
  startedAt: number | null;
  elapsedMs: number;
  isRunning: boolean;
  datasets: Array<{ dataset_id: string }>;
  selectedEmbeddingModels: string[];
  allEmbeddingModelsCount: number;
}

export const Footer: React.FC<FooterProps> = ({
  activeTab,
  selected,
  anyErrors,
  startedAt,
  elapsedMs,
  isRunning,
  datasets,
  selectedEmbeddingModels,
  allEmbeddingModelsCount
}) => {
  return (
    <footer className="text-xs text-zinc-500 dark:text-zinc-400 flex justify-between">
      {activeTab === "chat" ? (
        <>
          <span>{selected.length} selected</span>
          {anyErrors && <span className="text-orange-600 dark:text-orange-400">Some models returned errors</span>}
          {startedAt && (
            <span>
              Elapsed: {(elapsedMs / 1000).toFixed(1)}s{isRunning ? " (live)" : ""}
            </span>
          )}
        </>
      ) : (
        <>
          <span>{datasets.length} datasets • {selectedEmbeddingModels.length} embedding models selected</span>
          <span>
            {allEmbeddingModelsCount} embedding models available
          </span>
        </>
      )}
    </footer>
  );
};