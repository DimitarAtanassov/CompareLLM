// components/embedding/DatasetManager.tsx
import React from 'react';
import { Dataset } from '../../types';

interface DatasetManagerProps {
  datasets: Dataset[];
  onDeleteDataset: (id: string) => void;
}

export const DatasetManager: React.FC<DatasetManagerProps> = ({
  datasets,
  onDeleteDataset
}) => {
  return (
    <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 sm:p-5 bg-white dark:bg-zinc-950 shadow-sm">
      <h3 className="text-lg font-semibold mb-4 text-orange-600 dark:text-orange-400">Manage Datasets</h3>
      <div className="space-y-2">
        {datasets.length === 0 ? (
          <p className="text-sm text-zinc-500 dark:text-zinc-400">No datasets uploaded yet.</p>
        ) : (
          datasets.map((dataset) => (
            <div
              key={dataset.dataset_id}
              className="flex items-center justify-between p-3 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800/50"
            >
              <div>
                <div className="font-mono text-sm">{dataset.dataset_id}</div>
                <div className="text-xs text-zinc-500 dark:text-zinc-400">
                  {dataset.document_count} documents
                </div>
              </div>
              <button
                onClick={() => onDeleteDataset(dataset.dataset_id)}
                className="px-3 py-1 text-xs rounded-lg border border-red-200 text-red-700 bg-red-50 hover:bg-red-100 dark:border-red-800 dark:text-red-400 dark:bg-red-900/20 dark:hover:bg-red-900/40 transition"
              >
                Delete
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
};