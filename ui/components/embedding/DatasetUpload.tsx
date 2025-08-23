// components/embedding/DatasetUpload.tsx
import React from 'react';

interface DatasetUploadProps {
  datasetId: string;
  onDatasetIdChange: (id: string) => void;
  textField: string;
  onTextFieldChange: (field: string) => void;
  selectedEmbeddingModels: string[];
  jsonInput: string;
  onJsonInputChange: (json: string) => void;
  uploadingDataset: boolean;
  onUploadDataset: () => void;
}

export const DatasetUpload: React.FC<DatasetUploadProps> = ({
  datasetId,
  onDatasetIdChange,
  textField,
  onTextFieldChange,
  selectedEmbeddingModels,
  jsonInput,
  onJsonInputChange,
  uploadingDataset,
  onUploadDataset
}) => {
  return (
    <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 sm:p-5 bg-white dark:bg-zinc-950 shadow-sm">
      <h3 className="text-lg font-semibold mb-4 text-orange-600 dark:text-orange-400">Upload Dataset</h3>
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Dataset ID</label>
          <input
            type="text"
            value={datasetId}
            onChange={(e) => onDatasetIdChange(e.target.value)}
            placeholder="my-documents"
            className="w-full rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 outline-none focus:ring-2 focus:ring-orange-300/60 bg-white dark:bg-zinc-900"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Text Field Name</label>
          <input
            type="text"
            value={textField}
            onChange={(e) => onTextFieldChange(e.target.value)}
            placeholder="text"
            className="w-full rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 outline-none focus:ring-2 focus:ring-orange-300/60 bg-white dark:bg-zinc-900"
          />
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
            Field containing the text to embed in each document
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Selected Models for Upload</label>
          <div className="text-sm text-zinc-600 dark:text-zinc-300">
            {selectedEmbeddingModels.length === 0 ? (
              <span className="text-zinc-500">No models selected</span>
            ) : (
              selectedEmbeddingModels.map(model => (
                <span key={model} className="inline-block bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400 px-2 py-1 rounded-md text-xs mr-2 mb-1">
                  {model}
                </span>
              ))
            )}
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">JSON Data</label>
          <textarea
            value={jsonInput}
            onChange={(e) => onJsonInputChange(e.target.value)}
            placeholder={`[
  {"text": "Document 1 content", "title": "Doc 1"},
  {"text": "Document 2 content", "title": "Doc 2"}
]`}
            rows={8}
            className="w-full rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 outline-none focus:ring-2 focus:ring-orange-300/60 bg-white dark:bg-zinc-900 font-mono text-sm"
          />
        </div>

        <button
          onClick={onUploadDataset}
          disabled={uploadingDataset || !jsonInput.trim() || !datasetId.trim() || selectedEmbeddingModels.length === 0}
          className="w-full rounded-xl py-2 px-4 font-medium text-white bg-orange-600 hover:bg-orange-700 dark:bg-orange-500 dark:hover:bg-orange-600 transition disabled:opacity-50"
        >
          {uploadingDataset ? "Uploading…" : `Upload with ${selectedEmbeddingModels.length} model(s)`}
        </button>
      </div>
    </div>
  );
};