// components/embeddings/EmbeddingLeftRail.tsx
"use client";

import { PROVIDER_TEXT_COLOR } from "@/app/lib/colors";
import { Dataset, ProviderBrand } from "@/app/lib/types";
import React, { JSX, useCallback, useRef, useState } from "react";
import Spinner from "../ui/Spinner";

type EmbeddingLeftRailProps = {
  // model selection
  allEmbeddingModels: string[];
  selectedEmbeddingModels: string[];
  toggleEmbeddingModel: (m: string) => void;
  selectAllEmbedding: () => void;
  clearAllEmbedding: () => void;
  getProviderType: (model: string) => ProviderBrand;

  // upload dataset
  datasetId: string;
  setDatasetId: (v: string) => void;
  textField: string;
  setTextField: (v: string) => void;
  jsonInput: string;
  setJsonInput: (v: string) => void;
  uploadingDataset: boolean;
  uploadDataset: () => void | Promise<void>;

  // single search
  selectedSearchModel: string;
  setSelectedSearchModel: (v: string) => void;
  datasets: Dataset[];
  selectedDataset: string;
  setSelectedDataset: (v: string) => void;
  searchQuery: string;
  setSearchQuery: (v: string) => void;
  topKSingle: number;
  setTopKSingle: (n: number) => void;
  performSearch: () => void | Promise<void>;
  isSearchingSingle: boolean;
  hasAnyDataset: boolean;

  // compare (now with its own dataset selector)
  compareQuery: string;
  setCompareQuery: (v: string) => void;
  topKCompare: number;
  setTopKCompare: (n: number) => void;
  performMultiSearch: () => void | Promise<void>;
  isComparing: boolean;
  selectedCompareDataset: string;
  setSelectedCompareDataset: (v: string) => void;

  // datasets list actions
  deleteDataset: (id: string) => void | Promise<void>;
};

export default function EmbeddingLeftRail(props: EmbeddingLeftRailProps): JSX.Element {
  const {
    allEmbeddingModels,
    selectedEmbeddingModels,
    toggleEmbeddingModel,
    selectAllEmbedding,
    clearAllEmbedding,
    getProviderType,

    datasetId,
    setDatasetId,
    textField,
    setTextField,
    jsonInput,
    setJsonInput,
    uploadingDataset,
    uploadDataset,

    selectedSearchModel,
    setSelectedSearchModel,
    datasets,
    selectedDataset,
    setSelectedDataset,
    searchQuery,
    setSearchQuery,
    topKSingle,
    setTopKSingle,
    performSearch,
    isSearchingSingle,
    hasAnyDataset,

    compareQuery,
    setCompareQuery,
    topKCompare,
    setTopKCompare,
    performMultiSearch,
    isComparing,
    selectedCompareDataset,
    setSelectedCompareDataset,

    deleteDataset,
  } = props;

  // --- Drag-to-resize for Embedding Models list ---
  const MIN_H = 140;
  const MAX_H = 520;
  const [embedListHeight, setEmbedListHeight] = useState<number>(260);
  const dragRef = useRef<{ startY: number; startH: number } | null>(null);

  const beginDrag = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      dragRef.current = { startY: e.clientY, startH: embedListHeight };
      const onMove = (ev: MouseEvent) => {
        if (!dragRef.current) return;
        const dy = ev.clientY - dragRef.current.startY;
        const next = Math.min(MAX_H, Math.max(MIN_H, dragRef.current.startH + dy));
        setEmbedListHeight(next);
      };
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        dragRef.current = null;
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    [embedListHeight]
  );

  return (
    <section className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 sm:p-5 bg-white dark:bg-zinc-950 shadow-sm space-y-4">
      {/* Embedding Models */}
      <div>
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium">Embedding Models</label>
          <div className="flex gap-2 text-xs">
            <button
              onClick={selectAllEmbedding}
              className="px-2 py-1 rounded-lg border border-orange-200 bg-orange-50 hover:bg-orange-100 dark:bg-orange-400/10 dark:hover:bg-orange-400/20"
            >
              Select all
            </button>
            <button
              onClick={clearAllEmbedding}
              className="px-2 py-1 rounded-lg border border-orange-200 bg-orange-50 hover:bg-orange-100 dark:bg-orange-400/10 dark:hover:bg-orange-400/20"
            >
              Clear
            </button>
          </div>
        </div>

        <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 mt-2">
          <div
            className="overflow-auto p-2 grid grid-cols-1 gap-1"
            style={{ height: embedListHeight }}
            aria-label="Embedding model selection list"
          >
            {allEmbeddingModels.length === 0 && (
              <div className="text-sm text-zinc-500 dark:text-zinc-400">
                No embedding models discovered yet.
              </div>
            )}
            {allEmbeddingModels.map((m) => {
              const brand = getProviderType(m);
              const checked = selectedEmbeddingModels.includes(m);
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
                    onChange={() => toggleEmbeddingModel(m)}
                  />
                  <span className="text-sm font-mono flex-1 truncate">{m}</span>
                  <span className={`text-xs px-1.5 py-0.5 rounded ${PROVIDER_TEXT_COLOR[brand]} bg-current/10`}>
                    {brand}
                  </span>
                </label>
              );
            })}
          </div>

          {/* Drag handle */}
          <div
            onMouseDown={beginDrag}
            className="relative h-3 w-full cursor-row-resize select-none"
            title="Drag to resize"
          >
            <div className="absolute left-1/2 -translate-x-1/2 top-0.5 h-2 w-16 rounded-full bg-zinc-300 dark:bg-zinc-700" />
          </div>
        </div>
      </div>

      {/* Upload Dataset */}
      <div className="space-y-2">
        <h3 className="text-sm font-medium">Upload Dataset</h3>
        <input
          type="text"
          placeholder="dataset id"
          className="w-full rounded-md border border-zinc-200 dark:border-zinc-800 p-2 bg-white dark:bg-zinc-900 text-sm"
          value={datasetId}
          onChange={(e) => setDatasetId(e.target.value)}
        />
        <input
          type="text"
          placeholder="text field (default: text)"
          className="w-full rounded-md border border-zinc-200 dark:border-zinc-800 p-2 bg-white dark:bg-zinc-900 text-sm"
          value={textField}
          onChange={(e) => setTextField(e.target.value)}
        />
        <textarea
          placeholder='[{"id":"1","text":"hello"},{"id":"2","text":"world"}]'
          className="w-full rounded-md border border-zinc-200 dark:border-zinc-800 p-2 bg-white dark:bg-zinc-900 text-sm h-28"
          value={jsonInput}
          onChange={(e) => setJsonInput(e.target.value)}
        />
        <button
          onClick={() => void uploadDataset()}
          disabled={uploadingDataset}
          className="w-full rounded-xl py-2 px-4 font-medium text-white bg-orange-600 hover:bg-orange-700 dark:bg-orange-500 dark:hover:bg-orange-600 transition disabled:opacity-50"
        >
          {uploadingDataset ? "Uploading…" : "Upload"}
        </button>
      </div>

      {/* Similarity Search (single-model) */}
      <div className="space-y-3 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4">
        <h3 className="text-sm font-semibold">Similarity Search</h3>
        <div>
          <label className="block text-xs font-medium mb-1">Provider model</label>
          <select
            className="w-full rounded-md border border-zinc-200 dark:border-zinc-800 p-2 bg-white dark:bg-zinc-900 text-sm"
            value={selectedSearchModel}
            onChange={(e) => setSelectedSearchModel(e.target.value)}
          >
            {allEmbeddingModels.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs font-medium mb-1">Dataset</label>
          <select
            className="w-full rounded-md border border-zinc-200 dark:border-zinc-800 p-2 bg-white dark:bg-zinc-900 text-sm"
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
          >
            <option value="">-- Select a dataset --</option>
            {datasets.map((d) => (
              <option key={d.dataset_id} value={d.dataset_id}>
                {d.dataset_id}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs font-medium mb-1">Query</label>
          <input
            type="text"
            placeholder="Search query"
            className="w-full rounded-md border border-zinc-200 dark:border-zinc-800 p-2 bg-white dark:bg-zinc-900 text-sm"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        <div>
          <label className="block text-xs font-medium mb-1">Top-K</label>
          <input
            type="number"
            min={1}
            max={50}
            value={topKSingle}
            onChange={(e) => setTopKSingle(Math.max(1, Math.min(50, Number(e.target.value) || 5)))}
            className="w-full rounded-md border border-zinc-200 dark:border-zinc-800 p-2 bg-white dark:bg-zinc-900 text-sm"
            placeholder="5"
          />
        </div>
        <button
          onClick={() => void performSearch()}
          disabled={
            isSearchingSingle ||
            uploadingDataset ||
            !hasAnyDataset ||
            !selectedDataset ||
            !selectedSearchModel ||
            !searchQuery.trim()
          }
          className="w-full rounded-lg px-4 py-2 bg-orange-600 text-white hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          aria-busy={isSearchingSingle}
          title={
            !hasAnyDataset
              ? "Upload a dataset first"
              : !selectedDataset
              ? "Select a dataset"
              : !selectedSearchModel
              ? "Pick a provider model"
              : !searchQuery.trim()
              ? "Enter a search query"
              : uploadingDataset
              ? "Uploading dataset…"
              : "Cmd/Ctrl+Enter"
          }
        >
          {isSearchingSingle && <Spinner />}
          {isSearchingSingle ? "Searching…" : "Run similarity search"}
        </button>
      </div>

      {/* Compare Across Selected Models (separate dataset selector) */}
      <div className="space-y-3 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4">
        <h3 className="text-sm font-semibold">Compare Across Selected Models</h3>
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          Uses the models you checked above. Choose a dataset below just for comparison.
        </p>

        <div>
          <label className="block text-xs font-medium mb-1">Dataset for compare</label>
          <select
            className="w-full rounded-md border border-zinc-200 dark:border-zinc-800 p-2 bg-white dark:bg-zinc-900 text-sm"
            value={selectedCompareDataset}
            onChange={(e) => setSelectedCompareDataset(e.target.value)}
          >
            <option value="">-- Select a dataset --</option>
            {datasets.map((d) => (
              <option key={d.dataset_id} value={d.dataset_id}>
                {d.dataset_id}
              </option>
            ))}
          </select>
        </div>

        <input
          type="text"
          placeholder="Comparison query"
          className="w-full rounded-md border border-zinc-200 dark:border-zinc-800 p-2 bg-white dark:bg-zinc-900 text-sm"
          value={compareQuery}
          onChange={(e) => setCompareQuery(e.target.value)}
        />
        <div>
          <label className="block text-xs font-medium mb-1">Top-K</label>
          <input
            type="number"
            min={1}
            max={50}
            value={topKCompare}
            onChange={(e) => setTopKCompare(Math.max(1, Math.min(50, Number(e.target.value) || 5)))}
            className="w-full rounded-md border border-zinc-200 dark:border-zinc-800 p-2 bg-white dark:bg-zinc-900 text-sm"
            placeholder="5"
          />
        </div>
        <button
          onClick={() => void performMultiSearch()}
          disabled={
            isComparing ||
            uploadingDataset ||
            selectedEmbeddingModels.length === 0 ||
            !compareQuery.trim() ||
            !hasAnyDataset ||
            !selectedCompareDataset
          }
          className="w-full rounded-lg px-4 py-2 font-medium text-white bg-orange-600 hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          aria-busy={isComparing}
          title={
            !hasAnyDataset
              ? "Upload a dataset first"
              : selectedEmbeddingModels.length === 0
              ? "Select embedding models to compare"
              : !selectedCompareDataset
              ? "Select a dataset for comparison"
              : !compareQuery.trim()
              ? "Enter a comparison query"
              : uploadingDataset
              ? "Uploading dataset…"
              : "Shift+Cmd/Ctrl+Enter"
          }
        >
          {isComparing && <Spinner />}
          {isComparing ? "Comparing…" : "Compare against selected models"}
        </button>
      </div>

      {/* Datasets list */}
      <div className="space-y-2">
        <h3 className="text-sm font-medium">Datasets</h3>
        <div className="max-h-[160px] overflow-auto rounded-xl border border-zinc-200 dark:border-zinc-800">
          {datasets.length === 0 ? (
            <div className="p-3 text-sm text-zinc-500 dark:text-zinc-400">No datasets uploaded.</div>
          ) : (
            <ul className="divide-y divide-zinc-200 dark:divide-zinc-800">
              {datasets.map((d) => (
                <li key={d.dataset_id} className="p-2 flex items-center justify-between">
                  <button
                    className={`text-left font-mono text-xs ${
                      selectedDataset === d.dataset_id ? "text-orange-600" : ""
                    }`}
                    onClick={() => setSelectedDataset(d.dataset_id)}
                    title={`Docs: ${d.document_count}`}
                  >
                    {d.dataset_id}
                  </button>
                  <button
                    onClick={() => void deleteDataset(d.dataset_id)}
                    className="text-xs px-2 py-1 rounded-md border border-red-200 text-red-700 bg-red-50 hover:bg-red-100 dark:border-red-800 dark:text-red-400 dark:bg-red-900/20 dark:hover:bg-red-900/40 transition"
                  >
                    delete
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </section>
  );
}
