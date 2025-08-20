"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// ---------------- Types kept in sync with your backend ----------------
type ProviderInfo = {
  name: string;
  type: string;
  base_url: string;
  models: string[];
  embedding_models: string[];
  auth_required: boolean;
};

type PerModelParam = { temperature?: number; max_tokens?: number; min_tokens?: number };
type ModelParamsMap = Record<string, PerModelParam>;

type ProvidersResp = { providers: ProviderInfo[] };
type AskAnswers = Record<string, { answer?: string; error?: string; latency_ms?: number }>;

type StreamEvent =
  | { type: "meta"; models: string[] }
  | { type: "chunk"; model: string; answer?: string; error?: string; latency_ms: number }
  | { type: "done" };

type Dataset = {
  dataset_id: string;
  document_count: number;
  sample_fields: string[];
};

type SearchResult = {
  similarity_score: number;
  [key: string]: string | number | boolean | null | undefined;
};

// ---------------- Config ----------------
const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "/backend");

// ---------------- Page ----------------
export default function Page() {
  const [activeTab, setActiveTab] = useState<"chat" | "embedding">("chat");
  
  // Providers and models
  const [loadingProviders, setLoadingProviders] = useState(false);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [allModels, setAllModels] = useState<string[]>([]);
  const [allEmbeddingModels, setAllEmbeddingModels] = useState<string[]>([]);

  // Chat functionality (existing)
  const [selected, setSelected] = useState<string[]>([]);
  const [prompt, setPrompt] = useState<string>("");
  const [isRunning, setIsRunning] = useState(false);
  const [answers, setAnswers] = useState<AskAnswers>({});
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [endedAt, setEndedAt] = useState<number | null>(null);

  // Embedding functionality (new)
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedEmbeddingModels, setSelectedEmbeddingModels] = useState<string[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [selectedSearchModel, setSelectedSearchModel] = useState<string>("");
  const [uploadingDataset, setUploadingDataset] = useState(false);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [jsonInput, setJsonInput] = useState<string>("");
  const [datasetId, setDatasetId] = useState<string>("");
  const [textField, setTextField] = useState<string>("text");

  // --- NEW: freeze the model/dataset/query that produced current results
  const [searchContext, setSearchContext] = useState<{
    model: string;
    dataset: string;
    query: string;
    startedAt: number;
  } | null>(null);

  const streamAbortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  // Guard against out-of-order search responses
  const requestIdRef = useRef(0);

  // global + per-model params
  const [modelParams, setModelParams] = useState<ModelParamsMap>({});
  const [globalTemp, setGlobalTemp] = useState<number>(0.7);
  const [globalMax, setGlobalMax] = useState<number>(8192);
  const [globalMin, setGlobalMin] = useState<number | undefined>(undefined);

  const updateParam = useCallback(
    (model: string, key: keyof PerModelParam, value: number | undefined) => {
      setModelParams((prev) => ({ ...prev, [model]: { ...(prev[model] || {}), [key]: value } }));
    },
    []
  );

  // -------- Load providers/models once --------
  useEffect(() => {
    const load = async () => {
      setLoadingProviders(true);
      try {
        const res = await fetch(`${API_BASE}/providers`, { cache: "no-store" });
        if (!res.ok) throw new Error(`Failed to load providers: ${res.statusText}`);
        const data = (await res.json()) as ProvidersResp;
        setProviders(data.providers);
        const models = [...new Set(data.providers.flatMap((p) => p.models))].sort();
        const embeddingModels = [...new Set(data.providers.flatMap((p) => p.embedding_models || []))].sort();
        setAllModels(models);
        setAllEmbeddingModels(embeddingModels);
        if (embeddingModels.length > 0) {
          setSelectedSearchModel(embeddingModels[0]);
        }
      } catch (err) {
        console.error(err);
      } finally {
        setLoadingProviders(false);
      }
    };
    load();
  }, []);

  // Load datasets
  const loadDatasets = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/datasets`);
      if (res.ok) {
        const data = await res.json();
        setDatasets(data.datasets || []);
      }
    } catch (err) {
      console.error("Failed to load datasets:", err);
    }
  }, []);

  useEffect(() => {
    if (activeTab === "embedding") {
      loadDatasets();
    }
  }, [activeTab, loadDatasets]);

  // -------- Chat helpers (existing) --------
  const toggleModel = (m: string) =>
    setSelected((prev) => (prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]));

  const selectAll = () => setSelected(allModels);
  const clearAll = () => setSelected([]);

  // -------- Embedding helpers (new) --------
  const toggleEmbeddingModel = (m: string) =>
    setSelectedEmbeddingModels((prev) => (prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]));

  const selectAllEmbedding = () => setSelectedEmbeddingModels(allEmbeddingModels);
  const clearAllEmbedding = () => setSelectedEmbeddingModels([]);

  const uploadDataset = useCallback(async () => {
    if (!jsonInput.trim() || !datasetId.trim() || selectedEmbeddingModels.length === 0) {
      alert("Please provide dataset ID, JSON data, and select at least one embedding model.");
      return;
    }

    try {
      const documents = JSON.parse(jsonInput);
      if (!Array.isArray(documents)) {
        alert("JSON must be an array of documents.");
        return;
      }

      setUploadingDataset(true);

      // Upload dataset with each selected embedding model
      const uploadPromises = selectedEmbeddingModels.map(async (embeddingModel) => {
        const res = await fetch(`${API_BASE}/upload-dataset`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            dataset_id: datasetId,
            documents,
            embedding_model: embeddingModel,
            text_field: textField,
          }),
        });

        if (!res.ok) {
          const error = await res.text();
          throw new Error(`${embeddingModel}: ${error}`);
        }

        return await res.json();
      });

      const results = await Promise.allSettled(uploadPromises);
      
      const successful = results.filter(r => r.status === 'fulfilled').length;
      const failed = results.filter(r => r.status === 'rejected').length;
      
      let message = `Successfully uploaded with ${successful} embedding model(s).`;
      if (failed > 0) {
        const errors = results
          .filter((r): r is PromiseRejectedResult => r.status === 'rejected')
          .map(r => r.reason.message)
          .join('\n');
        message += `\n\nFailed with ${failed} model(s):\n${errors}`;
      }
      
      alert(message);
      setJsonInput("");
      setDatasetId("");
      await loadDatasets();
    } catch (err) {
      console.error("Upload failed:", err);
      alert(`Upload failed: ${err instanceof Error ? err.message : "Unknown error"}`);
    } finally {
      setUploadingDataset(false);
    }
  }, [jsonInput, datasetId, selectedEmbeddingModels, textField, loadDatasets]);

  const performSearch = useCallback(async () => {
    if (!searchQuery.trim() || !selectedDataset || !selectedSearchModel) {
      alert("Please provide search query, select a dataset, and select a search model.");
      return;
    }

    // Snapshot inputs so UI changes during the request won't affect in-flight search
    const snapshot = {
      query: searchQuery,
      dataset: selectedDataset,
      model: selectedSearchModel,
      startedAt: Date.now(),
    };

    setIsSearching(true);
    const myId = ++requestIdRef.current; // bump request id for race protection

    try {
      const res = await fetch(`${API_BASE}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: snapshot.query,
          embedding_model: snapshot.model,
          dataset_id: snapshot.dataset,
          top_k: 5,
        }),
      });

      if (!res.ok) {
        const error = await res.text();
        throw new Error(error);
      }

      const result = await res.json();

      // Apply only if this is the latest request
      if (myId === requestIdRef.current) {
        setSearchResults(result.results || []);
        setSearchContext(snapshot); // freeze label to executed model
      }
    } catch (err) {
      console.error("Search failed:", err);
      if (myId === requestIdRef.current) {
        alert(`Search failed: ${err instanceof Error ? err.message : "Unknown error"}`);
        setSearchResults([]);
        setSearchContext(null);
      }
    } finally {
      if (myId === requestIdRef.current) setIsSearching(false);
    }
  }, [searchQuery, selectedDataset, selectedSearchModel]);

  const deleteDataset = useCallback(async (id: string) => {
    if (!confirm(`Are you sure you want to delete dataset "${id}"?`)) return;

    try {
      const res = await fetch(`${API_BASE}/datasets/${id}`, { method: "DELETE" });
      if (res.ok) {
        await loadDatasets();
        if (selectedDataset === id) {
          setSelectedDataset("");
        }
      }
    } catch (err) {
      console.error("Delete failed:", err);
    }
  }, [loadDatasets, selectedDataset]);

  const canRun = prompt.trim().length > 0 && selected.length > 0 && !isRunning;

  const resetRun = useCallback(() => {
    setAnswers(Object.fromEntries(selected.map((m) => [m, { answer: "", error: undefined, latency_ms: 0 }])));
    setStartedAt(Date.now());
    setEndedAt(null);
  }, [selected]);

  // -------- Streaming runner (JSONL) --------
  const runPrompt = useCallback(async () => {
    if (!canRun) return;

    streamAbortRef.current?.abort();
    const controller = new AbortController();
    streamAbortRef.current = controller;

    resetRun();
    setIsRunning(true);

    const processEvent = (evt: StreamEvent) => {
      if (evt.type === "chunk") {
        setAnswers((prev) => ({
          ...prev,
          [evt.model]: {
            answer: (prev[evt.model]?.answer || "") + (evt.answer || ""),
            error: evt.error,
            latency_ms: evt.latency_ms,
          },
        }));
      } else if (evt.type === "done") {
        setIsRunning(false);
        setEndedAt(Date.now());
        streamAbortRef.current = null;
      }
    };

    try {
      const body = JSON.stringify({
        prompt,
        models: selected,
        temperature: globalTemp,
        max_tokens: globalMax,
        min_tokens: globalMin,
        model_params: modelParams,
      });

      const res = await fetch(`${API_BASE}/ask/ndjson`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
        signal: controller.signal,
      });
      if (!res.ok || !res.body) throw new Error(`Bad response: ${res.status} ${res.statusText}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buf += decoder.decode(value, { stream: true });

        const lines = buf.split("\n");
        buf = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          try {
            processEvent(JSON.parse(trimmed) as StreamEvent);
          } catch (e) {
            console.warn("Could not parse line", trimmed, e);
          }
        }
        bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
      }

      buf += decoder.decode();
      const tailLines = buf.split("\n").map((l) => l.trim()).filter(Boolean);
      for (const t of tailLines) {
        try {
          processEvent(JSON.parse(t) as StreamEvent);
        } catch (e) {
          console.warn("Could not parse tail line", t, e);
        }
      }
    } catch (err: unknown) {
      const isAbort = err instanceof DOMException && err.name === "AbortError";
      if (!isAbort) {
        console.error(err);
        setIsRunning(false);
        setEndedAt(Date.now());
      }
    } finally {
      setIsRunning(false);
      streamAbortRef.current = null;
    }
  }, [canRun, prompt, selected, resetRun, globalTemp, globalMax, globalMin, modelParams]);

  // Keyboard: Cmd/Ctrl + Enter to run
  useEffect(() => {
    const onKey = (evt: KeyboardEvent) => {
      if ((evt.metaKey || evt.ctrlKey) && evt.key === "Enter") {
        evt.preventDefault();
        if (activeTab === "chat" && canRun) {
          void runPrompt();
        } else if (activeTab === "embedding" && searchQuery.trim()) {
          void performSearch();
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [canRun, runPrompt, activeTab, searchQuery, performSearch]);

  // Small UX helpers
  const anyErrors = useMemo(() => Object.values(answers).some((a) => a?.error), [answers]);
  const elapsedMs = useMemo(() => {
    if (!startedAt) return 0;
    if (isRunning) return Date.now() - startedAt;
    if (endedAt) return Math.max(0, endedAt - startedAt);
    return 0;
  }, [startedAt, endedAt, isRunning]);

  return (
    <div className="min-h-screen grid grid-rows-[auto_auto_1fr_auto] gap-6 p-6 sm:p-8 bg-white text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      {/* Header */}
      <header className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-orange-600 dark:text-orange-400">
            Multi-LLM AI Platform
          </h1>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Chat with multiple models and perform semantic search with embeddings.
          </p>
        </div>
        <div className="text-sm text-zinc-500 dark:text-zinc-400">
          {loadingProviders ? "Loading providers…" : `${providers.length} provider(s), ${allModels.length} chat model(s), ${allEmbeddingModels.length} embedding model(s)`}
        </div>
      </header>

      {/* Tab Navigation */}
      <nav className="flex border-b border-zinc-200 dark:border-zinc-800">
        <button
          onClick={() => setActiveTab("chat")}
          className={`px-6 py-3 text-sm font-medium border-b-2 transition ${
            activeTab === "chat"
              ? "border-orange-500 text-orange-600 dark:text-orange-400"
              : "border-transparent text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300"
          }`}
        >
          Chat Models
        </button>
        <button
          onClick={() => setActiveTab("embedding")}
          className={`px-6 py-3 text-sm font-medium border-b-2 transition ${
            activeTab === "embedding"
              ? "border-orange-500 text-orange-600 dark:text-orange-400"
              : "border-transparent text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300"
          }`}
        >
          Embeddings
        </button>
      </nav>

      {/* Tab Content */}
      {activeTab === "chat" && (
        <main className="grid grid-cols-1 xl:grid-cols-[360px_1fr] gap-6 items-start">
          {/* Left rail: Chat Controls */}
          <section className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 sm:p-5 bg-white dark:bg-zinc-950 shadow-sm">
            <div className="space-y-4">
              <label className="text-sm font-medium">Prompt</label>
              <textarea
                className="w-full rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 outline-none focus:ring-2 focus:ring-orange-300/60 bg-white dark:bg-zinc-900"
                placeholder="Ask your question once. e.g., 'Explain RAG vs fine-tuning for my use case.'"
                value={prompt}
                onChange={(evt) => setPrompt(evt.target.value)}
              />

              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Models</label>
                <div className="flex gap-2 text-xs">
                  <button
                    onClick={selectAll}
                    className="px-2 py-1 rounded-lg border border-orange-200 text-zinc-800 dark:text-zinc-100 bg-orange-50 hover:bg-orange-100 dark:bg-orange-400/10 dark:hover:bg-orange-400/20 transition"
                  >
                    Select all
                  </button>
                  <button
                    onClick={clearAll}
                    className="px-2 py-1 rounded-lg border border-orange-200 text-zinc-800 dark:text-zinc-100 bg-orange-50 hover:bg-orange-100 dark:bg-orange-400/10 dark:hover:bg-orange-400/20 transition"
                  >
                    Clear
                  </button>
                </div>
              </div>

              <div className="max-h-[280px] overflow-auto rounded-xl border border-zinc-200 dark:border-zinc-800 p-2 grid grid-cols-1 sm:grid-cols-2 gap-1">
                {allModels.length === 0 && (
                  <div className="text-sm text-zinc-500 dark:text-zinc-400">No models discovered yet.</div>
                )}
                {allModels.map((m) => (
                  <label
                    key={m}
                    className="flex items-center gap-2 px-2 py-1 rounded-lg cursor-pointer hover:bg-orange-50 dark:hover:bg-orange-400/10"
                  >
                    <input
                      type="checkbox"
                      className="accent-orange-600 dark:accent-orange-500 cursor-pointer"
                      checked={selected.includes(m)}
                      onChange={() => toggleModel(m)}
                    />
                    <span className="text-sm font-mono">{m}</span>
                  </label>
                ))}
              </div>

              {/* Global defaults */}
              <div className="space-y-3 text-sm">
                <div>
                  <label className="block mb-1 font-medium">Global temp</label>
                  <input
                    type="number" step={0.1} min={0} max={2}
                    value={globalTemp}
                    onChange={(e) => setGlobalTemp(Number(e.target.value))}
                    className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900"
                  />
                </div>
                <div>
                  <label className="block mb-1 font-medium">Global max_tokens</label>
                  <input
                    type="number" min={1}
                    value={globalMax}
                    onChange={(e) => setGlobalMax(Number(e.target.value))}
                    className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900"
                  />
                </div>
                <div>
                  <label className="block mb-1 font-medium">Global min_tokens</label>
                  <input
                    type="number" min={1}
                    value={globalMin ?? ""}
                    placeholder="optional"
                    onChange={(e) => setGlobalMin(e.target.value ? Number(e.target.value) : undefined)}
                    className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900"
                  />
                </div>
              </div>

              {/* Per-model overrides */}
              <div className="mt-3 space-y-2">
                {allModels.map((m) => (
                  <details
                    key={m}
                    className="rounded-lg border border-orange-200 dark:border-orange-500/40 p-2 bg-orange-50/60 dark:bg-orange-400/10"
                  >
                    <summary className="cursor-pointer flex items-center justify-between">
                      <span className="font-mono text-sm">{m}</span>
                      <span className="text-xs text-zinc-500 dark:text-zinc-400">Overrides</span>
                    </summary>
                    <div className="mt-2 grid grid-cols-3 gap-3 text-sm">
                      <div>
                        <label className="block mb-1">temp</label>
                        <input
                          type="number" step={0.1} min={0} max={2}
                          value={modelParams[m]?.temperature ?? ""}
                          placeholder={`↳ ${globalTemp}`}
                          onChange={(e) => updateParam(m, "temperature", e.target.value ? Number(e.target.value) : undefined)}
                          className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900"
                        />
                      </div>
                      <div>
                        <label className="block mb-1">max_tokens</label>
                        <input
                          type="number" min={1}
                          value={modelParams[m]?.max_tokens ?? ""}
                          placeholder={`↳ ${globalMax}`}
                          onChange={(e) => updateParam(m, "max_tokens", e.target.value ? Number(e.target.value) : undefined)}
                          className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900"
                        />
                      </div>
                      <div>
                        <label className="block mb-1">min_tokens</label>
                        <input
                          type="number" min={1}
                          value={modelParams[m]?.min_tokens ?? ""}
                          placeholder={globalMin ? `↳ ${globalMin}` : "optional"}
                          onChange={(e) => updateParam(m, "min_tokens", e.target.value ? Number(e.target.value) : undefined)}
                          className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900"
                        />
                      </div>
                    </div>
                  </details>
                ))}
              </div>

              <button
                onClick={() => void runPrompt()}
                disabled={!canRun}
                className="w-full rounded-xl py-2 px-4 font-medium text-white bg-orange-600 hover:bg-orange-700 dark:bg-orange-500 dark:hover:bg-orange-600 transition disabled:opacity-50"
              >
                {isRunning ? "Running…" : "Run prompt"}
              </button>
            </div>
          </section>

          {/* Right rail: Results */}
          <section className="space-y-4">
            {Object.entries(answers).map(([model, { answer, error, latency_ms }]) => (
              <div
                key={model}
                className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 bg-white dark:bg-zinc-950 shadow-sm"
              >
                <div className="flex items-center justify-between mb-2">
                  <h2 className="text-sm font-semibold font-mono">{model}</h2>
                  <span className="text-xs text-zinc-500 dark:text-zinc-400">
                    {error ? "⌘ Error" : latency_ms ? `${(latency_ms / 1000).toFixed(1)}s` : isRunning ? "running…" : ""}
                  </span>
                </div>
                <pre className="whitespace-pre-wrap text-sm">{error ? error : answer}</pre>
              </div>
            ))}
            <div ref={bottomRef} />
          </section>
        </main>
      )}

      {activeTab === "embedding" && (
        <main className="grid grid-cols-1 xl:grid-cols-[400px_1fr] gap-6 items-start">
          {/* Left rail: Embedding Controls */}
          <section className="space-y-6">
            {/* Embedding Model Selection */}
            <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 sm:p-5 bg-white dark:bg-zinc-950 shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-orange-600 dark:text-orange-400">Select Embedding Models</h3>
                <div className="flex gap-2 text-xs">
                  <button
                    onClick={selectAllEmbedding}
                    className="px-2 py-1 rounded-lg border border-orange-200 text-zinc-800 dark:text-zinc-100 bg-orange-50 hover:bg-orange-100 dark:bg-orange-400/10 dark:hover:bg-orange-400/20 transition"
                  >
                    Select all
                  </button>
                  <button
                    onClick={clearAllEmbedding}
                    className="px-2 py-1 rounded-lg border border-orange-200 text-zinc-800 dark:text-zinc-100 bg-orange-50 hover:bg-orange-100 dark:bg-orange-400/10 dark:hover:bg-orange-400/20 transition"
                  >
                    Clear
                  </button>
                </div>
              </div>
              
              <div className="max-h-[200px] overflow-auto rounded-xl border border-zinc-200 dark:border-zinc-800 p-2 grid grid-cols-1 gap-1">
                {allEmbeddingModels.length === 0 && (
                  <div className="text-sm text-zinc-500 dark:text-zinc-400">No embedding models discovered yet.</div>
                )}
                {allEmbeddingModels.map((m) => (
                  <label
                    key={m}
                    className="flex items-center gap-2 px-2 py-1 rounded-lg cursor-pointer hover:bg-orange-50 dark:hover:bg-orange-400/10"
                  >
                    <input
                      type="checkbox"
                      className="accent-orange-600 dark:accent-orange-500 cursor-pointer"
                      checked={selectedEmbeddingModels.includes(m)}
                      onChange={() => toggleEmbeddingModel(m)}
                    />
                    <span className="text-sm font-mono">{m}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Dataset Upload */}
            <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 sm:p-5 bg-white dark:bg-zinc-950 shadow-sm">
              <h3 className="text-lg font-semibold mb-4 text-orange-600 dark:text-orange-400">Upload Dataset</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Dataset ID</label>
                  <input
                    type="text"
                    value={datasetId}
                    onChange={(e) => setDatasetId(e.target.value)}
                    placeholder="my-documents"
                    className="w-full rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 outline-none focus:ring-2 focus:ring-orange-300/60 bg-white dark:bg-zinc-900"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">Text Field Name</label>
                  <input
                    type="text"
                    value={textField}
                    onChange={(e) => setTextField(e.target.value)}
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
                    onChange={(e) => setJsonInput(e.target.value)}
                    placeholder={`[
  {"text": "Document 1 content", "title": "Doc 1"},
  {"text": "Document 2 content", "title": "Doc 2"}
]`}
                    rows={8}
                    className="w-full rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 outline-none focus:ring-2 focus:ring-orange-300/60 bg-white dark:bg-zinc-900 font-mono text-sm"
                  />
                </div>

                <button
                  onClick={uploadDataset}
                  disabled={uploadingDataset || !jsonInput.trim() || !datasetId.trim() || selectedEmbeddingModels.length === 0}
                  className="w-full rounded-xl py-2 px-4 font-medium text-white bg-orange-600 hover:bg-orange-700 dark:bg-orange-500 dark:hover:bg-orange-600 transition disabled:opacity-50"
                >
                  {uploadingDataset ? "Uploading…" : `Upload with ${selectedEmbeddingModels.length} model(s)`}
                </button>
              </div>
            </div>

            {/* Search Interface */}
            <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 sm:p-5 bg-white dark:bg-zinc-950 shadow-sm">
              <h3 className="text-lg font-semibold mb-4 text-orange-600 dark:text-orange-400">Semantic Search</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Search Model</label>
                  <select
                    value={selectedSearchModel}
                    onChange={(e) => setSelectedSearchModel(e.target.value)}
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
                    onChange={(e) => setSelectedDataset(e.target.value)}
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
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="What are you looking for?"
                    rows={3}
                    className="w-full rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 outline-none focus:ring-2 focus:ring-orange-300/60 bg-white dark:bg-zinc-900"
                  />
                </div>

                <button
                  onClick={performSearch}
                  disabled={isSearching || !searchQuery.trim() || !selectedDataset || !selectedSearchModel}
                  className="w-full rounded-xl py-2 px-4 font-medium text-white bg-orange-600 hover:bg-orange-700 dark:bg-orange-500 dark:hover:bg-orange-600 transition disabled:opacity-50"
                >
                  {isSearching ? "Searching…" : "Search"}
                </button>
              </div>
            </div>

            {/* Dataset Management */}
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
                        onClick={() => deleteDataset(dataset.dataset_id)}
                        className="px-3 py-1 text-xs rounded-lg border border-red-200 text-red-700 bg-red-50 hover:bg-red-100 dark:border-red-800 dark:text-red-400 dark:bg-red-900/20 dark:hover:bg-red-900/40 transition"
                      >
                        Delete
                      </button>
                    </div>
                  ))
                )}
              </div>
            </div>
          </section>

          {/* Right rail: Search Results */}
          <section className="space-y-4">
            {searchResults.length > 0 && (
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
            )}

            {searchResults.length === 0 && searchQuery && !isSearching && (
              <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-8 bg-white dark:bg-zinc-950 shadow-sm text-center">
                <p className="text-zinc-500 dark:text-zinc-400">No results found for your search query.</p>
              </div>
            )}

            {!searchQuery && (
              <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-8 bg-white dark:bg-zinc-950 shadow-sm text-center">
                <p className="text-zinc-500 dark:text-zinc-400">
                  Upload a dataset and perform a search to see results here.
                </p>
              </div>
            )}
          </section>
        </main>
      )}

      {/* Footer */}
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
              {allEmbeddingModels.length} embedding models available
            </span>
          </>
        )}
      </footer>
    </div>
  );
}
