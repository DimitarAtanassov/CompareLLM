"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// ---------------- Types kept in sync with your backend ----------------
type ProviderInfo = {
  name: string;
  type: string;
  base_url: string;
  models: string[];
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

// ---------------- Config ----------------
const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "/backend");

// ---------------- Page ----------------
export default function Page() {
  const [loadingProviders, setLoadingProviders] = useState(false);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [allModels, setAllModels] = useState<string[]>([]);

  const [selected, setSelected] = useState<string[]>([]);
  const [prompt, setPrompt] = useState<string>("");

  const [isRunning, setIsRunning] = useState(false);
  const [answers, setAnswers] = useState<AskAnswers>({});
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [endedAt, setEndedAt] = useState<number | null>(null);

  const streamAbortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

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
        setAllModels(models);
      } catch (err) {
        console.error(err);
      } finally {
        setLoadingProviders(false);
      }
    };
    load();
  }, []);

  // -------- Helpers --------
  const toggleModel = (m: string) =>
    setSelected((prev) => (prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]));

  const selectAll = () => setSelected(allModels);
  const clearAll = () => setSelected([]);

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
        if (canRun) void runPrompt();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [canRun, runPrompt]);

  // Small UX helpers
  const anyErrors = useMemo(() => Object.values(answers).some((a) => a?.error), [answers]);
  const elapsedMs = useMemo(() => {
    if (!startedAt) return 0;
    if (isRunning) return Date.now() - startedAt;
    if (endedAt) return Math.max(0, endedAt - startedAt);
    return 0;
  }, [startedAt, endedAt, isRunning]);

  return (
    <div className="min-h-screen grid grid-rows-[auto_1fr_auto] gap-6 p-6 sm:p-8 bg-white text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      {/* Header */}
      <header className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-orange-600 dark:text-orange-400">
            Multi-LLM Prompt Runner
          </h1>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Prompt multiple models at once and compare responses side-by-side.
          </p>
        </div>
        <div className="text-sm text-zinc-500 dark:text-zinc-400">
          {loadingProviders ? "Loading providers…" : `${providers.length} provider(s), ${allModels.length} model(s)`}
        </div>
      </header>

      {/* Main content */}
      <main className="grid grid-cols-1 xl:grid-cols-[360px_1fr] gap-6 items-start">
        {/* Left rail: Controls */}
        <section className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 sm:p-5 bg-white dark:bg-zinc-950 shadow-sm">
          <div className="space-y-4">
            <label className="text-sm font-medium">Prompt</label>
            <textarea
              className="w-full rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 outline-none focus:ring-2 focus:ring-orange-300/60 bg-white dark:bg-zinc-900"
              placeholder="Ask your question once. e.g., ‘Explain RAG vs fine-tuning for my use case.’"
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
                  {error ? "❌ Error" : latency_ms ? `${(latency_ms / 1000).toFixed(1)}s` : isRunning ? "running…" : ""}
                </span>
              </div>
              <pre className="whitespace-pre-wrap text-sm">{error ? error : answer}</pre>
            </div>
          ))}
          <div ref={bottomRef} />
        </section>
      </main>

      {/* Footer */}
      <footer className="text-xs text-zinc-500 dark:text-zinc-400 flex justify-between">
        <span>{selected.length} selected</span>
        {anyErrors && <span className="text-orange-600 dark:text-orange-400">Some models returned errors</span>}
        {startedAt && (
          <span>
            Elapsed: {(elapsedMs / 1000).toFixed(1)}s{isRunning ? " (live)" : ""}
          </span>
        )}
      </footer>
    </div>
  );
}
