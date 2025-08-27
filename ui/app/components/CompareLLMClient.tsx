// app/components/CompareLLMClient.tsx
"use client";

import { JSX, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AskAnswers,
  Dataset,
  ModelChat,
  ModelParamsMap,
  MultiSearchResponse,
  PerModelParam,
  ProviderBrand,
  ProviderInfo,
  ProvidersResp,
  ProviderWire,
  SearchResult,
} from "../lib/types";
import { coerceBrand, coerceWire } from "../lib/utils";
import { API_BASE } from "../lib/config";
import InteractiveChatModal from "./chat/InteractiveChatModal";
import Tabs from "./ui/Tabs";
import ModelList from "./chat/ModelList";
import ProviderParameterEditor from "./chat/ProviderParameterEditor";
import ChatResults from "./chat/ChatResults";
import { PROVIDER_BADGE_BG } from "../lib/colors";
import EmbeddingLeftRail from "./embeddings/EmbeddingLeftRail";
import EmbeddingRightRail from "./embeddings/EmbeddingRightRail";

/** Streamed event from /v2/chat/completions/enhanced/ndjson */
type StreamEvent =
  | { type: "meta"; models: string[] }
  | { type: "chunk"; model: string; answer?: string; error?: string; latency_ms: number }
  | { type: "done" };

/** Body shape for the enhanced chat endpoint */
type EnhancedChatRequest = {
  messages: { role: string; content: string }[];
  models: string[];
  temperature?: number;
  max_tokens?: number;
  min_tokens?: number;
  anthropic_params?: {
    thinking_enabled?: boolean;
    thinking_budget_tokens?: number;
    top_k?: number;
    top_p?: number;
    stop_sequences?: string[];
  };
  openai_params?: {
    top_p?: number;
    frequency_penalty?: number;
    presence_penalty?: number;
    seed?: number;
  };
  gemini_params?: {
    top_k?: number;
    top_p?: number;
    candidate_count?: number;
    safety_settings?: unknown[];
  };
  ollama_params?: {
    mirostat?: number;
    mirostat_eta?: number;
    mirostat_tau?: number;
    num_ctx?: number;
    repeat_penalty?: number;
  };
  model_params?: Record<string, Partial<PerModelParam>>;
};

export default function CompareLLMClient(): JSX.Element {
  // ==== STATE (copied 1:1 from your page.tsx) ====
  const [activeTab, setActiveTab] = useState<"chat" | "embedding">("chat");
  const [embedView, setEmbedView] = useState<"single" | "compare">("single");
  const [loadingProviders, setLoadingProviders] = useState(false);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [allModels, setAllModels] = useState<string[]>([]);
  const [allEmbeddingModels, setAllEmbeddingModels] = useState<string[]>([]);

  const [selected, setSelected] = useState<string[]>([]);
  const [prompt, setPrompt] = useState<string>("");
  const [isRunning, setIsRunning] = useState(false);
  const [answers, setAnswers] = useState<AskAnswers>({});
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [endedAt, setEndedAt] = useState<number | null>(null);

  const [activeModel, setActiveModel] = useState<string | null>(null);
  const [modelChats, setModelChats] = useState<Record<string, ModelChat>>({});
  const [interactivePrompt, setInteractivePrompt] = useState<string>("");

  // Embeddings state (unchanged)
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedEmbeddingModels, setSelectedEmbeddingModels] = useState<string[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [selectedSearchModel, setSelectedSearchModel] = useState<string>("");
  const [uploadingDataset, setUploadingDataset] = useState(false);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false); // (unused, kept to preserve API shape)
  const [jsonInput, setJsonInput] = useState<string>("");
  const [datasetId, setDatasetId] = useState<string>("");
  const [textField, setTextField] = useState<string>("text");
  const [compareQuery, setCompareQuery] = useState<string>("");
  const [isSearchingSingle, setIsSearchingSingle] = useState(false);
  const [isComparing, setIsComparing] = useState(false);
  const [searchContext, setSearchContext] = useState<{
    model: string;
    dataset: string;
    query: string;
    startedAt: number;
  } | null>(null);
  const [multiSearchResults, setMultiSearchResults] = useState<MultiSearchResponse | null>(null);

  const bottomRef = useRef<HTMLDivElement | null>(null);
  const streamAbortRef = useRef<AbortController | null>(null);
  const interactiveAbortRef = useRef<AbortController | null>(null);
  const chatBottomRef = useRef<HTMLDivElement | null>(null); // (unused)
  const requestIdRef = useRef(0);

  const [modelParams, setModelParams] = useState<ModelParamsMap>({});
  const [globalTemp, setGlobalTemp] = useState<number | undefined>(undefined);
  const [globalMax, setGlobalMax] = useState<number | undefined>(undefined);
  const [globalMin, setGlobalMin] = useState<number | undefined>(undefined);
  const [topKSingle, setTopKSingle] = useState<number>(5);
  const [topKCompare, setTopKCompare] = useState<number>(5);
  const useEnhancedAPI = true;
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set());

  // ==== MEMO MAPS ====
  const modelToProvider = useMemo(() => {
    const map: Record<string, ProviderBrand> = {};
    providers.forEach((p) => {
      (p.models || []).forEach((m) => (map[m] = coerceBrand(p.type)));
      (p.embedding_models || []).forEach((m) => (map[m] = coerceBrand(p.type)));
    });
    return map;
  }, [providers]);

  const modelToWire = useMemo(() => {
    const map: Record<string, ProviderWire> = {};
    providers.forEach((p) => {
      const wire = coerceWire(p);
      (p.models || []).forEach((m) => (map[m] = wire));
      (p.embedding_models || []).forEach((m) => (map[m] = wire));
    });
    return map;
  }, [providers]);

  const getProviderType = useCallback((m: string): ProviderBrand => modelToProvider[m] ?? "unknown", [modelToProvider]);
  const getProviderWire = useCallback((m: string): ProviderWire => modelToWire[m] ?? "unknown", [modelToWire]);

  // ==== EFFECTS & LOADERS ====
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
        if (embeddingModels.length > 0) setSelectedSearchModel(embeddingModels[0]);
      } finally {
        setLoadingProviders(false);
      }
    };
    void load();
  }, []);

  // ==== Handlers ====
  const toggleModel = (m: string) =>
    setSelected((prev) => (prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]));
  const selectAll = () => setSelected(allModels);
  const clearAll = () => setSelected([]);

  const updateParam = useCallback((model: string, params: PerModelParam) => {
    setModelParams((prev) => ({ ...prev, [model]: params }));
  }, []);
  const toggleModelExpansion = useCallback((model: string) => {
    setExpandedModels((prev) => {
      const next = new Set(prev);
      next.has(model) ? next.delete(model) : next.add(model);
      return next;
    });
  }, []);

  // =============================
  // Interactive Chat
  // =============================
  const openModelChat = useCallback(
    (model: string) => {
      setActiveModel(model);
      setModelChats((prev) => {
        if (prev[model]) return prev;
        const initialMessages: ModelChat["messages"] = [];
        if (prompt.trim()) {
          initialMessages.push({ role: "user", content: prompt.trim(), timestamp: Date.now() });
        }
        const modelAnswer = answers[model];
        if (modelAnswer?.answer && !modelAnswer.error) {
          initialMessages.push({ role: "assistant", content: modelAnswer.answer, timestamp: Date.now() });
        }
        return { ...prev, [model]: { messages: initialMessages, isStreaming: false, currentResponse: "" } };
      });
    },
    [answers, prompt]
  );

  const closeModelChat = useCallback(() => {
    setActiveModel(null);
    interactiveAbortRef.current?.abort();
  }, []);

  const pruneUndefined = <T extends Record<string, unknown>>(obj: T): Partial<T> => {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(obj)) if (v !== undefined) out[k] = v;
    return out as Partial<T>;
  };

  const sendInteractiveMessage = useCallback(async () => {
    if (!activeModel || !interactivePrompt.trim()) return;

    const message = interactivePrompt.trim();
    setInteractivePrompt("");

    setModelChats((prev) => ({
      ...prev,
      [activeModel]: {
        ...prev[activeModel],
        messages: [...(prev[activeModel]?.messages || []), { role: "user", content: message, timestamp: Date.now() }],
        isStreaming: true,
        currentResponse: "",
      },
    }));

    interactiveAbortRef.current?.abort();
    const controller = new AbortController();
    interactiveAbortRef.current = controller;

    try {
      const currentChat = modelChats[activeModel] || { messages: [] as ModelChat["messages"] };
      const conversationHistory = [
        ...currentChat.messages,
        { role: "user" as const, content: message, timestamp: Date.now() },
      ];
      const apiMessages = conversationHistory.map((m) => ({ role: m.role, content: m.content }));

      const modelParam = modelParams[activeModel] || {};
      const providerWire = getProviderWire(activeModel);
      const hasProviderParams = Object.keys(modelParam).some(
        (k) => !["temperature", "max_tokens", "min_tokens"].includes(k)
      );

      let endpoint = "";
      let body = "";

      if (useEnhancedAPI || hasProviderParams) {
        endpoint = `${API_BASE}/v2/chat/completions/enhanced`;
        const enhancedRequest: EnhancedChatRequest = {
          messages: apiMessages,
          models: [activeModel],
        };
        const t = modelParam.temperature ?? globalTemp;
        const mx = modelParam.max_tokens ?? globalMax;
        const mn = modelParam.min_tokens ?? globalMin;
        if (t !== undefined) enhancedRequest.temperature = t;
        if (mx !== undefined) enhancedRequest.max_tokens = mx;
        if (mn !== undefined) enhancedRequest.min_tokens = mn;

        if (providerWire === "anthropic") {
          const p: NonNullable<EnhancedChatRequest["anthropic_params"]> = pruneUndefined({
            thinking_enabled: modelParam.thinking_enabled,
            thinking_budget_tokens: modelParam.thinking_budget_tokens,
            top_k: modelParam.top_k,
            top_p: modelParam.top_p,
            stop_sequences: modelParam.stop_sequences,
          });
          if (Object.keys(p).length) enhancedRequest.anthropic_params = p;
        } else if (providerWire === "openai") {
          const p: NonNullable<EnhancedChatRequest["openai_params"]> = pruneUndefined({
            top_p: modelParam.top_p,
            frequency_penalty: modelParam.frequency_penalty,
            presence_penalty: modelParam.presence_penalty,
            seed: modelParam.seed,
          });
          if (Object.keys(p).length) enhancedRequest.openai_params = p;
        } else if (providerWire === "gemini") {
          const p: NonNullable<EnhancedChatRequest["gemini_params"]> = pruneUndefined({
            top_k: modelParam.top_k,
            top_p: modelParam.top_p,
            candidate_count: modelParam.candidate_count,
            safety_settings: modelParam.safety_settings,
          });
          if (Object.keys(p).length) enhancedRequest.gemini_params = p;
        } else if (providerWire === "ollama") {
          const p: NonNullable<EnhancedChatRequest["ollama_params"]> = pruneUndefined({
            mirostat: modelParam.mirostat,
            mirostat_eta: modelParam.mirostat_eta,
            mirostat_tau: modelParam.mirostat_tau,
            num_ctx: modelParam.num_ctx,
            repeat_penalty: modelParam.repeat_penalty,
          });
          if (Object.keys(p).length) enhancedRequest.ollama_params = p;
        }

        body = JSON.stringify(enhancedRequest);
      } else {
        endpoint = `${API_BASE}/v1/chat/completions`;
        const stdPayload: Record<string, unknown> = {
          model: activeModel,
          messages: apiMessages,
          stream: false,
        };
        const temp = modelParam.temperature ?? globalTemp;
        const maxTok = modelParam.max_tokens ?? globalMax;
        if (temp !== undefined) stdPayload.temperature = temp;
        if (maxTok !== undefined) stdPayload.max_tokens = maxTok;
        body = JSON.stringify(stdPayload);
      }

      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
        signal: controller.signal,
      });

      if (!res.ok) {
        const errorData = (await res.json().catch(() => ({ detail: res.statusText }))) as { detail?: string };
        throw new Error(errorData.detail || `HTTP ${res.status}`);
      }

      type EnhancedResp = { answers?: AskAnswers };
      type OpenAIResp = { choices?: { message?: { content?: string } }[] };

      const result = (await res.json()) as unknown;

      const assistantMessage: string =
        useEnhancedAPI || hasProviderParams
          ? (result as EnhancedResp).answers?.[activeModel]?.answer || "No response"
          : (result as OpenAIResp).choices?.[0]?.message?.content || "No response";

      setModelChats((prev) => ({
        ...prev,
        [activeModel]: {
          ...prev[activeModel],
          messages: [
            ...(prev[activeModel]?.messages || []),
            { role: "assistant", content: assistantMessage, timestamp: Date.now() },
          ],
          isStreaming: false,
          currentResponse: "",
        },
      }));
    } catch (err) {
      const isAbort = err instanceof DOMException && err.name === "AbortError";
      if (!isAbort) {
        const msg = err instanceof Error ? err.message : "Unknown error";
        setModelChats((prev) => ({
          ...prev,
          [activeModel!]: {
            ...prev[activeModel!],
            isStreaming: false,
            currentResponse: "",
            messages: [
              ...(prev[activeModel!]?.messages || []),
              { role: "assistant", content: `Error: ${msg}`, timestamp: Date.now() },
            ],
          },
        }));
      }
    } finally {
      interactiveAbortRef.current = null;
    }
  }, [activeModel, interactivePrompt, modelChats, modelParams, getProviderWire, globalTemp, globalMax, globalMin]);

  // =============================
  // Providers / datasets
  // =============================
  const loadDatasets = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/datasets`);
      if (!res.ok) return;
      const data = (await res.json()) as { datasets?: Dataset[] };
      setDatasets(data.datasets || []);
    } catch (err) {
      console.error("Failed to load datasets:", err);
    }
  }, []);

  useEffect(() => {
    if (activeTab === "embedding") void loadDatasets();
  }, [activeTab, loadDatasets]);

  const uploadDataset = useCallback(async () => {
    if (!jsonInput.trim() || !datasetId.trim() || selectedEmbeddingModels.length === 0) {
      alert("Please provide dataset ID, JSON data, and select at least one embedding model.");
      return;
    }
    try {
      const documents = JSON.parse(jsonInput) as unknown;
      if (!Array.isArray(documents)) {
        alert("JSON must be an array of documents.");
        return;
      }
      setUploadingDataset(true);
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
        return (await res.json()) as unknown;
      });
      const results = await Promise.allSettled(uploadPromises);
      const successful = results.filter((r) => r.status === "fulfilled").length;
      const failed = results.filter((r) => r.status === "rejected").length;
      let message = `Successfully uploaded with ${successful} embedding model(s).`;
      if (failed > 0) {
        const errors = results
          .filter((r): r is PromiseRejectedResult => r.status === "rejected")
          .map((r) => (r.reason as Error)?.message)
          .join("\n");
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
    const snapshot = {
      query: searchQuery,
      dataset: selectedDataset,
      model: selectedSearchModel,
      startedAt: Date.now(),
    };

    setIsSearchingSingle(true);
    const myId = ++requestIdRef.current;
    try {
      const res = await fetch(`${API_BASE}/search/semantic`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: snapshot.query,
          embedding_model: snapshot.model,
          dataset_id: snapshot.dataset,
          top_k: topKSingle || 5,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const result = (await res.json()) as { results?: SearchResult[] };
      if (myId === requestIdRef.current) {
        setSearchResults(result.results || []);
        setSearchContext(snapshot);
      }
    } catch (err) {
      console.error("Search failed:", err);
      if (myId === requestIdRef.current) {
        alert(`Search failed: ${err instanceof Error ? err.message : "Unknown error"}`);
        setSearchResults([]);
        setSearchContext(null);
      }
    } finally {
      if (myId === requestIdRef.current) setIsSearchingSingle(false);
    }
  }, [searchQuery, selectedDataset, selectedSearchModel, topKSingle]);

  const performMultiSearch = useCallback(async () => {
    if (!compareQuery.trim()) {
      alert("Please provide a comparison query.");
      return;
    }
    if (selectedEmbeddingModels.length === 0) {
      alert("Select at least one embedding model (left rail) to compare.");
      return;
    }

    setIsComparing(true);
    setMultiSearchResults(null);
    const myId = ++requestIdRef.current;

    try {
      const res = await fetch(`${API_BASE}/v2/search/self-dataset-compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: compareQuery,
          embedding_models: selectedEmbeddingModels,
          top_k: topKCompare || 5,
        }),
      });

      if (!res.ok) {
        let msg = await res.text();
        try {
          msg = (JSON.parse(msg) as { detail?: string }).detail || msg;
        } catch {
          /* noop */
        }
        throw new Error(msg || `HTTP ${res.status}`);
      }

      const json = (await res.json()) as MultiSearchResponse;
      if (myId === requestIdRef.current) setMultiSearchResults(json);
    } catch (err) {
      console.error("Self-dataset compare failed:", err);
      if (myId === requestIdRef.current) {
        alert(`Compare failed: ${err instanceof Error ? err.message : "Unknown error"}`);
        setMultiSearchResults(null);
      }
    } finally {
      if (myId === requestIdRef.current) setIsComparing(false);
    }
  }, [compareQuery, selectedEmbeddingModels, topKCompare]);

  const deleteDataset = useCallback(
    async (id: string) => {
      if (!confirm(`Are you sure you want to delete dataset "${id}"?`)) return;
      try {
        const res = await fetch(`${API_BASE}/datasets/${id}`, { method: "DELETE" });
        if (res.ok) {
          await loadDatasets();
          if (selectedDataset === id) setSelectedDataset("");
        }
      } catch (err) {
        console.error("Delete failed:", err);
      }
    },
    [loadDatasets, selectedDataset]
  );

  // =============================
  // Run Prompt (streaming + non-stream)
  // =============================
  const canRun = prompt.trim().length > 0 && selected.length > 0 && !isRunning;

  const resetRun = useCallback(() => {
    setAnswers(
      Object.fromEntries(selected.map((m) => [m, { answer: "", error: undefined, latency_ms: 0 }])) as AskAnswers
    );
    setStartedAt(Date.now());
    setEndedAt(null);
  }, [selected]);

  const runPrompt = useCallback(async () => {
    if (!canRun) return;

    // cancel previous stream
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
      const ndjsonPayload: Record<string, unknown> = {
        messages: [{ role: "user", content: prompt }],
        models: selected,
        ...(globalTemp !== undefined ? { temperature: globalTemp } : {}),
        ...(globalMax !== undefined ? { max_tokens: globalMax } : {}),
        ...(globalMin !== undefined ? { min_tokens: globalMin } : {}),
        model_params: modelParams,
      };

      const res = await fetch(`${API_BASE}/v2/chat/completions/enhanced/ndjson`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(ndjsonPayload),
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
            processEvent(JSON.parse(trimmed) as unknown as StreamEvent);
          } catch (e) {
            console.warn("Could not parse line", trimmed, e);
          }
        }
      }

      buf += decoder.decode();
      const tail = buf
        .split("\n")
        .map((l) => l.trim())
        .filter(Boolean);
      for (const t of tail) {
        try {
          processEvent(JSON.parse(t) as unknown as StreamEvent);
        } catch (e) {
          console.warn("Could not parse tail line", t, e);
        }
      }
    } catch (err) {
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
  }, [canRun, prompt, selected, modelParams, globalTemp, globalMax, globalMin, resetRun]);

  const runEnhancedPrompt = useCallback(async () => {
    if (!canRun) return;
    setIsRunning(true);
    resetRun();
    try {
      const perModel: Record<string, Partial<PerModelParam>> = {};
      for (const m of selected) {
        const p = modelParams[m] || {};
        const trimmed = pruneUndefined({
          temperature: p.temperature,
          max_tokens: p.max_tokens,
          min_tokens: p.min_tokens,
        });
        if (Object.keys(trimmed).length > 0) perModel[m] = trimmed;
      }

      const enhancedRequest: EnhancedChatRequest = {
        messages: [{ role: "user", content: prompt }],
        models: selected,
      };
      if (globalTemp !== undefined) enhancedRequest.temperature = globalTemp;
      if (globalMax !== undefined) enhancedRequest.max_tokens = globalMax;
      if (globalMin !== undefined) enhancedRequest.min_tokens = globalMin;
      if (Object.keys(perModel).length > 0) enhancedRequest.model_params = perModel;

      const anthropicModels = selected.filter((m) => getProviderWire(m) === "anthropic");
      const openaiModels = selected.filter((m) => getProviderWire(m) === "openai");
      const geminiModels = selected.filter((m) => getProviderWire(m) === "gemini");
      const ollamaModels = selected.filter((m) => getProviderWire(m) === "ollama");

      if (anthropicModels.length > 0) {
        const merged = pruneUndefined(
          anthropicModels.reduce((acc, model) => {
            const p = modelParams[model] || {};
            acc.thinking_enabled = acc.thinking_enabled ?? p.thinking_enabled;
            acc.thinking_budget_tokens = acc.thinking_budget_tokens ?? p.thinking_budget_tokens;
            acc.top_k = acc.top_k ?? p.top_k;
            acc.top_p = acc.top_p ?? p.top_p;
            acc.stop_sequences = acc.stop_sequences ?? p.stop_sequences;
            return acc;
          }, {} as NonNullable<EnhancedChatRequest["anthropic_params"]>)
        );
        if (Object.keys(merged).length > 0) enhancedRequest.anthropic_params = merged;
      }

      if (openaiModels.length > 0) {
        const merged = pruneUndefined(
          openaiModels.reduce((acc, model) => {
            const p = modelParams[model] || {};
            acc.top_p = acc.top_p ?? p.top_p;
            acc.frequency_penalty = acc.frequency_penalty ?? p.frequency_penalty;
            acc.presence_penalty = acc.presence_penalty ?? p.presence_penalty;
            acc.seed = acc.seed ?? p.seed;
            return acc;
          }, {} as NonNullable<EnhancedChatRequest["openai_params"]>)
        );
        if (Object.keys(merged).length > 0) enhancedRequest.openai_params = merged;
      }

      if (geminiModels.length > 0) {
        const merged = pruneUndefined(
          geminiModels.reduce((acc, model) => {
            const p = modelParams[model] || {};
            acc.top_k = acc.top_k ?? p.top_k;
            acc.top_p = acc.top_p ?? p.top_p;
            acc.candidate_count = acc.candidate_count ?? p.candidate_count;
            acc.safety_settings = acc.safety_settings ?? p.safety_settings;
            return acc;
          }, {} as NonNullable<EnhancedChatRequest["gemini_params"]>)
        );
        if (Object.keys(merged).length > 0) enhancedRequest.gemini_params = merged;
      }

      if (ollamaModels.length > 0) {
        const merged = pruneUndefined(
          ollamaModels.reduce((acc, model) => {
            const p = modelParams[model] || {};
            acc.mirostat = acc.mirostat ?? p.mirostat;
            acc.mirostat_eta = acc.mirostat_eta ?? p.mirostat_eta;
            acc.mirostat_tau = acc.mirostat_tau ?? p.mirostat_tau;
            acc.num_ctx = acc.num_ctx ?? p.num_ctx;
            acc.repeat_penalty = acc.repeat_penalty ?? p.repeat_penalty;
            return acc;
          }, {} as NonNullable<EnhancedChatRequest["ollama_params"]>)
        );
        if (Object.keys(merged).length > 0) enhancedRequest.ollama_params = merged;
      }

      const res = await fetch(`${API_BASE}/v2/chat/completions/enhanced`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(enhancedRequest),
      });
      if (!res.ok) {
        const error = (await res.json().catch(() => ({ detail: res.statusText }))) as { detail?: string };
        throw new Error(error.detail || `HTTP ${res.status}`);
      }
      const result = (await res.json()) as { answers?: AskAnswers };
      const newAnswers: AskAnswers = {};
      for (const model of selected) {
        const modelResult = result.answers?.[model];
        newAnswers[model] = {
          answer: modelResult?.answer || "",
          error: modelResult?.error,
          latency_ms: modelResult?.latency_ms || 0,
        };
      }
      setAnswers(newAnswers);
      setEndedAt(Date.now());
    } catch (err) {
      console.error("Enhanced API error:", err);
      alert(`Error: ${err instanceof Error ? err.message : "Unknown error"}`);
    } finally {
      setIsRunning(false);
    }
  }, [canRun, selected, prompt, modelParams, globalTemp, globalMax, globalMin, getProviderWire, resetRun]);

  const executePrompt = useCallback(async () => {
    if (activeTab === "chat") {
      await runPrompt();
    } else {
      await runEnhancedPrompt();
    }
  }, [activeTab, runPrompt, runEnhancedPrompt]);

  // =============================
  // Keyboard shortcuts
  // =============================
  useEffect(() => {
    const onKey = (evt: KeyboardEvent) => {
      if ((evt.metaKey || evt.ctrlKey) && evt.shiftKey && evt.key === "Enter") {
        evt.preventDefault();
        if (activeTab === "embedding" && compareQuery.trim() && selectedEmbeddingModels.length > 0 && datasets.length > 0) {
          void performMultiSearch();
          return;
        }
      }

      if ((evt.metaKey || evt.ctrlKey) && evt.key === "Enter") {
        evt.preventDefault();
        if (activeModel && interactivePrompt.trim()) {
          void sendInteractiveMessage();
        } else if (activeTab === "chat" && canRun) {
          void executePrompt();
        } else if (activeTab === "embedding" && searchQuery.trim()) {
          void performSearch();
        }
      }
      if (evt.key === "Escape" && activeModel) {
        closeModelChat();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    activeTab,
    searchQuery,
    selectedEmbeddingModels,
    datasets.length,
    activeModel,
    interactivePrompt,
    canRun,
    executePrompt,
    performSearch,
    performMultiSearch,
    sendInteractiveMessage,
    closeModelChat,
  ]);

  // ==== RENDER ====
  return (
    <div className="min-h-screen grid grid-rows-[auto_auto_1fr_auto] gap-6 p-6 sm:p-8 bg-white text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      {/* Interactive Chat Modal */}
      {activeModel && (
        <InteractiveChatModal
          activeModel={activeModel}
          messages={modelChats[activeModel]?.messages || []}
          isStreaming={!!modelChats[activeModel]?.isStreaming}
          currentResponse={modelChats[activeModel]?.currentResponse || ""}
          onClose={closeModelChat}
          prompt={interactivePrompt}
          setPrompt={setInteractivePrompt}
          onSend={() => {
            void sendInteractiveMessage();
          }}
        />
      )}

      {/* Header */}
      <header className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-orange-600 dark:text-orange-400">
            CompareLLM
          </h1>
        </div>
        <div className="text-sm text-zinc-500 dark:text-zinc-400">
          {loadingProviders
            ? "Loading providers…"
            : `${providers.length} provider(s), ${allModels.length} chat model(s), ${allEmbeddingModels.length} embedding model(s)`}
        </div>
      </header>

      <Tabs
        activeId={activeTab}
        onChange={(id) => setActiveTab(id as "chat" | "embedding")}
        tabs={[
          { id: "chat", label: "Chat Models" },
          { id: "embedding", label: "Embeddings" },
        ]}
      />

      {/* Chat Tab */}
      {activeTab === "chat" && (
        <main className="grid grid-cols-1 xl:grid-cols-[400px_1fr] gap-6 items-start">
          <section className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 sm:p-5 bg-white dark:bg-zinc-950 shadow-sm">
            {/* Prompt */}
            <div className="space-y-4">
              <label className="text-sm font-medium">Prompt</label>
              <textarea
                className="w-full rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 outline-none focus:ring-2 focus:ring-orange-300/60 bg-white dark:bg-zinc-900"
                placeholder="Ask your question once. e.g., 'Explain RAG vs fine-tuning for my use case.'"
                value={prompt}
                onChange={(evt) => setPrompt(evt.target.value)}
              />

              {/* Model list */}
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

              <ModelList models={allModels} selected={selected} onToggle={toggleModel} brandOf={getProviderType} />

              {/* Global defaults */}
              <div className="space-y-3 text-sm">
                <div>
                  <label className="block mb-1 font-medium">Global temp</label>
                  <input
                    type="number"
                    step={0.1}
                    min={0}
                    max={2}
                    value={globalTemp ?? ""}
                    placeholder="Model default"
                    onChange={(e) => setGlobalTemp(e.target.value ? Number(e.target.value) : undefined)}
                    className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900"
                  />
                </div>
                <div>
                  <label className="block mb-1 font-medium">Global max_tokens</label>
                  <input
                    type="number"
                    min={1}
                    value={globalMax ?? ""}
                    placeholder="Model default"
                    onChange={(e) => setGlobalMax(e.target.value ? Number(e.target.value) : undefined)}
                    className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900"
                  />
                </div>
                <div>
                  <label className="block mb-1 font-medium">Global min_tokens</label>
                  <input
                    type="number"
                    min={1}
                    value={globalMin ?? ""}
                    placeholder="Model default"
                    onChange={(e) => setGlobalMin(e.target.value ? Number(e.target.value) : undefined)}
                    className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900"
                  />
                </div>
              </div>

              {/* Per-model parameters */}
              <div className="space-y-2">
                <h3 className="text-sm font-medium">Model-Specific Parameters</h3>
                {allModels.map((m) => {
                  const brand = getProviderType(m);
                  const wire = getProviderWire(m);
                  const isExpanded = expandedModels.has(m);
                  const hasParams = modelParams[m] && Object.keys(modelParams[m]).length > 0;

                  return (
                    <div
                      key={m}
                      className="rounded-lg border border-orange-200 dark:border-orange-500/40 bg-orange-50/30 dark:bg-orange-400/5"
                    >
                      <div
                        className="p-3 cursor-pointer flex items-center justify-between hover:bg-orange-50 dark:hover:bg-orange-400/10 rounded-lg transition"
                        onClick={() => toggleModelExpansion(m)}
                      >
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-sm">{m}</span>
                          {hasParams && (
                            <span className="text-xs px-1.5 py-0.5 rounded bg-orange-200 dark:bg-orange-800 text-orange-800 dark:text-orange-200">
                              configured
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <span className={`text-xs px-1.5 py-0.5 rounded ${PROVIDER_BADGE_BG[brand]}`}>{brand}</span>
                          <svg
                            className={`w-4 h-4 transition-transform ${isExpanded ? "rotate-180" : ""}`}
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                          </svg>
                        </div>
                      </div>

                      {isExpanded && (
                        <div className="px-3 pb-3 border-t border-orange-200 dark:border-orange-700">
                          <div className="grid grid-cols-3 gap-3 mb-4 mt-3">
                            <div>
                              <label className="block mb-1 text-xs font-medium">Temperature</label>
                              <input
                                type="number"
                                step={0.1}
                                min={0}
                                max={2}
                                value={modelParams[m]?.temperature ?? ""}
                                placeholder={`↳ ${globalTemp ?? "backend default"}`}
                                onChange={(e) =>
                                  updateParam(m, {
                                    ...modelParams[m],
                                    temperature: e.target.value ? Number(e.target.value) : undefined,
                                  })
                                }
                                className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
                              />
                            </div>
                            <div>
                              <label className="block mb-1 text-xs font-medium">Max Tokens</label>
                              <input
                                type="number"
                                min={1}
                                value={modelParams[m]?.max_tokens ?? ""}
                                placeholder={`↳ ${globalMax ?? "backend default"}`}
                                onChange={(e) =>
                                  updateParam(m, {
                                    ...modelParams[m],
                                    max_tokens: e.target.value ? Number(e.target.value) : undefined,
                                  })
                                }
                                className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
                              />
                            </div>
                            <div>
                              <label className="block mb-1 text-xs font-medium">Min Tokens</label>
                              <input
                                type="number"
                                min={1}
                                value={modelParams[m]?.min_tokens ?? ""}
                                placeholder={globalMin ? `↳ ${globalMin}` : "optional"}
                                onChange={(e) =>
                                  updateParam(m, {
                                    ...modelParams[m],
                                    min_tokens: e.target.value ? Number(e.target.value) : undefined,
                                  })
                                }
                                className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
                              />
                            </div>
                          </div>

                          <ProviderParameterEditor
                            model={m}
                            providerWire={wire}
                            params={modelParams[m] || {}}
                            onUpdate={(params) => updateParam(m, params)}
                          />

                          {hasParams && (
                            <div className="mt-3 pt-3 border-t border-orange-200 dark:border-orange-700">
                              <button
                                onClick={() => updateParam(m, {})}
                                className="text-xs px-2 py-1 rounded-md border border-red-200 text-red-700 bg-red-50 hover:bg-red-100 dark:border-red-800 dark:text-red-400 dark:bg-red-900/20 dark:hover:bg-red-900/40 transition"
                              >
                                Clear all parameters
                              </button>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              <button
                onClick={() => {
                  void executePrompt();
                }}
                disabled={!canRun}
                className="w-full rounded-xl py-2 px-4 font-medium text-white bg-orange-600 hover:bg-orange-700 dark:bg-orange-500 dark:hover:bg-orange-600 transition disabled:opacity-50"
              >
                {isRunning ? "Running…" : "Run"}
              </button>
            </div>
          </section>

          {/* Right rail: Chat results */}
          <section className="space-y-4">
            <ChatResults
              answers={answers}
              isRunning={isRunning}
              brandOf={getProviderType}
              onOpenModel={(m) => {
                openModelChat(m);
              }}
            />
            <div ref={bottomRef} />
          </section>
        </main>
      )}

      {/* Embeddings Tab (left/right rails) */}
      {activeTab === "embedding" && (
        <main className="grid grid-cols-1 xl:grid-cols-[400px_1fr] gap-6 items-start">
          {/* Left rail */}
          <EmbeddingLeftRail
            allEmbeddingModels={allEmbeddingModels}
            selectedEmbeddingModels={selectedEmbeddingModels}
            toggleEmbeddingModel={(m) =>
              setSelectedEmbeddingModels((prev) => (prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]))
            }
            selectAllEmbedding={() => setSelectedEmbeddingModels(allEmbeddingModels)}
            clearAllEmbedding={() => setSelectedEmbeddingModels([])}
            getProviderType={getProviderType}
            datasetId={datasetId}
            setDatasetId={setDatasetId}
            textField={textField}
            setTextField={setTextField}
            jsonInput={jsonInput}
            setJsonInput={setJsonInput}
            uploadingDataset={uploadingDataset}
            uploadDataset={uploadDataset}
            selectedSearchModel={selectedSearchModel}
            setSelectedSearchModel={setSelectedSearchModel}
            datasets={datasets}
            selectedDataset={selectedDataset}
            setSelectedDataset={setSelectedDataset}
            searchQuery={searchQuery}
            setSearchQuery={setSearchQuery}
            topKSingle={topKSingle}
            setTopKSingle={setTopKSingle}
            performSearch={performSearch}
            isSearchingSingle={isSearchingSingle}
            hasAnyDataset={datasets.length > 0}
            compareQuery={compareQuery}
            setCompareQuery={setCompareQuery}
            topKCompare={topKCompare}
            setTopKCompare={setTopKCompare}
            performMultiSearch={performMultiSearch}
            isComparing={isComparing}
            deleteDataset={deleteDataset}
          />

          {/* Right rail */}
          <EmbeddingRightRail
            embedView={embedView}
            setEmbedView={setEmbedView}
            isSearchingSingle={isSearchingSingle}
            searchContext={searchContext}
            searchResults={searchResults}
            isComparing={isComparing}
            multiSearchResults={multiSearchResults}
            getProviderType={getProviderType}
          />
        </main>
      )}
    </div>
  );
}
