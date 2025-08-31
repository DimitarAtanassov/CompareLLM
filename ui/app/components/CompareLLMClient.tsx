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
  ProviderWire,
  SearchResult,
} from "../lib/types";
import { coerceBrand, pruneUndefined as pruneUndefBrandUtils } from "../lib/utils";
import { API_BASE } from "../lib/config";
import InteractiveChatModal from "./chat/InteractiveChatModal";
import Tabs from "./ui/Tabs";
import ModelList from "./chat/ModelList";
import ProviderParameterEditor from "./chat/ProviderParameterEditor";
import ChatResults from "./chat/ChatResults";
import { PROVIDER_BADGE_BG } from "../lib/colors";
import EmbeddingLeftRail from "./embeddings/EmbeddingLeftRail";
import EmbeddingRightRail from "./embeddings/EmbeddingRightRail";

// ---- NEW: embeddings API helpers ----
import {
  listEmbeddingModels,
  listStores,
  createStore,
  indexDocs,
  queryStore,
  toDocsFromJsonArray,
  makeStoreId,
  groupStoresByDataset,
  storesForModel,
  deleteStore,
  type IndexDoc,
} from "../lib/utils";

const PREFIX_TO_BRAND: Record<string, ProviderBrand> = {
  openai: "openai",
  anthropic: "anthropic",
  cohere: "cohere",
  gemini: "google",
  google: "google",
  ollama: "ollama",
  voyage: "voyage",
  cerebras: "cerebras",
  deepseek: "deepseek",
};
// === NDJSON event shape from /chat/stream ===
type NDJSONEvent =
  | { type: "start"; provider: string; model: string }
  | { type: "delta"; provider: string; model: string; text: string }
  | { type: "end"; provider: string; model: string; text?: string }
  | { type: "error"; provider: string; model: string; error: string }
  | { type: "all_done" };

type Selection = { provider: string; model: string };
type ChatMsg = { role: string; content: string };
type WithDatasetId = SearchResult & { _dataset_id?: string };
// --- Group param shapes (same as backend expects) ---
interface AnthropicGroupParams {
  thinking_enabled?: boolean;
  thinking_budget_tokens?: number;
  top_k?: number;
  top_p?: number;
  stop_sequences?: string[];
}

interface OpenAIGroupParams {
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  seed?: number;
}

interface GeminiGroupParams {
  top_k?: number;
  top_p?: number;
  candidate_count?: number;
  safety_settings?: unknown[];
}

interface OllamaGroupParams {
  mirostat?: number;
  mirostat_eta?: number;
  mirostat_tau?: number;
  num_ctx?: number;
  repeat_penalty?: number;
}

interface CohereGroupParams {
  stop_sequences?: string[];
  seed?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  k?: number;
  p?: number;
  logprobs?: boolean;
}

interface DeepseekGroupParams {
  frequency_penalty?: number;
  presence_penalty?: number;
  top_p?: number;
  logprobs?: boolean;
  top_logprobs?: number;
}

interface ChatRequestBody {
  prompt: string;
  selections: Selection[];
  history?: ChatMsg[];
  system?: string;
  temperature?: number;
  max_tokens?: number;
  min_tokens?: number;
  model_params?: ModelParamsMap;
  anthropic_params?: AnthropicGroupParams;
  openai_params?: OpenAIGroupParams;
  gemini_params?: GeminiGroupParams;
  ollama_params?: OllamaGroupParams;
  cohere_params?: CohereGroupParams;
  deepseek_params?: DeepseekGroupParams;
}

// === Helper: model -> backend provider key (registry YAML `type`) ===
function buildModelToProviderKeyMap(providers: ProviderInfo[]) {
  const map: Record<string, string> = {};
  for (const p of providers) {
    const key = (p.type || "").toLowerCase().trim();
    const models = Array.isArray(p.models) ? p.models : [];
    const embeds = Array.isArray(p.embedding_models) ? p.embedding_models : [];
    for (const m of models) map[m] = key;
    for (const m of embeds) map[m] = key;
  }
  return map;
}

// === UI-only: map provider key -> param editor wire
function uiWireForProviderKey(provKey: string): ProviderWire {
  if (provKey === "openai") return "openai";
  if (provKey === "anthropic") return "anthropic";
  if (provKey === "gemini") return "gemini";
  if (provKey === "ollama") return "ollama";
  if (provKey === "cohere") return "cohere";
  if (provKey === "cerebras") return "openai";
  if (provKey === "deepseek") return "deepseek";
  return "unknown";
}

// === Build selections for /chat endpoints
function buildSelections(selected: string[], providerOf: (m: string) => string): Selection[] {
  return selected.map((model) => ({ provider: providerOf(model), model }));
}

// === Build ChatRequest body for /chat/*
function buildChatRequestBody(opts: {
  prompt: string;
  selected: string[];
  providerOf: (m: string) => string;
  history?: ChatMsg[];
  system?: string;
  temperature?: number;
  max_tokens?: number;
  min_tokens?: number;
  modelParams: ModelParamsMap;
}): ChatRequestBody {
  const {
    prompt,
    selected,
    providerOf,
    history,
    system,
    temperature,
    max_tokens,
    min_tokens,
    modelParams,
  } = opts;

  const selections = buildSelections(selected, providerOf);

  const provOf = providerOf;
  const anthropic_models = selected.filter((m) => provOf(m) === "anthropic");
  const openai_like_models = selected.filter((m) => ["openai", "cerebras"].includes(provOf(m)));
  const gemini_models = selected.filter((m) => provOf(m) === "gemini");
  const ollama_models = selected.filter((m) => provOf(m) === "ollama");
  const cohere_models = selected.filter((m) => provOf(m) === "cohere");
  const deepseek_models = selected.filter((m) => provOf(m) === "deepseek");

  const anthropic_params: AnthropicGroupParams = {};
  for (const m of anthropic_models) {
    const p = modelParams[m] || {};
    if (anthropic_params.thinking_enabled === undefined) anthropic_params.thinking_enabled = p.thinking_enabled;
    if (anthropic_params.thinking_budget_tokens === undefined) anthropic_params.thinking_budget_tokens = p.thinking_budget_tokens;
    if (anthropic_params.top_k === undefined) anthropic_params.top_k = p.top_k;
    if (anthropic_params.top_p === undefined) anthropic_params.top_p = p.top_p;
    if (anthropic_params.stop_sequences === undefined) anthropic_params.stop_sequences = p.stop_sequences as string[] | undefined;
  }

  const openai_params: OpenAIGroupParams = {};
  for (const m of openai_like_models) {
    const p = modelParams[m] || {};
    if (openai_params.top_p === undefined) openai_params.top_p = p.top_p;
    if (openai_params.frequency_penalty === undefined) openai_params.frequency_penalty = p.frequency_penalty;
    if (openai_params.presence_penalty === undefined) openai_params.presence_penalty = p.presence_penalty;
    if (openai_params.seed === undefined) openai_params.seed = p.seed;
  }

  const gemini_params: GeminiGroupParams = {};
  for (const m of gemini_models) {
    const p = modelParams[m] || {};
    if (gemini_params.top_k === undefined) gemini_params.top_k = p.top_k;
    if (gemini_params.top_p === undefined) gemini_params.top_p = p.top_p;
    if (gemini_params.candidate_count === undefined) gemini_params.candidate_count = p.candidate_count;
    if (gemini_params.safety_settings === undefined) gemini_params.safety_settings = p.safety_settings as unknown[];
  }

  const ollama_params: OllamaGroupParams = {};
  for (const m of ollama_models) {
    const p = modelParams[m] || {};
    if (ollama_params.mirostat === undefined) ollama_params.mirostat = p.mirostat;
    if (ollama_params.mirostat_eta === undefined) ollama_params.mirostat_eta = p.mirostat_eta;
    if (ollama_params.mirostat_tau === undefined) ollama_params.mirostat_tau = p.mirostat_tau;
    if (ollama_params.num_ctx === undefined) ollama_params.num_ctx = p.num_ctx;
    if (ollama_params.repeat_penalty === undefined) ollama_params.repeat_penalty = p.repeat_penalty;
  }

  const cohere_params: CohereGroupParams = {};
  for (const m of cohere_models) {
    const p = modelParams[m] || {};
    if (cohere_params.stop_sequences === undefined) cohere_params.stop_sequences = p.stop_sequences as string[] | undefined;
    if (cohere_params.seed === undefined) cohere_params.seed = p.seed;
    if (cohere_params.frequency_penalty === undefined) cohere_params.frequency_penalty = p.frequency_penalty;
    if (cohere_params.presence_penalty === undefined) cohere_params.presence_penalty = p.presence_penalty;
    if (cohere_params.k === undefined) cohere_params.k = p.k;
    if (cohere_params.p === undefined) cohere_params.p = p.p;
    if (cohere_params.logprobs === undefined) cohere_params.logprobs = p.logprobs as boolean | undefined;
  }

  const deepseek_params: DeepseekGroupParams = {};
  for (const m of deepseek_models) {
    const p = modelParams[m] || {};
    if (deepseek_params.frequency_penalty === undefined) deepseek_params.frequency_penalty = p.frequency_penalty;
    if (deepseek_params.presence_penalty === undefined) deepseek_params.presence_penalty = p.presence_penalty;
    if (deepseek_params.top_p === undefined) deepseek_params.top_p = p.top_p;
    if (deepseek_params.logprobs === undefined) deepseek_params.logprobs = p.logprobs as boolean | undefined;
    if (deepseek_params.top_logprobs === undefined) deepseek_params.top_logprobs = p.top_logprobs;
  }

  const body: ChatRequestBody = { prompt, selections };

  if (history?.length) body.history = history;
  if (system) body.system = system;
  if (temperature !== undefined) body.temperature = temperature;
  if (max_tokens !== undefined) body.max_tokens = max_tokens;
  if (min_tokens !== undefined) body.min_tokens = min_tokens;
  if (Object.keys(modelParams).length) body.model_params = modelParams;

  const has = (o: object) => Object.values(o).some((v) => v !== undefined);
  if (has(anthropic_params)) body.anthropic_params = anthropic_params;
  if (has(openai_params)) body.openai_params = openai_params;
  if (has(gemini_params)) body.gemini_params = gemini_params;
  if (has(ollama_params)) body.ollama_params = ollama_params;
  if (has(cohere_params)) body.cohere_params = cohere_params;
  if (has(deepseek_params)) body.deepseek_params = deepseek_params;

  return body;
}

export default function CompareLLMClient(): JSX.Element {
  // ==== STATE ====
  const [activeTab, setActiveTab] = useState<"chat" | "embedding">("chat");
  const [embedView, setEmbedView] = useState<"single" | "compare">("single");
  const [loadingProviders, setLoadingProviders] = useState(false);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [allModels, setAllModels] = useState<string[]>([]);

  // ---- NEW: embeddings state
  const [allEmbeddingModels, setAllEmbeddingModels] = useState<string[]>([]);
  const [stores, setStores] = useState<Record<string, string>>({}); // {storeId: embeddingKey}
  const [datasets, setDatasets] = useState<Dataset[]>([]);

  const [selected, setSelected] = useState<string[]>([]);
  const [prompt, setPrompt] = useState<string>("");
  const [isRunning, setIsRunning] = useState(false);
  const [answers, setAnswers] = useState<AskAnswers>({});
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [endedAt, setEndedAt] = useState<number | null>(null);

  const [activeModel, setActiveModel] = useState<string | null>(null);
  const [modelChats, setModelChats] = useState<Record<string, ModelChat>>({});
  const [interactivePrompt, setInteractivePrompt] = useState<string>("");

  // Embeddings UI state
  const [selectedEmbeddingModels, setSelectedEmbeddingModels] = useState<string[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [selectedSearchModel, setSelectedSearchModel] = useState<string>("");
  const [uploadingDataset, setUploadingDataset] = useState(false);
  const [jsonInput, setJsonInput] = useState<string>("");
  const [datasetId, setDatasetId] = useState<string>("");
  const [textField, setTextField] = useState<string>("text");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [compareQuery, setCompareQuery] = useState<string>("");

  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearchingSingle, setIsSearchingSingle] = useState(false);
  const [isComparing, setIsComparing] = useState(false);
  const [searchContext, setSearchContext] = useState<{ model: string; dataset: string; query: string; startedAt: number } | null>(null);
  const [multiSearchResults, setMultiSearchResults] = useState<MultiSearchResponse | null>(null);

  const bottomRef = useRef<HTMLDivElement | null>(null);
  const streamAbortRef = useRef<AbortController | null>(null);
  const interactiveAbortRef = useRef<AbortController | null>(null);
  const promptRef = useRef<HTMLTextAreaElement | null>(null);
  const requestIdRef = useRef(0);

  const [modelParams, setModelParams] = useState<ModelParamsMap>({});
  const [globalTemp, setGlobalTemp] = useState<number | undefined>(undefined);
  const [globalMax, setGlobalMax] = useState<number | undefined>(undefined);
  const [globalMin, setGlobalMin] = useState<number | undefined>(undefined);
  const [topKSingle, setTopKSingle] = useState<number>(5);
  const [topKCompare, setTopKCompare] = useState<number>(5);
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set());
  const [lastRunPrompt, setLastRunPrompt] = useState<string>("");

  // ==== MEMO MAPS ====
  const modelToBrand = useMemo(() => {
    const map: Record<string, ProviderBrand> = {};
    providers.forEach((p) => {
      (p.models || []).forEach((m) => (map[m] = coerceBrand(p.type)));
      (p.embedding_models || []).forEach((m) => (map[m] = coerceBrand(p.type)));
    });
    return map;
  }, [providers]);

  const modelToProviderKey = useMemo(() => buildModelToProviderKeyMap(providers), [providers]);
  const getProviderType = useCallback((m: string): ProviderBrand => {
    const s = (m || "").toLowerCase();
    // If it's an embedding key like "openai:xxx" or "cohere:xxx", use the prefix directly
    const prefix = s.split(":", 1)[0];
    if (PREFIX_TO_BRAND[prefix]) return PREFIX_TO_BRAND[prefix];

    // Otherwise, try the map built from /providers (works for chat models and some embedding names)
    if (modelToBrand[m]) return modelToBrand[m];

    // Final fallback: heuristic brand detection
    return coerceBrand(m);
  }, [modelToBrand]);
  const getProviderKey = useCallback((m: string): string => modelToProviderKey[m] ?? "unknown", [modelToProviderKey]);

  // ==== LOAD PROVIDERS + EMBEDDINGS (models & stores) ====
  useEffect(() => {
    const isString = (x: unknown): x is string => typeof x === "string";
    const pickModelName = (v: unknown): string | null => {
      if (typeof v === "string") return v;
      if (v && typeof v === "object") {
        const o = v as Record<string, unknown>;
        if (isString(o.model)) return o.model;
        if (isString(o.name)) return o.name;
        if (isString(o.id)) return o.id;
      }
      return null;
    };

    const load = async () => {
      setLoadingProviders(true);
      try {
        // Providers (for chat + embedding branding)
        const res = await fetch(`${API_BASE}/providers`, { cache: "no-store" });
        if (!res.ok) throw new Error(`Failed to load providers: ${res.status} ${res.statusText}`);
        const raw = await res.json();

        const provsUnknown: unknown = Array.isArray(raw)
          ? raw
          : Array.isArray(raw?.providers)
          ? raw.providers
          : raw?.providers && typeof raw.providers === "object"
          ? Object.values(raw.providers as Record<string, unknown>)
          : [];

        const arr = Array.isArray(provsUnknown) ? provsUnknown : [];

        const normalized: ProviderInfo[] = arr.map((item, idx) => {
          const p = (item ?? {}) as Record<string, unknown>;
          const name =
            isString(p.name) ? p.name : isString(p.key) ? p.key : isString(p.id) ? p.id : `prov_${idx}`;
          const type = isString(p.type) ? p.type : "unknown";
          const base_url = isString(p.base_url) ? p.base_url : "";

          const rawModels = Array.isArray(p.models)
            ? p.models
            : Array.isArray(p.chat_models)
            ? p.chat_models
            : Array.isArray(p.llm_models)
            ? p.llm_models
            : [];
          const models = rawModels.map(pickModelName).filter((m): m is string => !!m);

          const rawEmb = Array.isArray(p.embedding_models)
            ? p.embedding_models
            : Array.isArray(p.embed_models)
            ? p.embed_models
            : [];
          const embedding_models = rawEmb.map(pickModelName).filter((m): m is string => !!m);

          const auth_required =
            typeof p.auth_required === "boolean"
              ? p.auth_required
              : typeof p.requires_api_key === "boolean"
              ? p.requires_api_key
              : isString(p.api_key_env);

          return { name, type, base_url, models, embedding_models, auth_required };
        });

        setProviders(normalized);
        setAllModels([...new Set(normalized.flatMap((p) => p.models ?? []))].sort());

        // ---- NEW: embedding models & stores from /embeddings
        const [embM, storesRes] = await Promise.all([listEmbeddingModels(), listStores()]);
        setAllEmbeddingModels(embM.embedding_models || []);
        setStores(storesRes.stores || {});
        if (embM.embedding_models?.length && !selectedSearchModel) {
          setSelectedSearchModel(embM.embedding_models[0]);
        }

        // build datasets from stores
        const byDataset = groupStoresByDataset(storesRes.stores || {});
        const ds: Dataset[] = Object.keys(byDataset)
          .sort()
          .map((id) => ({ dataset_id: id } as Dataset)); // document_count not available
        setDatasets(ds);
      } finally {
        setLoadingProviders(false);
      }
    };

    void load();
  }, [selectedSearchModel]);

  // =============================
  // Embeddings: upload/index via /embeddings/*
  // =============================
  const refreshStoresAndDatasets = useCallback(async () => {
    const s = await listStores();
    setStores(s.stores || {});
    const byDataset = groupStoresByDataset(s.stores || {});
    const ds: Dataset[] = Object.keys(byDataset)
      .sort()
      .map((id) => ({ dataset_id: id } as Dataset));
    setDatasets(ds);
  }, []);

  const uploadDataset = useCallback(async () => {
    if (!jsonInput.trim() || !datasetId.trim() || selectedEmbeddingModels.length === 0) {
      alert("Please provide dataset ID, JSON data, and select at least one embedding model.");
      return;
    }

    let docs: IndexDoc[] = [];
    try {
      const parsed = JSON.parse(jsonInput);
      const arr = Array.isArray(parsed) ? parsed : [parsed];
      docs = toDocsFromJsonArray(arr, textField);
      if (!docs.length) {
        alert(`No valid docs. Ensure "${textField}" exists and is a non-empty string.`);
        return;
      }
    } catch {
      alert("Invalid JSON");
      return;
    }

    setUploadingDataset(true);
    try {
      const results = await Promise.allSettled(
        selectedEmbeddingModels.map(async (embeddingModel) => {
          const storeId = makeStoreId(datasetId.trim(), embeddingModel.trim());
          try {
            await createStore(storeId, embeddingModel);
          } catch {
            // store may already exist; continue
          }
          await indexDocs(storeId, docs);
          return storeId;
        })
      );

      const ok = results.filter((r) => r.status === "fulfilled").length;
      const bad = results.filter((r) => r.status === "rejected");
      let msg = `Embedded & indexed for ${ok} model(s).`;
      if (bad.length) {
        msg += `\n${bad.length} failed:\n` + bad.map((r) => (r as PromiseRejectedResult).reason?.message || String((r as PromiseRejectedResult).reason)).join("\n");
      }
      alert(msg);

      setJsonInput("");
      setDatasetId("");
      await refreshStoresAndDatasets();
    } catch (err) {
      console.error("Upload failed:", err);
      alert(`Upload failed: ${err instanceof Error ? err.message : "Unknown error"}`);
    } finally {
      setUploadingDataset(false);
    }
  }, [jsonInput, datasetId, selectedEmbeddingModels, textField, refreshStoresAndDatasets]);

  // =============================
  // Embeddings: single-model search
  // =============================
  const performSearch = useCallback(async () => {
    if (!searchQuery.trim() || !selectedDataset || !selectedSearchModel) {
      alert("Please provide search query, select a dataset, and select a search model.");
      return;
    }
    const snapshot = {
      query: searchQuery.trim(),
      dataset: selectedDataset.trim(),
      model: selectedSearchModel.trim(),
      startedAt: Date.now(),
    };

    setIsSearchingSingle(true);
    const myId = ++requestIdRef.current;

    try {
      const sid = makeStoreId(snapshot.dataset, snapshot.model);
      const { matches } = await queryStore(sid, snapshot.query, { k: topKSingle, with_scores: true, search_type: "similarity" });

      // Map backend shape -> SearchResult shape your UI expects
      const rows: SearchResult[] = matches.map((m) => {
        const base: Record<string, unknown> = { ...(m.metadata || {}) };
        base.text = m.page_content;
        if (typeof m.score === "number") base.similarity_score = m.score;
        return base as SearchResult;
      });

      if (myId === requestIdRef.current) {
        setSearchResults(rows);
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

  // =============================
  // Embeddings: multi-model compare (fan-out)
  // =============================
  const performMultiSearch = useCallback(async () => {
    if (!compareQuery.trim()) {
      alert("Please provide a comparison query.");
      return;
    }
    if (selectedEmbeddingModels.length === 0) {
      alert("Select at least one embedding model (left rail) to compare.");
      return;
    }

    const started = Date.now();
    setIsComparing(true);
    setMultiSearchResults(null);
    const myId = ++requestIdRef.current;

    try {
      const resultsByModel: Record<string, { items: SearchResult[]; error?: string }> = {};

      await Promise.all(
        selectedEmbeddingModels.map(async (embeddingKey) => {
          const sids = storesForModel(stores, embeddingKey); // all stores that use this model
          const collected: SearchResult[] = [];

          await Promise.all(
            sids.map(async (sid) => {
              try {
                const { matches } = await queryStore(sid, compareQuery.trim(), { k: topKCompare, with_scores: true });
                const datasetId = sid.split("::", 1)[0] ?? sid;
                const mapped: WithDatasetId[] = matches.map((m) => {
                  const meta = (m.metadata ?? {}) as Record<string, unknown>;
                  const row: WithDatasetId = {
                    ...meta,
                    text: m.page_content,
                    similarity_score: typeof m.score === "number" ? m.score : 0, // <-- always set
                    _dataset_id: datasetId,
                  };
                  return row;
                });
                collected.push(...mapped);
              } catch (e) {
                console.warn("queryStore failed for", sid, e);
              }
            })
          );

          collected.sort((a, b) => (b.similarity_score ?? 0) - (a.similarity_score ?? 0));
          resultsByModel[embeddingKey] = { items: collected.slice(0, topKCompare) };
        })
      );

      const payload: MultiSearchResponse = {
        query: compareQuery.trim(),
        duration_ms: Date.now() - started,
        results: resultsByModel,
      };

      if (myId === requestIdRef.current) setMultiSearchResults(payload);
    } catch (err) {
      console.error("Compare failed:", err);
      if (myId === requestIdRef.current) {
        alert(`Compare failed: ${err instanceof Error ? err.message : "Unknown error"}`);
        setMultiSearchResults(null);
      }
    } finally {
      if (myId === requestIdRef.current) setIsComparing(false);
    }
  }, [compareQuery, selectedEmbeddingModels, topKCompare, stores]);

  // =============================
  // Embeddings: delete dataset (delete all its stores)
  // =============================
  const deleteDataset = useCallback(
    async (id: string) => {
      if (!confirm(`Delete dataset "${id}"? This will remove all of its embedding stores.`)) return;
      try {
        const toDelete = Object.keys(stores).filter((sid) => sid.startsWith(`${id}::`));
        await Promise.allSettled(toDelete.map((sid) => deleteStore(sid)));
        await refreshStoresAndDatasets();
        if (selectedDataset === id) setSelectedDataset("");
      } catch (err) {
        console.error("Delete failed:", err);
      }
    },
    [stores, refreshStoresAndDatasets, selectedDataset]
  );

  // =============================
  // Chat tab (unchanged)
  // =============================
  const toggleModel = (m: string) =>
    setSelected((prev) => (prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]));
  const selectAll = () => setSelected(allModels);
  const clearAll = () => setSelected([]);

  const prevSelectedRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    const prev = prevSelectedRef.current;
    const next = new Set(selected);
    const removed = [...prev].filter((m) => !next.has(m));
    setExpandedModels((prevExp) => {
      const ns = new Set(prevExp);
      removed.forEach((m) => ns.delete(m));
      return ns;
    });
    prevSelectedRef.current = next;
  }, [selected]);

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
  const handleRemoveModel = useCallback((model: string) => {
    setAnswers((prev) => {
      const next = { ...prev };
      delete next[model];
      return next;
    });
  }, []);

  // Interactive chat helpers (unchanged except using getProviderKey)
  const openModelChat = useCallback(
    (model: string) => {
      setActiveModel(model);
      setModelChats((prev) => {
        if (prev[model]) return prev;
        const initial: ModelChat["messages"] = [];
        if (prompt.trim()) initial.push({ role: "user", content: prompt.trim(), timestamp: Date.now() });
        const modelAnswer = answers[model];
        if (modelAnswer?.answer && !modelAnswer.error) {
          initial.push({ role: "assistant", content: modelAnswer.answer, timestamp: Date.now() });
        }
        return { ...prev, [model]: { messages: initial, isStreaming: false, currentResponse: "" } };
      });
    },
    [answers, prompt]
  );
  const closeModelChat = useCallback(() => {
    setActiveModel(null);
    interactiveAbortRef.current?.abort();
  }, []);
const pruneUndefined = <T extends Record<string, unknown>>(obj: T): Partial<T> =>
  pruneUndefBrandUtils(obj);

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
      const apiMessages: ChatMsg[] = conversationHistory.map((m) => ({ role: m.role, content: m.content }));

      const body = buildChatRequestBody({
        prompt: message,
        selected: [activeModel],
        providerOf: (m) => getProviderKey(m),
        history: apiMessages.slice(0, -1),
        system: undefined,
        temperature: modelParams[activeModel]?.temperature ?? globalTemp,
        max_tokens: modelParams[activeModel]?.max_tokens ?? globalMax,
        min_tokens: modelParams[activeModel]?.min_tokens ?? globalMin,
        modelParams: { [activeModel]: modelParams[activeModel] || {} },
      });

      const res = await fetch(`${API_BASE}/chat/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!res.ok) {
        const errorData = (await res.json().catch(() => ({ detail: res.statusText }))) as { detail?: string };
        throw new Error(errorData.detail || `HTTP ${res.status}`);
      }
      const json = (await res.json()) as { results: { provider: string; model: string; response?: string; error?: string }[] };
      const entry = json.results.find((r) => r.model === activeModel);
      const assistantMessage = entry?.response || entry?.error || "No response";

      setModelChats((prev) => ({
        ...prev,
        [activeModel]: {
          ...prev[activeModel],
          messages: [...(prev[activeModel]?.messages || []), { role: "assistant", content: assistantMessage, timestamp: Date.now() }],
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
            messages: [...(prev[activeModel!]?.messages || []), { role: "assistant", content: `Error: ${msg}`, timestamp: Date.now() }],
          },
        }));
      }
    } finally {
      interactiveAbortRef.current = null;
    }
  }, [activeModel, interactivePrompt, modelChats, modelParams, getProviderKey, globalTemp, globalMax, globalMin]);

  // Providers-only chat streaming bits (unchanged except getProviderKey)
  const canRun = prompt.trim().length > 0 && selected.length > 0 && !isRunning;
  const resetRun = useCallback(() => {
    setAnswers(Object.fromEntries(selected.map((m) => [m, { answer: "", error: undefined, latency_ms: 0 }])) as AskAnswers);
    setStartedAt(Date.now());
    setEndedAt(null);
  }, [selected]);

  const runPrompt = useCallback(async () => {
    if (!canRun) return;
    streamAbortRef.current?.abort();
    const controller = new AbortController();
    streamAbortRef.current = controller;

    resetRun();
    setIsRunning(true);
    setLastRunPrompt(prompt.trim());

    const startedAt = Date.now();
    const modelStart: Record<string, number> = {};

    const processEvent = (evt: NDJSONEvent) => {
      if (evt.type === "start") {
        modelStart[evt.model] = Date.now();
      } else if (evt.type === "delta") {
        setAnswers((prev) => ({
          ...prev,
          [evt.model]: {
            answer: (prev[evt.model]?.answer || "") + (evt.text || ""),
            error: prev[evt.model]?.error,
            latency_ms: prev[evt.model]?.latency_ms ?? 0,
          },
        }));
      } else if (evt.type === "error") {
        setAnswers((prev) => ({
          ...prev,
          [evt.model]: { answer: prev[evt.model]?.answer || "", error: evt.error || "Unknown error", latency_ms: prev[evt.model]?.latency_ms ?? 0 },
        }));
      } else if (evt.type === "end") {
        const t0 = modelStart[evt.model] ?? startedAt;
        setAnswers((prev) => {
          const prevAns = prev[evt.model]?.answer || "";
          return { ...prev, [evt.model]: { ...prev[evt.model], answer: prevAns || evt.text || "", latency_ms: Date.now() - t0 } };
        });
      } else if (evt.type === "all_done") {
        setIsRunning(false);
        setEndedAt(Date.now());
        streamAbortRef.current = null;
      }
    };

    try {
      const body = buildChatRequestBody({
        prompt: prompt.trim(),
        selected,
        providerOf: (m) => getProviderKey(m),
        history: undefined,
        system: undefined,
        temperature: globalTemp,
        max_tokens: globalMax,
        min_tokens: globalMin,
        modelParams,
      });

      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
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
            processEvent(JSON.parse(trimmed) as NDJSONEvent);
          } catch {
            /* ignore malformed */
          }
        }
      }
      if (buf.trim()) {
        try {
          processEvent(JSON.parse(buf.trim()) as NDJSONEvent);
        } catch {
          /* ignore */
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
  }, [canRun, prompt, selected, modelParams, globalTemp, globalMax, globalMin, resetRun, getProviderKey]);

  const retryModel = useCallback(
    async (model: string) => {
      const retryPrompt = (lastRunPrompt || prompt).trim();
      if (!retryPrompt) {
        alert("No prompt to retry. Please enter a prompt first.");
        return;
      }
      setAnswers((prev) => ({ ...prev, [model]: { answer: "", error: undefined, latency_ms: 0 } }));
      const controller = new AbortController();
      const started = Date.now();

      try {
        const body = buildChatRequestBody({
          prompt: retryPrompt,
          selected: [model],
          providerOf: (m) => getProviderKey(m),
          history: undefined,
          system: undefined,
          temperature: modelParams[model]?.temperature ?? globalTemp,
          max_tokens: modelParams[model]?.max_tokens ?? globalMax,
          min_tokens: modelParams[model]?.min_tokens ?? globalMin,
          modelParams: { [model]: modelParams[model] || {} },
        });

        const res = await fetch(`${API_BASE}/chat/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: controller.signal,
        });

        if (!res.ok || !res.body) {
          const err = (await res.json().catch(() => ({ detail: res.statusText }))) as { detail?: string };
          throw new Error(err.detail || `HTTP ${res.status}`);
        }

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
              const evt = JSON.parse(trimmed) as NDJSONEvent;
              if (evt.type === "delta" && evt.model === model) {
                setAnswers((prev) => ({
                  ...prev,
                  [model]: {
                    answer: (prev[model]?.answer || "") + (evt.text || ""),
                    error: prev[model]?.error,
                    latency_ms: prev[model]?.latency_ms ?? 0,
                  },
                }));
              } else if (evt.type === "error" && evt.model === model) {
                setAnswers((prev) => ({ ...prev, [model]: { answer: prev[model]?.answer || "", error: evt.error, latency_ms: 0 } }));
              } else if (evt.type === "end" && evt.model === model) {
                setAnswers((prev) => ({ ...prev, [model]: { ...prev[model], latency_ms: Date.now() - started } }));
              }
            } catch {
              /* ignore */
            }
          }
        }

        if (buf.trim()) {
          try {
            const evt = JSON.parse(buf.trim()) as NDJSONEvent;
            if (evt.type === "delta" && evt.model === model) {
              setAnswers((prev) => ({
                ...prev,
                [model]: {
                  answer: (prev[model]?.answer || "") + (evt.text || ""),
                  error: prev[model]?.error,
                  latency_ms: prev[model]?.latency_ms ?? 0,
                },
              }));
            } else if (evt.type === "end" && evt.model === model) {
              setAnswers((prev) => ({ ...prev, [model]: { ...prev[model], latency_ms: Date.now() - started } }));
            }
          } catch {
            /* noop */
          }
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Unknown error";
        setAnswers((prev) => ({ ...prev, [model]: { answer: "", error: `Retry failed: ${msg}`, latency_ms: 0 } }));
      }
    },
    [lastRunPrompt, prompt, modelParams, globalTemp, globalMax, globalMin, getProviderKey]
  );

  const repromptFromResults = useCallback(() => {
    if (!lastRunPrompt) return;
    setActiveTab("chat");
    setPrompt(lastRunPrompt);
    setTimeout(() => promptRef.current?.focus(), 0);
  }, [lastRunPrompt]);

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
          void runPrompt();
        } else if (activeTab === "embedding" && searchQuery.trim()) {
          void performSearch();
        }
      }
      if (evt.key === "Escape" && activeModel) closeModelChat();
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
    runPrompt,
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
          onSend={() => void sendInteractiveMessage()}
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
                ref={promptRef}
                className="w-full h-48 resize-y rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 outline-none focus:ring-2 focus:ring-orange-300/60 bg-white dark:bg-zinc-900 leading-relaxed"
                placeholder="Paste or write a long prompt here..."
                value={prompt}
                onChange={(evt) => setPrompt(evt.target.value)}
                spellCheck={true}
              />

              {/* Model list header */}
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Models</label>
                <div className="flex gap-2 text-xs">
                  <button
                    onClick={() => setSelected(allModels)}
                    className="px-2 py-1 rounded-lg border border-orange-200 text-zinc-800 dark:text-zinc-100 bg-orange-50 hover:bg-orange-100 dark:bg-orange-400/10 dark:hover:bg-orange-400/20 transition"
                  >
                    Select all
                  </button>
                  <button
                    onClick={() => setSelected([])}
                    className="px-2 py-1 rounded-lg border border-orange-200 text-zinc-800 dark:text-zinc-100 bg-orange-50 hover:bg-orange-100 dark:bg-orange-400/10 dark:hover:bg-orange-400/20 transition"
                  >
                    Clear
                  </button>
                </div>
              </div>

              {/* Resizable ModelList */}
              <ModelList
                models={allModels}
                selected={selected}
                onToggle={(m) => setSelected((prev) => (prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]))}
                brandOf={getProviderType}
                initialHeightPx={260}
                minHeightPx={140}
                maxHeightPx={520}
              />

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
                    placeholder="optional"
                    onChange={(e) => setGlobalMin(e.target.value ? Number(e.target.value) : undefined)}
                    className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900"
                  />
                </div>
              </div>

              {/* Per-model parameters */}
              <div className="space-y-2">
                <h3 className="text-sm font-medium">Model-Specific Parameters</h3>
                {selected.length === 0 && (
                  <div className="text-sm text-zinc-500 dark:text-zinc-400">Select one or more models above to configure their parameters.</div>
                )}
                {selected.map((m) => {
                  const brand = getProviderType(m);
                  const provKey = getProviderKey(m);
                  const wire: ProviderWire = uiWireForProviderKey(provKey);
                  const isExpanded = expandedModels.has(m);
                  const hasParams = modelParams[m] && Object.keys(modelParams[m]).length > 0;

                  return (
                    <div key={m} className="rounded-lg border border-orange-200 dark:border-orange-500/40 bg-orange-50/30 dark:bg-orange-400/5">
                      <div
                        className="p-3 cursor-pointer flex items-center justify-between hover:bg-orange-50 dark:hover:bg-orange-400/10 rounded-lg transition"
                        onClick={() => setExpandedModels((prev) => {
                          const next = new Set(prev); next.has(m) ? next.delete(m) : next.add(m); return next;
                        })}
                        title={`Configure ${m}`}
                      >
                        <div className="flex items-center gap-2 min-w-0">
                          <span className="font-mono text-sm truncate">{m}</span>
                          {hasParams && (
                            <span className="text-xs px-1.5 py-0.5 rounded bg-orange-200 dark:bg-orange-800 text-orange-800 dark:text-orange-200">
                              configured
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <span className={`text-xs px-1.5 py-0.5 rounded ${PROVIDER_BADGE_BG[brand]}`}>{brand}</span>
                          <svg className={`w-4 h-4 transition-transform ${isExpanded ? "rotate-180" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
                                type="number" step={0.1} min={0} max={2}
                                value={modelParams[m]?.temperature ?? ""}
                                placeholder="↳ global / default"
                                onChange={(e) => setModelParams((prev) => ({ ...prev, [m]: { ...prev[m], temperature: e.target.value ? Number(e.target.value) : undefined } }))}
                                className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
                              />
                            </div>
                            <div>
                              <label className="block mb-1 text-xs font-medium">Max Tokens</label>
                              <input
                                type="number" min={1}
                                value={modelParams[m]?.max_tokens ?? ""}
                                placeholder="↳ global / default"
                                onChange={(e) => setModelParams((prev) => ({ ...prev, [m]: { ...prev[m], max_tokens: e.target.value ? Number(e.target.value) : undefined } }))}
                                className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
                              />
                            </div>
                            <div>
                              <label className="block mb-1 text-xs font-medium">Min Tokens</label>
                              <input
                                type="number" min={1}
                                value={modelParams[m]?.min_tokens ?? ""}
                                placeholder="optional"
                                onChange={(e) => setModelParams((prev) => ({ ...prev, [m]: { ...prev[m], min_tokens: e.target.value ? Number(e.target.value) : undefined } }))}
                                className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
                              />
                            </div>
                          </div>

                          <ProviderParameterEditor
                            model={m}
                            providerWire={wire}
                            params={modelParams[m] || {}}
                            onUpdate={(params) => setModelParams((prev) => ({ ...prev, [m]: params }))}
                          />

                          {hasParams && (
                            <div className="mt-3 pt-3 border-t border-orange-200 dark:border-orange-700">
                              <button
                                onClick={() => setModelParams((prev) => ({ ...prev, [m]: {} }))}
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
                onClick={() => void runPrompt()}
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
              onOpenModel={(m) => openModelChat(m)}
              onRemoveModel={handleRemoveModel}
              onRetryModel={retryModel}
            />
            <div ref={bottomRef} />
          </section>
        </main>
      )}

      {/* Embeddings Tab */}
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
