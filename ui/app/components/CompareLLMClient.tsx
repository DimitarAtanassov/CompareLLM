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
import ImageLeftRail, { ImageEndpoint } from "./image/ImageLeftRail";
import ImageRightRail from "./image/ImageRightRail";
// ---- embeddings API helpers ----
import {
  listEmbeddingModels,
  listStores,
  createStore,
  indexDocs,
  queryStore,
  toDocsFromJsonArray,
  makeStoreId,
  groupStoresByDataset,
  deleteStore,
  type IndexDoc,
} from "../lib/utils";
import ImageResults from "./image/ImageResults";

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

// === Legacy NDJSON event shape from /chat/stream ===
type NDJSONEvent =
  | { type: "start"; provider: string; model: string }
  | { type: "delta"; provider: string; model: string; text: string }
  | { type: "end"; provider: string; model: string; text?: string }
  | { type: "error"; provider: string; model: string; error: string }
  | { type: "all_done" };

// === NEW LangGraph stream shapes ===
type LGEvent =
  | { type: "delta"; scope: "multi"; model: string; node?: string; text: string; done?: boolean }
  | { type: "done"; scope: "multi"; model?: string; done: true }
  | { type: "delta"; scope: "single"; node?: string; delta: string; done?: boolean }
  | { type: "done"; scope: "single"; done: true };

type Selection = { provider: string; model: string };
type ChatMsg = { role: string; content: string };

// Safe helpers for runtime narrowing
const isRecord = (v: unknown): v is Record<string, unknown> =>
  !!v && typeof v === "object";

// Legacy /chat/* events
const isNDJSONEvent = (e: unknown): e is NDJSONEvent => {
  if (!isRecord(e) || typeof e.type !== "string") return false;
  if (e.type === "all_done") return true;
  return (
    typeof (e as { provider?: unknown }).provider === "string" &&
    typeof (e as { model?: unknown }).model === "string" &&
    (e.type === "start" || e.type === "delta" || e.type === "end" || e.type === "error")
  );
};

// /langgraph/* events
const isLGEvent = (e: unknown): e is LGEvent => {
  if (!isRecord(e) || typeof e.type !== "string" || typeof (e as { scope?: unknown }).scope !== "string") return false;

  const scope = (e as { scope: string }).scope;
  if (scope === "multi") {
    if (e.type === "delta") {
      return typeof (e as { model?: unknown }).model === "string" && typeof (e as { text?: unknown }).text === "string";
    }
    if (e.type === "done") return true;
  }
  if (scope === "single") {
    if (e.type === "delta") return typeof (e as { delta?: unknown }).delta === "string";
    if (e.type === "done") return true;
  }
  return false;
};

// --- Minimal SSE parser: collects "data:" lines until a blank line, then yields JSON payloads
async function readSSE<T>(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  onEvent: (obj: T) => void
): Promise<void> {
  const decoder = new TextDecoder();
  let buf = "";
  let dataBuf = ""; // accumulates data: lines for the current SSE event

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";

    for (const raw of lines) {
      const line = raw.endsWith("\r") ? raw.slice(0, -1) : raw;
      if (line === "") {
        // blank line = end of one SSE event
        const payload = dataBuf.trim();
        if (payload) {
          try {
            const obj = JSON.parse(payload) as T;
            onEvent(obj);
          } catch {
            // ignore malformed JSON
          }
        }
        dataBuf = "";
        continue;
      }

      if (line.startsWith(":")) {
        // comment/heartbeat
        continue;
      }
      if (line.startsWith("event:")) {
        // event name unused — the data payload includes "type"
        continue;
      }
      if (line.startsWith("data:")) {
        // Accumulate multiple data lines for multi-line JSON payloads
        const part = line.slice(5).trimStart();
        dataBuf += part;
        continue;
      }

      // Fallback: raw JSON without SSE prefix
      if (line.startsWith("{") || line.startsWith("[")) {
        try {
          const obj = JSON.parse(line) as T;
          onEvent(obj);
        } catch {
          // ignore
        }
      }
    }
  }

  // Flush any trailing JSON that might still be in the buffer
  const tail = buf.trim();
  if (tail.startsWith("data:")) {
    try {
      const obj = JSON.parse(tail.slice(5).trim()) as T;
      onEvent(obj);
    } catch {
      // ignore
    }
  } else if (tail.startsWith("{") || tail.startsWith("[")) {
    try {
      const obj = JSON.parse(tail) as T;
      onEvent(obj);
    } catch {
      // ignore
    }
  }
}

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
function buildModelToProviderKeyMap(providers: ProviderInfo[]): Record<string, string> {
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

// === Build selections for /chat endpoints (legacy)
function buildSelections(selected: string[], providerOf: (m: string) => string): Selection[] {
  return selected.map((model) => ({ provider: providerOf(model), model }));
}

// === Build ChatRequest body for /chat/* (legacy)
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

  const selections: Selection[] = buildSelections(selected, providerOf);

  // Per-provider model splits
  const provOf = providerOf;
  const anthropicModels = selected.filter((m) => provOf(m) === "anthropic");
  const openaiLikeModels = selected.filter((m) => ["openai", "cerebras"].includes(provOf(m)));
  const geminiModels = selected.filter((m) => provOf(m) === "gemini");
  const ollamaModels = selected.filter((m) => provOf(m) === "ollama");
  const cohereModels = selected.filter((m) => provOf(m) === "cohere");
  const deepseekModels = selected.filter((m) => provOf(m) === "deepseek");

  // Helper: does an object contain any defined value?
  const hasValues = <T extends object>(o: T): boolean =>
    Object.values(o as Record<string, unknown>).some((v) => v !== undefined);

  // Build group params by peeking the first model's params for that provider group
  const anthropic_params: AnthropicGroupParams = {};
  for (const m of anthropicModels) {
    const p = modelParams[m] || {};
    if (anthropic_params.thinking_enabled === undefined) anthropic_params.thinking_enabled = p.thinking_enabled as boolean | undefined;
    if (anthropic_params.thinking_budget_tokens === undefined) anthropic_params.thinking_budget_tokens = p.thinking_budget_tokens as number | undefined;
    if (anthropic_params.top_k === undefined) anthropic_params.top_k = p.top_k as number | undefined;
    if (anthropic_params.top_p === undefined) anthropic_params.top_p = p.top_p as number | undefined;
    if (anthropic_params.stop_sequences === undefined) anthropic_params.stop_sequences = p.stop_sequences as string[] | undefined;
  }

  const openai_params: OpenAIGroupParams = {};
  for (const m of openaiLikeModels) {
    const p = modelParams[m] || {};
    if (openai_params.top_p === undefined) openai_params.top_p = p.top_p as number | undefined;
    if (openai_params.frequency_penalty === undefined) openai_params.frequency_penalty = p.frequency_penalty as number | undefined;
    if (openai_params.presence_penalty === undefined) openai_params.presence_penalty = p.presence_penalty as number | undefined;
    if (openai_params.seed === undefined) openai_params.seed = p.seed as number | undefined;
  }

  const gemini_params: GeminiGroupParams = {};
  for (const m of geminiModels) {
    const p = modelParams[m] || {};
    if (gemini_params.top_k === undefined) gemini_params.top_k = p.top_k as number | undefined;
    if (gemini_params.top_p === undefined) gemini_params.top_p = p.top_p as number | undefined;
    if (gemini_params.candidate_count === undefined) gemini_params.candidate_count = p.candidate_count as number | undefined;
    if (gemini_params.safety_settings === undefined) gemini_params.safety_settings = p.safety_settings as unknown[] | undefined;
  }

  const ollama_params: OllamaGroupParams = {};
  for (const m of ollamaModels) {
    const p = modelParams[m] || {};
    if (ollama_params.mirostat === undefined) ollama_params.mirostat = p.mirostat as number | undefined;
    if (ollama_params.mirostat_eta === undefined) ollama_params.mirostat_eta = p.mirostat_eta as number | undefined;
    if (ollama_params.mirostat_tau === undefined) ollama_params.mirostat_tau = p.mirostat_tau as number | undefined;
    if (ollama_params.num_ctx === undefined) ollama_params.num_ctx = p.num_ctx as number | undefined;
    if (ollama_params.repeat_penalty === undefined) ollama_params.repeat_penalty = p.repeat_penalty as number | undefined;
  }

  const cohere_params: CohereGroupParams = {};
  for (const m of cohereModels) {
    const p = modelParams[m] || {};
    if (cohere_params.stop_sequences === undefined) cohere_params.stop_sequences = p.stop_sequences as string[] | undefined;
    if (cohere_params.seed === undefined) cohere_params.seed = p.seed as number | undefined;
    if (cohere_params.frequency_penalty === undefined) cohere_params.frequency_penalty = p.frequency_penalty as number | undefined;
    if (cohere_params.presence_penalty === undefined) cohere_params.presence_penalty = p.presence_penalty as number | undefined;
    if (cohere_params.k === undefined) cohere_params.k = p.k as number | undefined;
    if (cohere_params.p === undefined) cohere_params.p = p.p as number | undefined;
    if (cohere_params.logprobs === undefined) cohere_params.logprobs = p.logprobs as boolean | undefined;
  }

  const deepseek_params: DeepseekGroupParams = {};
  for (const m of deepseekModels) {
    const p = modelParams[m] || {};
    if (deepseek_params.frequency_penalty === undefined) deepseek_params.frequency_penalty = p.frequency_penalty as number | undefined;
    if (deepseek_params.presence_penalty === undefined) deepseek_params.presence_penalty = p.presence_penalty as number | undefined;
    if (deepseek_params.top_p === undefined) deepseek_params.top_p = p.top_p as number | undefined;
    if (deepseek_params.logprobs === undefined) deepseek_params.logprobs = p.logprobs as boolean | undefined;
    if (deepseek_params.top_logprobs === undefined) deepseek_params.top_logprobs = p.top_logprobs as number | undefined;
  }

  // Assemble final body
  const body: ChatRequestBody = { prompt, selections };

  if (history?.length) body.history = history;
  if (system) body.system = system;
  if (temperature !== undefined) body.temperature = temperature;
  if (max_tokens !== undefined) body.max_tokens = max_tokens;
  if (min_tokens !== undefined) body.min_tokens = min_tokens;
  if (Object.keys(modelParams).length) body.model_params = modelParams;

  if (hasValues(anthropic_params)) body.anthropic_params = anthropic_params;
  if (hasValues(openai_params)) body.openai_params = openai_params;
  if (hasValues(gemini_params)) body.gemini_params = gemini_params;
  if (hasValues(ollama_params)) body.ollama_params = ollama_params;
  if (hasValues(cohere_params)) body.cohere_params = cohere_params;
  if (hasValues(deepseek_params)) body.deepseek_params = deepseek_params;

  return body;
}


// === LangGraph helpers ===
function toWire(providerKey: string, model: string): string {
  return `${providerKey}:${model}`;
}

function buildLangGraphMultiBody(opts: {
  prompt: string;
  selected: string[];
  providerOf: (m: string) => string;
  history?: { role: string; content: string }[];
  perModelParams: Record<string, Record<string, unknown>>;
}): { targets: string[]; messages: ChatMsg[]; per_model_params: Record<string, Record<string, unknown>> } {
  const { prompt, selected, providerOf, history, perModelParams } = opts;
  const targets = selected.map((m) => toWire(providerOf(m), m));
  const messages: ChatMsg[] = [...(history || []), { role: "user", content: prompt }];
  const per_model_params: Record<string, Record<string, unknown>> = {};
  for (const m of selected) {
    const wire = toWire(providerOf(m), m);
    per_model_params[wire] = perModelParams[m] || {};
  }
  return { targets, messages, per_model_params };
}

export default function CompareLLMClient(): JSX.Element {
  // ==== STATE ====
  const [activeTab, setActiveTab] = useState<"chat" | "embedding" | "image">("chat");
  const [embedView, setEmbedView] = useState<"single" | "compare">("single");
  const [loadingProviders, setLoadingProviders] = useState(false);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [allModels, setAllModels] = useState<string[]>([]);

  // ---- embeddings state
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
  const [selectedCompareDataset, setSelectedCompareDataset] = useState<string>(""); // NEW: compare dataset
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

  // One shared conversation id for both /multi and /single calls
  const [threadId] = useState<string>(() => {
    const uuid =
      (typeof crypto !== "undefined" && "randomUUID" in crypto && crypto.randomUUID()) ||
      `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
    return `thread:${uuid}`;
  });

  // ---- image processing state ----
  const [allVisionModels, setAllVisionModels] = useState<string[]>([]);
  const [selectedVisionModels, setSelectedVisionModels] = useState<string[]>([]);

  type ImgResp = {
    text?: string;
    json?: unknown;
    image_base64?: string;
    image_mime?: string;
    image_url?: string;
  };
  const [imageOutputs, setImageOutputs] = useState<
    Record<string, { response: ImgResp | null; error: string | null }>
  >({});

  const IMAGE_ENDPOINTS: ImageEndpoint[] = [
    // Adjust to your actual backend routes/names
    { id: "analyze",   label: "Analyze Image",     path: "/vision/analyze",   help: "Sends image + optional prompt and returns JSON/text." },
    { id: "transform", label: "Transform/Generate", path: "/vision/transform", help: "Returns a processed image and/or JSON/text." },
  ];

  const [imageEndpointId, setImageEndpointId] = useState<string>(IMAGE_ENDPOINTS[0].id);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePrompt, setImagePrompt] = useState<string>("");
  const [isProcessingImage, setIsProcessingImage] = useState<boolean>(false);
  const [imageError, setImageError] = useState<string | null>(null);
  const [imageResponse, setImageResponse] = useState<{
    text?: string;
    json?: unknown;
    image_base64?: string;
    image_mime?: string;
    image_url?: string;
  } | null>(null);

  // ==== LIVE REFS to avoid stale-closure on immediate Run after typing/changing params ====
  const modelParamsRef = useRef<ModelParamsMap>({});
  const globalTempRef = useRef<number | undefined>(undefined);
  const globalMaxRef = useRef<number | undefined>(undefined);
  const globalMinRef = useRef<number | undefined>(undefined);
  const promptRefLive = useRef<string>("");
  const selectedRef = useRef<string[]>([]);
  const modelChatsRef = useRef<Record<string, ModelChat>>({});

  useEffect(() => { modelParamsRef.current = modelParams; }, [modelParams]);
  useEffect(() => { globalTempRef.current = globalTemp; }, [globalTemp]);
  useEffect(() => { globalMaxRef.current = globalMax; }, [globalMax]);
  useEffect(() => { globalMinRef.current = globalMin; }, [globalMin]);
  useEffect(() => { promptRefLive.current = prompt; }, [prompt]);
  useEffect(() => { selectedRef.current = selected; }, [selected]);
  useEffect(() => { modelChatsRef.current = modelChats; }, [modelChats]);

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
    const prefix = s.split(":", 1)[0];
    if (PREFIX_TO_BRAND[prefix]) return PREFIX_TO_BRAND[prefix] as ProviderBrand;
    if (modelToBrand[m]) return modelToBrand[m];
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

    const load = async (): Promise<void> => {
      setLoadingProviders(true);
      try {
        // Providers
        const res = await fetch(`${API_BASE}/providers`, { cache: "no-store" });
        if (!res.ok) throw new Error(`Failed to load providers: ${res.status} ${res.statusText}`);
        const raw = (await res.json()) as unknown;

        const provsUnknown: unknown = Array.isArray(raw)
          ? raw
          : Array.isArray((raw as { providers?: unknown })?.providers)
          ? (raw as { providers: unknown[] }).providers
          : (raw as { providers?: unknown })?.providers && typeof (raw as { providers?: unknown })?.providers === "object"
          ? Object.values((raw as { providers: Record<string, unknown> }).providers)
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
            : Array.isArray((p as { chat_models?: unknown[] }).chat_models)
            ? (p as { chat_models: unknown[] }).chat_models
            : Array.isArray((p as { llm_models?: unknown[] }).llm_models)
            ? (p as { llm_models: unknown[] }).llm_models
            : [];
          const models = rawModels.map(pickModelName).filter((m): m is string => !!m);

          const rawEmb = Array.isArray(p.embedding_models)
            ? p.embedding_models
            : Array.isArray((p as { embed_models?: unknown[] }).embed_models)
            ? (p as { embed_models: unknown[] }).embed_models
            : [];
          const embedding_models = rawEmb.map(pickModelName).filter((m): m is string => !!m);

          const auth_required =
            typeof p.auth_required === "boolean"
              ? (p.auth_required as boolean)
              : typeof (p as { requires_api_key?: unknown }).requires_api_key === "boolean"
              ? Boolean((p as { requires_api_key: boolean }).requires_api_key)
              : isString((p as { api_key_env?: unknown }).api_key_env);

          return { name, type, base_url, models, embedding_models, auth_required };
        });

        setProviders(normalized);
        setAllModels([...new Set(normalized.flatMap((p) => p.models ?? []))].sort());
        // Vision-capable models (via /providers/vision). Fallback: all chat models.
        try {
          const vres = await fetch(`${API_BASE}/providers/vision`, { cache: "no-store" });
          if (!vres.ok) throw new Error(`Failed to load vision providers: ${vres.status} ${vres.statusText}`);
          const vraw = (await vres.json()) as unknown;

          const vProvsUnknown: unknown = Array.isArray(vraw)
            ? vraw
            : Array.isArray((vraw as { providers?: unknown })?.providers)
            ? (vraw as { providers: unknown[] }).providers
            : (vraw as { providers?: unknown })?.providers && typeof (vraw as { providers?: unknown })?.providers === "object"
            ? Object.values((vraw as { providers: Record<string, unknown> }).providers)
            : [];

          const vArr = Array.isArray(vProvsUnknown) ? vProvsUnknown : [];
          const vProviders: ProviderInfo[] = vArr.map((item, idx) => {
            const p = (item ?? {}) as Record<string, unknown>;
            const name =
              isString(p.name) ? p.name : isString(p.key) ? p.key : isString(p.id) ? p.id : `prov_${idx}`;
            const type = isString(p.type) ? p.type : "unknown";
            const base_url = isString(p.base_url) ? p.base_url : "";

            const rawModels = Array.isArray(p.models)
              ? p.models
              : Array.isArray((p as { chat_models?: unknown[] }).chat_models)
              ? (p as { chat_models: unknown[] }).chat_models
              : Array.isArray((p as { llm_models?: unknown[] }).llm_models)
              ? (p as { llm_models: unknown[] }).llm_models
              : [];
            const models = rawModels.map(pickModelName).filter((m): m is string => !!m);

            const rawEmb = Array.isArray(p.embedding_models)
              ? p.embedding_models
              : Array.isArray((p as { embed_models?: unknown[] }).embed_models)
              ? (p as { embed_models: unknown[] }).embed_models
              : [];
            const embedding_models = rawEmb.map(pickModelName).filter((m): m is string => !!m);

            const auth_required =
              typeof p.auth_required === "boolean"
                ? (p.auth_required as boolean)
                : typeof (p as { requires_api_key?: unknown }).requires_api_key === "boolean"
                ? Boolean((p as { requires_api_key: boolean }).requires_api_key)
                : isString((p as { api_key_env?: unknown }).api_key_env);

            return { name, type, base_url, models, embedding_models, auth_required };
          });

          const visionModels = [...new Set(vProviders.flatMap((p) => p.models ?? []))].sort();
          setAllVisionModels(visionModels);
        } catch {
          // Graceful fallback: treat all chat models as vision-capable
          setAllVisionModels([...new Set(normalized.flatMap((p) => p.models ?? []))].sort());
        }

        // Embedding models & stores
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
          .map((id) => ({ dataset_id: id } as Dataset));
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
      const parsed = JSON.parse(jsonInput) as unknown;
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
        msg +=
          `\n${bad.length} failed:\n` +
          bad
            .map((r) => (r as PromiseRejectedResult).reason?.message || String((r as PromiseRejectedResult).reason))
            .join("\n");
      }
      alert(msg);

      setJsonInput("");
      setDatasetId("");
      await refreshStoresAndDatasets();
    } catch (err) {
      // eslint-disable-next-line no-console
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

      const rows: SearchResult[] = matches.map((m) => {
        const base: Record<string, unknown> = { ...(m.metadata || {}) };
        (base as { text: string }).text = m.page_content;
        if (typeof m.score === "number") (base as { similarity_score: number }).similarity_score = m.score;
        return base as SearchResult;
      });

      if (myId === requestIdRef.current) {
        setSearchResults(rows);
        setSearchContext(snapshot);
      }
    } catch (err) {
      // eslint-disable-next-line no-console
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
  // Embeddings: multi-model compare (NEW: hits backend /embeddings/compare)
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
    if (!selectedCompareDataset.trim()) {
      alert("Select a dataset for comparison.");
      return;
    }

    const started = Date.now();
    setIsComparing(true);
    setMultiSearchResults(null);
    const myId = ++requestIdRef.current;

    try {
      const res = await fetch(`${API_BASE}/embeddings/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: selectedCompareDataset.trim(),
          embedding_models: selectedEmbeddingModels.map((s) => s.trim()),
          query: compareQuery.trim(),
          k: topKCompare,
        }),
      });

      if (!res.ok) {
        const err = (await res.json().catch(() => ({}))) as { detail?: string };
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const json = (await res.json()) as MultiSearchResponse;
      const payload: MultiSearchResponse = {
        ...json,
        duration_ms: typeof json.duration_ms === "number" ? json.duration_ms : Date.now() - started,
      };

      if (myId === requestIdRef.current) setMultiSearchResults(payload);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("Compare failed:", err);
      if (myId === requestIdRef.current) {
        alert(`Compare failed: ${err instanceof Error ? err.message : "Unknown error"}`);
        setMultiSearchResults(null);
      }
    } finally {
      if (myId === requestIdRef.current) setIsComparing(false);
    }
  }, [API_BASE, compareQuery, selectedEmbeddingModels, selectedCompareDataset, topKCompare]);

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
        if (selectedCompareDataset === id) setSelectedCompareDataset("");
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error("Delete failed:", err);
      }
    },
    [stores, refreshStoresAndDatasets, selectedDataset, selectedCompareDataset]
  );

  const runImageProcessing = useCallback(async () => {
    if (!imageFile || selectedVisionModels.length === 0) return;

    setIsProcessingImage(true);
    setImageError(null);
    // prime per-model outputs
    setImageOutputs(Object.fromEntries(selectedVisionModels.map((m) => [m, { response: null, error: null }])));

    const endpoint = IMAGE_ENDPOINTS.find((e) => e.id === imageEndpointId) || IMAGE_ENDPOINTS[0];
    const url = `${API_BASE}${endpoint.path}`;

    await Promise.allSettled(
      selectedVisionModels.map(async (model) => {
        try {
          const form = new FormData();
          form.append("file", imageFile);
          if (imagePrompt.trim()) form.append("prompt", imagePrompt.trim());

          // Pass model info; your backend can use either or ignore
          form.append("model", model);
          const provKey = getProviderKey(model);
          if (provKey) form.append("provider", provKey);
          form.append("wire", toWire(provKey, model));

          const res = await fetch(url, { method: "POST", body: form });
          if (!res.ok) {
            let msg = `HTTP ${res.status}`;
            try {
              const j = (await res.json()) as { detail?: string };
              if (j?.detail) msg = j.detail;
            } catch {}
            throw new Error(msg);
          }

          const payload = (await res.json()) as {
            text?: string; message?: string; json?: unknown; result?: unknown;
            image_base64?: string; image_mime?: string; image_url?: string;
          };

          const normalized: ImgResp = {
            text:
              typeof payload.text === "string" ? payload.text :
              typeof payload.message === "string" ? payload.message : undefined,
            json: typeof payload.json !== "undefined" ? payload.json : payload.result,
            image_base64: payload.image_base64,
            image_mime: payload.image_mime,
            image_url: payload.image_url,
          };

          setImageOutputs((prev) => ({ ...prev, [model]: { response: normalized, error: null } }));
        } catch (e) {
          const msg = e instanceof Error ? e.message : "Unknown error";
          setImageOutputs((prev) => ({ ...prev, [model]: { response: null, error: msg } }));
        }
      })
    );

    setIsProcessingImage(false);
  }, [API_BASE, imageEndpointId, imageFile, imagePrompt, selectedVisionModels, getProviderKey]);


  const clearImageInputs = useCallback(() => {
    setImageFile(null);
    setImagePrompt("");
    setImageError(null);
    setImageResponse(null);
  }, []);


  // =============================
  // Chat tab
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
      const next: AskAnswers = { ...prev };
      delete next[model];
      return next;
    });
  }, []);

  const openModelChat = useCallback(
    (model: string) => {
      setActiveModel(model);
      setModelChats((prev) => {
        if (prev[model]) return prev;
        const initial: ModelChat["messages"] = [];
        if (promptRefLive.current.trim()) initial.push({ role: "user", content: promptRefLive.current.trim(), timestamp: Date.now() });
        const modelAnswer = answers[model];
        if (modelAnswer?.answer && !modelAnswer.error) {
          initial.push({ role: "assistant", content: modelAnswer.answer, timestamp: Date.now() });
        }
        return { ...prev, [model]: { messages: initial, isStreaming: false, currentResponse: "" } };
      });
    },
    [answers]
  );
  const closeModelChat = useCallback(() => {
    setActiveModel(null);
    interactiveAbortRef.current?.abort();
  }, []);

  const pruneUndefined = <T extends Record<string, unknown>>(obj: T): Partial<T> =>
    pruneUndefBrandUtils(obj);

  // === Interactive chat via LangGraph single-stream (SSE) ===
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
      const currentChat = modelChatsRef.current[activeModel] || { messages: [] as ModelChat["messages"] };
      const conversationHistory = [
        ...currentChat.messages,
        { role: "user" as const, content: message, timestamp: Date.now() },
      ];
      const apiMessages: ChatMsg[] = conversationHistory.map((m) => ({ role: m.role, content: m.content }));

      const wire = toWire(getProviderKey(activeModel), activeModel);
      const res = await fetch(`${API_BASE}/langgraph/chat/single/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          wire,
          messages: apiMessages, // include the latest user message
          model_params: modelParamsRef.current[activeModel] || {},
          thread_id: threadId,   // shared thread for memory
        }),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

      const reader = res.body.getReader();
      await readSSE<LGEvent>(reader, (evt) => {
        if (evt.type === "delta" && evt.scope === "single") {
          const piece = evt.delta || "";
          setModelChats((prev) => ({
            ...prev,
            [activeModel]: {
              ...(prev[activeModel] || { messages: [] }),
              isStreaming: true,
              currentResponse: (prev[activeModel]?.currentResponse || "") + piece,
            },
          }));
        } else if (evt.type === "done" && evt.scope === "single") {
          setModelChats((prev) => {
            const streamed = prev[activeModel]?.currentResponse || "";
            return {
              ...prev,
              [activeModel]: {
                ...(prev[activeModel] || { messages: [] }),
                isStreaming: false,
                currentResponse: "",
                messages: [
                  ...(prev[activeModel]?.messages || []),
                  { role: "assistant", content: streamed, timestamp: Date.now() },
                ],
              },
            };
          });
        }
      });
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
  }, [activeModel, interactivePrompt, getProviderKey, threadId]);

  // Providers-only chat streaming bits
  const canRun = prompt.trim().length > 0 && selected.length > 0 && !isRunning;

  const resetRun = useCallback(() => {
    const initial: AskAnswers = Object.fromEntries(
      selectedRef.current.map((m) => [m, { answer: "", error: undefined, latency_ms: 0 }])
    ) as AskAnswers;
    setAnswers(initial);
    setStartedAt(Date.now());
    setEndedAt(null);
  }, []);

  // === Run (LangGraph MULTI stream via SSE) ===
  const runPrompt = useCallback(async () => {
    if (!canRun) return;
    streamAbortRef.current?.abort();
    const controller = new AbortController();
    streamAbortRef.current = controller;

    resetRun();
    setIsRunning(true);
    setLastRunPrompt(promptRefLive.current.trim());

    // Build wire<->model maps for this run
    const selectedNow = [...selectedRef.current];
    const wireForModel: Record<string, string> = Object.fromEntries(
      selectedNow.map((m) => [m, toWire(getProviderKey(m), m)])
    );
    const modelForWire: Record<string, string> = Object.fromEntries(
      selectedNow.map((m) => [wireForModel[m], m])
    );

    const startedAtRun = Date.now();
    const modelStart: Record<string, number> = {}; // latency starts on first delta per model

    const processEvent = (evt: NDJSONEvent | LGEvent | unknown): void => {
      // Legacy /chat/* shape
      if (isNDJSONEvent(evt)) {
        if (evt.type === "start") {
          modelStart[evt.model] = Date.now();
          return;
        }
        if (evt.type === "delta") {
          setAnswers((prev) => ({
            ...prev,
            [evt.model]: {
              answer: (prev[evt.model]?.answer || "") + (evt.text || ""),
              error: prev[evt.model]?.error,
              latency_ms: prev[evt.model]?.latency_ms ?? 0,
            },
          }));
          return;
        }
        if (evt.type === "error") {
          setAnswers((prev) => ({
            ...prev,
            [evt.model]: {
              answer: prev[evt.model]?.answer || "",
              error: evt.error || "Unknown error",
              latency_ms: prev[evt.model]?.latency_ms ?? 0,
            },
          }));
          return;
        }
        if (evt.type === "end") {
          const t0 = modelStart[evt.model] ?? startedAtRun;
          setAnswers((prev) => {
            const prevAns = prev[evt.model]?.answer || "";
            return {
              ...prev,
              [evt.model]: {
                ...prev[evt.model],
                answer: prevAns || evt.text || "",
                latency_ms: Date.now() - t0,
              },
            };
          });
          return;
        }
        if (evt.type === "all_done") {
          setIsRunning(false);
          setEndedAt(Date.now());
          streamAbortRef.current = null;
          return;
        }
      }

      // New /langgraph/* shape
      if (isLGEvent(evt)) {
        if (evt.scope === "multi") {
          if (evt.type === "delta" && evt.model) {
            const modelKey = modelForWire[evt.model] || evt.model;
            if (!modelStart[modelKey]) modelStart[modelKey] = Date.now();
            setAnswers((prev) => ({
              ...prev,
              [modelKey]: {
                ...(prev[modelKey] || { answer: "", latency_ms: 0 }),
                answer: (prev[modelKey]?.answer || "") + (evt.text || ""),
              },
            }));
            return;
          }
          if (evt.type === "done" && evt.model) {
            const modelKey = modelForWire[evt.model] || evt.model;
            const t0 = modelStart[modelKey] ?? startedAtRun;
            setAnswers((prev) => ({
              ...prev,
              [modelKey]: {
                ...(prev[modelKey] || { answer: "" }),
                latency_ms: Date.now() - t0,
              },
            }));
            return;
          }
        }
      }
      // ignore anything else
    };

    try {
      const body = buildLangGraphMultiBody({
        prompt: promptRefLive.current.trim(),
        selected: selectedNow,
        providerOf: (m) => getProviderKey(m),
        history: undefined,
        perModelParams: modelParamsRef.current,
      });

      const res = await fetch(`${API_BASE}/langgraph/chat/multi/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          ...body,
          thread_id: threadId, // shared thread id so single+multi share memory
        }),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) throw new Error(`Bad response: ${res.status} ${res.statusText}`);

      const reader = res.body.getReader();
      await readSSE<NDJSONEvent | LGEvent>(reader, (evt) => {
        try {
          processEvent(evt);
        } catch {
          // ignore malformed
        }
      });
    } catch (err) {
      const isAbort = err instanceof DOMException && err.name === "AbortError";
      if (!isAbort) {
        // eslint-disable-next-line no-console
        console.error(err);
        setIsRunning(false);
        setEndedAt(Date.now());
      }
    } finally {
      setIsRunning(false);
      streamAbortRef.current = null;
    }
  }, [canRun, resetRun, getProviderKey, threadId]);

  // === Retry single model via LangGraph single-stream (SSE) ===
  const retryModel = useCallback(
    async (model: string) => {
      const retryPrompt = (lastRunPrompt || promptRefLive.current).trim();
      if (!retryPrompt) {
        alert("No prompt to retry. Please enter a prompt first.");
        return;
      }
      setAnswers((prev) => ({ ...prev, [model]: { answer: "", error: undefined, latency_ms: 0 } }));
      const controller = new AbortController();
      const started = Date.now();

      try {
        const wire = toWire(getProviderKey(model), model);
        const res = await fetch(`${API_BASE}/langgraph/chat/single/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
          },
          body: JSON.stringify({
            wire,
            messages: [{ role: "user", content: retryPrompt }],
            model_params: modelParamsRef.current[model] || {},
            thread_id: threadId, // shared thread id
          }),
          signal: controller.signal,
        });

        if (!res.ok || !res.body) {
          const err = (await res.json().catch(() => ({ detail: res.statusText }))) as { detail?: string };
          throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const reader = res.body.getReader();
        await readSSE<LGEvent>(reader, (evt) => {
          if (evt.scope === "single" && evt.type === "delta") {
            const piece = evt.delta || "";
            setAnswers((prev) => ({
              ...prev,
              [model]: {
                answer: (prev[model]?.answer || "") + piece,
                error: prev[model]?.error,
                latency_ms: prev[model]?.latency_ms ?? 0,
              },
            }));
          } else if (evt.scope === "single" && evt.type === "done") {
            setAnswers((prev) => ({
              ...prev,
              [model]: { ...(prev[model] || { answer: "" }), latency_ms: Date.now() - started },
            }));
          }
        });
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Unknown error";
        setAnswers((prev) => ({ ...prev, [model]: { answer: "", error: `Retry failed: ${msg}`, latency_ms: 0 } }));
      }
    },
    [lastRunPrompt, getProviderKey, threadId]
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
        if (
          activeTab === "embedding" &&
          compareQuery.trim() &&
          selectedEmbeddingModels.length > 0 &&
          datasets.length > 0 &&
          selectedCompareDataset
        ) {
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
        } else if (activeTab === "image" && imageFile) {            // NEW
          void runImageProcessing();
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
    compareQuery,
    selectedCompareDataset,
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
        onChange={(id) => setActiveTab(id as "chat" | "embedding" | "image")}
        tabs={[
          { id: "chat", label: "Chat Models" },
          { id: "embedding", label: "Embeddings" },
          { id: "image", label: "Image Processing" }, // NEW
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
                        onClick={() =>
                          setExpandedModels((prev) => {
                            const next = new Set(prev);
                            next.has(m) ? next.delete(m) : next.add(m);
                            return next;
                          })
                        }
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
                                type="number"
                                step={0.1}
                                min={0}
                                max={2}
                                value={modelParams[m]?.temperature ?? ""}
                                placeholder="↳ global / default"
                                onChange={(e) =>
                                  setModelParams((prev) => ({
                                    ...prev,
                                    [m]: { ...prev[m], temperature: e.target.value ? Number(e.target.value) : undefined },
                                  }))
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
                                placeholder="↳ global / default"
                                onChange={(e) =>
                                  setModelParams((prev) => ({
                                    ...prev,
                                    [m]: { ...prev[m], max_tokens: e.target.value ? Number(e.target.value) : undefined },
                                  }))
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
                                placeholder="optional"
                                onChange={(e) =>
                                  setModelParams((prev) => ({
                                    ...prev,
                                    [m]: { ...prev[m], min_tokens: e.target.value ? Number(e.target.value) : undefined },
                                  }))
                                }
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
            // NEW: dedicated compare dataset selector
            selectedCompareDataset={selectedCompareDataset}
            setSelectedCompareDataset={setSelectedCompareDataset}
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
      {activeTab === "image" && (
        <main className="grid grid-cols-1 xl:grid-cols-[400px_1fr] gap-6 items-start">
          {/* Left rail: model picker, endpoint, uploader, prompt */}
          <ImageLeftRail
            // NEW: model selection
            allVisionModels={allVisionModels}
            selectedVisionModels={selectedVisionModels}
            toggleVisionModel={(m) =>
              setSelectedVisionModels((prev) => (prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]))
            }
            selectAllVision={() => setSelectedVisionModels(allVisionModels)}
            clearAllVision={() => setSelectedVisionModels([])}
            getProviderType={getProviderType}

            // Endpoint/prompt/uploader
            endpoints={IMAGE_ENDPOINTS}
            selectedEndpointId={imageEndpointId}
            setSelectedEndpointId={setImageEndpointId}
            imageFile={imageFile}
            setImageFile={setImageFile}
            prompt={imagePrompt}
            setPrompt={setImagePrompt}
            isProcessing={isProcessingImage}
            onRun={runImageProcessing}
            onClear={clearImageInputs}
          />

          {/* Right rail: multi-model results */}
          <ImageResults
            isProcessing={isProcessingImage}
            error={imageError}
            outputs={imageOutputs}
            brandOf={getProviderType}
          />
        </main>
      )}


    </div>
  );
}
