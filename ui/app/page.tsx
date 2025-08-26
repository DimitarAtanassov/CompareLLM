"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
/* ==========================================================================
   Types (keep in sync with your backend)
   ========================================================================== */
type ProviderInfo = {
  name: string;
  type: string;
  base_url: string;
  models: string[];
  embedding_models: string[];
  auth_required: boolean;
  wire?: ProviderWire;
};

type PerModelParam = {
  temperature?: number;
  max_tokens?: number;
  min_tokens?: number;
  // Anthropic
  thinking_enabled?: boolean;
  thinking_budget_tokens?: number;
  top_k?: number;
  top_p?: number;
  stop_sequences?: string[];
  // OpenAI
  frequency_penalty?: number;
  presence_penalty?: number;
  seed?: number;
  // Gemini
  candidate_count?: number;
  safety_settings?: unknown[];
  // Ollama
  mirostat?: number;
  mirostat_eta?: number;
  mirostat_tau?: number;
  num_ctx?: number;
  repeat_penalty?: number;
};

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

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  timestamp: number;
};

type ModelChat = {
  messages: ChatMessage[];
  isStreaming: boolean;
  currentResponse: string;
};

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
};

/* ---------- Multi-embedding search types ---------- */
type MultiBucket = {
  error?: string;
  items: SearchResult[];
  dataset_id?: string;
  total_documents?: number;
};
type MultiSearchResponse = {
  query: string;
  results: Record<string, MultiBucket>;
  duration_ms?: number;
};

/* ==========================================================================
   Provider branding vs wire (API dialect)
   ========================================================================== */
const BRAND_TO_DEFAULT_WIRE: Partial<Record<ProviderBrand, ProviderWire>> = {
  deepseek: "openai",   // DeepSeek speaks OpenAI wire
  voyage: "openai"   // ← new
};
type ProviderBrand =
    | "anthropic"
    | "openai"
    | "gemini"
    | "ollama"
    | "deepseek"
    | "voyage"   // ← new
    | "unknown";

type ProviderWire = "anthropic" | "openai" | "gemini" | "ollama" | "unknown";

const isProviderWire = (x?: string | null): x is ProviderWire =>
  x === "anthropic" || x === "openai" || x === "gemini" || x === "ollama" || x === "unknown";


const isProviderBrand = (x?: string | null): x is ProviderBrand =>
  x === "anthropic" || x === "openai" || x === "gemini" || x === "ollama" ||
  x === "deepseek" || x === "voyage" || x === "unknown";

const coerceWire = (p: ProviderInfo): ProviderWire => {
  if (isProviderWire(p.wire)) return p.wire;           // backend explicit wire wins
  const brand = coerceBrand(p.type);                    // normalize branding
  const inferred = BRAND_TO_DEFAULT_WIRE[brand];        // brand→wire default
  if (inferred) return inferred;
  if (isProviderWire(p.type)) return p.type as ProviderWire;  // rare case
  return "unknown";
};

const coerceBrand = (t?: string): ProviderBrand => {
  const s = (t || "").toLowerCase();
  if (isProviderBrand(s as ProviderBrand)) return s as ProviderBrand;
  if (s.includes("voyage")) return "voyage"; // ← new
  return "unknown";
};

/* Typed color maps for badges/text */
const PROVIDER_TEXT_COLOR: Record<ProviderBrand, string> = {
  anthropic: "text-orange-600 dark:text-orange-400",
  openai: "text-blue-600 dark:text-blue-400",
  gemini: "text-green-600 dark:text-green-400",
  ollama: "text-purple-600 dark:text-purple-400",
  deepseek: "text-sky-600 dark:text-sky-400",
  voyage: "text-cyan-600 dark:text-cyan-400",    // ← new
  unknown: "text-zinc-600 dark:text-zinc-400",
};

const PROVIDER_BADGE_BG: Record<ProviderBrand, string> = {
  anthropic: "bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400",
  openai: "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400",
  gemini: "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400",
  ollama: "bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400",
  deepseek: "bg-sky-100 dark:bg-sky-900/30 text-sky-700 dark:text-sky-400",
  voyage: "bg-cyan-100 dark:bg-cyan-900/30 text-cyan-700 dark:text-cyan-400", // ← new
  unknown: "bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-400",
};
function Spinner({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/>
    </svg>
  );
}

function LoadingBar() {
  return (
    <div className="relative h-1 overflow-hidden rounded bg-orange-100 dark:bg-orange-900/30" role="status" aria-live="polite">
      <div className="absolute inset-y-0 left-0 w-1/3 animate-[loading_1.2s_infinite] bg-orange-500/80" />
      <style jsx>{`
        @keyframes loading {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(300%); }
        }
      `}</style>
    </div>
  );
}


/* ==========================================================================
   Config
   ========================================================================== */
const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "/backend");

function redactResult(row: SearchResult): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(row)) {
    if (k === "similarity_score") continue;   // keep badge only
    if (k.startsWith("_")) continue;          // hides _text_field, _embedding_model, etc.
    out[k] = v;
  }
  return out;
}
function primarySnippet(row: SearchResult, maxLen = 160): string {
  // Prefer explicit "text" field if present and string
  const textVal = row["text"];
  if (typeof textVal === "string") {
    return textVal.length > maxLen ? textVal.slice(0, maxLen) + "…" : textVal;
  }

  // Otherwise, find the first non-empty string field
  for (const [key, value] of Object.entries(row)) {
    if (key === "similarity_score" || key.startsWith("_")) continue;
    if (typeof value === "string" && value.trim().length > 0) {
      return value.length > maxLen ? value.slice(0, maxLen) + "…" : value;
    }
  }

  // Fallback: stringify the row
  const s = JSON.stringify(row);
  return s.length > maxLen ? s.slice(0, maxLen) + "…" : s;
}


/* ==========================================================================
   Provider-specific parameter editor (driven by WIRE)
   ========================================================================== */
function ProviderParameterEditor({
  model,
  providerWire,
  params,
  onUpdate,
}: {
  model: string;
  providerWire: ProviderWire;
  params: PerModelParam;
  onUpdate: (params: PerModelParam) => void;
}) {
  const updateParam = (key: keyof PerModelParam, value: unknown) => {
    onUpdate({ ...params, [key]: value });
  };

  const formatStopSequences = (sequences?: string[]): string => (sequences ? sequences.join(", ") : "");
  const parseStopSequences = (value: string): string[] =>
    value.split(",").map((s) => s.trim()).filter((s) => s.length > 0);

  if (providerWire === "anthropic") {
    return (
      <div className="space-y-3">
        <h4 className="text-sm font-semibold text-orange-600 dark:text-orange-400">Anthropic Parameters</h4>

        <div className="bg-orange-50/50 dark:bg-orange-900/20 p-3 rounded-lg border border-orange-200 dark:border-orange-800">
          <label className="flex items-center gap-2 mb-2">
            <input
              type="checkbox"
              checked={params.thinking_enabled ?? false}
              onChange={(e) => updateParam("thinking_enabled", e.target.checked)}
              className="accent-orange-600 dark:accent-orange-500"
            />
            <span className="text-sm font-medium">Enable Extended Thinking</span>
          </label>

          {params.thinking_enabled && (
            <div>
              <label className="block text-xs font-medium mb-1">Thinking Budget (tokens)</label>
              <input
                type="number"
                min={1024}
                value={params.thinking_budget_tokens ?? 2048}
                onChange={(e) => updateParam("thinking_budget_tokens", Number(e.target.value))}
                className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
                placeholder="2048"
              />
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">Minimum 1024 tokens for thinking</p>
            </div>
          )}
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium mb-1">Top-K</label>
            <input
              type="number"
              min={1}
              value={params.top_k ?? ""}
              onChange={(e) => updateParam("top_k", e.target.value ? Number(e.target.value) : undefined)}
              className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
              placeholder="40"
            />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1">Top-P</label>
            <input
              type="number"
              step={0.1}
              min={0}
              max={1}
              value={params.top_p ?? ""}
              onChange={(e) => updateParam("top_p", e.target.value ? Number(e.target.value) : undefined)}
              className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
              placeholder="0.9"
            />
          </div>
        </div>

        <div>
          <label className="block text-xs font-medium mb-1">Stop Sequences</label>
          <input
            type="text"
            value={formatStopSequences(params.stop_sequences)}
            onChange={(e) => updateParam("stop_sequences", parseStopSequences(e.target.value))}
            className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
            placeholder="Human:, Assistant:, Stop"
          />
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">Comma-separated</p>
        </div>
      </div>
    );
  }

  if (providerWire === "openai") {
    return (
      <div className="space-y-3">
        <h4 className="text-sm font-semibold text-blue-600 dark:text-blue-400">OpenAI Parameters</h4>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium mb-1">Top-P</label>
            <input
              type="number"
              step={0.1}
              min={0}
              max={1}
              value={params.top_p ?? ""}
              onChange={(e) => updateParam("top_p", e.target.value ? Number(e.target.value) : undefined)}
              className="w-full rounded-md border border-blue-200 dark:border-blue-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
              placeholder="0.9"
            />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1">Seed</label>
            <input
              type="number"
              value={params.seed ?? ""}
              onChange={(e) => updateParam("seed", e.target.value ? Number(e.target.value) : undefined)}
              className="w-full rounded-md border border-blue-200 dark:border-blue-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
              placeholder="42"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium mb-1">Frequency Penalty</label>
            <input
              type="number"
              step={0.1}
              min={-2}
              max={2}
              value={params.frequency_penalty ?? ""}
              onChange={(e) => updateParam("frequency_penalty", e.target.value ? Number(e.target.value) : undefined)}
              className="w-full rounded-md border border-blue-200 dark:border-blue-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
              placeholder="0.0"
            />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1">Presence Penalty</label>
            <input
              type="number"
              step={0.1}
              min={-2}
              max={2}
              value={params.presence_penalty ?? ""}
              onChange={(e) => updateParam("presence_penalty", e.target.value ? Number(e.target.value) : undefined)}
              className="w-full rounded-md border border-blue-200 dark:border-blue-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
              placeholder="0.0"
            />
          </div>
        </div>
      </div>
    );
  }

  if (providerWire === "gemini") {
    return (
      <div className="space-y-3">
        <h4 className="text-sm font-semibold text-green-600 dark:text-green-400">Gemini Parameters</h4>
        <div className="grid grid-cols-3 gap-3">
          <div>
            <label className="block text-xs font-medium mb-1">Top-K</label>
            <input
              type="number"
              min={1}
              value={params.top_k ?? ""}
              onChange={(e) => updateParam("top_k", e.target.value ? Number(e.target.value) : undefined)}
              className="w-full rounded-md border border-green-200 dark:border-green-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
              placeholder="40"
            />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1">Top-P</label>
            <input
              type="number"
              step={0.1}
              min={0}
              max={1}
              value={params.top_p ?? ""}
              onChange={(e) => updateParam("top_p", e.target.value ? Number(e.target.value) : undefined)}
              className="w-full rounded-md border border-green-200 dark:border-green-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
              placeholder="0.9"
            />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1">Candidates</label>
            <input
              type="number"
              min={1}
              max={4}
              value={params.candidate_count ?? ""}
              onChange={(e) => updateParam("candidate_count", e.target.value ? Number(e.target.value) : undefined)}
              className="w-full rounded-md border border-green-200 dark:border-green-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
              placeholder="1"
            />
          </div>
        </div>
      </div>
    );
  }

  if (providerWire === "ollama") {
    return (
      <div className="space-y-3">
        <h4 className="text-sm font-semibold text-purple-600 dark:text-purple-400">Ollama Parameters</h4>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium mb-1">Mirostat</label>
            <select
              value={params.mirostat ?? ""}
              onChange={(e) => updateParam("mirostat", e.target.value ? Number(e.target.value) : undefined)}
              className="w-full rounded-md border border-purple-200 dark:border-purple-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
            >
              <option value="">Off</option>
              <option value="1">Mirostat 1</option>
              <option value="2">Mirostat 2</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium mb-1">Context Size</label>
            <input
              type="number"
              min={1}
              value={params.num_ctx ?? ""}
              onChange={(e) => updateParam("num_ctx", e.target.value ? Number(e.target.value) : undefined)}
              className="w-full rounded-md border border-purple-200 dark:border-purple-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
              placeholder="4096"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium mb-1">Mirostat Eta</label>
            <input
              type="number"
              step={0.01}
              min={0}
              value={params.mirostat_eta ?? ""}
              onChange={(e) => updateParam("mirostat_eta", e.target.value ? Number(e.target.value) : undefined)}
              className="w-full rounded-md border border-purple-200 dark:border-purple-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
              placeholder="0.1"
            />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1">Mirostat Tau</label>
            <input
              type="number"
              step={0.1}
              min={0}
              value={params.mirostat_tau ?? ""}
              onChange={(e) => updateParam("mirostat_tau", e.target.value ? Number(e.target.value) : undefined)}
              className="w-full rounded-md border border-purple-200 dark:border-purple-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
              placeholder="5.0"
            />
          </div>
        </div>

        <div>
          <label className="block text-xs font-medium mb-1">Repeat Penalty</label>
          <input
            type="number"
            step={0.1}
            min={0}
            value={params.repeat_penalty ?? ""}
            onChange={(e) => updateParam("repeat_penalty", e.target.value ? Number(e.target.value) : undefined)}
            className="w-full rounded-md border border-purple-200 dark:border-purple-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
            placeholder="1.1"
          />
        </div>
      </div>
    );
  }

  return null;
}

/* ==========================================================================
   Main Page
   ========================================================================== */
export default function Page() {
  const [activeTab, setActiveTab] = useState<"chat" | "embedding">("chat");

  // Providers and models
  const [loadingProviders, setLoadingProviders] = useState(false);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [allModels, setAllModels] = useState<string[]>([]);
  const [allEmbeddingModels, setAllEmbeddingModels] = useState<string[]>([]);

  // Chat
  const [selected, setSelected] = useState<string[]>([]);
  const [prompt, setPrompt] = useState<string>("");
  const [isRunning, setIsRunning] = useState(false);
  const [answers, setAnswers] = useState<AskAnswers>({});
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [endedAt, setEndedAt] = useState<number | null>(null);

  // Interactive chat
  const [activeModel, setActiveModel] = useState<string | null>(null);
  const [modelChats, setModelChats] = useState<Record<string, ModelChat>>({});
  const [interactivePrompt, setInteractivePrompt] = useState<string>("");

  // Embeddings
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
  const chatBottomRef = useRef<HTMLDivElement | null>(null);

  const requestIdRef = useRef(0);

  // Params
  const [modelParams, setModelParams] = useState<ModelParamsMap>({});
  const [globalTemp, setGlobalTemp] = useState<number | undefined>(undefined);
  const [globalMax, setGlobalMax] = useState<number | undefined>(undefined);
  const [globalMin, setGlobalMin] = useState<number | undefined>(undefined);
  const useEnhancedAPI = true;

  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set());

  // Branding/type map
  const modelToProvider = useMemo(() => {
    const map: Record<string, ProviderBrand> = {};
    providers.forEach((p) => {
      (p.models || []).forEach((m) => {
        map[m] = coerceBrand(p.type);
      });
      (p.embedding_models || []).forEach((m) => {
        map[m] = coerceBrand(p.type);
      });
    });
    return map;
  }, [providers]);
  
  const hasAnyDataset = useMemo(() => datasets.length > 0, [datasets]);

  // WIRE map
  const modelToWire = useMemo(() => {
    const map: Record<string, ProviderWire> = {};
    providers.forEach((p) => {
      const wire = coerceWire(p);
      (p.models || []).forEach((m) => {
        map[m] = wire;
      });
      (p.embedding_models || []).forEach((m) => {
        map[m] = wire;
      });
    });
    return map;
  }, [providers]);

  const getProviderType = useCallback((modelName: string): ProviderBrand => {
    return modelToProvider[modelName] ?? "unknown";
  }, [modelToProvider]);

  const getProviderWire = useCallback((modelName: string): ProviderWire => {
    return modelToWire[modelName] ?? "unknown";
  }, [modelToWire]);

  const updateParam = useCallback((model: string, params: PerModelParam) => {
    setModelParams((prev) => ({ ...prev, [model]: params }));
  }, []);

  const toggleModelExpansion = useCallback((model: string) => {
    setExpandedModels((prev) => {
      const next = new Set(prev);
      if (next.has(model)) next.delete(model);
      else next.add(model);
      return next;
    });
  }, []);

  /* ---------- Interactive Chat Functions ---------- */
  const openModelChat = useCallback((model: string) => {
    setActiveModel(model);
    setModelChats((prev) => {
      if (prev[model]) return prev;
      const initialMessages: ChatMessage[] = [];
      if (prompt.trim()) {
        initialMessages.push({
          role: "user",
          content: prompt.trim(),
          timestamp: Date.now(),
        });
      }
      const modelAnswer = answers[model];
      if (modelAnswer?.answer && !modelAnswer.error) {
        initialMessages.push({
          role: "assistant",
          content: modelAnswer.answer,
          timestamp: Date.now(),
        });
      }
      return {
        ...prev,
        [model]: { messages: initialMessages, isStreaming: false, currentResponse: "" },
      };
    });
  }, [answers, prompt]);

  const closeModelChat = useCallback(() => {
    setActiveModel(null);
    interactiveAbortRef.current?.abort();
  }, []);

  const sendInteractiveMessage = useCallback(async () => {
    if (!activeModel || !interactivePrompt.trim()) return;

    const message = interactivePrompt.trim();
    setInteractivePrompt("");

    setModelChats((prev) => ({
      ...prev,
      [activeModel]: {
        ...prev[activeModel],
        messages: [...prev[activeModel].messages, { role: "user", content: message, timestamp: Date.now() }],
        isStreaming: true,
        currentResponse: "",
      },
    }));

    interactiveAbortRef.current?.abort();
    const controller = new AbortController();
    interactiveAbortRef.current = controller;

    try {
      const currentChat = modelChats[activeModel] || { messages: [] as ChatMessage[] };
      const conversationHistory = [...currentChat.messages, { role: "user" as const, content: message, timestamp: Date.now() }];
      const apiMessages = conversationHistory.map((m) => ({ role: m.role, content: m.content }));

      const modelParam = modelParams[activeModel] || {};
      const providerWire = getProviderWire(activeModel);
      const hasProviderParams = Object.keys(modelParam).some((k) => !["temperature", "max_tokens", "min_tokens"].includes(k));

      let endpoint = "";
      let body = "";

      if (useEnhancedAPI || hasProviderParams) {
        endpoint = `${API_BASE}/v2/chat/completions/enhanced`;
        const enhancedRequest: EnhancedChatRequest = {
          messages: apiMessages,
          models: [activeModel],
          ...(modelParam.temperature ?? globalTemp) !== undefined && { temperature: modelParam.temperature ?? globalTemp },
          ...(modelParam.max_tokens ?? globalMax) !== undefined && { max_tokens: modelParam.max_tokens ?? globalMax },
          ...(modelParam.min_tokens ?? globalMin) !== undefined && { min_tokens: modelParam.min_tokens ?? globalMin },
        };

        if (providerWire === "anthropic") {
          enhancedRequest.anthropic_params = {
            thinking_enabled: modelParam.thinking_enabled,
            thinking_budget_tokens: modelParam.thinking_budget_tokens,
            top_k: modelParam.top_k,
            top_p: modelParam.top_p,
            stop_sequences: modelParam.stop_sequences,
          };
        } else if (providerWire === "openai") {
          enhancedRequest.openai_params = {
            top_p: modelParam.top_p,
            frequency_penalty: modelParam.frequency_penalty,
            presence_penalty: modelParam.presence_penalty,
            seed: modelParam.seed,
          };
        } else if (providerWire === "gemini") {
          enhancedRequest.gemini_params = {
            top_k: modelParam.top_k,
            top_p: modelParam.top_p,
            candidate_count: modelParam.candidate_count,
            safety_settings: modelParam.safety_settings,
          };
        } else if (providerWire === "ollama") {
          enhancedRequest.ollama_params = {
            mirostat: modelParam.mirostat,
            mirostat_eta: modelParam.mirostat_eta,
            mirostat_tau: modelParam.mirostat_tau,
            num_ctx: modelParam.num_ctx,
            repeat_penalty: modelParam.repeat_penalty,
          };
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
        const errorData = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errorData.detail || `HTTP ${res.status}`);
      }

      const result = await res.json();
      const assistantMessage: string = (useEnhancedAPI || hasProviderParams)
        ? result.answers?.[activeModel]?.answer || "No response"
        : result.choices?.[0]?.message?.content || "No response";

      setModelChats((prev) => ({
        ...prev,
        [activeModel]: {
          ...prev[activeModel],
          messages: [...prev[activeModel].messages, { role: "assistant", content: assistantMessage, timestamp: Date.now() }],
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
            messages: [...prev[activeModel!].messages, { role: "assistant", content: `Error: ${msg}`, timestamp: Date.now() }],
          },
        }));
      }
    } finally {
      interactiveAbortRef.current = null;
    }
  }, [activeModel, interactivePrompt, modelChats, modelParams, getProviderWire, globalTemp, globalMax, globalMin, useEnhancedAPI]);

  /* ---------- Load providers/models ---------- */
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
      } catch (err) {
        console.error(err);
      } finally {
        setLoadingProviders(false);
      }
    };
    load();
  }, []);

  /* ---------- Datasets ---------- */
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

  /* ---------- Chat helpers ---------- */
  const toggleModel = (m: string) =>
    setSelected((prev) => (prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]));
  const selectAll = () => setSelected(allModels);
  const clearAll = () => setSelected([]);

  /* ---------- Embedding helpers ---------- */
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
      const successful = results.filter((r) => r.status === "fulfilled").length;
      const failed = results.filter((r) => r.status === "rejected").length;
      let message = `Successfully uploaded with ${successful} embedding model(s).`;
      if (failed > 0) {
        const errors = results.filter((r): r is PromiseRejectedResult => r.status === "rejected").map((r) => r.reason.message).join("\n");
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
      if (!res.ok) throw new Error(await res.text());
      const result = await res.json();
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
  }, [searchQuery, selectedDataset, selectedSearchModel]);


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
          top_k: 5,
        }),
      });

      if (!res.ok) {
        let msg = await res.text();
        try { msg = JSON.parse(msg).detail || msg; } catch {}
        throw new Error(msg || `HTTP ${res.status}`);
      }

      const json: MultiSearchResponse = await res.json();
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
  }, [compareQuery, selectedEmbeddingModels]);



  const deleteDataset = useCallback(async (id: string) => {
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
  }, [loadDatasets, selectedDataset]);

  const canRun = prompt.trim().length > 0 && selected.length > 0 && !isRunning;

  const resetRun = useCallback(() => {
    setAnswers(Object.fromEntries(selected.map((m) => [m, { answer: "", error: undefined, latency_ms: 0 }])));
    setStartedAt(Date.now());
    setEndedAt(null);
  }, [selected]);

  const runPrompt = useCallback(async () => {
    if (!canRun) return;

    // cancel previous stream (if any)
    streamAbortRef.current?.abort();
    const controller = new AbortController();
    streamAbortRef.current = controller;

    // init result slots + timers
    resetRun();
    setIsRunning(true);

    // helper to process each NDJSON event
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
      // payload matches your backend /ask/ndjson contract
      const ndjsonPayload: Record<string, unknown> = {
        messages: [{ role: "user", content: prompt }],
        models: selected,
        ...(globalTemp !== undefined ? { temperature: globalTemp } : {}),
        ...(globalMax !== undefined ? { max_tokens: globalMax } : {}),
        ...(globalMin !== undefined ? { min_tokens: globalMin } : {}),
        model_params: modelParams,
      };
      if (globalTemp !== undefined) ndjsonPayload.temperature = globalTemp;
      if (globalMax !== undefined) ndjsonPayload.max_tokens = globalMax;
      if (globalMin !== undefined) ndjsonPayload.min_tokens = globalMin;

      const res = await fetch(`${API_BASE}/v2/chat/completions/enhanced/ndjson`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(ndjsonPayload),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) throw new Error(`Bad response: ${res.status} ${res.statusText}`);

      // stream & parse NDJSON
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


      }

      // flush tail
      buf += decoder.decode();
      const tail = buf.split("\n").map((l) => l.trim()).filter(Boolean);
      for (const t of tail) {
        try {
          processEvent(JSON.parse(t) as StreamEvent);
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
  // --- end add

  function pruneUndefined<T extends Record<string, unknown>>(obj: T): Partial<T> {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(obj)) if (v !== undefined) out[k] = v;
    return out as Partial<T>;
  }

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

      const enhancedRequest: EnhancedChatRequest & { model_params?: Record<string, Partial<PerModelParam>> } = {
        messages: [{ role: "user", content: prompt }],
        models: selected,
        ...(globalTemp !== undefined && { temperature: globalTemp }),
        ...(globalMax !== undefined && { max_tokens: globalMax }),
        ...(globalMin !== undefined && { min_tokens: globalMin }),
      };
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
        const error = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(error.detail || `HTTP ${res.status}`);
      }
      const result = await res.json();
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
    // Stream when we're on the Chat tab so users see incremental loading.
    if (activeTab === "chat") {
      await runPrompt();
    } else {
      // Keep embeddings behavior unchanged (doesn't use this anyway).
      await runEnhancedPrompt();
    }
  }, [activeTab, runPrompt, runEnhancedPrompt]);

  /* ---------- Keyboard shortcuts ---------- */
  useEffect(() => {
    const onKey = (evt: KeyboardEvent) => {
      if ((evt.metaKey || evt.ctrlKey) && evt.shiftKey && evt.key === "Enter") {
        evt.preventDefault();
        if (
          activeTab === "embedding" &&
          compareQuery.trim() &&
          selectedEmbeddingModels.length > 0 &&
          hasAnyDataset
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
  }, [activeTab, searchQuery, selectedEmbeddingModels, activeModel, interactivePrompt, canRun, executePrompt, performSearch, performMultiSearch, sendInteractiveMessage, closeModelChat]);

  const anyErrors = useMemo(() => Object.values(answers).some((a) => a?.error), [answers]);
  const elapsedMs = useMemo(() => {
    if (!startedAt) return 0;
    if (isRunning) return Date.now() - startedAt;
    if (endedAt) return Math.max(0, endedAt - startedAt);
    return 0;
  }, [startedAt, endedAt, isRunning]);

  /* ---------- Render ---------- */
  return (
    <div className="min-h-screen grid grid-rows-[auto_auto_1fr_auto] gap-6 p-6 sm:p-8 bg-white text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      {/* Interactive Chat Modal */}
      {activeModel && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="bg-white dark:bg-zinc-900 rounded-2xl shadow-2xl w-full max-w-4xl h-[80vh] max-h-[800px] flex flex-col border border-zinc-200 dark:border-zinc-700">
            <div className="flex items-center justify-between p-4 border-b border-zinc-200 dark:border-zinc-700">
              <h2 className="text-lg font-semibold">
                Chat with <span className="font-mono text-orange-600 dark:text-orange-400">{activeModel}</span>
              </h2>
              <button onClick={closeModelChat} className="p-2 rounded-lg hover:bg-zinc-100 dark:hover:bg-zinc-800 transition">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {modelChats[activeModel]?.messages.map((message, index) => (
                <div key={index} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                  <div className="max-w-[80%] space-y-1">
                    <div
                      className={`rounded-2xl px-4 py-2 ${
                        message.role === "user"
                          ? "bg-orange-600 text-white"
                          : "bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100"
                      }`}
                    >
                      <div className="prose prose-sm dark:prose-invert max-w-none">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {message.content}
                      </ReactMarkdown>
                    </div>
                      <div className="text-xs opacity-70 mt-1">{new Date(message.timestamp).toLocaleTimeString()}</div>
                    </div>
                  </div>
                </div>
              ))}
              {modelChats[activeModel]?.isStreaming && modelChats[activeModel]?.currentResponse && (
                <div className="flex justify-start">
                  <div className="max-w-[80%] rounded-2xl px-4 py-2 bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100">
                    <div className="prose prose-sm dark:prose-invert max-w-none">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {modelChats[activeModel].currentResponse}
                      </ReactMarkdown>
                    </div>
                    <div className="text-xs opacity-70 mt-1">Typing...</div>
                  </div>
                </div>
              )}
              <div ref={chatBottomRef} />
            </div>

            <div className="p-4 border-t border-zinc-200 dark:border-zinc-700">
              <div className="flex gap-3">
                <textarea
                  value={interactivePrompt}
                  onChange={(e) => setInteractivePrompt(e.target.value)}
                  placeholder="Type your message..."
                  rows={2}
                  className="flex-1 rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 outline-none focus:ring-2 focus:ring-orange-300/60 bg-white dark:bg-zinc-800 resize-none"
                  onKeyDown={(e) => {
                    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                      e.preventDefault();
                      void sendInteractiveMessage();
                    }
                  }}
                />
                <button
                  onClick={sendInteractiveMessage}
                  disabled={!interactivePrompt.trim() || modelChats[activeModel]?.isStreaming}
                  className="px-6 py-2 rounded-xl font-medium text-white bg-orange-600 hover:bg-orange-700 dark:bg-orange-500 dark:hover:bg-orange-600 transition disabled:opacity-50"
                >
                  Send
                </button>
              </div>
              <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-2">Press Cmd/Ctrl + Enter to send • Esc to close</div>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <header className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-orange-600 dark:text-orange-400">
            CompareLLM
          </h1>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Chat with multiple models using provider-specific parameters and perform semantic search.
          </p>
        </div>
        <div className="text-sm text-zinc-500 dark:text-zinc-400">
          {loadingProviders
            ? "Loading providers…"
            : `${providers.length} provider(s), ${allModels.length} chat model(s), ${allEmbeddingModels.length} embedding model(s)`}
        </div>
      </header>

      {/* Tabs */}
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

      {/* Chat Tab */}
      {activeTab === "chat" && (
        <main className="grid grid-cols-1 xl:grid-cols-[400px_1fr] gap-6 items-start">
          {/* Left rail */}
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

              <div className="max-h-[200px] overflow-auto rounded-xl border border-zinc-200 dark:border-zinc-800 p-2 grid grid-cols-1 gap-1">
                {allModels.length === 0 && (
                  <div className="text-sm text-zinc-500 dark:text-zinc-400">No models discovered yet.</div>
                )}
                {allModels.map((m) => {
                  const brand = getProviderType(m);
                  const providerColor = PROVIDER_TEXT_COLOR[brand];
                  return (
                    <label key={m} className="flex items-center gap-2 px-2 py-1 rounded-lg cursor-pointer hover:bg-orange-50 dark:hover:bg-orange-400/10">
                      <input
                        type="checkbox"
                        className="accent-orange-600 dark:accent-orange-500 cursor-pointer"
                        checked={selected.includes(m)}
                        onChange={() => toggleModel(m)}
                      />
                      <span className="text-sm font-mono flex-1">{m}</span>
                      <span className={`text-xs px-1.5 py-0.5 rounded ${providerColor} bg-current/10`}>{brand}</span>
                    </label>
                  );
                })}
              </div>

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
                    <div key={m} className="rounded-lg border border-orange-200 dark:border-orange-500/40 bg-orange-50/30 dark:bg-orange-400/5">
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
                onClick={() => void executePrompt()}
                disabled={!canRun}
                className="w-full rounded-xl py-2 px-4 font-medium text-white bg-orange-600 hover:bg-orange-700 dark:bg-orange-500 dark:hover:bg-orange-600 transition disabled:opacity-50"
              >
                {isRunning ? "Running…" : "Run"}
              </button>
            </div>
          </section>

          {/* Right rail: Results */}
          {/* Right rail: Search Results */}
          {/* Right rail: Chat results (streaming, stacked per model) */}
          <section className="space-y-4">
            {Object.entries(answers).map(([model, { answer, error, latency_ms }]) => {
              const brand = getProviderType(model);
              const badge = PROVIDER_BADGE_BG[brand];
              const hasErr = Boolean(error);

              return (
                <div
                  key={model}
                  className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 bg-white dark:bg-zinc-950 shadow-sm cursor-pointer hover:border-orange-300 dark:hover:border-orange-600 transition-colors group"
                  onClick={() => openModelChat(model)}
                  title="Click to continue chatting with this model"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <h2 className="text-sm font-semibold font-mono">{model}</h2>
                      <span className={`text-xs px-1.5 py-0.5 rounded ${badge}`}>{brand}</span>
                    </div>
                    <span className="text-xs text-zinc-500 dark:text-zinc-400">
                      {hasErr ? "⚠ Error" : latency_ms ? `${(latency_ms / 1000).toFixed(1)}s` : isRunning ? "running…" : ""}
                    </span>
                  </div>
                  <div className="prose prose-sm dark:prose-invert max-w-none">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {hasErr ? error || "" : (answer || (isRunning ? "…" : ""))}
                    </ReactMarkdown>
                  </div>

                  <div className="mt-2 text-xs text-zinc-500 dark:text-zinc-400 opacity-0 group-hover:opacity-100 transition-opacity">
                    Click to continue chatting with this model
                  </div>
                </div>
              );
            })}
            <div ref={bottomRef} />
          </section>

        </main>
      )}

      {/* Embeddings Tab */}
      {activeTab === "embedding" && (
        <main className="grid grid-cols-1 xl:grid-cols-[400px_1fr] gap-6 items-start">
          {/* Left rail */}
          <section className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 sm:p-5 bg-white dark:bg-zinc-950 shadow-sm space-y-4">
            {/* Embedding Models */}
            <div>
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Embedding Models</label>
                <div className="flex gap-2 text-xs">
                  <button onClick={selectAllEmbedding}
                    className="px-2 py-1 rounded-lg border border-orange-200 bg-orange-50 hover:bg-orange-100 dark:bg-orange-400/10 dark:hover:bg-orange-400/20">
                    Select all
                  </button>
                  <button onClick={clearAllEmbedding}
                    className="px-2 py-1 rounded-lg border border-orange-200 bg-orange-50 hover:bg-orange-100 dark:bg-orange-400/10 dark:hover:bg-orange-400/20">
                    Clear
                  </button>
                </div>
              </div>
              <div className="max-h-[160px] overflow-auto rounded-xl border border-zinc-200 dark:border-zinc-800 p-2 grid grid-cols-1 gap-1 mt-2">
                {allEmbeddingModels.length === 0 && (
                  <div className="text-sm text-zinc-500 dark:text-zinc-400">No embedding models discovered yet.</div>
                )}
                {allEmbeddingModels.map((m) => {
                  const brand = getProviderType(m);
                  return (
                    <label key={m} className="flex items-center gap-2 px-2 py-1 rounded-lg cursor-pointer hover:bg-orange-50 dark:hover:bg-orange-400/10">
                      <input type="checkbox" className="accent-orange-600 dark:accent-orange-500 cursor-pointer"
                            checked={selectedEmbeddingModels.includes(m)}
                            onChange={() => toggleEmbeddingModel(m)} />
                      <span className="text-sm font-mono flex-1">{m}</span>
                      <span className={`text-xs px-1.5 py-0.5 rounded ${PROVIDER_TEXT_COLOR[brand]} bg-current/10`}>{brand}</span>
                    </label>
                  );
                })}
              </div>
            </div>

            {/* Upload Dataset */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium">Upload Dataset</h3>
              <input type="text" placeholder="dataset id"
                    className="w-full rounded-md border border-zinc-200 dark:border-zinc-800 p-2 bg-white dark:bg-zinc-900 text-sm"
                    value={datasetId} onChange={(e) => setDatasetId(e.target.value)} />
              <input type="text" placeholder="text field (default: text)"
                    className="w-full rounded-md border border-zinc-200 dark:border-zinc-800 p-2 bg-white dark:bg-zinc-900 text-sm"
                    value={textField} onChange={(e) => setTextField(e.target.value)} />
              <textarea placeholder='[{"id":"1","text":"hello"},{"id":"2","text":"world"}]'
                        className="w-full rounded-md border border-zinc-200 dark:border-zinc-800 p-2 bg-white dark:bg-zinc-900 text-sm h-28"
                        value={jsonInput} onChange={(e) => setJsonInput(e.target.value)} />
              <button onClick={() => void uploadDataset()} disabled={uploadingDataset}
                      className="w-full rounded-xl py-2 px-4 font-medium text-white bg-orange-600 hover:bg-orange-700 dark:bg-orange-500 dark:hover:bg-orange-600 transition disabled:opacity-50">
                {uploadingDataset ? "Uploading…" : "Upload"}
              </button>
            </div>

            {/* Similarity Search (single model + dataset + query) */}
            <div className="space-y-3 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4">
              <h3 className="text-sm font-semibold">Similarity Search</h3>
              <div>
                <label className="block text-xs font-medium mb-1">Provider model</label>
                <select className="w-full rounded-md border border-zinc-200 dark:border-zinc-800 p-2 bg-white dark:bg-zinc-900 text-sm"
                        value={selectedSearchModel} onChange={(e) => setSelectedSearchModel(e.target.value)}>
                  {allEmbeddingModels.map((m) => (<option key={m} value={m}>{m}</option>))}
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
                      {d.dataset_id} ({d.document_count})
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium mb-1">Query</label>
                <input type="text" placeholder="Search query"
                      className="w-full rounded-md border border-zinc-200 dark:border-zinc-800 p-2 bg-white dark:bg-zinc-900 text-sm"
                      value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} />
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

            {/* Compare Across Selected Models */}
            <div className="space-y-3 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4">
              <h3 className="text-sm font-semibold">Compare Across Selected Models</h3>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">
                Uses the models you checked above. (Dataset(s) required for this self-dataset compare endpoint.)
              </p>
              <input type="text" placeholder="Comparison query"
                    className="w-full rounded-md border border-zinc-200 dark:border-zinc-800 p-2 bg-white dark:bg-zinc-900 text-sm"
                    value={compareQuery} onChange={(e) => setCompareQuery(e.target.value)} />
              <button
                onClick={() => void performMultiSearch()}
                disabled={
                  isComparing ||
                  uploadingDataset ||
                  selectedEmbeddingModels.length === 0 ||
                  !compareQuery.trim() ||
                  !hasAnyDataset
                }
                className="w-full rounded-lg px-4 py-2 font-medium text-white bg-orange-600 hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                aria-busy={isComparing}
                title={
                  !hasAnyDataset
                    ? "Upload a dataset first"
                    : selectedEmbeddingModels.length === 0
                    ? "Select embedding models to compare"
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
                        <button className={`text-left font-mono text-xs ${selectedDataset === d.dataset_id ? "text-orange-600" : ""}`}
                                onClick={() => setSelectedDataset(d.dataset_id)} title={`Docs: ${d.document_count}`}>
                          {d.dataset_id}
                        </button>
                        <button onClick={() => void deleteDataset(d.dataset_id)}
                                className="text-xs px-2 py-1 rounded-md border border-red-200 text-red-700 bg-red-50 hover:bg-red-100 dark:border-red-800 dark:text-red-400 dark:bg-red-900/20 dark:hover:bg-red-900/40 transition">
                          delete
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </section>




          {/* Right: search */}
          <section className="space-y-4">
          {(isSearchingSingle || isComparing) && (
            <div className="rounded-xl border border-orange-200 dark:border-orange-900/40 p-3 bg-orange-50/40 dark:bg-orange-900/10">
              <div className="flex items-center gap-2 text-sm text-orange-700 dark:text-orange-300 mb-2">
                <Spinner className="h-4 w-4" />
                {isSearchingSingle ? "Running similarity search…" : "Running side-by-side comparison…"}
              </div>
              <LoadingBar />
            </div>
          )}            
            {/* Single-model search results */}
            {searchContext && (
              <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 bg-white dark:bg-zinc-950 shadow-sm">
                <div className="flex items-center justify-between mb-3">
                  <div className="text-sm">
                    <span className="font-medium">Results</span>{" "}
                    <span className="text-zinc-500">for</span>{" "}
                    <span className="font-mono">{searchContext.query}</span>{" "}
                    <span className="text-zinc-500">with</span>{" "}
                    <span className="font-mono">{searchContext.model}</span>
                  </div>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="text-left">
                      <tr className="text-zinc-500">
                        <th className="py-2 pr-2">#</th>
                        <th className="py-2 pr-2">similarity</th>
                        <th className="py-2 pr-2">document</th>
                      </tr>
                    </thead>
                    <tbody>
                      {searchResults.map((row, i) => (
                        <tr key={i} className="border-t border-zinc-200 dark:border-zinc-800">
                          <td className="py-2 pr-2 text-zinc-500">{i + 1}</td>
                          <td className="py-2 pr-2 font-mono">
                            {(row.similarity_score ?? 0).toFixed(3)}
                          </td>
                          <td className="py-2 pr-2">
                            <pre className="whitespace-pre-wrap text-[13px] leading-snug max-w-[80ch]">
                              {JSON.stringify(redactResult(row), null, 2)}
                            </pre>
                          </td>
                        </tr>
                      ))}
                      {searchResults.length === 0 && (
                        <tr>
                          <td colSpan={3} className="py-3 text-zinc-500">
                            No results yet.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Multi-compare results */}
            {multiSearchResults && (
              <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 bg-white dark:bg-zinc-950 shadow-sm space-y-4">
                <div className="text-sm">
                  <span className="text-zinc-500">Compare query</span>{" "}
                  <span className="font-mono">{multiSearchResults.query}</span>{" "}
                  {typeof multiSearchResults.duration_ms === "number" && (
                    <span className="text-zinc-500"> • {(multiSearchResults.duration_ms / 1000).toFixed(1)}s</span>
                  )}
                </div>

                {(() => {
                  const entries = Object.entries(multiSearchResults.results);
                  const modelKeys = entries.map(([k]) => k);
                  const maxRows = Math.max(
                    0,
                    ...entries.map(([, b]) => (b?.items?.length ?? 0))
                  );

                  return (
                    <div className="overflow-x-auto">
                      {/* Dynamic column grid: header row + item rows aligned by rank */}
                      <div
                        className="grid gap-3"
                        style={{ gridTemplateColumns: `repeat(${modelKeys.length}, minmax(280px, 1fr))` }}
                      >
                        {/* Header cells */}
                        {modelKeys.map((modelKey) => {
                          const brand = getProviderType(modelKey);
                          return (
                            <div
                              key={`hdr-${modelKey}`}
                              className="rounded-lg border border-zinc-200 dark:border-zinc-800 px-3 py-2 bg-zinc-50 dark:bg-zinc-900/40 flex items-center justify-between"
                            >
                              <div className="font-mono text-xs">{modelKey}</div>
                              <span className={`text-[11px] px-2 py-0.5 rounded ${PROVIDER_BADGE_BG[brand]}`}>
                                {brand}
                              </span>
                            </div>
                          );
                        })}

                        {/* Rows by rank: i = 0..maxRows-1 */}
                        {Array.from({ length: maxRows }).map((_, i) =>
                          modelKeys.map((modelKey) => {
                            const bucket = multiSearchResults.results[modelKey];
                            const brand = getProviderType(modelKey);
                            const providerBadge = PROVIDER_BADGE_BG[brand];

                            // If error, show error card once per column
                            if (bucket?.error) {
                              return (
                                <div
                                  key={`${modelKey}-err`}
                                  className="rounded-lg border border-red-200 dark:border-red-900/40 p-3 text-xs text-red-600 dark:text-red-400 bg-red-50/40 dark:bg-red-900/10"
                                >
                                  Error: {bucket.error}
                                </div>
                              );
                            }

                            const item = bucket?.items?.[i];
                            if (!item) {
                              // Empty cell placeholder for this rank
                              return (
                                <div
                                  key={`${modelKey}-${i}-empty`}
                                  className="rounded-lg border border-dashed border-zinc-200 dark:border-zinc-800 p-3 text-xs text-zinc-500 dark:text-zinc-400"
                                >
                                  No result for rank {i + 1}.
                                </div>
                              );
                            }

                            return (
                              <div
                                key={`${modelKey}-${i}`}
                                className="rounded-lg border border-zinc-200 dark:border-zinc-800 p-3"
                              >
                                {/* Top row: rank badge + similarity pill pinned right */}
                                <div className="grid grid-cols-[auto_1fr_auto] items-start gap-2">
                                  <div className="shrink-0 w-6 h-6 rounded-md bg-zinc-100 dark:bg-zinc-800 text-[11px] flex items-center justify-center text-zinc-700 dark:text-zinc-300">
                                    {i + 1}
                                  </div>
                                  <div className="min-w-0" />
                                  <span
                                    className={`shrink-0 inline-block px-2 py-1 rounded-full text-xs font-medium ${providerBadge}`}
                                    title="cosine similarity"
                                  >
                                    {((item.similarity_score ?? 0) * 100).toFixed(1)}%
                                  </span>
                                </div>

                                {/* Content */}
                                <div className="mt-2 grid grid-cols-[1fr]">
                                  {/* Prefer readable snippet */}
                                  <div className="text-[13px] leading-snug">
                                    {primarySnippet(item, 220)}
                                  </div>
                                  {/* Small meta line */}
                                  <div className="mt-1 text-[11px] text-zinc-500 dark:text-zinc-400 font-mono">
                                    {Object.entries(item)
                                      .filter(([k]) => !["similarity_score", "embedding"].includes(k) && !k.startsWith("_"))
                                      .slice(0, 3)
                                      .map(([k, v]) => `${k}: ${typeof v === "string" ? v.slice(0, 40) : String(v)}`)
                                      .join(" • ")}
                                  </div>

                                  {/* Expandable JSON if you want (optional): 
                                  <details className="mt-2">
                                    <summary className="text-[11px] cursor-pointer text-zinc-500 dark:text-zinc-400">raw</summary>
                                    <pre className="mt-1 whitespace-pre-wrap break-words text-[12px] leading-snug max-w-[80ch]">
                                      {JSON.stringify(redactResult(item), null, 2)}
                                    </pre>
                                  </details>
                                  */}
                                </div>
                              </div>
                            );
                          })
                        )}
                      </div>
                    </div>
                  );
                })()}
              </div>
            )}


          </section>
        </main>
      )}
    </div>
  );
}
