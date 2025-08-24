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

type PerModelParam = { 
  temperature?: number; 
  max_tokens?: number; 
  min_tokens?: number;
  // Enhanced Anthropic parameters
  thinking_enabled?: boolean;
  thinking_budget_tokens?: number;
  top_k?: number;
  top_p?: number;
  stop_sequences?: string[];
  //service_tier?: "auto" | "standard_only";
  //tool_choice_type?: "auto" | "any" | "tool" | "none";
  //user_id?: string;
  // OpenAI parameters
  frequency_penalty?: number;
  presence_penalty?: number;
  seed?: number;
  // Gemini parameters
  candidate_count?: number;
  safety_settings?: unknown[];
  // Ollama parameters
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

// Enhanced API request types
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
    //service_tier?: string;
    //tool_choice_type?: string;
    //user_id?: string;
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

// ---------------- Config ----------------
const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "/backend");

// ---------------- Component ----------------
function ProviderParameterEditor({ 
  model, 
  providerType, 
  params, 
  onUpdate 
}: {
  model: string;
  providerType: string;
  params: PerModelParam;
  onUpdate: (params: PerModelParam) => void;
}) {
  const updateParam = (key: keyof PerModelParam, value: unknown) => {
    onUpdate({ ...params, [key]: value });
  };

  const formatStopSequences = (sequences?: string[]): string => {
    return sequences ? sequences.join(', ') : '';
  };

  const parseStopSequences = (value: string): string[] => {
    return value.split(',').map(s => s.trim()).filter(s => s.length > 0);
  };

  if (providerType === "anthropic") {
    return (
      <div className="space-y-3">
        <h4 className="text-sm font-semibold text-orange-600 dark:text-orange-400">Anthropic Parameters</h4>
        
        {/* Extended Thinking */}
        <div className="bg-orange-50/50 dark:bg-orange-900/20 p-3 rounded-lg border border-orange-200 dark:border-orange-800">
          <label className="flex items-center gap-2 mb-2">
            <input
              type="checkbox"
              checked={params.thinking_enabled ?? false}
              onChange={(e) => updateParam('thinking_enabled', e.target.checked)}
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
                onChange={(e) => updateParam('thinking_budget_tokens', Number(e.target.value))}
                className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
                placeholder="2048"
              />
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
                Minimum 1024 tokens for thinking process
              </p>
            </div>
          )}
        </div>

        {/* Sampling Parameters */}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium mb-1">Top-K</label>
            <input
              type="number"
              min={1}
              value={params.top_k ?? ""}
              onChange={(e) => updateParam('top_k', e.target.value ? Number(e.target.value) : undefined)}
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
              onChange={(e) => updateParam('top_p', e.target.value ? Number(e.target.value) : undefined)}
              className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
              placeholder="0.9"
            />
          </div>
        </div>

        {/* Stop Sequences */}
        <div>
          <label className="block text-xs font-medium mb-1">Stop Sequences</label>
          <input
            type="text"
            value={formatStopSequences(params.stop_sequences)}
            onChange={(e) => updateParam('stop_sequences', parseStopSequences(e.target.value))}
            className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
            placeholder="Human:, Assistant:, Stop"
          />
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
            Comma-separated sequences that will stop generation
          </p>
        </div>
      </div>
    );
  }

  if (providerType === "openai") {
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
              onChange={(e) => updateParam('top_p', e.target.value ? Number(e.target.value) : undefined)}
              className="w-full rounded-md border border-blue-200 dark:border-blue-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
              placeholder="0.9"
            />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1">Seed</label>
            <input
              type="number"
              value={params.seed ?? ""}
              onChange={(e) => updateParam('seed', e.target.value ? Number(e.target.value) : undefined)}
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
              onChange={(e) => updateParam('frequency_penalty', e.target.value ? Number(e.target.value) : undefined)}
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
              onChange={(e) => updateParam('presence_penalty', e.target.value ? Number(e.target.value) : undefined)}
              className="w-full rounded-md border border-blue-200 dark:border-blue-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
              placeholder="0.0"
            />
          </div>
        </div>
      </div>
    );
  }

  if (providerType === "gemini") {
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
              onChange={(e) => updateParam('top_k', e.target.value ? Number(e.target.value) : undefined)}
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
              onChange={(e) => updateParam('top_p', e.target.value ? Number(e.target.value) : undefined)}
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
              onChange={(e) => updateParam('candidate_count', e.target.value ? Number(e.target.value) : undefined)}
              className="w-full rounded-md border border-green-200 dark:border-green-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
              placeholder="1"
            />
          </div>
        </div>
      </div>
    );
  }

  if (providerType === "ollama") {
    return (
      <div className="space-y-3">
        <h4 className="text-sm font-semibold text-purple-600 dark:text-purple-400">Ollama Parameters</h4>
        
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium mb-1">Mirostat</label>
            <select
              value={params.mirostat ?? ""}
              onChange={(e) => updateParam('mirostat', e.target.value ? Number(e.target.value) : undefined)}
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
              onChange={(e) => updateParam('num_ctx', e.target.value ? Number(e.target.value) : undefined)}
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
              onChange={(e) => updateParam('mirostat_eta', e.target.value ? Number(e.target.value) : undefined)}
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
              onChange={(e) => updateParam('mirostat_tau', e.target.value ? Number(e.target.value) : undefined)}
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
            onChange={(e) => updateParam('repeat_penalty', e.target.value ? Number(e.target.value) : undefined)}
            className="w-full rounded-md border border-purple-200 dark:border-purple-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
            placeholder="1.1"
          />
        </div>
      </div>
    );
  }

  return null;
}

// ---------------- Main Page Component ----------------
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

  // Interactive chat functionality
  const [activeModel, setActiveModel] = useState<string | null>(null);
  const [modelChats, setModelChats] = useState<Record<string, ModelChat>>({});
  const [interactivePrompt, setInteractivePrompt] = useState<string>("");

  // Embedding functionality
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

  // Search context for results display
  const [searchContext, setSearchContext] = useState<{
    model: string;
    dataset: string;
    query: string;
    startedAt: number;
  } | null>(null);

  const streamAbortRef = useRef<AbortController | null>(null);
  const interactiveAbortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const chatBottomRef = useRef<HTMLDivElement | null>(null);

  // Guard against out-of-order search responses
  const requestIdRef = useRef(0);

  // Enhanced parameters
  const [modelParams, setModelParams] = useState<ModelParamsMap>({});
  const [globalTemp, setGlobalTemp] = useState<number | undefined>(undefined);
  const [globalMax, setGlobalMax] = useState<number | undefined>(undefined);
  const [globalMin, setGlobalMin] = useState<number | undefined>(undefined);
  // Always use Enhanced API
  const useEnhancedAPI = true;

  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set());

  // Helper function to get provider type for a model
  const modelToProvider = useMemo(() => {
    const map: Record<string, string> = {};
    providers.forEach((p) => {
      (p.models || []).forEach((m) => { map[m] = p.type; });
      (p.embedding_models || []).forEach((m) => { map[m] = p.type; });
    });
    return map;
  }, [providers]);

  const getProviderType = useCallback(
    (modelName: string) => modelToProvider[modelName] ?? "unknown",
    [modelToProvider]
  );


  const updateParam = useCallback(
    (model: string, params: PerModelParam) => {
      setModelParams((prev) => ({ ...prev, [model]: params }));
    },
    []
  );

  const toggleModelExpansion = useCallback((model: string) => {
    setExpandedModels(prev => {
      const newSet = new Set(prev);
      if (newSet.has(model)) {
        newSet.delete(model);
      } else {
        newSet.add(model);
      }
      return newSet;
    });
  }, []);

  // -------- Interactive Chat Functions --------
  const openModelChat = useCallback((model: string) => {
    setActiveModel(model);
    if (!modelChats[model]) {
      const initialMessages: ChatMessage[] = [];
      
      if (prompt.trim()) {
        initialMessages.push({
          role: "user",
          content: prompt.trim(),
          timestamp: startedAt || Date.now()
        });
      }
      
      const modelAnswer = answers[model];
      if (modelAnswer?.answer && !modelAnswer.error) {
        initialMessages.push({
          role: "assistant",
          content: modelAnswer.answer,
          timestamp: (startedAt || Date.now()) + (modelAnswer.latency_ms || 1000)
        });
      }
      
      setModelChats(prev => ({
        ...prev,
        [model]: { 
          messages: initialMessages, 
          isStreaming: false, 
          currentResponse: "" 
        }
      }));
    }
  }, [modelChats, prompt, answers, startedAt]);

  const closeModelChat = useCallback(() => {
    setActiveModel(null);
    interactiveAbortRef.current?.abort();
  }, []);

  const sendInteractiveMessage = useCallback(async () => {
    if (!activeModel || !interactivePrompt.trim()) return;

    const message = interactivePrompt.trim();
    setInteractivePrompt("");

    setModelChats(prev => ({
      ...prev,
      [activeModel]: {
        ...prev[activeModel],
        messages: [
          ...prev[activeModel].messages,
          { role: "user", content: message, timestamp: Date.now() }
        ],
        isStreaming: true,
        currentResponse: ""
      }
    }));

    interactiveAbortRef.current?.abort();
    const controller = new AbortController();
    interactiveAbortRef.current = controller;

    try {
      const currentChat = modelChats[activeModel];
      const conversationHistory = [
        ...currentChat.messages,
        { role: "user" as const, content: message, timestamp: Date.now() }
      ];

      const apiMessages = conversationHistory.map(msg => ({
        role: msg.role,
        content: msg.content
      }));

      // Use enhanced API if enabled and has provider-specific params
      const modelParam = modelParams[activeModel] || {};
      const providerType = getProviderType(activeModel);
      const hasProviderParams = Object.keys(modelParam).some(key => 
        !['temperature', 'max_tokens', 'min_tokens'].includes(key)
      );

      let body: string | undefined;
      let endpoint: string;

      if (useEnhancedAPI || hasProviderParams) {
        // Use enhanced API
        endpoint = `${API_BASE}/v2/chat/completions/enhanced`;
        
        const enhancedRequest: EnhancedChatRequest = {
          messages: apiMessages,
          models: [activeModel],
          ...(modelParam.temperature ?? globalTemp) !== undefined && { temperature: (modelParam.temperature ?? globalTemp) },
          ...(modelParam.max_tokens ?? globalMax) !== undefined && { max_tokens: (modelParam.max_tokens ?? globalMax) },
          ...(modelParam.min_tokens ?? globalMin) !== undefined && { min_tokens: (modelParam.min_tokens ?? globalMin) },
        } as EnhancedChatRequest;

        // Add provider-specific parameters
        if (providerType === "anthropic") {
          enhancedRequest.anthropic_params = {
            thinking_enabled: modelParam.thinking_enabled,
            thinking_budget_tokens: modelParam.thinking_budget_tokens,
            top_k: modelParam.top_k,
            top_p: modelParam.top_p,
            stop_sequences: modelParam.stop_sequences,
            // service_tier: modelParam.service_tier,
            // tool_choice_type: modelParam.tool_choice_type,
          };
        } else if (providerType === "openai") {
          enhancedRequest.openai_params = {
            top_p: modelParam.top_p,
            frequency_penalty: modelParam.frequency_penalty,
            presence_penalty: modelParam.presence_penalty,
            seed: modelParam.seed,
          };
        } else if (providerType === "gemini") {
          enhancedRequest.gemini_params = {
            top_k: modelParam.top_k,
            top_p: modelParam.top_p,
            candidate_count: modelParam.candidate_count,
            safety_settings: modelParam.safety_settings,
          };
        } else if (providerType === "ollama") {
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
        // Use standard OpenAI-compatible API
        endpoint = `${API_BASE}/v1/chat/completions`;
        const stdPayload: Record<string, unknown> = {
          model: activeModel,
          messages: apiMessages,
        };
        const temp = modelParam.temperature ?? globalTemp;
        const maxTok = modelParam.max_tokens ?? globalMax;
        if (temp !== undefined) stdPayload.temperature = temp;
        if (maxTok !== undefined) stdPayload.max_tokens = maxTok;
        stdPayload.stream = false;
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
      let assistantMessage: string;

      if (useEnhancedAPI || hasProviderParams) {
        // Enhanced API response format
        assistantMessage = result.answers?.[activeModel]?.answer || "No response";
      } else {
        // Standard OpenAI response format
        assistantMessage = result.choices[0]?.message?.content || "No response";
      }

      setModelChats(prev => ({
        ...prev,
        [activeModel]: {
          ...prev[activeModel],
          messages: [
            ...prev[activeModel].messages,
            { role: "assistant", content: assistantMessage, timestamp: Date.now() }
          ],
          isStreaming: false,
          currentResponse: ""
        }
      }));

    } catch (err: unknown) {
      const isAbort = err instanceof DOMException && err.name === "AbortError";
      if (!isAbort) {
        console.error(err);
        setModelChats(prev => ({
          ...prev,
          [activeModel]: {
            ...prev[activeModel],
            isStreaming: false,
            currentResponse: "",
            messages: [
              ...prev[activeModel].messages,
              { 
                role: "assistant", 
                content: `Error: ${err instanceof Error ? err.message : "Unknown error"}`, 
                timestamp: Date.now() 
              }
            ]
          }
        }));
      }
    } finally {
      interactiveAbortRef.current = null;
    }
  }, [activeModel, interactivePrompt, modelParams, globalTemp, globalMax, globalMin, modelChats, getProviderType]);

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

  // -------- Chat helpers --------
  const toggleModel = (m: string) =>
    setSelected((prev) => (prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]));

  const selectAll = () => setSelected(allModels);
  const clearAll = () => setSelected([]);

  // -------- Embedding helpers --------
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

    const snapshot = {
      query: searchQuery,
      dataset: selectedDataset,
      model: selectedSearchModel,
      startedAt: Date.now(),
    };

    setIsSearching(true);
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

      if (!res.ok) {
        const error = await res.text();
        throw new Error(error);
      }

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

  // -------- Enhanced API runner --------
  // -------- Enhanced API runner --------
  function pruneUndefined<T extends Record<string, unknown>>(obj: T): Partial<T> {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(obj)) {
      if (v !== undefined) out[k] = v;
    }
    return out as Partial<T>;
  }  
  // 2) Replace your current runEnhancedPrompt with this version (or edit in place)
  const runEnhancedPrompt = useCallback(async () => {
    if (!canRun) return;

    setIsRunning(true);
    resetRun();

    try {
      // Build per-model overrides: include only defined keys
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
        ...(globalTemp !== undefined && { temperature: globalTemp }),
        ...(globalMax !== undefined && { max_tokens: globalMax }),
        ...(globalMin !== undefined && { min_tokens: globalMin }),
      };

      // 👉 Only attach model_params if there’s at least one override
      if (Object.keys(perModel).length > 0) {
        // @ts-expect-error backend supports model_params
        enhancedRequest.model_params = perModel;
      }

      // Group models by provider type
      const anthropicModels = selected.filter(m => getProviderType(m) === "anthropic");
      const openaiModels    = selected.filter(m => getProviderType(m) === "openai");
      const geminiModels    = selected.filter(m => getProviderType(m) === "gemini");
      const ollamaModels    = selected.filter(m => getProviderType(m) === "ollama");

      // For each provider, collect only defined keys across selected models.
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
        const modelResult = result.answers[model];
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
  }, [canRun, prompt, selected, modelParams, globalTemp, globalMax, globalMin, getProviderType, resetRun]);




  // -------- Streaming runner (JSONL) - existing --------
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
      const ndjsonPayload: Record<string, unknown> = {
        prompt,
        models: selected,
        model_params: modelParams,
      };
      if (globalTemp !== undefined) ndjsonPayload.temperature = globalTemp;
      if (globalMax !== undefined) ndjsonPayload.max_tokens = globalMax;
      if (globalMin !== undefined) ndjsonPayload.min_tokens = globalMin;

      const res = await fetch(`${API_BASE}/ask/ndjson`, {
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

  // Decide which runner to use
  const executePrompt = useCallback(async () => {
    await runEnhancedPrompt();
  }, [runEnhancedPrompt]);

  // Keyboard shortcuts
  useEffect(() => {
    const onKey = (evt: KeyboardEvent) => {
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
  }, [canRun, executePrompt, activeTab, searchQuery, performSearch, activeModel, interactivePrompt, sendInteractiveMessage, closeModelChat]);

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
      {/* Interactive Chat Modal */}
      {activeModel && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="bg-white dark:bg-zinc-900 rounded-2xl shadow-2xl w-full max-w-4xl h-[80vh] max-h-[800px] flex flex-col border border-zinc-200 dark:border-zinc-700">
            {/* Chat Header */}
            <div className="flex items-center justify-between p-4 border-b border-zinc-200 dark:border-zinc-700">
              <h2 className="text-lg font-semibold">
                Chat with <span className="font-mono text-orange-600 dark:text-orange-400">{activeModel}</span>
              </h2>
              <button
                onClick={closeModelChat}
                className="p-2 rounded-lg hover:bg-zinc-100 dark:hover:bg-zinc-800 transition"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Chat Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {modelChats[activeModel]?.messages.map((message, index) => {
                const isFromOriginalRun = index <= 1 && prompt.trim() && message.content === prompt.trim();
                const isOriginalResponse = index === 1 && modelChats[activeModel]?.messages[0]?.content === prompt.trim();
                
                return (
                  <div
                    key={index}
                    className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div className="max-w-[80%] space-y-1">
                      {(isFromOriginalRun || isOriginalResponse) && (
                        <div className="text-xs text-zinc-500 dark:text-zinc-400 px-2">
                          From comparison run
                        </div>
                      )}
                      <div
                        className={`rounded-2xl px-4 py-2 ${
                          message.role === "user"
                            ? "bg-orange-600 text-white"
                            : "bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100"
                        }`}
                      >
                        <pre className="whitespace-pre-wrap text-sm">{message.content}</pre>
                        <div className="text-xs opacity-70 mt-1">
                          {new Date(message.timestamp).toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}

              {modelChats[activeModel]?.isStreaming && modelChats[activeModel]?.currentResponse && (
                <div className="flex justify-start">
                  <div className="max-w-[80%] rounded-2xl px-4 py-2 bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100">
                    <pre className="whitespace-pre-wrap text-sm">{modelChats[activeModel].currentResponse}</pre>
                    <div className="text-xs opacity-70 mt-1">Typing...</div>
                  </div>
                </div>
              )}

              <div ref={chatBottomRef} />
            </div>

            {/* Chat Input */}
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
              <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-2">
                Press Cmd/Ctrl + Enter to send • Esc to close
              </div>
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
        <main className="grid grid-cols-1 xl:grid-cols-[400px_1fr] gap-6 items-start">
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

              <div className="max-h-[200px] overflow-auto rounded-xl border border-zinc-200 dark:border-zinc-800 p-2 grid grid-cols-1 gap-1">
                {allModels.length === 0 && (
                  <div className="text-sm text-zinc-500 dark:text-zinc-400">No models discovered yet.</div>
                )}
                {allModels.map((m) => {
                  const providerType = getProviderType(m);
                  const providerColor = {
                    anthropic: "text-orange-600 dark:text-orange-400",
                    openai: "text-blue-600 dark:text-blue-400", 
                    gemini: "text-green-600 dark:text-green-400",
                    ollama: "text-purple-600 dark:text-purple-400",
                    unknown: "text-zinc-600 dark:text-zinc-400"
                  }[providerType] || "text-zinc-600 dark:text-zinc-400";
                  
                  return (
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
                      <span className="text-sm font-mono flex-1">{m}</span>
                      <span className={`text-xs px-1.5 py-0.5 rounded ${providerColor} bg-current/10`}>
                        {providerType}
                      </span>
                    </label>
                  );
                })}
              </div>

              {/* Global defaults */}
              <div className="space-y-3 text-sm">
                <div>
                  <label className="block mb-1 font-medium">Global temp</label>
                  <input
                    type="number" step={0.1} min={0} max={2}
                    value={globalTemp ?? ""}
                    placeholder="Model default"
                    onChange={(e) => setGlobalTemp(e.target.value ? Number(e.target.value) : undefined)}
                    className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900"
                  />
                </div>
                <div>
                  <label className="block mb-1 font-medium">Global max_tokens</label>
                  <input
                    type="number" min={1}
                    value={globalMax ?? ""}
                    placeholder="Model default"
                    onChange={(e) => setGlobalMax(e.target.value ? Number(e.target.value) : undefined)}
                    className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900"
                  />
                </div>
                <div>
                  <label className="block mb-1 font-medium">Global min_tokens</label>
                  <input
                    type="number" min={1}
                    value={globalMin ?? ""}
                    placeholder="Model default"
                    onChange={(e) => setGlobalMin(e.target.value ? Number(e.target.value) : undefined)}
                    className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900"
                  />
                </div>
              </div>

              {/* Per-model parameters with enhanced controls */}
              <div className="space-y-2">
                <h3 className="text-sm font-medium">Model-Specific Parameters</h3>
                {allModels.map((m) => {
                  const providerType = getProviderType(m);
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
                          <span className={`text-xs px-1.5 py-0.5 rounded ${
                            providerType === "anthropic" ? "bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400" :
                            providerType === "openai" ? "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400" :
                            providerType === "gemini" ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400" :
                            providerType === "ollama" ? "bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400" :
                            "bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-400"
                          }`}>
                            {providerType}
                          </span>
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
                          {/* Basic parameters */}
                          <div className="grid grid-cols-3 gap-3 mb-4 mt-3">
                            <div>
                              <label className="block mb-1 text-xs font-medium">Temperature</label>
                              <input
                                type="number" step={0.1} min={0} max={2}
                                value={modelParams[m]?.temperature ?? ""}
                                placeholder={`↳ ${globalTemp ?? "backend default"}`}
                                onChange={(e) => updateParam(m, { 
                                  ...modelParams[m], 
                                  temperature: e.target.value ? Number(e.target.value) : undefined 
                                })}
                                className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
                              />
                            </div>
                            <div>
                              <label className="block mb-1 text-xs font-medium">Max Tokens</label>
                            <input
                                type="number" min={1}
                                value={modelParams[m]?.max_tokens ?? ""}
                                placeholder={`↳ ${globalMax ?? "backend default"}`}
                                onChange={(e) => updateParam(m, { 
                                  ...modelParams[m], 
                                  max_tokens: e.target.value ? Number(e.target.value) : undefined 
                                })}
                                className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
                              />
                            </div>
                            <div>
                              <label className="block mb-1 text-xs font-medium">Min Tokens</label>
                              <input
                                type="number" min={1}
                                value={modelParams[m]?.min_tokens ?? ""}
                                placeholder={globalMin ? `↳ ${globalMin}` : "optional"}
                                onChange={(e) => updateParam(m, { 
                                  ...modelParams[m], 
                                  min_tokens: e.target.value ? Number(e.target.value) : undefined 
                                })}
                                className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
                              />
                            </div>
                          </div>

                          {/* Provider-specific parameters */}
                          <ProviderParameterEditor
                            model={m}
                            providerType={providerType}
                            params={modelParams[m] || {}}
                            onUpdate={(params) => updateParam(m, params)}
                          />
                          
                          {/* Clear parameters button */}
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
          <section className="space-y-4">
            {Object.entries(answers).map(([model, { answer, error, latency_ms }]) => {
              const providerType = getProviderType(model);
              const modelParams_forModel = modelParams[model] || {};
              const hasEnhancedParams = Object.keys(modelParams_forModel).some(key => 
                !['temperature', 'max_tokens', 'min_tokens'].includes(key)
              );
              
              return (
                <div
                  key={model}
                  className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 bg-white dark:bg-zinc-950 shadow-sm cursor-pointer hover:border-orange-300 dark:hover:border-orange-600 transition-colors group"
                  onClick={() => openModelChat(model)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <h2 className="text-sm font-semibold font-mono">{model}</h2>
                      <span className={`text-xs px-1.5 py-0.5 rounded ${
                        providerType === "anthropic" ? "bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400" :
                        providerType === "openai" ? "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400" :
                        providerType === "gemini" ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400" :
                        providerType === "ollama" ? "bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400" :
                        "bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-400"
                      }`}>
                        {providerType}
                      </span>
                      {hasEnhancedParams && (
                        <span className="text-xs px-1.5 py-0.5 rounded bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400">
                          enhanced
                        </span>
                      )}
                      {modelParams_forModel.thinking_enabled && (
                        <span className="text-xs px-1.5 py-0.5 rounded bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400">
                          🧠 thinking
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-zinc-500 dark:text-zinc-400">
                        {error ? "⚠ Error" : latency_ms ? `${(latency_ms / 1000).toFixed(1)}s` : isRunning ? "running…" : ""}
                      </span>
                      <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                        <svg className="w-4 h-4 text-orange-600 dark:text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-3.582 8-8 8a8.001 8.001 0 01-7.93-6.94c-.04-.24-.04-.46-.04-.68l.01-.08c.05-4.345 3.578-7.88 7.93-7.93.24 0 .46.04.68.04.08 0 .16-.01.24-.01" />
                        </svg>
                      </div>
                    </div>
                  </div>
                  <pre className="whitespace-pre-wrap text-sm">{error ? error : answer}</pre>
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
            <span>{selected.length} selected • {useEnhancedAPI ? "Enhanced API" : "Standard API"}</span>
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
