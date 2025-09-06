// lib/types.ts
// lib/types.ts
export type ProviderWire = "anthropic" | "openai" | "gemini" | "ollama" | "cohere" | "unknown" | "deepseek";

export type ProviderBrand =
  | "openai"
  | "anthropic"
  | "ollama"
  | "deepseek"
  | "cerebras"
  | "voyage"
  | "google"
  | "cohere" 
  | "unknown";

export type KnownProviderBrand = Exclude<ProviderBrand, "unknown">;

export type ProviderInfo = {
  name: string;
  type: string;
  base_url: string;
  models: string[];
  embedding_models: string[];
  auth_required: boolean;
  wire?: ProviderWire;
};

export type PerModelParam = {
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
  // Google
  candidate_count?: number;
  safety_settings?: unknown[];
  // Ollama
  mirostat?: number;
  mirostat_eta?: number;
  mirostat_tau?: number;
  num_ctx?: number;
  repeat_penalty?: number;
  // Cohere
  k?: number;
  p?: number;
  logprobs?: boolean;
  raw_prompting?: boolean;
  //DeepSeek
  top_logprobs?: number; 
};

export type ModelParamsMap = Record<string, PerModelParam>;

export type ProvidersResp = { providers: ProviderInfo[] };
export type AskAnswers = Record<string, { answer?: string; error?: string; latency_ms?: number }>;

export type StreamEvent =
  | { type: "meta"; models: string[] }
  | { type: "chunk"; model: string; answer?: string; error?: string; latency_ms: number }
  | { type: "done" };

export type Dataset = {
  dataset_id: string;
  document_count: number;
  sample_fields: string[];
};

export type SearchResult = {
  similarity_score: number;
  [key: string]: string | number | boolean | null | undefined;
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  timestamp: number;
};

export type ModelChat = {
  messages: ChatMessage[];
  isStreaming: boolean;
  currentResponse: string;
};

export type EnhancedChatRequest = {
  messages: { role: string; content: string }[];
  models: string[];
  system?: string; // Optional system message for all models
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
  google_params?: {
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

export type MultiBucket = {
  error?: string;
  items: SearchResult[];
  dataset_id?: string;
  total_documents?: number;
};
export type MultiSearchResponse = {
  query: string;
  results: Record<string, MultiBucket>;
  duration_ms?: number;
};
