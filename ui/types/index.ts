// types/index.ts
export type ProviderInfo = {
  name: string;
  type: string;
  base_url: string;
  models: string[];
  embedding_models: string[];
  auth_required: boolean;
};

export type PerModelParam = { 
  temperature?: number; 
  max_tokens?: number; 
  min_tokens?: number; 
};

export type ModelParamsMap = Record<string, PerModelParam>;

export type ProvidersResp = { providers: ProviderInfo[] };

export type AskAnswers = Record<string, { 
  answer?: string; 
  error?: string; 
  latency_ms?: number; 
}>;

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
