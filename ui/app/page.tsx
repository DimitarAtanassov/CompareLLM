"use client";

import { ChatControls } from "@/components/chat/ChatControls";
import { ChatResults } from "@/components/chat/ChatResults";
import { InteractiveChatModal } from "@/components/chat/InteractiveChatModal";
import { DatasetManager } from "@/components/embedding/DatasetManager";
import { DatasetUpload } from "@/components/embedding/DatasetUpload";
import { EmbeddingModelSelector } from "@/components/embedding/EmbeddingModelSelector";
import { SearchInterface } from "@/components/embedding/SearchInterface";
import { SearchResults } from "@/components/embedding/SearchResults";
import { Footer } from "@/components/Footer";
import { Header } from "@/components/Header";
import { TabNavigation } from "@/components/TabNavigation";
import { useDatasets } from "@/hooks/useDatasets";
import { useProviders } from "@/hooks/useProviders";
import { AskAnswers, ChatMessage, ModelChat, ModelParamsMap, PerModelParam, SearchResult, StreamEvent } from "@/types";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";


const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "/backend");

export default function Page() {
  const [activeTab, setActiveTab] = useState<"chat" | "embedding">("chat");
  
  // Use custom hooks
  const { loadingProviders, providers, allModels, allEmbeddingModels } = useProviders();
  const { datasets, loadDatasets, deleteDataset } = useDatasets();

  // Chat functionality
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

  // Search context
  const [searchContext, setSearchContext] = useState<{
    model: string;
    dataset: string;
    query: string;
    startedAt: number;
  } | null>(null);

  // Refs
  const streamAbortRef = useRef<AbortController | null>(null);
  const interactiveAbortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const requestIdRef = useRef(0);

  // Parameters
  const [modelParams, setModelParams] = useState<ModelParamsMap>({});
  const [globalTemp, setGlobalTemp] = useState<number>(0.7);
  const [globalMax, setGlobalMax] = useState<number>(8192);
  const [globalMin, setGlobalMin] = useState<number | undefined>(undefined);

  // Set initial embedding model
  useEffect(() => {
    if (allEmbeddingModels.length > 0 && !selectedSearchModel) {
      setSelectedSearchModel(allEmbeddingModels[0]);
    }
  }, [allEmbeddingModels, selectedSearchModel]);

  // Load datasets when switching to embedding tab
  useEffect(() => {
    if (activeTab === "embedding") {
      loadDatasets();
    }
  }, [activeTab, loadDatasets]);

  // Helper functions
  const updateParam = useCallback(
    (model: string, key: keyof PerModelParam, value: number | undefined) => {
      setModelParams((prev) => ({ ...prev, [model]: { ...(prev[model] || {}), [key]: value } }));
    },
    []
  );

  const toggleModel = (m: string) =>
    setSelected((prev) => (prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]));

  const selectAll = () => setSelected(allModels);
  const clearAll = () => setSelected([]);

  const toggleEmbeddingModel = (m: string) =>
    setSelectedEmbeddingModels((prev) => (prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]));

  const selectAllEmbedding = () => setSelectedEmbeddingModels(allEmbeddingModels);
  const clearAllEmbedding = () => setSelectedEmbeddingModels([]);

  const canRun = prompt.trim().length > 0 && selected.length > 0 && !isRunning;

  const resetRun = useCallback(() => {
    setAnswers(Object.fromEntries(selected.map((m) => [m, { answer: "", error: undefined, latency_ms: 0 }])));
    setStartedAt(Date.now());
    setEndedAt(null);
  }, [selected]);

  // Interactive chat functions
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

      const body = JSON.stringify({
        model: activeModel,
        messages: apiMessages,
        temperature: modelParams[activeModel]?.temperature ?? globalTemp,
        max_tokens: modelParams[activeModel]?.max_tokens ?? globalMax,
        stream: false
      });

      const res = await fetch(`${API_BASE}/v1/chat/completions`, {
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
      const assistantMessage = result.choices[0]?.message?.content || "No response";

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
  }, [activeModel, interactivePrompt, modelParams, globalTemp, globalMax, modelChats]);

  // Streaming runner
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

  // Upload dataset
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

  // Perform search
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

  // Handle dataset selection change
  const handleSelectedDatasetChange = useCallback((dataset: string) => {
    setSelectedDataset(dataset);
    if (dataset !== selectedDataset) {
      setSearchResults([]);
      setSearchContext(null);
    }
  }, [selectedDataset]);

  // Keyboard shortcuts
  useEffect(() => {
    const onKey = (evt: KeyboardEvent) => {
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
      if (evt.key === "Escape" && activeModel) {
        closeModelChat();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [canRun, runPrompt, activeTab, searchQuery, performSearch, activeModel, interactivePrompt, sendInteractiveMessage, closeModelChat]);

  // Computed values
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
      <InteractiveChatModal
        activeModel={activeModel}
        modelChats={modelChats}
        interactivePrompt={interactivePrompt}
        onInteractivePromptChange={setInteractivePrompt}
        onSendMessage={sendInteractiveMessage}
        onCloseChat={closeModelChat}
        originalPrompt={prompt}
      />

      {/* Header */}
      <Header
        loadingProviders={loadingProviders}
        providers={providers}
        allModelsCount={allModels.length}
        allEmbeddingModelsCount={allEmbeddingModels.length}
      />

      {/* Tab Navigation */}
      <TabNavigation
        activeTab={activeTab}
        setActiveTab={setActiveTab}
      />

      {/* Tab Content */}
      {activeTab === "chat" && (
        <main className="grid grid-cols-1 xl:grid-cols-[360px_1fr] gap-6 items-start">
          {/* Chat Controls */}
          <ChatControls
            prompt={prompt}
            onPromptChange={setPrompt}
            allModels={allModels}
            selected={selected}
            onToggleModel={toggleModel}
            onSelectAll={selectAll}
            onClearAll={clearAll}
            globalTemp={globalTemp}
            globalMax={globalMax}
            globalMin={globalMin}
            modelParams={modelParams}
            onGlobalTempChange={setGlobalTemp}
            onGlobalMaxChange={setGlobalMax}
            onGlobalMinChange={setGlobalMin}
            onUpdateParam={updateParam}
            canRun={canRun}
            isRunning={isRunning}
            onRunPrompt={runPrompt}
          />

          {/* Chat Results */}
          <ChatResults
            ref={bottomRef}
            answers={answers}
            isRunning={isRunning}
            onOpenModelChat={openModelChat}
          />
        </main>
      )}

      {activeTab === "embedding" && (
        <main className="grid grid-cols-1 xl:grid-cols-[400px_1fr] gap-6 items-start">
          {/* Embedding Controls */}
          <section className="space-y-6">
            <EmbeddingModelSelector
              allEmbeddingModels={allEmbeddingModels}
              selectedEmbeddingModels={selectedEmbeddingModels}
              onToggleEmbeddingModel={toggleEmbeddingModel}
              onSelectAllEmbedding={selectAllEmbedding}
              onClearAllEmbedding={clearAllEmbedding}
            />

            <DatasetUpload
              datasetId={datasetId}
              onDatasetIdChange={setDatasetId}
              textField={textField}
              onTextFieldChange={setTextField}
              selectedEmbeddingModels={selectedEmbeddingModels}
              jsonInput={jsonInput}
              onJsonInputChange={setJsonInput}
              uploadingDataset={uploadingDataset}
              onUploadDataset={uploadDataset}
            />

            <SearchInterface
              selectedSearchModel={selectedSearchModel}
              onSearchModelChange={setSelectedSearchModel}
              allEmbeddingModels={allEmbeddingModels}
              selectedDataset={selectedDataset}
              onSelectedDatasetChange={handleSelectedDatasetChange}
              datasets={datasets}
              searchQuery={searchQuery}
              onSearchQueryChange={setSearchQuery}
              isSearching={isSearching}
              onPerformSearch={performSearch}
            />

            <DatasetManager
              datasets={datasets}
              onDeleteDataset={deleteDataset}
            />
          </section>

          {/* Search Results */}
          <SearchResults
            searchResults={searchResults}
            searchQuery={searchQuery}
            isSearching={isSearching}
            searchContext={searchContext}
            selectedSearchModel={selectedSearchModel}
          />
        </main>
      )}

      {/* Footer */}
      <Footer
        activeTab={activeTab}
        selected={selected}
        anyErrors={anyErrors}
        startedAt={startedAt}
        elapsedMs={elapsedMs}
        isRunning={isRunning}
        datasets={datasets}
        selectedEmbeddingModels={selectedEmbeddingModels}
        allEmbeddingModelsCount={allEmbeddingModels.length}
      />
    </div>
  );
}