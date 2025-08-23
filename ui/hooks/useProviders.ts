import { useState, useEffect } from 'react';
import { ProviderInfo, ProvidersResp } from '../types';

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "/backend");

export const useProviders = () => {
  const [loadingProviders, setLoadingProviders] = useState(false);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [allModels, setAllModels] = useState<string[]>([]);
  const [allEmbeddingModels, setAllEmbeddingModels] = useState<string[]>([]);

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
      } catch (err) {
        console.error(err);
      } finally {
        setLoadingProviders(false);
      }
    };
    load();
  }, []);

  return {
    loadingProviders,
    providers,
    allModels,
    allEmbeddingModels
  };
};
