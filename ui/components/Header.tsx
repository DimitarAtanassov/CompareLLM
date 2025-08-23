// components/Header.tsx
import React from 'react';
import { ProviderInfo } from '../types';

interface HeaderProps {
  loadingProviders: boolean;
  providers: ProviderInfo[];
  allModelsCount: number;
  allEmbeddingModelsCount: number;
}

export const Header: React.FC<HeaderProps> = ({
  loadingProviders,
  providers,
  allModelsCount,
  allEmbeddingModelsCount
}) => {
  return (
    <header className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
      <div>
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-orange-600 dark:text-orange-400">
          Multi-LLM AI Platform
        </h1>
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Chat with multiple models and perform semantic search with embeddings.
        </p>
      </div>
      <div className="text-sm text-zinc-500 dark:text-zinc-400">
        {loadingProviders 
          ? "Loading providers…" 
          : `${providers.length} provider(s), ${allModelsCount} chat model(s), ${allEmbeddingModelsCount} embedding model(s)`
        }
      </div>
    </header>
  );
};
