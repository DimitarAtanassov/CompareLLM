// components/chat/ChatControls.tsx
import React from 'react';
import { ModelSelector } from './ModelSelector';
import { ParameterControls } from './ParameterControls';
import { ModelParamsMap, PerModelParam } from '../../types';

interface ChatControlsProps {
  prompt: string;
  onPromptChange: (prompt: string) => void;
  allModels: string[];
  selected: string[];
  onToggleModel: (model: string) => void;
  onSelectAll: () => void;
  onClearAll: () => void;
  globalTemp: number;
  globalMax: number;
  globalMin: number | undefined;
  modelParams: ModelParamsMap;
  onGlobalTempChange: (temp: number) => void;
  onGlobalMaxChange: (max: number) => void;
  onGlobalMinChange: (min: number | undefined) => void;
  onUpdateParam: (model: string, key: keyof PerModelParam, value: number | undefined) => void;
  canRun: boolean;
  isRunning: boolean;
  onRunPrompt: () => void;
}

export const ChatControls: React.FC<ChatControlsProps> = ({
  prompt,
  onPromptChange,
  allModels,
  selected,
  onToggleModel,
  onSelectAll,
  onClearAll,
  globalTemp,
  globalMax,
  globalMin,
  modelParams,
  onGlobalTempChange,
  onGlobalMaxChange,
  onGlobalMinChange,
  onUpdateParam,
  canRun,
  isRunning,
  onRunPrompt
}) => {
  return (
    <section className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 sm:p-5 bg-white dark:bg-zinc-950 shadow-sm">
      <div className="space-y-4">
        <div>
          <label className="text-sm font-medium">Prompt</label>
          <textarea
            className="w-full rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 outline-none focus:ring-2 focus:ring-orange-300/60 bg-white dark:bg-zinc-900 mt-2"
            placeholder="Ask your question once. e.g., 'Explain RAG vs fine-tuning for my use case.'"
            value={prompt}
            onChange={(evt) => onPromptChange(evt.target.value)}
          />
        </div>

        <ModelSelector
          allModels={allModels}
          selected={selected}
          onToggleModel={onToggleModel}
          onSelectAll={onSelectAll}
          onClearAll={onClearAll}
        />

        <ParameterControls
          globalTemp={globalTemp}
          globalMax={globalMax}
          globalMin={globalMin}
          modelParams={modelParams}
          allModels={allModels}
          onGlobalTempChange={onGlobalTempChange}
          onGlobalMaxChange={onGlobalMaxChange}
          onGlobalMinChange={onGlobalMinChange}
          onUpdateParam={onUpdateParam}
        />

        <button
          onClick={onRunPrompt}
          disabled={!canRun}
          className="w-full rounded-xl py-2 px-4 font-medium text-white bg-orange-600 hover:bg-orange-700 dark:bg-orange-500 dark:hover:bg-orange-600 transition disabled:opacity-50"
        >
          {isRunning ? "Running…" : "Run prompt"}
        </button>
      </div>
    </section>
  );
};