// components/chat/ParameterControls.tsx
import React from 'react';
import { ModelParamsMap, PerModelParam } from '../../types';

interface ParameterControlsProps {
  globalTemp: number;
  globalMax: number;
  globalMin: number | undefined;
  modelParams: ModelParamsMap;
  allModels: string[];
  onGlobalTempChange: (temp: number) => void;
  onGlobalMaxChange: (max: number) => void;
  onGlobalMinChange: (min: number | undefined) => void;
  onUpdateParam: (model: string, key: keyof PerModelParam, value: number | undefined) => void;
}

export const ParameterControls: React.FC<ParameterControlsProps> = ({
  globalTemp,
  globalMax,
  globalMin,
  modelParams,
  allModels,
  onGlobalTempChange,
  onGlobalMaxChange,
  onGlobalMinChange,
  onUpdateParam
}) => {
  return (
    <div className="space-y-4">
      {/* Global defaults */}
      <div className="space-y-3 text-sm">
        <div>
          <label className="block mb-1 font-medium">Global temp</label>
          <input
            type="number" 
            step={0.1} 
            min={0} 
            max={2}
            value={globalTemp}
            onChange={(e) => onGlobalTempChange(Number(e.target.value))}
            className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900"
          />
        </div>
        <div>
          <label className="block mb-1 font-medium">Global max_tokens</label>
          <input
            type="number" 
            min={1}
            value={globalMax}
            onChange={(e) => onGlobalMaxChange(Number(e.target.value))}
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
            onChange={(e) => onGlobalMinChange(e.target.value ? Number(e.target.value) : undefined)}
            className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900"
          />
        </div>
      </div>

      {/* Per-model overrides */}
      <div className="mt-3 space-y-2">
        {allModels.map((model) => (
          <details
            key={model}
            className="rounded-lg border border-orange-200 dark:border-orange-500/40 p-2 bg-orange-50/60 dark:bg-orange-400/10"
          >
            <summary className="cursor-pointer flex items-center justify-between">
              <span className="font-mono text-sm">{model}</span>
              <span className="text-xs text-zinc-500 dark:text-zinc-400">Overrides</span>
            </summary>
            <div className="mt-2 grid grid-cols-3 gap-3 text-sm">
              <div>
                <label className="block mb-1">temp</label>
                <input
                  type="number" 
                  step={0.1} 
                  min={0} 
                  max={2}
                  value={modelParams[model]?.temperature ?? ""}
                  placeholder={`↳ ${globalTemp}`}
                  onChange={(e) => onUpdateParam(model, "temperature", e.target.value ? Number(e.target.value) : undefined)}
                  className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900"
                />
              </div>
              <div>
                <label className="block mb-1">max_tokens</label>
                <input
                  type="number" 
                  min={1}
                  value={modelParams[model]?.max_tokens ?? ""}
                  placeholder={`↳ ${globalMax}`}
                  onChange={(e) => onUpdateParam(model, "max_tokens", e.target.value ? Number(e.target.value) : undefined)}
                  className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900"
                />
              </div>
              <div>
                <label className="block mb-1">min_tokens</label>
                <input
                  type="number" 
                  min={1}
                  value={modelParams[model]?.min_tokens ?? ""}
                  placeholder={globalMin ? `↳ ${globalMin}` : "optional"}
                  onChange={(e) => onUpdateParam(model, "min_tokens", e.target.value ? Number(e.target.value) : undefined)}
                  className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900"
                />
              </div>
            </div>
          </details>
        ))}
      </div>
    </div>
  );
};