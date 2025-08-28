// components/chat/ProviderParameterEditor.tsx
"use client";

import { PerModelParam, ProviderWire } from "@/app/lib/types";


export default function ProviderParameterEditor({
  providerWire, params, onUpdate,
}: {
  model: string;
  providerWire: ProviderWire;
  params: PerModelParam;
  onUpdate: (params: PerModelParam) => void;
}) {
  const updateParam = (key: keyof PerModelParam, value: unknown) =>
    onUpdate({ ...params, [key]: value });

  const fmtStops = (seq?: string[]) => (seq ? seq.join(", ") : "");
  const parseStops = (v: string) => v.split(",").map(s => s.trim()).filter(Boolean);

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
            </div>
          )}
        </div>
        <div className="grid grid-cols-2 gap-3">
          <Num label="Top-K" val={params.top_k} onChange={(n)=>updateParam("top_k",n)} placeholder="40" />
          <Num label="Top-P" val={params.top_p} min={0} max={1} step={0.1} onChange={(n)=>updateParam("top_p",n)} placeholder="0.9" />
        </div>
        <div>
          <label className="block text-xs font-medium mb-1">Stop Sequences</label>
          <input
            type="text"
            value={fmtStops(params.stop_sequences)}
            onChange={(e) => updateParam("stop_sequences", parseStops(e.target.value))}
            className="w-full rounded-md border border-orange-200 dark:border-orange-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
            placeholder="Human:, Assistant:"
          />
        </div>
      </div>
    );
  }

  if (providerWire === "openai") {
    return (
      <div className="space-y-3">
        <h4 className="text-sm font-semibold text-blue-600 dark:text-blue-400">OpenAI Parameters</h4>
        <div className="grid grid-cols-2 gap-3">
          <Num label="Top-P" val={params.top_p} min={0} max={1} step={0.1} onChange={(n)=>updateParam("top_p",n)} placeholder="0.9" />
          <Num label="Seed" val={params.seed} onChange={(n)=>updateParam("seed",n)} placeholder="42" />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <Num label="Frequency Penalty" val={params.frequency_penalty} min={-2} max={2} step={0.1}
               onChange={(n)=>updateParam("frequency_penalty",n)} placeholder="0.0" />
          <Num label="Presence Penalty" val={params.presence_penalty} min={-2} max={2} step={0.1}
               onChange={(n)=>updateParam("presence_penalty",n)} placeholder="0.0" />
        </div>
      </div>
    );
  }

  if (providerWire === "gemini") {
    return (
      <div className="space-y-3">
        <h4 className="text-sm font-semibold text-green-600 dark:text-green-400">Gemini Parameters</h4>
        <div className="grid grid-cols-3 gap-3">
          <Num label="Top-K" val={params.top_k} onChange={(n)=>updateParam("top_k",n)} placeholder="40" />
          <Num label="Top-P" val={params.top_p} min={0} max={1} step={0.1} onChange={(n)=>updateParam("top_p",n)} placeholder="0.9" />
          <Num label="Candidates" val={params.candidate_count} onChange={(n)=>updateParam("candidate_count",n)} placeholder="1" />
        </div>
      </div>
    );
  }

  if (providerWire === "ollama") {
    return (
      <div className="space-y-3">
        <h4 className="text-sm font-semibold text-purple-600 dark:text-purple-400">Ollama Parameters</h4>
        <div className="grid grid-cols-2 gap-3">
          <Select
            label="Mirostat"
            val={params.mirostat}
            onChange={(v)=>onUpdate({ ...params, mirostat: v ?? undefined })}
          />
          <Num label="Context Size" val={params.num_ctx} onChange={(n)=>updateParam("num_ctx",n)} placeholder="4096" />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <Num label="Mirostat Eta" val={params.mirostat_eta} step={0.01} min={0} onChange={(n)=>updateParam("mirostat_eta",n)} placeholder="0.1" />
          <Num label="Mirostat Tau" val={params.mirostat_tau} step={0.1} min={0} onChange={(n)=>updateParam("mirostat_tau",n)} placeholder="5.0" />
        </div>
        <Num label="Repeat Penalty" val={params.repeat_penalty} step={0.1} min={0} onChange={(n)=>updateParam("repeat_penalty",n)} placeholder="1.1" />
      </div>
    );
  }

  return null;
}

function Num({
  label, val, onChange, min, max, step, placeholder,
}: {
  label: string; val?: number; onChange: (n?: number) => void;
  min?: number; max?: number; step?: number; placeholder?: string;
}) {
  return (
    <div>
      <label className="block text-xs font-medium mb-1">{label}</label>
      <input
        type="number"
        {...(min !== undefined ? {min} : {})}
        {...(max !== undefined ? {max} : {})}
        {...(step !== undefined ? {step} : {})}
        value={val ?? ""}
        onChange={(e) => onChange(e.target.value ? Number(e.target.value) : undefined)}
        className="w-full rounded-md border p-2 border-orange-200 dark:border-orange-500/40 bg-white dark:bg-zinc-900 text-sm"
        placeholder={placeholder}
      />
    </div>
  );
}

function Select({
  label, val, onChange,
}: {
  label: string; val?: number; onChange: (v?: number) => void;
}) {
  return (
    <div>
      <label className="block text-xs font-medium mb-1">{label}</label>
      <select
        value={val ?? ""}
        onChange={(e) => onChange(e.target.value ? Number(e.target.value) : undefined)}
        className="w-full rounded-md border border-purple-200 dark:border-purple-500/40 p-2 bg-white dark:bg-zinc-900 text-sm"
      >
        <option value="">Off</option>
        <option value="1">Mirostat 1</option>
        <option value="2">Mirostat 2</option>
      </select>
    </div>
  );
}
