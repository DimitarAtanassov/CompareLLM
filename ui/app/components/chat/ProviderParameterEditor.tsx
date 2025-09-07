// components/chat/ProviderParameterEditor.tsx
"use client";

import { PerModelParam, ProviderWire } from "@/app/lib/types";
import React from "react";
import { PROVIDER_TEXT_COLOR } from "@/app/lib/colors";

export default function ProviderParameterEditor({
  model, providerWire, params, onUpdate,
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

  // ---------- Helpers specific to Gemini ----------
  const getGeminiVariant = (m: string): "pro" | "flash" | "flash-lite" | "unknown" => {
    const s = (m || "").toLowerCase();
    if (s.includes("pro")) return "pro";
    if (s.includes("flash-lite") || s.includes("flash_lite")) return "flash-lite";
    if (s.includes("flash")) return "flash";
    return "unknown";
  };

  const geminiVariant = providerWire === "gemini" ? getGeminiVariant(model) : "unknown";

  // Normalize the budget from flat or legacy key (so UI works with either)
  type MaybeBudget = { thinking_budget?: number; thinking_budget_tokens?: number };
  const paramsWithBudget = params as PerModelParam & MaybeBudget;

  const currentBudget: number | undefined =
    paramsWithBudget.thinking_budget ??
    paramsWithBudget.thinking_budget_tokens ??
    undefined;

  const setBudget = (val: number | undefined) => {
    // Use the flat key going forward; also clear legacy
    const next: PerModelParam & MaybeBudget = { ...paramsWithBudget };
    next.thinking_budget = val;
    delete next.thinking_budget_tokens;
    onUpdate(next);
  };

  type TBMode = "unset" | "disabled" | "dynamic" | "explicit";
  const modeFromBudget = (v: unknown): TBMode => {
    if (v === undefined || v === null || v === "") return "unset";
    const n = Number(v);
    if (Number.isNaN(n)) return "unset";
    if (n === 0) return "disabled";
    if (n === -1) return "dynamic";
    return "explicit";
  };

  const geminiBounds = (() => {
    switch (geminiVariant) {
      case "pro":        return { min: 128,  max: 32768, allowDisable: false };
      case "flash":      return { min: 1,    max: 24576, allowDisable: true  }; // 0 disables
      case "flash-lite": return { min: 512,  max: 24576, allowDisable: true  };
      default:           return { min: 1,    max: 24576, allowDisable: true  };
    }
  })();

  const tbMode = modeFromBudget(currentBudget);

  const setTBMode = (m: TBMode) => {
    if (m === "unset") {
      return setBudget(undefined);
    }
    if (m === "disabled") {
      return setBudget(0);
    }
    if (m === "dynamic") {
      return setBudget(-1);
    }
    // explicit: ALWAYS seed to a valid positive value,
    // even if currentBudget is 0 or -1
    const seed = Math.min(Math.max(geminiBounds.min, 1024), geminiBounds.max);
    return setBudget(seed);
  };

  const clampExplicit = (n: number) =>
    Math.max(geminiBounds.min, Math.min(geminiBounds.max, n));

  // ---------- Per-provider UIs ----------
  if (providerWire === "anthropic") {
    return (
      <div className="space-y-3">
        <h4 className={`text-sm font-semibold ${PROVIDER_TEXT_COLOR.anthropic}`}>Anthropic Parameters</h4>
        <div className="bg-orange-50/60 dark:bg-orange-900/10 p-3 rounded-lg border border-orange-200 dark:border-orange-800">
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
        <h4 className={`text-sm font-semibold ${PROVIDER_TEXT_COLOR.openai}`}>OpenAI Parameters</h4>
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
    // ---- Gemini UI w/ model-aware thinking budget ----
    const labelByVariant: Record<typeof geminiVariant, string> = {
      pro: "2.5 Pro — dynamic thinking by default; cannot disable. You may also set an explicit budget (128–32768).",
      flash: "2.5 Flash — Disable (0), Dynamic (−1), or set an explicit budget (1–24576).",
      "flash-lite": "2.5 Flash Lite — default minimal thinking; Disable (0), Dynamic (−1), or set 512–24576.",
      unknown: "Gemini reasoning — choose Disable (0), Dynamic (−1), or set an explicit budget.",
    };

    const disableHelp =
      geminiVariant === "pro"
        ? "Not available on Pro."
        : "Sets thinking_budget=0 (disables thinking)";
    const dynamicHelp = "Sets thinking_budget=−1 (model decides when/how much to think)";

    const rangeHint =
      geminiVariant === "pro"
        ? "128 – 32768"
        : geminiVariant === "flash-lite"
        ? "512 – 24576"
        : "1 – 24576";

    return (
      <div className="space-y-3">
        <h4 className={`text-sm font-semibold ${PROVIDER_TEXT_COLOR.google}`}>Google Gemini Parameters</h4>
        <div className="grid grid-cols-3 gap-3">
          <Num label="Top-K" val={params.top_k} onChange={(n)=>updateParam("top_k",n)} placeholder="40" />
          <Num label="Top-P" val={params.top_p} min={0} max={1} step={0.1} onChange={(n)=>updateParam("top_p",n)} placeholder="0.9" />
          <Num label="Candidates" val={params.candidate_count} onChange={(n)=>updateParam("candidate_count",n)} placeholder="1" />
        </div>

        <div className="bg-yellow-50/60 dark:bg-yellow-900/10 p-3 rounded-lg border border-yellow-200 dark:border-yellow-800 space-y-2">
          <div className="text-xs text-zinc-600 dark:text-zinc-400">{labelByVariant[geminiVariant]}</div>

          {/* Mode selector: 4 buttons */}
          <div className="grid sm:grid-cols-4 gap-2 mt-1">
            {/* Not set (default) */}
            <button
              type="button"
              onClick={() => setTBMode("unset")}
              className={[
                "text-xs px-2 py-1 rounded-md border",
                tbMode === "unset"
                  ? "border-yellow-400 bg-yellow-100/70 dark:border-yellow-700 dark:bg-yellow-900/30"
                  : "border-yellow-200 dark:border-yellow-700 hover:bg-yellow-100/40 dark:hover:bg-yellow-900/20",
              ].join(" ")}
              title="Do not set; use model defaults"
            >
              Not set (default)
            </button>

            {/* Disable (0) */}
            <button
              type="button"
              onClick={() => setTBMode("disabled")}
              disabled={!geminiBounds.allowDisable}
              className={[
                "text-xs px-2 py-1 rounded-md border",
                !geminiBounds.allowDisable
                  ? "opacity-50 cursor-not-allowed border-yellow-200 dark:border-yellow-800"
                  : tbMode === "disabled"
                  ? "border-yellow-400 bg-yellow-100/70 dark:border-yellow-700 dark:bg-yellow-900/30"
                  : "border-yellow-200 dark:border-yellow-700 hover:bg-yellow-100/40 dark:hover:bg-yellow-900/20",
              ].join(" ")}
              title={disableHelp}
            >
              Disable thinking (0)
            </button>

            {/* Dynamic (−1) */}
            <button
              type="button"
              onClick={() => setTBMode("dynamic")}
              className={[
                "text-xs px-2 py-1 rounded-md border",
                tbMode === "dynamic"
                  ? "border-yellow-400 bg-yellow-100/70 dark:border-yellow-700 dark:bg-yellow-900/30"
                  : "border-yellow-200 dark:border-yellow-700 hover:bg-yellow-100/40 dark:hover:bg-yellow-900/20",
              ].join(" ")}
              title={dynamicHelp}
            >
              Dynamic (−1)
            </button>

            {/* User specified (explicit) */}
            <button
              type="button"
              onClick={() => setTBMode("explicit")}
              className={[
                "text-xs px-2 py-1 rounded-md border",
                tbMode === "explicit"
                  ? "border-yellow-400 bg-yellow-100/70 dark:border-yellow-700 dark:bg-yellow-900/30"
                  : "border-yellow-200 dark:border-yellow-700 hover:bg-yellow-100/40 dark:hover:bg-yellow-900/20",
              ].join(" ")}
              title={`Set a specific budget in range ${rangeHint}`}
            >
              User specified
            </button>
          </div>

          {/* Explicit budget input — only visible when 'User specified' is active */}
          {tbMode === "explicit" && (
            <div className="mt-2">
              <label className="block text-xs font-medium mb-1">
                Thinking budget <span className="text-[11px] text-zinc-500">(tokens, {rangeHint})</span>
              </label>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min={geminiBounds.min}
                  max={geminiBounds.max}
                  step={1}
                  value={Number.isFinite(Number(currentBudget)) ? Number(currentBudget) : ""}
                  onChange={(e) => {
                    const raw = e.target.value ? Number(e.target.value) : NaN;
                    if (!Number.isNaN(raw)) setBudget(clampExplicit(raw));
                  }}
                  className="w-full rounded-md border border-yellow-200 dark:border-yellow-700 p-2 bg-white dark:bg-zinc-900 text-sm"
                  placeholder={geminiVariant === "pro" ? "e.g. 4096" : "e.g. 1024"}
                />
                <span className="text-[11px] text-zinc-500">
                  Sets <code>thinking_budget</code>
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  if (providerWire === "ollama") {
    return (
      <div className="space-y-3">
        <h4 className={`text-sm font-semibold ${PROVIDER_TEXT_COLOR.ollama}`}>Ollama Parameters</h4>
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

  if (providerWire === "cohere") {
    return (
      <div className="space-y-3">
        <h4 className={`text-sm font-semibold ${PROVIDER_TEXT_COLOR.cohere}`}>Cohere Parameters</h4>

        <div className="grid grid-cols-2 gap-3">
          <Num label="k (top-k)" val={params.k} min={0} max={500}
               onChange={(n)=>updateParam("k", n)} placeholder="0 (disabled)" />
          <Num label="p (top-p)" val={params.p} min={0.01} max={0.99} step={0.01}
               onChange={(n)=>updateParam("p", n)} placeholder="0.75" />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <Num label="Frequency Penalty" val={params.frequency_penalty} min={0} max={1} step={0.01}
               onChange={(n)=>updateParam("frequency_penalty", n)} placeholder="0.0" />
          <Num label="Presence Penalty" val={params.presence_penalty} min={0} max={1} step={0.01}
               onChange={(n)=>updateParam("presence_penalty", n)} placeholder="0.0" />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <Num label="Seed" val={params.seed} min={0}
               onChange={(n)=>updateParam("seed", n)} placeholder="42" />
        </div>

        <div>
          <label className="block text-xs font-medium mb-1">Stop Sequences (comma-separated)</label>
          <input
            type="text"
            value={fmtStops(params.stop_sequences as string[] | undefined)}
            onChange={(e)=>updateParam("stop_sequences", parseStops(e.target.value))}
            className="w-full rounded-md border p-2 border-cyan-200 dark:border-cyan-500/40 bg-white dark:bg-zinc-900 text-sm"
            placeholder="###, END, Human:"
          />
        </div>

        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={!!params.raw_prompting}
            onChange={(e)=>updateParam("raw_prompting", e.target.checked)}
            className="accent-cyan-600 dark:accent-cyan-400"
          />
          <span className="text-sm font-medium">Raw Prompting (no pre-processing)</span>
        </label>
      </div>
    );
  }

  if (providerWire === "deepseek") {
    return (
      <div className="space-y-3">
        <h4 className={`text-sm font-semibold ${PROVIDER_TEXT_COLOR.deepseek}`}>DeepSeek Parameters</h4>

        <div className="grid grid-cols-2 gap-3">
          <Num
            label="Frequency Penalty"
            val={params.frequency_penalty}
            min={-2}
            max={2}
            step={0.1}
            onChange={(n) => updateParam("frequency_penalty", n)}
            placeholder="0.0"
          />
          <Num
            label="Presence Penalty"
            val={params.presence_penalty}
            min={-2}
            max={2}
            step={0.1}
            onChange={(n) => updateParam("presence_penalty", n)}
            placeholder="0.0"
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <Num
            label="Top-P"
            val={params.top_p}
            min={0}
            max={1}
            step={0.01}
            onChange={(n) => updateParam("top_p", n)}
            placeholder="1.0"
          />
          <div>
            <label className="block text-xs font-medium mb-1">Logprobs</label>
            <input
              type="checkbox"
              checked={!!params.logprobs}
              onChange={(e) => updateParam("logprobs", e.target.checked)}
              className="accent-sky-600 dark:accent-sky-400"
            />
          </div>
        </div>

        <Num
          label="Top Logprobs"
          val={params.top_logprobs}
          min={0}
          max={20}
          step={1}
          onChange={(n) => updateParam("top_logprobs", n)}
          placeholder="0"
        />
      </div>
    );
  }

  return null;
}

// ---------- shared inputs ----------
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
