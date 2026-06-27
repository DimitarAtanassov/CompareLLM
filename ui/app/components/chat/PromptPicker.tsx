"use client";

import { JSX, useCallback, useEffect, useState } from "react";
import { API_BASE } from "../../lib/config";

// Minimal mirrors of the backend /prompts DTOs (only what the picker needs).
type ProjectRef = { slug: string; name: string };
type PromptRef = { name: string };
type VariableRef = { name: string; required: boolean; description: string | null };
type VersionRef = {
  version: number;
  system_prompt: string | null;
  user_prompt: string;
  variables: VariableRef[];
};
type Rendered = { system_prompt: string | null; user_prompt: string };

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return (await res.json()) as T;
}

export default function PromptPicker({
  active,
  onApply,
  onClear,
}: {
  active: boolean;
  onApply: (systemPrompt: string, label: string) => void;
  onClear: () => void;
}): JSX.Element | null {
  const [projects, setProjects] = useState<ProjectRef[]>([]);
  const [project, setProject] = useState<string>("");
  const [prompts, setPrompts] = useState<PromptRef[]>([]);
  const [prompt, setPrompt] = useState<string>("");
  const [versions, setVersions] = useState<VersionRef[]>([]);
  const [version, setVersion] = useState<number | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load projects once. Empty list ⇒ Floating Prompts not configured ⇒ hide.
  useEffect(() => {
    getJson<ProjectRef[]>("/prompts/projects")
      .then(setProjects)
      .catch(() => setProjects([]));
  }, []);

  useEffect(() => {
    if (!project) {
      setPrompts([]);
      return;
    }
    getJson<PromptRef[]>(`/prompts/projects/${project}/prompts`)
      .then(setPrompts)
      .catch(() => setPrompts([]));
    setPrompt("");
  }, [project]);

  useEffect(() => {
    if (!project || !prompt) {
      setVersions([]);
      setVersion(null);
      return;
    }
    getJson<VersionRef[]>(`/prompts/projects/${project}/prompts/${prompt}/versions`)
      .then((vs) => {
        setVersions(vs);
        setVersion(vs[0]?.version ?? null); // newest first
      })
      .catch(() => setVersions([]));
    setValues({});
  }, [project, prompt]);

  const selected = versions.find((v) => v.version === version);

  const apply = useCallback(async () => {
    if (!project || !prompt || version == null) return;
    setBusy(true);
    setError(null);
    try {
      const rendered = await fetch(
        `${API_BASE}/prompts/projects/${project}/prompts/${prompt}/render`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ variables: values, version }),
        }
      );
      if (!rendered.ok) {
        const body = await rendered.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${rendered.status}`);
      }
      const data = (await rendered.json()) as Rendered;
      const text = data.system_prompt?.trim() || data.user_prompt;
      onApply(text, `${project}/${prompt} v${version}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load prompt");
    } finally {
      setBusy(false);
    }
  }, [project, prompt, version, values, onApply]);

  if (projects.length === 0) return null; // feature disabled / nothing to pick

  return (
    <div className="space-y-2 rounded-xl border border-zinc-200 dark:border-zinc-800 p-3">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium">System prompt (Floating Prompts)</label>
        {active && (
          <button
            type="button"
            onClick={onClear}
            className="text-xs text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"
          >
            Clear
          </button>
        )}
      </div>

      <div className="grid grid-cols-2 gap-2">
        <select
          className="rounded-lg border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-2 text-sm"
          value={project}
          onChange={(e) => setProject(e.target.value)}
        >
          <option value="">Project…</option>
          {projects.map((p) => (
            <option key={p.slug} value={p.slug}>
              {p.name}
            </option>
          ))}
        </select>

        <select
          className="rounded-lg border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-2 text-sm disabled:opacity-50"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          disabled={!project}
        >
          <option value="">Prompt…</option>
          {prompts.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name}
            </option>
          ))}
        </select>
      </div>

      {selected && (
        <>
          <select
            className="w-full rounded-lg border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-2 text-sm"
            value={version ?? ""}
            onChange={(e) => setVersion(Number(e.target.value))}
          >
            {versions.map((v) => (
              <option key={v.version} value={v.version}>
                v{v.version}
              </option>
            ))}
          </select>

          {selected.variables.map((spec) => (
            <input
              key={spec.name}
              className="w-full rounded-lg border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-2 text-sm"
              placeholder={`${spec.name}${spec.required ? " *" : ""}`}
              value={values[spec.name] ?? ""}
              onChange={(e) =>
                setValues((prev) => ({ ...prev, [spec.name]: e.target.value }))
              }
            />
          ))}

          <button
            type="button"
            onClick={apply}
            disabled={busy}
            className="w-full rounded-lg bg-orange-500 px-3 py-2 text-sm font-medium text-white hover:bg-orange-600 disabled:opacity-50"
          >
            {busy ? "Loading…" : "Use as system prompt"}
          </button>
        </>
      )}

      {error && <p className="text-xs text-red-600">{error}</p>}
    </div>
  );
}
