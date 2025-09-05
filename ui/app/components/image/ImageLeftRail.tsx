// app/components/image/ImageLeftRail.tsx
"use client";

import React, { JSX, useEffect, useMemo, useRef, useState, useCallback } from "react";
import Spinner from "../ui/Spinner";
import { ProviderBrand } from "@/app/lib/types";
import { PROVIDER_TEXT_COLOR } from "@/app/lib/colors";

export type ImageEndpoint = { id: string; label: string; path: string; help?: string };

type Props = {
  // Model picker
  allVisionModels: string[];
  selectedVisionModels: string[];
  toggleVisionModel: (m: string) => void;
  selectAllVision: () => void;
  clearAllVision: () => void;
  getProviderType: (model: string) => ProviderBrand;

  // Uploader
  imageFile: File | null;
  setImageFile: (f: File | null) => void;

  // Optional prompt
  prompt: string;
  setPrompt: (v: string) => void;

  // Run + Clear
  isProcessing: boolean;
  onRun: () => void | Promise<void>;
  onClear: () => void;

  // accept
  accept?: string;
};

export default function ImageLeftRail({
  allVisionModels,
  selectedVisionModels,
  toggleVisionModel,
  selectAllVision,
  clearAllVision,
  getProviderType,

  imageFile,
  setImageFile,

  prompt,
  setPrompt,

  isProcessing,
  onRun,
  onClear,

  accept = "image/*",
}: Props): JSX.Element {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");

  useEffect(() => {
    if (!imageFile) {
      setPreviewUrl("");
      return;
    }
    const url = URL.createObjectURL(imageFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [imageFile]);

  // Resizable model list (like Embeddings)
  const MIN_H = 140;
  const MAX_H = 520;
  const [modelListHeight, setModelListHeight] = useState<number>(260);
  const dragRef = useRef<{ startY: number; startH: number } | null>(null);

  const beginDrag = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    dragRef.current = { startY: e.clientY, startH: modelListHeight };
    const onMove = (ev: MouseEvent) => {
      if (!dragRef.current) return;
      const dy = ev.clientY - dragRef.current.startY;
      const next = Math.min(MAX_H, Math.max(MIN_H, dragRef.current.startH + dy));
      setModelListHeight(next);
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      dragRef.current = null;
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, [modelListHeight]);

  const canRun = !!imageFile && !isProcessing && selectedVisionModels.length > 0;

  return (
    <section className="rounded-2xl border border-zinc-200 dark:border-zinc-800 p-4 sm:p-5 bg-white dark:bg-zinc-950 shadow-sm space-y-5">
      {/* Vision Models */}
      <div>
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium">Vision Models</label>
          <div className="flex gap-2 text-xs">
            <button
              onClick={selectAllVision}
              className="px-2 py-1 rounded-lg border border-orange-200 bg-orange-50 hover:bg-orange-100 dark:bg-orange-400/10 dark:hover:bg-orange-400/20"
            >
              Select all
            </button>
            <button
              onClick={clearAllVision}
              className="px-2 py-1 rounded-lg border border-orange-200 bg-orange-50 hover:bg-orange-100 dark:bg-orange-400/10 dark:hover:bg-orange-400/20"
            >
              Clear
            </button>
          </div>
        </div>

        <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 mt-2">
          <div
            className="overflow-auto p-2 grid grid-cols-1 gap-1"
            style={{ height: modelListHeight }}
            aria-label="Vision model selection list"
          >
            {allVisionModels.length === 0 && (
              <div className="text-sm text-zinc-500 dark:text-zinc-400">
                No models discovered yet.
              </div>
            )}
            {allVisionModels.map((m) => {
              const brand = getProviderType(m);
              const checked = selectedVisionModels.includes(m);
              return (
                <label
                  key={m}
                  className={[
                    "flex items-center gap-2 px-2 py-1 rounded-lg cursor-pointer",
                    "hover:bg-orange-50 dark:hover:bg-orange-400/10",
                    checked ? "ring-1 ring-orange-300/60 dark:ring-orange-500/40" : "",
                  ].join(" ")}
                  title={m}
                >
                  <input
                    type="checkbox"
                    className="accent-orange-600 dark:accent-orange-500 cursor-pointer"
                    checked={checked}
                    onChange={() => toggleVisionModel(m)}
                  />
                  <span className="text-sm font-mono flex-1 truncate">{m}</span>
                  <span className={`text-xs px-1.5 py-0.5 rounded ${PROVIDER_TEXT_COLOR[brand]} bg-current/10`}>
                    {brand}
                  </span>
                </label>
              );
            })}
          </div>

          {/* Drag handle */}
          <div
            onMouseDown={beginDrag}
            className="relative h-3 w-full cursor-row-resize select-none"
            title="Drag to resize"
          >
            <div className="absolute left-1/2 -translate-x-1/2 top-0.5 h-2 w-16 rounded-full bg-zinc-300 dark:bg-zinc-700" />
          </div>
        </div>
      </div>

      {/* Uploader */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const f = e.dataTransfer.files?.[0];
          if (f) setImageFile(f);
        }}
        className={[
          "rounded-xl border border-dashed p-4 transition",
          dragOver
            ? "border-orange-400 bg-orange-50/50 dark:border-orange-500/50 dark:bg-orange-500/10"
            : "border-zinc-300 dark:border-zinc-700",
        ].join(" ")}
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
        title="Click or drag & drop"
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          className="hidden"
          onChange={(e) => setImageFile(e.target.files?.[0] || null)}
        />
        <div className="flex items-center gap-3">
          <div className="shrink-0 w-12 h-12 rounded-lg bg-zinc-100 dark:bg-zinc-800 overflow-hidden flex items-center justify-center">
            {previewUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={previewUrl} alt="preview" className="w-full h-full object-cover" />
            ) : (
              <svg width="24" height="24" viewBox="0 0 24 24" className="text-zinc-500">
                <path fill="currentColor" d="M21 19V5a2 2 0 0 0-2-2H5C3.89 3 3 3.9 3 5v14c0 1.11.89 2 2 2h14a2 2 0 0 0 2-2M8.5 13.5 11 17l3.5-4.5L19 20H5l3.5-6.5Z" />
              </svg>
            )}
          </div>
          <div className="text-sm">
            <div className="font-medium">{imageFile ? imageFile.name : "Upload an image"}</div>
            <div className="text-zinc-500 dark:text-zinc-400">
              Click or drag & drop (PNG/JPG/WebP)
            </div>
          </div>
        </div>
      </div>

      {/* Optional prompt */}
      <div className="space-y-1">
        <label className="text-sm font-medium">Optional prompt</label>
        <textarea
          className="w-full h-28 resize-y rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 outline-none focus:ring-2 focus:ring-orange-300/60 bg-white dark:bg-zinc-900 leading-relaxed text-sm"
          placeholder="Add extra instructions (optional)…"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
      </div>

      <div className="flex gap-2">
        <button
          onClick={() => void onRun()}
          disabled={!canRun}
          className="flex-1 rounded-xl py-2 px-4 font-medium text-white bg-orange-600 hover:bg-orange-700 dark:bg-orange-500 dark:hover:bg-orange-600 transition disabled:opacity-50 flex items-center justify-center gap-2"
          title="Cmd/Ctrl+Enter"
        >
          {isProcessing && <Spinner />}
          {isProcessing ? "Processing…" : "Run"}
        </button>
        <button
          onClick={onClear}
          disabled={isProcessing && !!imageFile}
          className="px-3 py-2 rounded-xl border border-zinc-200 dark:border-zinc-800 text-sm hover:bg-zinc-50 dark:hover:bg-zinc-900"
        >
          Clear
        </button>
      </div>
    </section>
  );
}
