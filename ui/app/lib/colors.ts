// lib/colors.ts
import type { ProviderBrand } from "./types";

export const PROVIDER_TEXT_COLOR: Record<ProviderBrand, string> = {
  anthropic: "text-orange-600 dark:text-orange-400",
  openai: "text-green-600 dark:text-green-400",
  gemini: "text-green-600 dark:text-green-400",
  google: "text-yellow-600 dark:text-yellow-400",
  ollama: "text-purple-600 dark:text-purple-400",
  deepseek: "text-sky-600 dark:text-sky-400",
  cerebras: "text-red-600 dark:text-red-400",
  cohere: "text-cyan-600 dark:text-cyan-400", 
  voyage: "text-cyan-600 dark:text-cyan-400",
  unknown: "text-zinc-600 dark:text-zinc-400",
};

export const PROVIDER_BADGE_BG: Record<ProviderBrand, string> = {
  anthropic: "bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400",
  openai: "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400",
  gemini: "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400",
  google: "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400",
  ollama: "bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400",
  deepseek: "bg-sky-100 dark:bg-sky-900/30 text-sky-700 dark:text-sky-400",
  cerebras: "text-red-600 dark:text-red-400",
  cohere: "text-cyan-600 dark:text-cyan-400", 
  voyage: "bg-cyan-100 dark:bg-cyan-900/30 text-cyan-700 dark:text-cyan-400",
  unknown: "bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-400",
};
