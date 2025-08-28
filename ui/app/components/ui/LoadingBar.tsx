// components/ui/LoadingBar.tsx
"use client";
export default function LoadingBar() {
  return (
    <div className="relative h-1 overflow-hidden rounded bg-orange-100 dark:bg-orange-900/30" role="status" aria-live="polite">
      <div className="absolute inset-y-0 left-0 w-1/3 animate-[loading_1.2s_infinite] bg-orange-500/80" />
      <style jsx>{`
        @keyframes loading {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(300%); }
        }
      `}</style>
    </div>
  );
}
