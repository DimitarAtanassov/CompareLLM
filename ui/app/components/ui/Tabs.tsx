// components/ui/Tabs.tsx
"use client";
type Tab = { id: string; label: string };
export default function Tabs({
  activeId, tabs, onChange,
}: { activeId: string; tabs: Tab[]; onChange: (id: string) => void }) {
  return (
    <nav className="flex border-b border-zinc-200 dark:border-zinc-800">
      {tabs.map(t => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          className={`px-6 py-3 text-sm font-medium border-b-2 transition ${
            activeId === t.id
              ? "border-orange-500 text-orange-600 dark:text-orange-400"
              : "border-transparent text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300"
          }`}
        >
          {t.label}
        </button>
      ))}
    </nav>
  );
}
