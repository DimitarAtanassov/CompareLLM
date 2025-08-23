// components/TabNavigation.tsx
import React from 'react';

interface TabNavigationProps {
  activeTab: "chat" | "embedding";
  setActiveTab: (tab: "chat" | "embedding") => void;
}

export const TabNavigation: React.FC<TabNavigationProps> = ({
  activeTab,
  setActiveTab
}) => {
  return (
    <nav className="flex border-b border-zinc-200 dark:border-zinc-800">
      <button
        onClick={() => setActiveTab("chat")}
        className={`px-6 py-3 text-sm font-medium border-b-2 transition ${
          activeTab === "chat"
            ? "border-orange-500 text-orange-600 dark:text-orange-400"
            : "border-transparent text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300"
        }`}
      >
        Chat Models
      </button>
      <button
        onClick={() => setActiveTab("embedding")}
        className={`px-6 py-3 text-sm font-medium border-b-2 transition ${
          activeTab === "embedding"
            ? "border-orange-500 text-orange-600 dark:text-orange-400"
            : "border-transparent text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300"
        }`}
      >
        Embeddings
      </button>
    </nav>
  );
};