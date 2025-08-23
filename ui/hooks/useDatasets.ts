// hooks/useDatasets.ts
import { useState, useCallback } from 'react';
import { Dataset } from '../types';

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "/backend");

export const useDatasets = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);

  const loadDatasets = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/datasets`);
      if (res.ok) {
        const data = await res.json();
        setDatasets(data.datasets || []);
      }
    } catch (err) {
      console.error("Failed to load datasets:", err);
    }
  }, []);

  const deleteDataset = useCallback(async (id: string) => {
    if (!confirm(`Are you sure you want to delete dataset "${id}"?`)) return;

    try {
      const res = await fetch(`${API_BASE}/datasets/${id}`, { method: "DELETE" });
      if (res.ok) {
        await loadDatasets();
      }
    } catch (err) {
      console.error("Delete failed:", err);
    }
  }, [loadDatasets]);

  return {
    datasets,
    loadDatasets,
    deleteDataset
  };
};