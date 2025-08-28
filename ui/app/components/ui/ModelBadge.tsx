// components/ui/ModelBadge.tsx
"use client";

import { PROVIDER_BADGE_BG } from "@/app/lib/colors";
import { ProviderBrand } from "@/app/lib/types";


export default function ModelBadge({ brand }: { brand: ProviderBrand }) {
  return <span className={`text-xs px-1.5 py-0.5 rounded ${PROVIDER_BADGE_BG[brand]}`}>{brand}</span>;
}
