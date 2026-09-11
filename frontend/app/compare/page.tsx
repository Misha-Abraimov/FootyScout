import type { Metadata } from "next";

import { CompareClient } from "@/components/CompareClient";
import { PageHeader } from "@/components/PageHeader";
import { api } from "@/lib/api";
import type { ComparisonResponse } from "@/lib/types";

export const dynamic = "force-dynamic";
export const metadata: Metadata = { title: "Compare players" };

export default async function ComparePage({ searchParams }: { searchParams: Promise<{ player_ids?: string }> }) {
  const raw = (await searchParams).player_ids?.split(",").map(Number) ?? [];
  let initial: ComparisonResponse | null = null;
  if (raw.length === 2 && raw.every(Number.isInteger) && raw[0] !== raw[1]) {
    try { initial = await api.comparePlayers([raw[0], raw[1]]); } catch { initial = null; }
  }
  return (
    <main className="mx-auto min-h-screen max-w-6xl px-5 py-10 sm:px-8 sm:py-14">
      <PageHeader eyebrow="Side-by-side" title="Compare passing profiles" description="Compare raw model-derived metrics with optional same-position percentile context. Percentiles are not ratings." />
      <div className="mt-8"><CompareClient initial={initial} /></div>
    </main>
  );
}
