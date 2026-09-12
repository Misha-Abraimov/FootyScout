import type { Metadata } from "next";
import Link from "next/link";

import { PageHeader } from "@/components/PageHeader";
import { ScoutingRecommendationCard } from "@/components/ScoutingRecommendationCard";
import { ErrorState } from "@/components/States";
import { api } from "@/lib/api";
import { cx } from "@/lib/format";

export const dynamic = "force-dynamic";
export const metadata: Metadata = { title: "Scouting Recommendation System" };
const roles = ["DEF", "MID", "FWD"] as const;

export default async function ScoutingPage({ searchParams }: { searchParams: Promise<{ role?: string }> }) {
  const requested = (await searchParams).role;
  const role = roles.includes(requested as (typeof roles)[number])
    ? (requested as (typeof roles)[number])
    : "DEF";
  const recommendations = await api.getScoutingRecommendations(904, role, 6).catch(() => null);
  if (!recommendations) return <main className="mx-auto min-h-[70vh] max-w-7xl px-5 py-16 sm:px-8"><ErrorState /></main>;
  return (
    <main className="mx-auto min-h-screen max-w-7xl space-y-9 px-5 py-10 sm:px-8 sm:py-14">
      <PageHeader eyebrow="Bayer Leverkusen" title="Scouting Recommendation System" description={recommendations.definition} />
      <p className="-mt-5 max-w-4xl text-sm leading-6 text-[var(--muted)]">External players in the current product sample have limited match coverage. Recommendations rank observed style fit and may change as additional matches are added. {recommendations.disclaimer}</p>
      <nav aria-label="Target role" className="flex flex-wrap gap-2">
        {roles.map((item) => <Link key={item} href={`/scouting?role=${item}`} aria-current={role === item ? "page" : undefined} className={cx("rounded-lg border px-5 py-2.5 text-sm font-semibold", role === item ? "border-[var(--accent)] bg-emerald-300/10 text-[var(--accent-strong)]" : "border-[var(--border)] bg-[var(--panel)] text-[var(--muted)] hover:text-white")}>{item}</Link>)}
      </nav>
      {role === "FWD" ? <p className="rounded-xl border border-amber-200/20 bg-amber-200/5 px-4 py-3 text-sm leading-6 text-amber-100/80">{recommendations.role_support_message}</p> : null}
      <section aria-labelledby="shortlist-title">
        <div className="flex flex-wrap items-end justify-between gap-3"><div><p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">Target role · {role}</p><h2 id="shortlist-title" className="mt-2 text-2xl font-semibold">Scouting Recommendations</h2></div><p className="text-xs text-[var(--muted)]">Lower role distance indicates closer observed style.</p></div>
        <div className="mt-6 grid gap-4 lg:grid-cols-2">{recommendations.items.map((item) => <ScoutingRecommendationCard key={item.player.player_id} recommendation={item} />)}</div>
      </section>
    </main>
  );
}
