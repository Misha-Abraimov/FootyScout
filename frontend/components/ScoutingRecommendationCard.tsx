import Link from "next/link";

import { formatDecimal, humanizeField } from "@/lib/format";
import type { ScoutingRecommendation } from "@/lib/types";

export function ScoutingRecommendationCard({ recommendation }: { recommendation: ScoutingRecommendation }) {
  return (
    <article className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold text-[var(--accent)]">#{recommendation.rank}</p>
          <h3 className="mt-2 text-xl font-semibold"><Link href={`/players/${recommendation.player.player_id}`} className="hover:text-[var(--accent-strong)]">{recommendation.player.player_name}</Link></h3>
          <p className="mt-1 text-sm text-[var(--muted)]">{recommendation.player.team_name} · {recommendation.player.position}</p>
        </div>
        <div className="text-right"><span className="text-[11px] text-[var(--muted)]">Role distance</span><strong className="metric-tabular block text-xl">{formatDecimal(recommendation.role_distance, 3)}</strong></div>
      </div>
      <div className="mt-5">
        <p className="text-xs font-semibold uppercase tracking-wide text-[var(--muted)]">Closest alignment</p>
        <p className="mt-2 text-sm leading-6">{recommendation.closest_dimensions.map(humanizeField).join(" · ")}</p>
      </div>
      <p className="mt-3 text-xs text-[var(--muted)]">Largest difference: {humanizeField(recommendation.largest_difference)}</p>
      <div className="mt-5 flex flex-wrap items-center gap-2 text-xs">
        <span className="rounded-full border border-amber-200/20 bg-amber-200/5 px-2.5 py-1 font-semibold uppercase text-amber-100/80">{recommendation.sample_support}</span>
        <span className="text-[var(--muted)]">{recommendation.player_matches_observed} observed {recommendation.player_matches_observed === 1 ? "match" : "matches"}</span>
        {recommendation.archetype_name ? <span className="text-[var(--muted)]">· {recommendation.archetype_name}</span> : null}
      </div>
    </article>
  );
}
