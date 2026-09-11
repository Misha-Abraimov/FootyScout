import Link from "next/link";

import { formatSimilarity, humanizeField } from "@/lib/format";
import type { SimilarPlayerResponse } from "@/lib/types";

export function SimilarPlayerCard({ player }: { player: SimilarPlayerResponse }) {
  const supportLabel = player.sample_support === "higher" ? "Higher" : "Limited";
  return (
    <article className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs text-[var(--muted)]">#{player.rank} · Same position group</p>
          <Link href={`/players/${player.similar_player_id}`} className="mt-2 block text-lg font-semibold hover:text-[var(--accent-strong)]">{player.similar_player_name}</Link>
          <p className="mt-1 text-sm text-[var(--muted)]">{player.similar_team_name} · {player.similar_position}</p>
        </div>
        <p className="metric-tabular shrink-0 text-lg font-semibold text-[var(--accent-strong)]">{formatSimilarity(player.similarity_score)}</p>
      </div>
      <div className="mt-5 flex flex-wrap gap-2" aria-label="Closest style dimensions">
        {player.closest_style_dimensions.map((feature) => (
          <span key={feature} className="rounded-full border border-[var(--border)] bg-white/[0.025] px-2.5 py-1 text-xs text-[var(--muted)]">{humanizeField(feature)}</span>
        ))}
      </div>
      <div className="mt-4 border-t border-[var(--border)] pt-3 text-xs text-[var(--muted)]">
        <p><span className="font-medium text-[var(--foreground)]">Sample support: {supportLabel}</span> · {player.pair_support_matches} limiting matches</p>
        <p className="mt-1">Observed: {player.query_matches_observed} vs {player.candidate_matches_observed} matches</p>
      </div>
    </article>
  );
}
