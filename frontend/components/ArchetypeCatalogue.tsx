import Link from "next/link";

import { formatDecimal } from "@/lib/format";
import type { ArchetypeCatalogueResponse } from "@/lib/types";

export function ArchetypeCatalogue({ catalogue }: { catalogue: ArchetypeCatalogueResponse }) {
  return (
    <div className="space-y-6">
      <p className="max-w-3xl text-sm leading-6 text-[var(--muted)]">{catalogue.purpose}</p>
      <div className="grid gap-6 lg:grid-cols-2">
        {catalogue.definitions.map((definition) => (
          <article key={definition.id} className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-5 sm:p-7">
            <p className="text-xs font-semibold tracking-[0.15em] text-[var(--accent)] uppercase">Playing style</p>
            <div className="mt-2 flex items-start justify-between gap-4">
              <h2 className="text-2xl font-semibold">{definition.name}</h2>
              <span className="metric-tabular rounded-full border border-white/10 px-3 py-1 text-xs text-[var(--muted)]">
                {definition.player_count} players
              </span>
            </div>
            <p className="mt-3 text-sm leading-6 text-[var(--muted)]">{definition.description}</p>

            <h3 className="mt-6 text-sm font-semibold">Distinguishing relative style</h3>
            <ul className="mt-3 space-y-2">
              {definition.distinguishing_features.map((feature) => (
                <li key={feature.feature_name} className="flex items-center justify-between gap-4 text-sm">
                  <span><span aria-hidden="true" className="mr-2 text-[var(--accent-strong)]">{feature.direction === "higher" ? "↑" : "↓"}</span>{feature.label}</span>
                  <span className="metric-tabular text-xs text-[var(--muted)]">{feature.position_z >= 0 ? "+" : ""}{formatDecimal(feature.position_z, 2)} z</span>
                </li>
              ))}
            </ul>

            <h3 className="mt-6 text-sm font-semibold">Position composition</h3>
            <p className="mt-2 text-xs text-[var(--muted)]">
              {(["DEF", "MID", "FWD"] as const).map((position) => `${position} ${definition.position_composition[position].count}`).join(" · ")}
            </p>

            <h3 className="mt-6 text-sm font-semibold">Representative players</h3>
            <p className="mt-1 text-[11px] leading-5 text-[var(--muted)]">Closest to the centroid in the six-dimensional style space; not the best players.</p>
            <ul className="mt-3 divide-y divide-[var(--border)]">
              {definition.representative_players.map((player) => (
                <li key={player.player_id} className="flex items-center justify-between gap-4 py-2.5 text-sm">
                  <div>
                    <Link href={`/players/${player.player_id}`} className="font-medium text-[var(--accent-strong)] hover:underline">{player.player_name}</Link>
                    <span className="ml-2 text-xs text-[var(--muted)]">{player.team_name} · {player.position_group}</span>
                  </div>
                  <span className="metric-tabular text-xs text-[var(--muted)]">d {formatDecimal(player.centroid_distance, 2)}</span>
                </li>
              ))}
            </ul>
          </article>
        ))}
      </div>
      <section className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-5 text-sm leading-6 text-[var(--muted)]">
        <h2 className="font-semibold text-[var(--foreground)]">Methodology boundary</h2>
        <p className="mt-2">
          K-Means was fit in all six position-normalized dimensions for 133 eligible outfield players. The two clusters are descriptive tendencies, not objectively true football roles, ratings, or ability tiers.
        </p>
      </section>
    </div>
  );
}
