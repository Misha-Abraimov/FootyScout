import { formatDecimal } from "@/lib/format";
import type { PlayerArchetype } from "@/lib/types";

export function PlayerArchetypeCard({ archetype }: { archetype: PlayerArchetype }) {
  if (!archetype.eligible || !archetype.name) {
    return (
      <section aria-label="Playing style" className="rounded-xl border border-[var(--border)] bg-black/15 p-4 sm:p-5">
        <p className="text-xs font-semibold tracking-[0.15em] text-[var(--accent)] uppercase">Playing style</p>
        <h3 className="mt-2 text-lg font-semibold">Archetype unavailable</h3>
        <p className="mt-2 text-sm leading-6 text-[var(--muted)]">
          {archetype.eligibility_reason ?? "Limited sample for archetype analysis."}
        </p>
        <p className="mt-3 text-xs text-[var(--muted)]">
          Archetypes describe outfield playing tendencies, not quality, ability, or rank.
        </p>
      </section>
    );
  }

  return (
    <section aria-label="Playing style" className="rounded-xl border border-[var(--border)] bg-black/15 p-4 sm:p-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="text-xs font-semibold tracking-[0.15em] text-[var(--accent)] uppercase">Playing style</p>
          <h3 className="mt-2 text-xl font-semibold">{archetype.name}</h3>
          <p className="mt-2 text-xs text-[var(--muted)]">
            Position-relative style among eligible {archetype.position_group} players.
          </p>
        </div>
        {archetype.separation_margin !== null ? (
          <div className="rounded-lg border border-white/10 px-3 py-2 text-right">
            <span className="block text-[10px] tracking-wide text-[var(--muted)] uppercase">Style separation</span>
            <strong className="metric-tabular mt-1 block text-sm">{formatDecimal(archetype.separation_margin, 3)}</strong>
          </div>
        ) : null}
      </div>

      <div className="mt-5 grid gap-2 sm:grid-cols-2">
        {archetype.distinguishing_features.map((feature) => (
          <p key={feature.feature_name} className="text-sm text-[var(--muted)]">
            <span aria-hidden="true" className="mr-2 text-[var(--accent-strong)]">
              {feature.direction === "higher" ? "↑" : "↓"}
            </span>
            {feature.label} <span className="text-xs">({feature.direction})</span>
          </p>
        ))}
      </div>

      <div aria-label="Style dimensions relative to positional peers" className="mt-6 space-y-3">
        {archetype.style_dimensions.map((dimension) => (
          <RelativeStyleBar key={dimension.feature_name} dimension={dimension} position={archetype.position_group} />
        ))}
      </div>
      <p className="mt-4 text-[11px] leading-5 text-[var(--muted)]">
        Bars are z-scores relative to eligible positional peers, not percentiles or ratings. {archetype.separation_interpretation}
      </p>
    </section>
  );
}

function RelativeStyleBar({
  dimension,
  position,
}: {
  dimension: PlayerArchetype["style_dimensions"][number];
  position: string;
}) {
  const magnitude = Math.min(Math.abs(dimension.position_z) / 3, 1) * 50;
  const positive = dimension.position_z >= 0;
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3 text-xs">
        <span>{dimension.label}</span>
        <span className="metric-tabular text-[var(--muted)]">{dimension.position_z >= 0 ? "+" : ""}{formatDecimal(dimension.position_z, 2)} z</span>
      </div>
      <div
        role="img"
        aria-label={`${dimension.label}: ${formatDecimal(dimension.position_z, 2)} standard deviations relative to eligible ${position} peers`}
        className="relative mt-1.5 h-2 overflow-hidden rounded-full bg-white/8"
      >
        <span aria-hidden="true" className="absolute inset-y-0 left-1/2 w-px bg-white/35" />
        <span
          aria-hidden="true"
          className="absolute inset-y-0 bg-[var(--accent)]"
          style={positive ? { left: "50%", width: `${magnitude}%` } : { right: "50%", width: `${magnitude}%` }}
        />
      </div>
    </div>
  );
}
