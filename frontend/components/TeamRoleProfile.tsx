import Link from "next/link";

import { formatCount, formatDecimal, formatPercent } from "@/lib/format";
import type { TeamRoleDimension, TeamRoleResponse } from "@/lib/types";

function rawValue(dimension: TeamRoleDimension): string {
  if (dimension.feature_name === "positive_forward_distance_per_100_passes") {
    return formatDecimal(dimension.raw_value, 1);
  }
  return formatPercent(dimension.raw_value);
}

export function TeamRoleProfile({ role, compact = false }: { role: TeamRoleResponse; compact?: boolean }) {
  return (
    <article className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">Observed positional role</p>
          <h3 className="mt-2 text-2xl font-semibold">{role.position_group}</h3>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-[var(--muted)]">{role.position_context}</p>
        </div>
        <dl className="grid grid-cols-2 gap-x-5 gap-y-2 text-sm">
          <div><dt className="text-xs text-[var(--muted)]">Matches</dt><dd className="metric-tabular font-semibold">{role.matches_observed}</dd></div>
          <div><dt className="text-xs text-[var(--muted)]">Contributors</dt><dd className="metric-tabular font-semibold">{role.contributor_count}</dd></div>
          <div><dt className="text-xs text-[var(--muted)]">Actions</dt><dd className="metric-tabular font-semibold">{formatCount(role.actions)}</dd></div>
          <div><dt className="text-xs text-[var(--muted)]">Passes / carries</dt><dd className="metric-tabular font-semibold">{formatCount(role.passes)} / {formatCount(role.carries)}</dd></div>
        </dl>
      </div>

      {role.position_group === "FWD" ? (
        <p className="mt-5 rounded-xl border border-amber-200/20 bg-amber-200/5 px-4 py-3 text-sm leading-6 text-amber-100/80">
          {role.support_message}
        </p>
      ) : null}

      <div className="mt-6 grid gap-3 md:grid-cols-2">
        {role.dimensions.map((dimension) => {
          const width = Math.min(50, Math.abs(dimension.position_z) * 18);
          const positive = dimension.position_z >= 0;
          return (
            <div key={dimension.feature_name} className="rounded-xl border border-[var(--border)] bg-black/15 p-4">
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-sm font-medium">{dimension.label}</span>
                <span className="metric-tabular text-sm text-[var(--muted)]">{rawValue(dimension)} · z {dimension.position_z >= 0 ? "+" : ""}{formatDecimal(dimension.position_z, 2)}</span>
              </div>
              <div aria-label={`${dimension.label}: ${formatDecimal(dimension.position_z, 2)} position-relative z-score`} className="relative mt-3 h-2 rounded-full bg-white/8">
                <span className="absolute inset-y-0 left-1/2 w-px bg-white/30" />
                <span
                  className="absolute inset-y-0 rounded-full bg-[var(--accent)]"
                  style={positive ? { left: "50%", width: `${width}%` } : { right: "50%", width: `${width}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {!compact ? (
        <div className="mt-6 flex flex-wrap gap-2 text-xs text-[var(--muted)]">
          {role.contributors.map((player) => (
            <Link key={player.player_id} href={`/players/${player.player_id}`} className="rounded-full border border-[var(--border)] px-3 py-1.5 hover:text-white">
              {player.player_name} · {formatPercent(player.action_share, 1)}
            </Link>
          ))}
        </div>
      ) : null}
    </article>
  );
}
