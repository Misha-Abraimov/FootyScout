import { EmptyState } from "@/components/States";
import { formatDecimal, metricLabel } from "@/lib/format";
import type { PlayerRoleFitResponse } from "@/lib/types";

export function PlayerRoleFitSection({ fit }: { fit: PlayerRoleFitResponse }) {
  if (!fit.available || fit.role_distance === null) {
    return (
      <section aria-labelledby="role-fit-title">
        <h2 id="role-fit-title" className="text-2xl font-semibold">Role Fit — Bayer Leverkusen</h2>
        <div className="mt-5"><EmptyState title="Role Fit unavailable" message={fit.unavailable_reason ?? "This player is not eligible for Role Fit."} /></div>
      </section>
    );
  }
  const comparison = fit.is_target_team_player
    ? `Compared with the observed role of Leverkusen's other ${fit.position_group} contributors.`
    : `Compared with Bayer Leverkusen's observed ${fit.position_group} role.`;
  return (
    <section data-surface="neutral" aria-labelledby="role-fit-title" className="rounded-2xl border border-[#242424] bg-[#111111] p-5 sm:p-7">
      <p className="text-xs font-semibold tracking-[0.16em] text-[var(--muted)] uppercase">Role Fit</p>
      <div className="mt-2 flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h2 id="role-fit-title" className="text-2xl font-semibold">{fit.position_group} role</h2>
          <p className="mt-2 text-sm leading-6 text-[var(--muted)]">How closely a player&apos;s playing style matches this role. {comparison}</p>
        </div>
        <div className="lg:text-right">
          {fit.cohort_rank !== null && fit.cohort_size > 0 ? (
            <>
              <strong className="metric-tabular mt-1 block text-3xl">#{fit.cohort_rank} of {fit.cohort_size}</strong>
              <span className="mt-1 block text-xs text-[var(--muted)]">{cohortDescription(fit.position_group, fit.is_target_team_player)}</span>
            </>
          ) : null}
          <span className="mt-2 block text-xs text-[var(--muted)]">Raw distance: {formatDecimal(fit.role_distance, 3)}</span>
        </div>
      </div>
      <p className="mt-4 text-sm text-[var(--muted)]">Lower distance means closer stylistic resemblance.</p>
      <div className="mt-6 grid gap-4 md:grid-cols-3">
        <div className="rounded-xl border border-[#242424] bg-[#0a0a0a] p-4 md:col-span-2">
          <h3 className="text-sm font-semibold">Closest stylistic dimensions</h3>
          <ul className="mt-3 grid gap-2 sm:grid-cols-3">{fit.closest_dimensions.map((feature) => <li key={feature} className="text-sm text-[var(--muted)]">{metricLabel(feature)}</li>)}</ul>
        </div>
        <div className="rounded-xl border border-[#242424] bg-[#0a0a0a] p-4">
          <h3 className="text-sm font-semibold">Largest stylistic difference</h3>
          <p className="mt-3 text-sm text-[var(--muted)]">{metricLabel(fit.largest_difference ?? "Unavailable")}</p>
        </div>
      </div>
      <div className="mt-5 flex flex-wrap items-center gap-3 text-xs text-[var(--muted)]">
        <span className="rounded-full border border-[#242424] px-3 py-1.5 font-semibold uppercase">{fit.sample_support} sample</span>
        <span>Observed across {fit.player_matches_observed} matches</span>
        <span>Role: {fit.role_matches_observed} matches · {fit.role_contributor_count} contributors</span>
      </div>
      {fit.position_group === "FWD" ? <p className="mt-4 text-xs leading-5 text-amber-100/70">{fit.role_support_message}</p> : null}
      <p className="mt-5 text-xs leading-5 text-[var(--muted)]">Role Fit compares observed playing style. It does not predict player quality, transfer success, or future performance.</p>
    </section>
  );
}

function cohortDescription(positionGroup: PlayerRoleFitResponse["position_group"], isTargetTeamPlayer: boolean): string {
  const position = { DEF: "defensive", MID: "midfield", FWD: "forward", GK: "goalkeeper" }[positionGroup];
  return isTargetTeamPlayer
    ? `Comparable current-team ${position} profiles`
    : `Comparable external ${position} candidates`;
}
