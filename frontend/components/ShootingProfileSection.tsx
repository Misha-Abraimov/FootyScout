import { MetricCard } from "@/components/MetricCard";
import { ReliabilityBadge } from "@/components/ReliabilityBadge";
import { formatCount, formatDecimal, formatPercent } from "@/lib/format";
import type { ShootingProfileResponse } from "@/lib/types";

export function ShootingProfileSection({ profile }: { profile: ShootingProfileResponse | null }) {
  if (!profile) return null;
  return (
    <section aria-labelledby="shooting-title">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">Expected Goals</p>
          <h2 id="shooting-title" className="mt-2 text-2xl font-semibold">Non-penalty shooting</h2>
          <p className="mt-2 text-sm text-[var(--muted)]">Goals above expected is descriptive for this observed sample, not a definitive finishing-skill rating.</p>
        </div>
        <ReliabilityBadge reliable={profile.shooting_reliable} />
      </div>
      <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        <MetricCard label="Shots" value={formatCount(profile.shots)} note={`${formatCount(profile.matches_observed)} matches observed`} />
        <MetricCard label="Goals" value={formatCount(profile.goals)} />
        <MetricCard label="Total xG" value={formatDecimal(profile.total_xg)} />
        <MetricCard label="Goals above expected" value={`${profile.goals_minus_xg > 0 ? "+" : ""}${formatDecimal(profile.goals_minus_xg)}`} />
        <MetricCard label="xG / shot" value={formatDecimal(profile.xg_per_shot)} note={`${formatPercent(profile.goals_per_shot)} goals / shot`} />
      </div>
      {!profile.shooting_reliable ? <p className="mt-4 text-xs text-amber-100/70">Limited sample: fewer than 20 non-penalty shots.</p> : null}
    </section>
  );
}
