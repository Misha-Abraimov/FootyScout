import { MetricCard } from "@/components/MetricCard";
import { ReliabilityBadge } from "@/components/ReliabilityBadge";
import { formatPercent, formatSignedDecimal } from "@/lib/format";
import type { AttackingProfileResponse } from "@/lib/types";

export function AttackingValueSection({ profile }: { profile: AttackingProfileResponse | null }) {
  if (!profile) return null;
  const value = (number: number | null) => formatSignedDecimal(number, 3);
  return (
    <section aria-labelledby="attacking-value-title">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">Attacking impact</p>
          <h2 id="attacking-value-title" className="mt-2 text-2xl font-semibold">Impact of passes and carries</h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--muted)]">Measures how much each pass or carry changed the expected attacking value of a possession. Positive values improved the attack; negative values reduced it.</p>
        </div>
        <ReliabilityBadge reliable={profile.attacking_value_reliable} />
      </div>
      <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        <MetricCard label="Overall impact / 100 actions" value={value(profile.attacking_value_per_100_actions)} />
        <MetricCard label="Passing impact / 100 passes" value={value(profile.pass_value_per_100_passes)} />
        <MetricCard label="Carrying impact / 100 carries" value={value(profile.carry_value_per_100_carries)} />
        <MetricCard label="Progressive-action impact / 100 actions" value={value(profile.progressive_value_per_100_actions)} />
        <MetricCard label="Under-pressure impact / 100 actions" value={value(profile.pressure_value_per_100_actions)} />
      </div>
      <p className="mt-3 text-xs text-[var(--muted)]">{profile.actions.toLocaleString()} actions · {profile.passes.toLocaleString()} passes · {profile.carries.toLocaleString()} carries · {formatPercent(profile.positive_value_action_rate)} increased attacking value</p>
    </section>
  );
}
