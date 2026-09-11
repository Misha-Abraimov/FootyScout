import { MetricCard } from "@/components/MetricCard";
import { ReliabilityBadge } from "@/components/ReliabilityBadge";
import { formatDecimal, formatPercent } from "@/lib/format";
import type { AttackingProfileResponse } from "@/lib/types";

export function AttackingValueSection({ profile }: { profile: AttackingProfileResponse | null }) {
  if (!profile) return null;
  const value = (number: number | null) => number === null ? "—" : formatDecimal(number, 3);
  return (
    <section aria-labelledby="attacking-value-title">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">Attacking value</p>
          <h2 id="attacking-value-title" className="mt-2 text-2xl font-semibold">Possession value added</h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--muted)]">Held-out model estimate of V(after) − V(before). This is observational possession value, not causal impact or pass-success probability.</p>
        </div>
        <ReliabilityBadge reliable={profile.attacking_value_reliable} />
      </div>
      <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        <MetricCard label="Action value / 100" value={value(profile.attacking_value_per_100_actions)} />
        <MetricCard label="Pass value / 100" value={value(profile.pass_value_per_100_passes)} />
        <MetricCard label="Carry value / 100" value={value(profile.carry_value_per_100_carries)} />
        <MetricCard label="Progressive value / 100" value={value(profile.progressive_value_per_100_actions)} />
        <MetricCard label="Pressure value / 100" value={value(profile.pressure_value_per_100_actions)} />
      </div>
      <p className="mt-3 text-xs text-[var(--muted)]">{profile.actions.toLocaleString()} actions · {profile.passes.toLocaleString()} passes · {profile.carries.toLocaleString()} carries · {formatPercent(profile.positive_value_action_rate)} positive-value actions. Pass and carry reliability are evaluated separately.</p>
    </section>
  );
}
