import { formatCount, formatPercent, formatPercentagePoints } from "@/lib/format";
import type { PlayerProfileResponse } from "@/lib/types";

export function ExpectedVsActual({ player }: { player: PlayerProfileResponse }) {
  return (
    <section className="overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--panel)]">
      <div className="grid md:grid-cols-[1fr_1fr_1.15fr]">
        <div className="border-b border-[var(--border)] p-6 md:border-r md:border-b-0">
          <p className="text-sm text-[var(--muted)]">Actual completion</p>
          <p className="metric-tabular mt-3 text-4xl font-semibold">{formatPercent(player.actual_completion_rate)}</p>
          <p className="mt-2 text-xs text-[var(--muted)]">{formatCount(player.passes_completed)} completed passes</p>
        </div>
        <div className="border-b border-[var(--border)] p-6 md:border-r md:border-b-0">
          <p className="text-sm text-[var(--muted)]">Pass difficulty</p>
          <p className="metric-tabular mt-3 text-4xl font-semibold">{formatPercent(player.expected_completion_rate)}</p>
          <p className="mt-2 text-xs text-[var(--muted)]">How difficult a pass was to complete.</p>
        </div>
        <div className="bg-emerald-300/[0.045] p-6">
          <p className="text-sm text-[var(--muted)]">Passing vs. expected</p>
          <p className="metric-tabular mt-3 text-4xl font-semibold text-[var(--accent-strong)]">{formatPercentagePoints(player.completion_above_expected_pp)}</p>
          <p className="metric-tabular mt-2 text-xs text-[var(--muted)]">{player.completions_above_expected > 0 ? "+" : ""}{player.completions_above_expected.toFixed(1)} completions vs model expectation</p>
        </div>
      </div>
      <p className="border-t border-[var(--border)] px-6 py-4 text-sm leading-6 text-[var(--muted)]">Whether a player completed more or fewer passes than expected.</p>
    </section>
  );
}
