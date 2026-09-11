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
          <p className="text-sm text-[var(--muted)]">Expected completion</p>
          <p className="metric-tabular mt-3 text-4xl font-semibold">{formatPercent(player.expected_completion_rate)}</p>
          <p className="mt-2 text-xs text-[var(--muted)]">Given the difficulty of all attempts</p>
        </div>
        <div className="bg-emerald-300/[0.045] p-6">
          <p className="text-sm text-[var(--muted)]">Completion above expected</p>
          <p className="metric-tabular mt-3 text-4xl font-semibold text-[var(--accent-strong)]">{formatPercentagePoints(player.completion_above_expected_pp)}</p>
          <p className="metric-tabular mt-2 text-xs text-[var(--muted)]">{player.completions_above_expected > 0 ? "+" : ""}{player.completions_above_expected.toFixed(1)} completions vs model expectation</p>
        </div>
      </div>
      <p className="border-t border-[var(--border)] px-6 py-4 text-sm leading-6 text-[var(--muted)]">Positive values indicate that the player completed more passes than the model expected given the difficulty of those attempts.</p>
    </section>
  );
}
