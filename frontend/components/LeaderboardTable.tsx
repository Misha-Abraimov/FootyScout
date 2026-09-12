import Link from "next/link";

import { EmptyState } from "@/components/States";
import { formatCount, formatDecimal, formatPercent, formatPercentagePoints } from "@/lib/format";
import type { LeaderboardEntry, LeaderboardMetric } from "@/lib/types";

export const leaderboardLabels: Record<LeaderboardMetric, string> = {
  completion_above_expected_pp: "Actual vs. expected",
  pressure_above_expected_pp: "Under-pressure above expected",
  progressive_above_expected_pp: "Progressive above expected",
  long_pass_above_expected_pp: "Long-pass above expected",
  final_third_entries_per_100_passes: "Final-third entries / 100",
  expected_completion_rate: "Pass difficulty",
  progressive_pass_rate: "Progressive-pass rate",
  attacking_value_per_100_actions: "Attacking impact / 100 actions",
  pass_value_per_100_passes: "Pass value / 100 passes",
  carry_value_per_100_carries: "Carry value / 100 carries",
  progressive_value_per_100_actions: "Progressive value / 100 actions",
  pressure_value_per_100_actions: "Pressure value / 100 actions",
};

export function formatLeaderboardValue(metric: LeaderboardMetric, value: number): string {
  if (metric.endsWith("_pp")) return formatPercentagePoints(value);
  if (metric.endsWith("_rate")) return formatPercent(value);
  return formatDecimal(value);
}

export function LeaderboardTable({ entries }: { entries: LeaderboardEntry[] }) {
  if (entries.length === 0) {
    return <EmptyState title="No eligible players" message="No reliable player sample matches these filters." />;
  }
  return (
    <div className="overflow-x-auto rounded-2xl border border-[var(--border)] bg-[var(--panel)]">
      <table className="w-full min-w-[760px] border-collapse text-left text-sm">
        <thead className="border-b border-[var(--border)] text-xs tracking-wide text-[var(--muted)] uppercase">
          <tr>
            <th className="px-5 py-4 font-medium">Rank</th>
            <th className="px-4 py-4 font-medium">Player</th>
            <th className="px-4 py-4 font-medium">Position</th>
            <th className="px-4 py-4 text-right font-medium">Sample</th>
            <th className="px-5 py-4 text-right font-medium">Value</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((entry) => (
            <tr key={entry.player_id} className="border-b border-[var(--border)] last:border-0 hover:bg-white/[0.025]">
              <td className="metric-tabular px-5 py-4 text-lg font-semibold text-[var(--muted)]">{entry.rank}</td>
              <td className="px-4 py-4">
                <Link href={`/players/${entry.player_id}`} className="font-semibold hover:text-[var(--accent-strong)]">
                  {entry.player_name}
                </Link>
                <p className="mt-1 text-xs text-[var(--muted)]">{entry.team_name}</p>
              </td>
              <td className="px-4 py-4">{entry.position}</td>
              <td className="metric-tabular px-4 py-4 text-right text-[var(--muted)]">{formatCount(entry.relevant_attempts)} attempts</td>
              <td className="metric-tabular px-5 py-4 text-right text-lg font-semibold text-[var(--accent-strong)]">
                {formatLeaderboardValue(entry.metric, entry.metric_value)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
