import Link from "next/link";

import { ReliabilityBadge } from "@/components/ReliabilityBadge";
import { EmptyState } from "@/components/States";
import { formatCount, formatPercent, formatPercentagePoints } from "@/lib/format";
import type { PlayerSummary } from "@/lib/types";

export function PlayerTable({ players }: { players: PlayerSummary[] }) {
  if (players.length === 0) {
    return (
      <EmptyState
        title="No players match these filters"
        message="Try broadening the team, position, or minimum-attempt criteria."
      />
    );
  }
  return (
    <div className="overflow-x-auto rounded-2xl border border-[var(--border)] bg-[var(--panel)]">
      <table className="w-full min-w-[980px] border-collapse text-left text-sm">
        <thead className="border-b border-[var(--border)] text-xs tracking-wide text-[var(--muted)] uppercase">
          <tr>
            <th className="px-5 py-4 font-medium">Player</th>
            <th className="px-4 py-4 font-medium">Position</th>
            <th className="px-4 py-4 text-right font-medium">Passes</th>
            <th className="px-4 py-4 text-right font-medium">Actual</th>
            <th className="px-4 py-4 text-right font-medium">Pass difficulty</th>
            <th className="px-4 py-4 text-right font-medium">Actual vs. expected</th>
            <th className="px-5 py-4 text-right font-medium">Progressive rate</th>
          </tr>
        </thead>
        <tbody>
          {players.map((player) => (
            <tr key={player.player_id} className="border-b border-[var(--border)] last:border-0 hover:bg-white/[0.025]">
              <td className="px-5 py-4">
                <Link
                  href={`/players/${player.player_id}`}
                  className="font-semibold text-white hover:text-[var(--accent-strong)]"
                >
                  {player.player_name}
                </Link>
                <p className="mt-1 text-xs text-[var(--muted)]">{player.team_name}</p>
                <div className="mt-2">
                  <ReliabilityBadge reliable={player.overall_reliable} />
                </div>
              </td>
              <td className="px-4 py-4">
                <span>{player.position}</span>
                <span className="ml-2 rounded bg-white/5 px-2 py-1 text-xs text-[var(--muted)]">
                  {player.position_group}
                </span>
              </td>
              <td className="metric-tabular px-4 py-4 text-right">{formatCount(player.pass_attempts)}</td>
              <td className="metric-tabular px-4 py-4 text-right">{formatPercent(player.actual_completion_rate)}</td>
              <td className="metric-tabular px-4 py-4 text-right text-[var(--muted)]">{formatPercent(player.expected_completion_rate)}</td>
              <td className="metric-tabular px-4 py-4 text-right font-semibold text-[var(--accent-strong)]">
                {formatPercentagePoints(player.completion_above_expected_pp)}
              </td>
              <td className="metric-tabular px-5 py-4 text-right">{formatPercent(player.progressive_pass_rate)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
