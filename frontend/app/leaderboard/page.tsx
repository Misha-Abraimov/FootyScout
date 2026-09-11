import type { Metadata } from "next";

import { LeaderboardTable, leaderboardLabels } from "@/components/LeaderboardTable";
import { PageHeader } from "@/components/PageHeader";
import { ErrorState } from "@/components/States";
import { api } from "@/lib/api";
import type { LeaderboardMetric, PositionGroup } from "@/lib/types";

export const dynamic = "force-dynamic";
export const metadata: Metadata = { title: "Leaderboard" };

type SearchParams = Promise<Record<string, string | string[] | undefined>>;
const one = (value: string | string[] | undefined) => Array.isArray(value) ? value[0] : value;

export default async function LeaderboardPage({ searchParams }: { searchParams: SearchParams }) {
  const raw = await searchParams;
  const metric = (one(raw.metric) as LeaderboardMetric | undefined) ?? "completion_above_expected_pp";
  const positionGroup = one(raw.position_group) as PositionGroup | undefined;
  const team = one(raw.team);
  const result = await Promise.all([
      api.getLeaderboard({ metric, position_group: positionGroup, team, limit: 100 }),
      api.getMeta(),
    ]).catch(() => null);
  if (!result) {
    return <main className="mx-auto min-h-[70vh] max-w-7xl px-5 py-16 sm:px-8"><ErrorState /></main>;
  }
  const [leaderboard, meta] = result;
  return (
      <main className="mx-auto min-h-screen max-w-7xl px-5 py-10 sm:px-8 sm:py-14">
        <PageHeader eyebrow="Reliable samples" title="Player leaderboards" description="Rank passing execution, progression, or attacking possession value. Metric-specific reliability thresholds remain authoritative in the API." />
        <form action="/leaderboard" className="mt-8 grid gap-4 rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-5 md:grid-cols-4">
          <label className="grid gap-2 text-sm md:col-span-2"><span className="text-[var(--muted)]">Metric</span><select name="metric" defaultValue={metric} className="rounded-lg border border-[var(--border)] bg-[#0a100e] px-3 py-2.5">{Object.entries(leaderboardLabels).filter(([key]) => key !== "expected_completion_rate").map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
          <label className="grid gap-2 text-sm"><span className="text-[var(--muted)]">Position</span><select name="position_group" defaultValue={positionGroup ?? ""} className="rounded-lg border border-[var(--border)] bg-[#0a100e] px-3 py-2.5"><option value="">All positions</option>{meta.position_groups.map((group) => <option key={group}>{group}</option>)}</select></label>
          <label className="grid gap-2 text-sm"><span className="text-[var(--muted)]">Team</span><select name="team" defaultValue={team ?? ""} className="rounded-lg border border-[var(--border)] bg-[#0a100e] px-3 py-2.5"><option value="">All teams</option>{meta.teams.map((item) => <option key={item}>{item}</option>)}</select></label>
          <button className="rounded-lg bg-[var(--accent)] px-4 py-2.5 text-sm font-semibold text-[#07110d] md:col-start-4">Apply filters</button>
        </form>
        <div className="mt-6"><LeaderboardTable entries={leaderboard.items} /></div>
        <p className="mt-4 text-xs text-[var(--muted)]">Available StatsBomb matches; metric-specific sample thresholds apply. {leaderboard.total} eligible players{leaderboard.minimum_pass_attempts ? ` · minimum ${leaderboard.minimum_pass_attempts} passes` : ""}.</p>
      </main>
  );
}
