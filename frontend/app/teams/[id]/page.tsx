import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { MetricCard } from "@/components/MetricCard";
import { PageHeader } from "@/components/PageHeader";
import { ErrorState } from "@/components/States";
import { TeamRoleProfile } from "@/components/TeamRoleProfile";
import { ApiError, api } from "@/lib/api";
import { formatCount, formatDecimal, formatPercent, formatSignedDecimal } from "@/lib/format";

export const dynamic = "force-dynamic";
export const metadata: Metadata = { title: "Team Intelligence" };

export default async function TeamIntelligencePage({ params }: { params: Promise<{ id: string }> }) {
  const teamId = Number((await params).id);
  if (!Number.isInteger(teamId)) notFound();
  const intelligence = await api.getTeamIntelligence(teamId).catch((error: unknown) => {
    if (error instanceof ApiError && error.status === 404) notFound();
    return null;
  });
  if (!intelligence) return <main className="mx-auto min-h-[70vh] max-w-7xl px-5 py-16 sm:px-8"><ErrorState /></main>;
  const { team, roles } = intelligence;
  return (
    <main className="mx-auto min-h-screen max-w-7xl space-y-12 px-5 py-10 sm:px-8 sm:py-14">
      <PageHeader eyebrow="Team Intelligence" title={team.team_name} description={team.sample_scope} />

      <section aria-labelledby="team-overview-title">
        <h2 id="team-overview-title" className="text-2xl font-semibold">Team style overview</h2>
        <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard label="Matches observed" value={formatCount(team.matches_observed)} note={`${formatCount(team.contributors)} contributors`} />
          <MetricCard label="Pass difficulty" value={formatPercent(team.metrics.expected_completion_rate)} />
          <MetricCard label="Progressive-pass rate" value={formatPercent(team.metrics.progressive_pass_rate)} />
          <MetricCard label="Carry share" value={formatPercent(team.metrics.carry_share_of_actions)} />
        </div>
      </section>

      <section aria-labelledby="passing-title">
        <p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">Passing & progression</p>
        <h2 id="passing-title" className="mt-2 text-2xl font-semibold">How the observed team moved the ball</h2>
        <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard label="Under-pressure passes" value={formatPercent(team.metrics.pressure_pass_rate)} />
          <MetricCard label="Long-pass rate" value={formatPercent(team.metrics.long_pass_rate)} />
          <MetricCard label="Positive forward / 100" value={formatDecimal(team.metrics.positive_forward_distance_per_100_passes, 1)} />
          <MetricCard label="Final-third entries / 100" value={formatDecimal(team.metrics.final_third_entries_per_100_passes, 1)} />
        </div>
      </section>

      <section aria-labelledby="attacking-title">
        <p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">Carrying & attacking</p>
        <h2 id="attacking-title" className="mt-2 text-2xl font-semibold">Observed attacking description</h2>
        <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard label="Progressive-carry rate" value={formatPercent(team.metrics.progressive_carry_rate)} />
          <MetricCard label="Shots / match" value={formatDecimal(team.metrics.shots_per_match, 2)} />
          <MetricCard label="xG / shot" value={formatDecimal(team.metrics.xg_per_shot, 3)} />
          <MetricCard label="Overall impact / 100 actions" value={formatSignedDecimal(team.metrics.attacking_value_per_100_actions, 3)} />
        </div>
        <p className="mt-4 text-xs leading-5 text-[var(--muted)]">Attacking and value metrics describe the observed sample. They are not inputs to Role Fit.</p>
      </section>

      <section aria-labelledby="roles-title">
        <p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">Positional roles</p>
        <h2 id="roles-title" className="mt-2 text-2xl font-semibold">DEF, MID and FWD role profiles</h2>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--muted)]">Each profile describes the pooled observed style of Leverkusen players assigned to that broad position. It does not claim coaching intent or define what the club requires.</p>
        <div className="mt-6 grid gap-6">{roles.map((role) => <TeamRoleProfile key={role.position_group} role={role} />)}</div>
      </section>
    </main>
  );
}
