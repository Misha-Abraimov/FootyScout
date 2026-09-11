import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { ExpectedVsActual } from "@/components/ExpectedVsActual";
import { AttackingValueSection } from "@/components/AttackingValueSection";
import { MetricCard } from "@/components/MetricCard";
import { PassingSubsetCard } from "@/components/PassingSubsetCard";
import { PlayerIntelligenceSection } from "@/components/PlayerIntelligenceSection";
import { PlayerRoleFitSection } from "@/components/PlayerRoleFitSection";
import { ReliabilityBadge } from "@/components/ReliabilityBadge";
import { FootyScoutPassMap } from "@/components/FootyScoutPassMap";
import { FootyScoutShotMap } from "@/components/FootyScoutShotMap";
import { FootyScoutCarryMap } from "@/components/FootyScoutCarryMap";
import { SimilarPlayerCard } from "@/components/SimilarPlayerCard";
import { ShootingProfileSection } from "@/components/ShootingProfileSection";
import { EmptyState, ErrorState } from "@/components/States";
import { ApiError, api } from "@/lib/api";
import { formatCount, formatDecimal, formatPercent } from "@/lib/format";

export const dynamic = "force-dynamic";
export const metadata: Metadata = { title: "Player profile" };

export default async function PlayerPage({ params }: { params: Promise<{ id: string }> }) {
  const playerId = Number((await params).id);
  if (!Number.isInteger(playerId)) notFound();

  const result = await Promise.all([
      api.getPlayer(playerId),
      api.getSimilarPlayers(playerId, 6),
      api.getPlayerPasses(playerId, { limit: 200, offset: 0 }),
      api.getPlayerShooting(playerId).catch((error: unknown) => {
        if (error instanceof ApiError && error.status === 404) return null;
        throw error;
      }),
      api.getPlayerShots(playerId, { limit: 200, offset: 0 }),
      api.getPlayerAttacking(playerId).catch((error: unknown) => {
        if (error instanceof ApiError && error.status === 404) return null;
        throw error;
      }),
      api.getPlayerActions(playerId, { action_type: "Carry", limit: 200, offset: 0 }),
      api.getPlayerIntelligence(playerId).catch((error: unknown) => {
        if (error instanceof ApiError && error.status === 404) return null;
        throw error;
      }),
      api.getPlayerRoleFit(playerId),
    ]).catch((error: unknown) => {
      if (error instanceof ApiError && error.status === 404) notFound();
      return null;
    });
  if (!result) {
    return <main className="mx-auto min-h-[70vh] max-w-7xl px-5 py-16 sm:px-8"><ErrorState /></main>;
  }
  const [player, similar, passes, shooting, shots, attacking, carries, intelligence, roleFit] = result;
  return (
      <main className="mx-auto min-h-screen max-w-7xl space-y-10 px-5 py-10 sm:px-8 sm:py-14">
        <header className="flex flex-col gap-6 border-b border-[var(--border)] pb-8 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs font-semibold tracking-[0.18em] text-[var(--accent)] uppercase">Player profile</p>
            <h1 className="mt-3 text-4xl font-semibold tracking-[-0.04em] sm:text-5xl">{player.player_name}</h1>
            <p className="mt-3 text-[var(--muted)]">{player.team_name} · {player.position} · {player.position_group}</p>
          </div>
          <div className="flex flex-wrap items-center gap-5 text-sm">
            <div><span className="block text-xs text-[var(--muted)]">Matches observed</span><strong className="metric-tabular mt-1 block text-lg">{formatCount(player.matches_observed)}</strong></div>
            <div><span className="block text-xs text-[var(--muted)]">Pass attempts</span><strong className="metric-tabular mt-1 block text-lg">{formatCount(player.pass_attempts)}</strong></div>
            <ReliabilityBadge reliable={player.overall_reliable} />
          </div>
        </header>

        <ExpectedVsActual player={player} />

        {intelligence ? <PlayerIntelligenceSection profile={intelligence} /> : null}

        <PlayerRoleFitSection fit={roleFit} />

        <ShootingProfileSection profile={shooting} />

        {shooting ? <FootyScoutShotMap player={player} shots={shots} /> : null}

        <AttackingValueSection profile={attacking} />

        {attacking ? <FootyScoutCarryMap player={player} data={carries} /> : null}

        <section aria-labelledby="situational-title">
          <h2 id="situational-title" className="text-2xl font-semibold">Situational execution</h2>
          <div className="mt-5 grid gap-4 lg:grid-cols-3">
            <PassingSubsetCard title="Under pressure" attempts={player.pressure_attempts} actual={player.pressure_actual_completion_rate} expected={player.pressure_expected_completion_rate} aboveExpected={player.pressure_above_expected_pp} reliable={player.pressure_reliable} />
            <PassingSubsetCard title="Progressive passing" attempts={player.progressive_attempts} actual={player.progressive_actual_completion_rate} expected={player.progressive_expected_completion_rate} aboveExpected={player.progressive_above_expected_pp} reliable={player.progressive_reliable} />
            <PassingSubsetCard title="Long passing" attempts={player.long_pass_attempts} actual={player.long_pass_actual_completion_rate} expected={player.long_pass_expected_completion_rate} aboveExpected={player.long_pass_above_expected_pp} reliable={player.long_pass_reliable} />
          </div>
        </section>

        <section aria-labelledby="progression-title">
          <h2 id="progression-title" className="text-2xl font-semibold">Progression profile</h2>
          <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
            <MetricCard label="Progressive-pass rate" value={formatPercent(player.progressive_pass_rate)} />
            <MetricCard label="Pressure-pass rate" value={formatPercent(player.pressure_pass_rate)} />
            <MetricCard label="Average forward distance" value={`${formatDecimal(player.average_forward_distance)} m`} />
            <MetricCard label="Positive forward distance / 100" value={`${formatDecimal(player.positive_forward_distance_per_100_passes)} m`} />
            <MetricCard label="Final-third entries / 100" value={formatDecimal(player.final_third_entries_per_100_passes)} />
          </div>
        </section>

        <FootyScoutPassMap player={player} initialData={passes} />

        <section aria-labelledby="similar-title">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
            <div><h2 id="similar-title" className="text-2xl font-semibold">Similar playing styles</h2><p className="mt-2 max-w-3xl text-sm text-[var(--muted)]">Similarity compares position-relative playing style across passing and carrying tendencies. Scores are not ability ratings.</p></div>
            <p className="text-xs text-[var(--muted)]">100 means identical observed profiles; 50 is about the median same-position distance. Not a probability.</p>
          </div>
          {similar.query_sample_support === "limited" && similar.available ? <p className="mt-4 rounded-xl border border-[var(--border)] bg-[var(--panel)] px-4 py-3 text-sm text-[var(--muted)]"><span className="font-medium text-[var(--foreground)]">Limited sample:</span> this player&apos;s style profile is based on {similar.query_matches_observed} observed {similar.query_matches_observed === 1 ? "match" : "matches"}, so nearest-neighbor rankings may be less stable.</p> : null}
          {similar.items.length ? <div className="mt-5 grid gap-4 lg:grid-cols-2">{similar.items.map((item) => <SimilarPlayerCard key={item.similar_player_id} player={item} />)}</div> : <div className="mt-5"><EmptyState title="Playing-style similarity unavailable" message={similar.unavailable_reason ?? "No eligible same-position comparisons are available for this player."} /></div>}
          <p className="mt-4 text-xs leading-5 text-[var(--muted)]">Sample support uses the lower observed-match count in each pair. Lower coverage means a measured profile may vary more with additional matches; it never changes the similarity score or ranking.</p>
        </section>
      </main>
  );
}
