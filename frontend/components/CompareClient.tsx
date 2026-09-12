"use client";

import { useState } from "react";

import { ComparisonMetric } from "@/components/ComparisonMetric";
import { PlayerAutocomplete } from "@/components/PlayerAutocomplete";
import { ErrorState } from "@/components/States";
import { api } from "@/lib/api";
import { formatDecimal, formatPercent, formatPercentagePoints, formatPercentile } from "@/lib/format";
import { displayMetricLabel } from "@/lib/terminology";
import type { ComparisonResponse, PlayerIdentity, PlayerIntelligenceResponse, PlayerProfileResponse, PlayerSummary } from "@/lib/types";

export function CompareClient({ initial }: { initial: ComparisonResponse | null }) {
  const [left, setLeft] = useState<PlayerIdentity | null>(initial?.players[0] ?? null);
  const [right, setRight] = useState<PlayerIdentity | null>(initial?.players[1] ?? null);
  const [comparison, setComparison] = useState(initial);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function compare(nextLeft: PlayerIdentity | null, nextRight: PlayerIdentity | null) {
    if (!nextLeft || !nextRight || nextLeft.player_id === nextRight.player_id) return;
    setLoading(true);
    setError(null);
    try {
      const response = await api.comparePlayers([nextLeft.player_id, nextRight.player_id]);
      setComparison(response);
      window.history.replaceState(null, "", `/compare?player_ids=${nextLeft.player_id},${nextRight.player_id}`);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Comparison unavailable.");
    } finally {
      setLoading(false);
    }
  }

  function selectLeft(player: PlayerSummary) {
    if (player.player_id === right?.player_id) return;
    setLeft(player);
    void compare(player, right);
  }
  function selectRight(player: PlayerSummary) {
    if (player.player_id === left?.player_id) return;
    setRight(player);
    void compare(left, player);
  }

  const players = comparison?.players;
  return (
    <div>
      <div className="grid gap-5 rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-5 md:grid-cols-2">
        <PlayerAutocomplete label="Player A" initialValue={left?.player_name} selected={left} excludedPlayerIds={right ? [right.player_id] : []} onSelect={selectLeft} />
        <PlayerAutocomplete label="Player B" initialValue={right?.player_name} selected={right} excludedPlayerIds={left ? [left.player_id] : []} onSelect={selectRight} />
      </div>
      {left && right && left.player_id === right.player_id ? <p role="alert" className="mt-4 text-sm text-amber-200">Choose two different players.</p> : null}
      {loading ? <p aria-live="polite" className="mt-6 text-sm text-[var(--muted)]">Loading comparison…</p> : null}
      {error ? <div className="mt-6"><ErrorState title="Comparison unavailable" message={error} /></div> : null}
      {!error && players ? <ComparisonResults left={players[0]} right={players[1]} attacking={comparison.attacking ?? []} intelligence={comparison.intelligence ?? []} sameGroup={comparison.comparison.same_position_group} /> : !loading && !error ? <p className="mt-8 rounded-2xl border border-dashed border-[var(--border)] p-8 text-center text-sm text-[var(--muted)]">Select two players to compare their raw FootyScout passing metrics.</p> : null}
    </div>
  );
}

function ComparisonResults({ left, right, attacking, intelligence, sameGroup }: { left: PlayerProfileResponse; right: PlayerProfileResponse; attacking: ComparisonResponse["attacking"]; intelligence: ComparisonResponse["intelligence"]; sameGroup: boolean }) {
  const metrics: Array<[string, number | null, number | null, (value: number | null) => string]> = [
    ["Actual completion", left.actual_completion_rate, right.actual_completion_rate, formatPercent],
    ["Pass difficulty", left.expected_completion_rate, right.expected_completion_rate, formatPercent],
    ["Actual vs. expected passing", left.completion_above_expected_pp, right.completion_above_expected_pp, formatPercentagePoints],
    ["Under-pressure above expected", left.pressure_above_expected_pp, right.pressure_above_expected_pp, formatPercentagePoints],
    ["Progressive above expected", left.progressive_above_expected_pp, right.progressive_above_expected_pp, formatPercentagePoints],
    ["Long-pass above expected", left.long_pass_above_expected_pp, right.long_pass_above_expected_pp, formatPercentagePoints],
    ["Progressive-pass rate", left.progressive_pass_rate, right.progressive_pass_rate, formatPercent],
    ["Pressure-pass rate", left.pressure_pass_rate, right.pressure_pass_rate, formatPercent],
    ["Average forward distance", left.average_forward_distance, right.average_forward_distance, (value) => value === null ? "—" : `${formatDecimal(value)} m`],
    ["Positive forward distance / 100", left.positive_forward_distance_per_100_passes, right.positive_forward_distance_per_100_passes, (value) => value === null ? "—" : `${formatDecimal(value)} m`],
    ["Final-third entries / 100", left.final_third_entries_per_100_passes, right.final_third_entries_per_100_passes, formatDecimal],
  ];
  return (
    <section className="mt-8" aria-label="Player comparison">
      <div className="grid grid-cols-2 gap-4 rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-5">
        {[left, right].map((player, index) => <header key={player.player_id} className={index ? "text-right" : ""}><p className="text-xs text-[var(--muted)]">{index ? "Player B" : "Player A"}</p><h2 className="mt-1 text-xl font-semibold">{player.player_name}</h2><p className="mt-1 text-sm text-[var(--muted)]">{player.team_name} · {player.position}</p></header>)}
      </div>
      {!sameGroup ? <p className="mt-3 text-xs text-amber-100/70">Cross-position comparison: interpret role-dependent metrics with care.</p> : null}
      <div className="mt-5 grid gap-3 lg:grid-cols-2">{metrics.map(([label, a, b, formatter]) => <ComparisonMetric key={label} label={label} left={a} right={b} format={formatter} />)}</div>
      {attacking.length === 2 ? <><h3 className="mt-10 text-lg font-semibold">Attacking impact</h3><div className="mt-4 grid gap-3 lg:grid-cols-2">{[
        ["Action value / 100", attacking[0]?.attacking_value_per_100_actions ?? null, attacking[1]?.attacking_value_per_100_actions ?? null],
        ["Pass value / 100", attacking[0]?.pass_value_per_100_passes ?? null, attacking[1]?.pass_value_per_100_passes ?? null],
        ["Carry value / 100", attacking[0]?.carry_value_per_100_carries ?? null, attacking[1]?.carry_value_per_100_carries ?? null],
        ["Progressive value / 100", attacking[0]?.progressive_value_per_100_actions ?? null, attacking[1]?.progressive_value_per_100_actions ?? null],
        ["Pressure value / 100", attacking[0]?.pressure_value_per_100_actions ?? null, attacking[1]?.pressure_value_per_100_actions ?? null],
      ].map(([label, a, b]) => <ComparisonMetric key={String(label)} label={String(label)} left={a as number | null} right={b as number | null} format={(value) => formatDecimal(value, 3)} />)}</div></> : null}
      {intelligence.length === 2 && intelligence[0] && intelligence[1] ? (
        <>
          <ArchetypeComparison left={intelligence[0]} right={intelligence[1]} />
          <PositionPercentiles
            left={intelligence[0]}
            right={intelligence[1]}
            sameGroup={sameGroup}
          />
        </>
      ) : null}
    </section>
  );
}

export function ArchetypeComparison({ left, right }: { left: PlayerIntelligenceResponse; right: PlayerIntelligenceResponse }) {
  return (
    <section aria-label="Playing-style archetypes" className="mt-10">
      <h3 className="text-lg font-semibold">Playing style</h3>
      <p className="mt-2 text-xs leading-5 text-[var(--muted)]">Archetypes describe position-relative tendencies, not player quality or overall similarity.</p>
      <div className="mt-4 grid gap-3 md:grid-cols-2">
        {[left, right].map((profile, index) => (
          <article key={profile.player.player_id} className="rounded-xl border border-[var(--border)] bg-[var(--panel)] p-4">
            <p className="text-xs text-[var(--muted)]">{index ? "Player B" : "Player A"} · {profile.position_group} peers</p>
            <p className="mt-2 text-lg font-semibold">{profile.archetype.name ?? "Archetype unavailable"}</p>
            {!profile.archetype.eligible ? <p className="mt-2 text-xs leading-5 text-[var(--muted)]">{profile.archetype.eligibility_reason}</p> : null}
            {profile.archetype.separation_margin !== null ? <p className="metric-tabular mt-2 text-xs text-[var(--muted)]">Style separation {formatDecimal(profile.archetype.separation_margin, 3)}</p> : null}
          </article>
        ))}
      </div>
    </section>
  );
}

export function PositionPercentiles({ left, right, sameGroup }: { left: PlayerIntelligenceResponse; right: PlayerIntelligenceResponse; sameGroup: boolean }) {
  const preferred = [
    "completion_above_expected_pp", "progressive_pass_rate", "pressure_pass_rate",
    "long_pass_rate", "progressive_carry_rate", "carry_value_per_100_carries",
    "shots_per_match_observed",
  ];
  const leftByName = new Map([...left.style_metrics, ...left.performance_metrics].map((metric) => [metric.metric_name, metric]));
  const rightByName = new Map([...right.style_metrics, ...right.performance_metrics].map((metric) => [metric.metric_name, metric]));
  return (
    <section aria-label="Position percentiles" className="mt-10">
      <h3 className="text-lg font-semibold">Position percentiles</h3>
      <p className="mt-2 text-xs leading-5 text-[var(--muted)]">
        {sameGroup
          ? `Both players are ranked against eligible ${left.position_group} peers.`
          : `Different-position context: ${left.player.player_name} is ranked among ${left.position_group} peers and ${right.player.player_name} among ${right.position_group} peers. These are not the same peer distribution.`}
        {" "}Percentiles are not ratings.
      </p>
      <div className="mt-4 overflow-hidden rounded-xl border border-[var(--border)]">
        <div className="grid grid-cols-[minmax(130px,1fr)_minmax(100px,0.7fr)_minmax(100px,0.7fr)] bg-black/20 px-4 py-3 text-xs text-[var(--muted)]">
          <span>Metric</span><span>{left.player.player_name}</span><span>{right.player.player_name}</span>
        </div>
        {preferred.map((name) => {
          const first = leftByName.get(name);
          const second = rightByName.get(name);
          const label = displayMetricLabel(first?.label ?? second?.label ?? name);
          return (
            <div key={name} className="grid grid-cols-[minmax(130px,1fr)_minmax(100px,0.7fr)_minmax(100px,0.7fr)] border-t border-[var(--border)] px-4 py-3 text-sm">
              <span>{label}</span>
              <span className="metric-tabular">{formatPercentile(first?.percentile ?? null)} <small className="block text-[10px] text-[var(--muted)]">n = {first?.peer_count ?? 0}</small></span>
              <span className="metric-tabular">{formatPercentile(second?.percentile ?? null)} <small className="block text-[10px] text-[var(--muted)]">n = {second?.peer_count ?? 0}</small></span>
            </div>
          );
        })}
      </div>
    </section>
  );
}
