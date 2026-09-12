"use client";

import { DARK_THEME, RadarChart, ThemeProvider } from "@withqwerty/campos-react";

import {
  formatDecimal,
  formatPercent,
  formatPercentagePoints,
  formatPercentile,
} from "@/lib/format";
import type { PlayerIntelligenceMetric, PlayerIntelligenceResponse } from "@/lib/types";
import { PlayerArchetypeCard } from "@/components/PlayerArchetypeCard";
import { displayMetricLabel } from "@/lib/terminology";

function rawValue(metric: PlayerIntelligenceMetric): string {
  if (metric.raw_value === null) return "—";
  if (metric.unit === "rate") return formatPercent(metric.raw_value);
  if (metric.unit === "percentage_points") return formatPercentagePoints(metric.raw_value);
  if (metric.unit === "per_match") return `${formatDecimal(metric.raw_value, 2)} / match`;
  if (metric.unit === "distance") return `${formatDecimal(metric.raw_value, 1)} units`;
  if (metric.unit === "distance_per_100") return `${formatDecimal(metric.raw_value, 1)} units`;
  return formatDecimal(metric.raw_value, 3);
}

export function PlayerIntelligenceSection({ profile }: { profile: PlayerIntelligenceResponse }) {
  const radarRows = profile.radar_metrics.map((metric) => ({
    metric: displayMetricLabel(metric.label),
    value: metric.percentile ?? 0,
    percentile: metric.percentile ?? undefined,
    category: metric.family === "style" ? "Style" : "Performance",
    displayValue: rawValue(metric),
  }));
  return (
    <section aria-labelledby="player-intelligence-title" className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-5 sm:p-7">
      <div>
        <p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">Player intelligence</p>
        <h2 id="player-intelligence-title" className="mt-2 text-2xl font-semibold">Position profile</h2>
        <p className="mt-2 text-sm leading-6 text-[var(--muted)]">
          {profile.position_group} context · {profile.percentile_context}
        </p>
      </div>
      <div className="mt-6 grid items-start gap-7 xl:grid-cols-[minmax(360px,0.95fr)_1.05fr]">
        <div data-testid="position-profile-radar" className="mx-auto w-full max-w-[640px] min-w-0 rounded-xl border border-white/10 bg-[#0b1210] p-2 sm:p-3">
          {radarRows.length >= 3 ? (
            <ThemeProvider value={DARK_THEME}>
              <RadarChart
                rows={radarRows}
                valueMode="percentile"
                ringStyle="banded-inside-polygon"
                showLegend
                showVertexMarkers
                areas={{ fill: "#74d6a2", stroke: "#9aebbd", fillOpacity: 0.2 }}
                methodologyNotes={{
                  below: "0–100 empirical percentiles among eligible same-position peers.",
                }}
              />
            </ThemeProvider>
          ) : (
            <div className="grid min-h-64 place-items-center p-6 text-center text-sm leading-6 text-[var(--muted)]">
              {profile.radar_status}
            </div>
          )}
        </div>
        <div className="grid gap-7">
          <MetricGroup title="Style" note="Tendencies describe what a player attempts; higher is not automatically better." metrics={profile.style_metrics} />
          <MetricGroup title="Performance" note="Observed execution or model-derived output relative to opportunities." metrics={profile.performance_metrics} />
        </div>
      </div>
      <div className="mt-7">
        <PlayerArchetypeCard archetype={profile.archetype} />
      </div>
    </section>
  );
}

function MetricGroup({ title, note, metrics }: { title: string; note: string; metrics: PlayerIntelligenceMetric[] }) {
  return (
    <section aria-label={`${title} percentiles`}>
      <h3 className="text-lg font-semibold">{title}</h3>
      <p className="mt-1 text-xs leading-5 text-[var(--muted)]">{note}</p>
      <div className="mt-3 grid gap-2 sm:grid-cols-2">
        {metrics.map((metric) => (
          <article key={metric.metric_name} className="rounded-lg border border-[var(--border)] bg-black/15 p-3">
            <p className="text-sm font-medium">{displayMetricLabel(metric.label)}</p>
            <p className="metric-tabular mt-2 text-lg font-semibold">{rawValue(metric)}</p>
            <p className={metric.percentile === null ? "mt-1 text-xs text-[var(--muted)]" : "mt-1 text-xs text-[var(--accent-strong)]"}>
              {formatPercentile(metric.percentile)} among {metric.peer_position_group} players
            </p>
            <p className="mt-1 text-[11px] text-[var(--muted)]">
              n = {metric.peer_count} eligible peers · sample {metric.sample_count}
            </p>
            {!metric.eligible ? <p className="mt-1 text-[11px] text-amber-100/70">{metric.eligibility_reason.replaceAll("_", " ")}</p> : null}
            {metric.stability_note ? <p className="mt-2 text-[11px] leading-4 text-[var(--muted)]">{metric.stability_note}</p> : null}
          </article>
        ))}
      </div>
    </section>
  );
}
