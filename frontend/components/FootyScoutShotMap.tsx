"use client";

import { DARK_THEME, ShotMap, ThemeProvider } from "@withqwerty/campos-react";
import { useMemo } from "react";

import { toCamposShots } from "@/lib/shot-map";
import type { PlayerIdentity, ShotListResponse } from "@/lib/types";

const GOAL_OUTLINE_COLOR = "var(--accent)";
const GOAL_OUTLINE_WIDTH = 0.9;

export function FootyScoutShotMap({
  player,
  shots,
}: {
  player: PlayerIdentity;
  shots: ShotListResponse;
}) {
  const camposShots = useMemo(() => toCamposShots(shots.items, player), [shots.items, player]);
  if (!shots.items.length) return null;
  return (
    <section aria-labelledby="shot-map-title" className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-4 sm:p-6">
      <p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">Campos visualization</p>
      <h2 id="shot-map-title" className="mt-2 text-xl font-semibold">Shot map</h2>
      <p className="mt-2 text-sm text-[var(--muted)]">Non-penalty xG is encoded by marker size and color. Preserved penalties have no modeled xG.</p>
      <div className="campos-frame mt-6 overflow-hidden rounded-xl border border-white/10 bg-[#0b1210]">
        <ThemeProvider value={DARK_THEME}>
          <ShotMap
            shots={camposShots}
            preset="statsbomb"
            colorScale="magma"
            markers={{
              stroke: ({ shot }) => shot.outcome === "goal" ? GOAL_OUTLINE_COLOR : undefined,
              strokeWidth: ({ shot }) => shot.outcome === "goal" ? GOAL_OUTLINE_WIDTH : undefined,
            }}
            crop="half"
            attackingDirection="right"
            side="attack"
            pitchPreset="dark"
            framePadding={8}
            maxWidth={1200}
            showShotTrajectory={false}
            showHeaderStats
            showLegend
          />
        </ThemeProvider>
      </div>
      <div
        aria-label="Shot outcome legend"
        className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-[var(--muted)]"
      >
        <span className="inline-flex items-center gap-2">
          <span
            aria-hidden="true"
            className="size-3 rounded-full border-2 border-[var(--accent)] bg-[#b45f9d]"
          />
          Green outline = Goal
        </span>
        <span className="inline-flex items-center gap-2">
          <span
            aria-hidden="true"
            className="size-3 rounded-full border border-white/45 bg-[#b45f9d]"
          />
          Normal outline = Non-goal
        </span>
      </div>
    </section>
  );
}
