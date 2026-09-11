"use client";

import { DARK_THEME, PassMap, ThemeProvider, type PassEvent } from "@withqwerty/campos-react";
import { useMemo } from "react";

import { statsBombToCampos } from "@/lib/campos";
import type { AttackingActionListResponse, PlayerIdentity } from "@/lib/types";

export function FootyScoutCarryMap({ player, data }: { player: PlayerIdentity; data: AttackingActionListResponse }) {
  const carries = useMemo(() => data.items.map((action): PassEvent => {
    const start = statsBombToCampos(action.start_x, action.start_y);
    const end = statsBombToCampos(action.end_x, action.end_y);
    return {
      kind: "pass", id: `carry:${action.action_id}`, matchId: String(action.match_id), teamId: String(action.team_id),
      playerId: action.player_id === null ? null : String(action.player_id), playerName: player.player_name,
      minute: 0, addedMinute: null, second: 0, period: 1, x: start.x, y: start.y, endX: end.x, endY: end.y,
      length: Math.hypot(end.x - start.x, end.y - start.y), angle: Math.atan2(end.y - start.y, end.x - start.x),
      recipient: null, passType: "other", passResult: "complete", isAssist: false,
      provider: "footyscout-statsbomb", providerEventId: action.action_id,
      sourceMeta: { attackingValue: action.attacking_value, progressive: action.progressive, underPressure: action.under_pressure },
    };
  }), [data.items, player.player_name]);
  if (!carries.length) return null;
  return (
    <section aria-labelledby="carry-map-title" className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-4 sm:p-6">
      <p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">Campos visualization</p>
      <h2 id="carry-map-title" className="mt-2 text-xl font-semibold">Carry value map</h2>
      <p className="mt-2 text-sm text-[var(--muted)]">Exact StatsBomb carry start/end locations. Green increases modeled possession value; coral decreases it.</p>
      <div className="campos-frame mt-5 overflow-hidden rounded-xl border border-white/10 bg-[#0b1210]">
        <ThemeProvider value={DARK_THEME}>
          <PassMap passes={carries} attackingDirection="right" pitchPreset="dark" framePadding={8} maxWidth={1200} showHeaderStats={false} showLegend={false}
            lines={{ stroke: ({ pass }) => Number(pass.sourceMeta?.attackingValue ?? 0) >= 0 ? "#80e4a8" : "#ff8f86", opacity: ({ pass }) => Math.min(1, 0.3 + Math.abs(Number(pass.sourceMeta?.attackingValue ?? 0)) * 8), strokeWidth: ({ pass }) => pass.sourceMeta?.progressive ? 1.1 : 0.65 }}
            dots={{ fill: ({ pass }) => Number(pass.sourceMeta?.attackingValue ?? 0) >= 0 ? "#80e4a8" : "#ff8f86", radius: 0.6 }} />
        </ThemeProvider>
      </div>
    </section>
  );
}
