import type { PassEvent } from "@withqwerty/campos-react";

import type { PassResponse, PlayerIdentity } from "@/lib/types";

const STATSBOMB_LENGTH = 120;
const STATSBOMB_WIDTH = 80;

export interface FootyScoutCamposMeta {
  [key: string]: string | number | boolean;
  passIndex: number;
  completed: boolean;
  expectedCompletion: number;
  progressive: boolean;
  underPressure: boolean;
  rawLength: number;
  clockUnavailable: true;
}

function clamp(value: number): number {
  return Math.min(100, Math.max(0, value));
}

/** Match Campos's published StatsBomb adapter: scale x and invert/scale y. */
export function statsBombToCampos(x: number, y: number): { x: number; y: number } {
  return {
    x: clamp((x / STATSBOMB_LENGTH) * 100),
    y: clamp(100 - (y / STATSBOMB_WIDTH) * 100),
  };
}

function passType(pass: PassResponse): PassEvent["passType"] {
  const setPieceTypes: Record<string, PassEvent["passType"]> = {
    Corner: "corner",
    "Free Kick": "free-kick",
    "Goal Kick": "goal-kick",
    "Throw-in": "throw-in",
    "Kick Off": "kick-off",
  };
  const setPiece = setPieceTypes[pass.pass_type];
  if (setPiece) return setPiece;
  if (pass.pass_height === "Ground Pass") return "ground";
  if (pass.pass_height === "Low Pass") return "low";
  if (pass.pass_height === "High Pass") return "high";
  return "other";
}

export function toCamposPass(
  pass: PassResponse,
  player: Pick<PlayerIdentity, "player_name" | "team_id">,
): PassEvent {
  const start = statsBombToCampos(pass.start_x, pass.start_y);
  const end = statsBombToCampos(pass.end_x, pass.end_y);
  const sourceMeta: FootyScoutCamposMeta = {
    passIndex: pass.pass_index,
    completed: pass.completed,
    expectedCompletion: pass.expected_completion,
    progressive: pass.progressive,
    underPressure: pass.under_pressure,
    rawLength: pass.pass_length,
    clockUnavailable: true,
  };
  return {
    kind: "pass",
    id: `footyscout:${pass.pass_index}`,
    matchId: String(pass.match_id),
    teamId: String(player.team_id),
    playerId: pass.player_id === null ? null : String(pass.player_id),
    playerName: player.player_name,
    // The current pass API intentionally has no clock fields. Campos requires them,
    // so neutral values live only at this visualization boundary.
    minute: 0,
    addedMinute: null,
    second: 0,
    period: 1,
    x: start.x,
    y: start.y,
    endX: end.x,
    endY: end.y,
    length: pass.pass_length * (100 / STATSBOMB_LENGTH),
    angle: pass.pass_angle,
    recipient: null,
    passType: passType(pass),
    passResult: pass.completed ? "complete" : "incomplete",
    isAssist: false,
    provider: "footyscout-statsbomb",
    providerEventId: String(pass.pass_index),
    sourceMeta,
  };
}

export function toCamposPasses(
  passes: PassResponse[],
  player: Pick<PlayerIdentity, "player_name" | "team_id">,
): PassEvent[] {
  return passes.map((pass) => toCamposPass(pass, player));
}

export function camposMeta(pass: PassEvent): FootyScoutCamposMeta {
  const metadata = pass.sourceMeta;
  if (
    metadata === null ||
    metadata === undefined ||
    typeof metadata.passIndex !== "number" ||
    typeof metadata.expectedCompletion !== "number" ||
    typeof metadata.completed !== "boolean" ||
    typeof metadata.progressive !== "boolean" ||
    typeof metadata.underPressure !== "boolean" ||
    typeof metadata.rawLength !== "number"
  ) {
    throw new Error("Campos pass is missing FootyScout visualization metadata.");
  }
  return {
    passIndex: metadata.passIndex,
    expectedCompletion: metadata.expectedCompletion,
    completed: metadata.completed,
    progressive: metadata.progressive,
    underPressure: metadata.underPressure,
    rawLength: metadata.rawLength,
    clockUnavailable: true,
  };
}
