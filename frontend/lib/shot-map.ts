import type { ShotMapProps } from "@withqwerty/campos-react";

import { statsBombToCampos } from "@/lib/campos";
import type { PlayerIdentity, ShotResponse } from "@/lib/types";

type CamposShot = ShotMapProps["shots"][number];

function bodyPart(value: string): CamposShot["bodyPart"] {
  if (value === "Left Foot") return "left-foot";
  if (value === "Right Foot") return "right-foot";
  if (value === "Head") return "head";
  return "other";
}

function context(shot: ShotResponse): CamposShot["context"] {
  if (shot.penalty) return "penalty";
  if (shot.shot_type === "Free Kick") return "direct-free-kick";
  if (shot.play_pattern === "From Corner") return "from-corner";
  if (shot.play_pattern === "From Counter") return "fast-break";
  if (shot.play_pattern === "Regular Play") return "regular-play";
  if (shot.play_pattern.startsWith("From ")) return "set-piece";
  return "other";
}

function period(value: number): CamposShot["period"] {
  if (value === 2 || value === 3 || value === 4 || value === 5) return value;
  return 1;
}

/** Keep StatsBomb-to-Campos conversion isolated at the visualization boundary. */
export function toCamposShot(
  shot: ShotResponse,
  player: Pick<PlayerIdentity, "player_name" | "team_id">,
): CamposShot {
  const location = statsBombToCampos(shot.start_x, shot.start_y);
  return {
    kind: "shot",
    id: `footyscout:${shot.shot_id}`,
    matchId: String(shot.match_id),
    teamId: String(player.team_id),
    playerId: shot.player_id === null ? null : String(shot.player_id),
    playerName: player.player_name,
    minute: shot.minute,
    addedMinute: null,
    second: shot.second,
    period: period(shot.period),
    x: location.x,
    y: location.y,
    xg: shot.expected_goal,
    outcome: shot.goal ? "goal" : "other",
    bodyPart: bodyPart(shot.body_part),
    isOwnGoal: false,
    isPenalty: shot.penalty,
    context: context(shot),
    provider: "footyscout-statsbomb",
    providerEventId: shot.shot_id,
    sourceMeta: {
      technique: shot.technique,
      playPattern: shot.play_pattern,
      modelEligible: shot.model_eligible,
      underPressure: shot.under_pressure,
    },
  };
}

export function toCamposShots(
  shots: ShotResponse[],
  player: Pick<PlayerIdentity, "player_name" | "team_id">,
): CamposShot[] {
  return shots.map((shot) => toCamposShot(shot, player));
}
