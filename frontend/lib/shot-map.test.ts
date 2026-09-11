import { describe, expect, it } from "vitest";

import { toCamposShot } from "./shot-map";
import type { PlayerIdentity, ShotResponse } from "./types";

const player: PlayerIdentity = {
  player_id: 1,
  player_name: "Player One",
  team_id: 2,
  team_name: "Team",
  position: "Forward",
  position_group: "FWD",
};

const shot: ShotResponse = {
  shot_id: "shot-1", match_id: 3, player_id: 1, period: 1, minute: 9, second: 20,
  start_x: 108, start_y: 40, distance: 12, angle: 0.64, goal: true,
  expected_goal: 0.31, body_part: "Right Foot", shot_type: "Open Play",
  technique: "Normal", play_pattern: "Regular Play", under_pressure: false,
  first_time: false, one_on_one: false, open_goal: false, penalty: false,
  penalty_shootout: false, model_eligible: true,
};

describe("shot-map adapter", () => {
  it("converts real API fields to canonical Campos coordinates and semantics", () => {
    const converted = toCamposShot(shot, player);
    expect(converted.x).toBe(90);
    expect(converted.y).toBe(50);
    expect(converted.xg).toBe(0.31);
    expect(converted.outcome).toBe("goal");
    expect(converted.bodyPart).toBe("right-foot");
    expect(converted.context).toBe("regular-play");
  });

  it("preserves a penalty with nullable xG", () => {
    const converted = toCamposShot({ ...shot, goal: false, penalty: true, expected_goal: null }, player);
    expect(converted.isPenalty).toBe(true);
    expect(converted.xg).toBeNull();
    expect(converted.context).toBe("penalty");
  });
});
