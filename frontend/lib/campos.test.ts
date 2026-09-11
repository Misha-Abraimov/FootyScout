import { describe, expect, it } from "vitest";

import { statsBombToCampos, toCamposPass } from "./campos";
import type { PassResponse } from "./types";

const basePass: PassResponse = {
  pass_index: 12,
  match_id: 7,
  player_id: 3500,
  start_x: 0,
  start_y: 0,
  end_x: 120,
  end_y: 80,
  pass_length: 30,
  pass_angle: 0.2,
  forward_distance: 20,
  lateral_distance: 4,
  distance_to_goal_before: 80,
  distance_to_goal_after: 60,
  distance_toward_goal: 20,
  completed: true,
  expected_completion: 0.74,
  under_pressure: false,
  progressive: true,
  pass_height: "Ground Pass",
  body_part: "Right Foot",
  pass_type: "Open Play",
  start_zone: "middle_centre",
  end_zone: "attacking_centre",
  fold: 2,
};

describe("FootyScout to Campos adapter", () => {
  it("scales x and inverts/scales y at documented boundaries", () => {
    expect(statsBombToCampos(0, 0)).toEqual({ x: 0, y: 100 });
    expect(statsBombToCampos(120, 80)).toEqual({ x: 100, y: 0 });
    expect(statsBombToCampos(60, 40)).toEqual({ x: 50, y: 50 });
  });

  it("clamps malformed out-of-range coordinates to the pitch", () => {
    expect(statsBombToCampos(-10, 100)).toEqual({ x: 0, y: 0 });
  });

  it("converts completion outcome without changing API types", () => {
    const player = { player_name: "Granit Xhaka", team_id: 904 };
    expect(toCamposPass(basePass, player).passResult).toBe("complete");
    expect(toCamposPass({ ...basePass, completed: false }, player).passResult).toBe("incomplete");
  });
});
