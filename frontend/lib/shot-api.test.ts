import { describe, expect, it } from "vitest";

import { buildApiUrl } from "./api";
import type { ShotListResponse } from "./types";

describe("shot API contract", () => {
  it("types nullable penalty xG and builds pagination filters", () => {
    const response: ShotListResponse = {
      total: 1,
      limit: 50,
      offset: 0,
      items: [{
        shot_id: "shot-1", match_id: 1, player_id: 2, period: 1, minute: 3, second: 4,
        start_x: 108, start_y: 40, distance: 12, angle: 0.64, goal: false,
        expected_goal: null, body_part: "Right Foot", shot_type: "Penalty",
        technique: "Normal", play_pattern: "From Free Kick", under_pressure: false,
        first_time: false, one_on_one: false, open_goal: false, penalty: true,
        penalty_shootout: false, model_eligible: false,
      }],
    };
    const url = new URL(buildApiUrl("/api/players/2/shots", { limit: 50, offset: 0, goal: false }));
    expect(response.items[0].expected_goal).toBeNull();
    expect(url.pathname).toBe("/api/players/2/shots");
    expect(url.searchParams.get("goal")).toBe("false");
  });
});
