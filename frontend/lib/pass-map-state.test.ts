import { describe, expect, it } from "vitest";

import {
  createPassMapState,
  formatPassOption,
  passMapReducer,
  selectedPassForState,
} from "./pass-map-state";
import type { PassListResponse, PassResponse } from "./types";

function pass(passIndex: number, expectedCompletion: number): PassResponse {
  return {
    pass_index: passIndex,
    match_id: 1,
    player_id: 3500,
    start_x: 10,
    start_y: 20,
    end_x: 30,
    end_y: 40,
    pass_length: 25,
    pass_angle: 0.1,
    forward_distance: 20,
    lateral_distance: 20,
    distance_to_goal_before: 100,
    distance_to_goal_after: 80,
    distance_toward_goal: 20,
    completed: true,
    expected_completion: expectedCompletion,
    under_pressure: false,
    progressive: false,
    pass_height: "Ground Pass",
    body_part: "Left Foot",
    pass_type: "Open Play",
    start_zone: "defensive_centre",
    end_zone: "middle_centre",
    fold: 1,
  };
}

function response(items: PassResponse[], total = items.length, offset = 0): PassListResponse {
  return { items, total, limit: 200, offset };
}

describe("pass map state", () => {
  it("ignores stale responses so they cannot replace the active filter data", () => {
    let state = createPassMapState(response([pass(1, 0.8)]));
    state = passMapReducer(state, { type: "request", requestId: 1, filter: { view: "completed", difficulty: "all", offset: 0 } });
    state = passMapReducer(state, { type: "request", requestId: 2, filter: { view: "pressure", difficulty: "all", offset: 0 } });
    state = passMapReducer(state, { type: "success", requestId: 1, response: response([pass(2, 0.7)], 20) });
    expect(state.data.items[0]?.pass_index).toBe(1);
    state = passMapReducer(state, { type: "success", requestId: 2, response: response([pass(3, 0.6)], 10) });
    expect(state.data.items[0]?.pass_index).toBe(3);
    expect(state.data.total).toBe(10);
  });

  it("keeps a valid selection, resets a missing one, and clears an empty result", () => {
    let state = createPassMapState(response([pass(1, 0.8), pass(2, 0.7)]));
    state = passMapReducer(state, { type: "select", passIndex: 2 });
    state = passMapReducer(state, { type: "request", requestId: 1, filter: { view: "completed", difficulty: "all", offset: 0 } });
    state = passMapReducer(state, { type: "success", requestId: 1, response: response([pass(2, 0.7), pass(3, 0.6)]) });
    expect(state.selectedPassIndex).toBe(2);
    state = passMapReducer(state, { type: "request", requestId: 2, filter: { view: "pressure", difficulty: "all", offset: 0 } });
    state = passMapReducer(state, { type: "success", requestId: 2, response: response([pass(4, 0.5)]) });
    expect(state.selectedPassIndex).toBe(4);
    state = passMapReducer(state, { type: "request", requestId: 3, filter: { view: "pressure", difficulty: "difficult", offset: 0 } });
    state = passMapReducer(state, { type: "success", requestId: 3, response: response([], 0) });
    expect(state.selectedPassIndex).toBeNull();
    expect(selectedPassForState(state)).toBeNull();
  });

  it("uses one pass object for the selector and inspector xPass", () => {
    let state = createPassMapState(response([pass(14, 0.99), pass(27, 0.848)]));
    state = passMapReducer(state, { type: "select", passIndex: 27 });
    const selected = selectedPassForState(state);
    expect(selected?.pass_index).toBe(27);
    expect(formatPassOption(selected!)).toContain("xPass 84.8%");
    expect(selected?.expected_completion).toBe(0.848);
  });
});
