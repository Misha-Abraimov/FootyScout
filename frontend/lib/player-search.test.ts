import { describe, expect, it } from "vitest";

import { rankPlayerSuggestions } from "./player-search";
import type { PlayerSummary } from "./types";

function player(playerId: number, playerName: string): PlayerSummary {
  return {
    player_id: playerId,
    player_name: playerName,
    team_id: 1,
    team_name: "Test FC",
    position: "Midfield",
    position_group: "MID",
    matches_observed: 10,
    pass_attempts: 500,
    actual_completion_rate: 0.9,
    expected_completion_rate: 0.88,
    completion_above_expected_pp: 2,
    progressive_pass_rate: 0.2,
    pressure_pass_rate: 0.1,
    final_third_entries_per_100_passes: 4,
    overall_reliable: true,
  };
}

describe("player autocomplete matching", () => {
  const players = [
    player(1, "Granit Xhaka"),
    player(2, "Xavi Simons"),
    player(3, "Florian Wirtz"),
  ];

  it("matches a one-character contiguous substring", () => {
    expect(rankPlayerSuggestions(players, "x").map((item) => item.player_name)).toEqual([
      "Xavi Simons",
      "Granit Xhaka",
    ]);
  });

  it("matches multi-character substrings case-insensitively", () => {
    expect(rankPlayerSuggestions(players, "XH").map((item) => item.player_name)).toEqual([
      "Granit Xhaka",
    ]);
    expect(rankPlayerSuggestions(players, "aka").map((item) => item.player_name)).toEqual([
      "Granit Xhaka",
    ]);
  });

  it("does not perform fuzzy or subsequence matching", () => {
    expect(rankPlayerSuggestions(players, "xka")).toEqual([]);
  });

  it("ranks full-name starts, then word starts, then internal matches", () => {
    const ranked = rankPlayerSuggestions(
      [player(4, "Max Winter"), player(5, "Winter Max"), player(6, "Tom Amaxi")],
      "max",
    );
    expect(ranked.map((item) => item.player_name)).toEqual([
      "Max Winter",
      "Winter Max",
      "Tom Amaxi",
    ]);
  });
});
