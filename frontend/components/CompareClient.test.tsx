// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CompareClient } from "./CompareClient";
import type { PlayerListResponse, PlayerSummary } from "../lib/types";

const mocks = vi.hoisted(() => ({ getPlayers: vi.fn(), comparePlayers: vi.fn() }));
vi.mock("@/lib/api", () => ({
  api: {
    getPlayers: mocks.getPlayers,
    comparePlayers: mocks.comparePlayers,
  },
}));

function player(playerId: number, playerName: string): PlayerSummary {
  return {
    player_id: playerId,
    player_name: playerName,
    team_id: playerId,
    team_name: `${playerName} FC`,
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

function response(items: PlayerSummary[]): PlayerListResponse {
  return { items, total: items.length, limit: 50, offset: 0 };
}

afterEach(() => {
  cleanup();
  mocks.getPlayers.mockReset();
  mocks.comparePlayers.mockReset();
});

describe("CompareClient autocomplete", () => {
  it("selects both players by stable player_id and excludes Player A from Player B", async () => {
    const xhaka = player(3500, "Granit Xhaka");
    const wirtz = player(5500, "Florian Wirtz");
    mocks.getPlayers
      .mockResolvedValueOnce(response([xhaka]))
      .mockResolvedValueOnce(response([xhaka, wirtz]));
    mocks.comparePlayers.mockResolvedValueOnce(undefined);
    render(<CompareClient initial={null} />);
    const user = userEvent.setup();

    await user.type(screen.getByRole("combobox", { name: "Player A" }), "x");
    await user.click(await screen.findByRole("option", { name: /Granit Xhaka/ }));
    await user.type(screen.getByRole("combobox", { name: "Player B" }), "w");

    expect(await screen.findByRole("option", { name: /Florian Wirtz/ })).toBeTruthy();
    expect(screen.queryByRole("option", { name: /Granit Xhaka/ })).toBeNull();
    await user.click(screen.getByRole("option", { name: /Florian Wirtz/ }));

    await waitFor(() => expect(mocks.comparePlayers).toHaveBeenCalledWith([3500, 5500]));
  });
});
