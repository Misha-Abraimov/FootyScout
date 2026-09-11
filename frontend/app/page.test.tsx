// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import Home from "./page";

const mocks = vi.hoisted(() => ({
  getMeta: vi.fn(),
  getModel: vi.fn(),
  getLeaderboard: vi.fn(),
}));

vi.mock("@/lib/api", () => ({ api: mocks }));
vi.mock("@/components/LeaderboardTable", () => ({
  LeaderboardTable: () => <div data-testid="leaderboard" />,
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("homepage hero", () => {
  it("describes the current player-intelligence scope and links both CTAs", async () => {
    mocks.getMeta.mockResolvedValue({
      player_count: 369,
      reliable_player_count: 43,
      teams: Array.from({ length: 18 }),
    });
    mocks.getModel.mockResolvedValue({
      dataset: { pass_count: 39_214 },
      out_of_fold_metrics: { roc_auc: 0.9 },
      oof_fold_count: 5,
    });
    mocks.getLeaderboard.mockResolvedValue({ items: [] });

    render(await Home());

    expect(screen.getByText("Player intelligence for scouting")).toBeTruthy();
    expect(screen.getByText(/FootyScout models pass difficulty/)).toBeTruthy();
    expect(
      screen.getByRole("heading", {
        name: "Scout players beyond traditional statistics.",
      }),
    ).toBeTruthy();
    expect(screen.getByText(/models pass difficulty, expected goals, attacking value/)).toBeTruthy();
    expect(screen.getByText(/analyze Team Intelligence, and surface Role Fit scouting recommendations/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "Explore players" }).getAttribute("href")).toBe("/players");
    expect(screen.getByRole("link", { name: "View archetypes" }).getAttribute("href")).toBe("/archetypes");
    expect(screen.getByRole("link", { name: /Explore Team Intelligence/ }).getAttribute("href")).toBe("/teams/904");
    expect(screen.getByRole("link", { name: /Open Scouting Recommendations/ }).getAttribute("href")).toBe("/scouting");
    expect(screen.getByText("Based on available StatsBomb 2023/24 Bundesliga event data.")).toBeTruthy();
  });
});
