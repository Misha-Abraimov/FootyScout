// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import Home from "./page";

const mocks = vi.hoisted(() => ({
  getMeta: vi.fn(),
  getActionValueModel: vi.fn(),
  getArchetypes: vi.fn(),
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

describe("homepage", () => {
  it("presents the full scouting-intelligence scope and keeps its CTAs", async () => {
    mocks.getMeta.mockResolvedValue({
      player_count: 369,
      reliable_player_count: 43,
      teams: Array.from({ length: 18 }),
    });
    mocks.getActionValueModel.mockResolvedValue({
      training_corpus: { states: 667_062 },
    });
    mocks.getArchetypes.mockResolvedValue({
      definitions: [{ player_count: 80 }, { player_count: 53 }],
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
    expect(screen.getByRole("link", { name: /Read methodology/ }).getAttribute("href")).toBe("/model");
    expect(screen.getByRole("link", { name: /Explore Team Intelligence/ }).getAttribute("href")).toBe("/teams/904");
    expect(screen.getByRole("link", { name: /Open Scouting Recommendations/ }).getAttribute("href")).toBe("/scouting");
    expect(screen.getByText("Based on available StatsBomb 2023/24 Bundesliga event data.")).toBeTruthy();

    expect(screen.getByText("369")).toBeTruthy();
    expect(screen.getByText("667,000+")).toBeTruthy();
    expect(screen.getByText("133")).toBeTruthy();
    expect(screen.getByText("Player profiles across the available competition sample")).toBeTruthy();
    expect(screen.getByText("Teams represented in the available Bundesliga event data")).toBeTruthy();
    expect(screen.getByText("States evaluated for attacking-value estimation")).toBeTruthy();
    expect(screen.getByText("Eligible outfield players with position-aware style profiles")).toBeTruthy();

    expect(screen.getByText("How FootyScout builds scouting intelligence")).toBeTruthy();
    expect(screen.getByRole("heading", { name: "From event data to a focused shortlist" })).toBeTruthy();
    for (const step of [
      "Event data",
      "Predictive models",
      "Attacking value",
      "Player intelligence",
      "Team & role fit",
    ]) {
      expect(screen.getByRole("heading", { name: step })).toBeTruthy();
    }
    expect(screen.queryByText("How xPass becomes a profile")).toBeNull();
    expect(screen.getByText(/observed playing-style distance/)).toBeTruthy();
  });
});
