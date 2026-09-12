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
      training_corpus: { events: 823_553, matches: 233 },
    });
    mocks.getArchetypes.mockResolvedValue({
      definitions: [{ player_count: 80 }, { player_count: 53 }],
    });
    mocks.getLeaderboard.mockResolvedValue({ items: [] });

    render(await Home());

    expect(screen.getByText("Player intelligence for scouting")).toBeTruthy();
    expect(screen.getByText(/FootyScout analyzes pass difficulty/)).toBeTruthy();
    expect(
      screen.getByRole("heading", {
        name: "Scout players beyond traditional statistics.",
      }),
    ).toBeTruthy();
    expect(screen.getByText(/analyzes pass difficulty, expected goals, attacking impact/)).toBeTruthy();
    expect(screen.getByText(/understand team intelligence, and generate scouting recommendations/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "Explore players" }).getAttribute("href")).toBe("/players");
    expect(screen.getByRole("link", { name: "View archetypes" }).getAttribute("href")).toBe("/archetypes");
    expect(screen.getByRole("link", { name: /Read methodology/ }).getAttribute("href")).toBe("/model");
    expect(screen.getByRole("link", { name: /Explore Team Intelligence/ }).getAttribute("href")).toBe("/teams/904");
    expect(screen.getByRole("link", { name: /Open Scouting Recommendations/ }).getAttribute("href")).toBe("/scouting");
    expect(screen.getByText("Based on available StatsBomb 2023/24 Bundesliga event data.")).toBeTruthy();

    expect(screen.getByText("369")).toBeTruthy();
    expect(screen.getByText("823,000+")).toBeTruthy();
    expect(screen.getByText("133")).toBeTruthy();
    expect(screen.getByText("Player profiles across the available competition sample")).toBeTruthy();
    expect(screen.getByText("Teams represented in the available Bundesliga event data")).toBeTruthy();
    expect(screen.getByText("FootyScout by the numbers")).toBeTruthy();
    expect(screen.getByText("Events analyzed")).toBeTruthy();
    expect(screen.getByText("Across 233 matches used to build FootyScout's analytics models")).toBeTruthy();
    expect(screen.getByText("Detailed player profiles")).toBeTruthy();
    expect(screen.getByText("Eligible outfield players with same-position comparisons")).toBeTruthy();

    expect(screen.getByText("How FootyScout builds scouting intelligence")).toBeTruthy();
    for (const step of [
      "Event data",
      "Predictive models",
      "Attacking impact",
      "Player intelligence",
      "Team & role fit",
    ]) {
      expect(screen.getByRole("heading", { name: step })).toBeTruthy();
    }
    expect(screen.queryByText("How xPass becomes a profile")).toBeNull();
    expect(screen.queryByText("A transparent analytical base")).toBeNull();
    expect(screen.queryByText("From event data to a focused shortlist")).toBeNull();
    expect(screen.queryByText("From observed roles to a focused shortlist")).toBeNull();
    expect(screen.getByText(/observed playing style match/)).toBeTruthy();
  });
});
