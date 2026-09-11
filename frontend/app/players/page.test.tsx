// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import PlayersPage from "./page";

const mocks = vi.hoisted(() => ({ getPlayers: vi.fn(), getMeta: vi.fn() }));

vi.mock("@/lib/api", () => ({ api: mocks }));
vi.mock("@/components/PlayerExplorerAutocomplete", () => ({
  PlayerExplorerAutocomplete: () => <div data-testid="player-search" />,
}));
vi.mock("@/components/PlayerTable", () => ({
  PlayerTable: () => <div data-testid="player-table" />,
}));
vi.mock("@/components/Pagination", () => ({
  Pagination: () => <div data-testid="pagination" />,
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("player explorer copy", () => {
  it("describes players and full FootyScout profiles rather than passers", async () => {
    mocks.getPlayers.mockResolvedValue({ total: 0, limit: 25, offset: 0, items: [] });
    mocks.getMeta.mockResolvedValue({
      teams: [],
      position_groups: [],
      player_count: 0,
      reliable_player_count: 0,
    });

    render(await PlayersPage({ searchParams: Promise.resolve({}) }));

    expect(screen.getByText("Player explorer")).toBeTruthy();
    expect(
      screen.getByRole("heading", {
        name: "Find players by role, team, and performance.",
      }),
    ).toBeTruthy();
    expect(
      screen.getByText(
        "Explore the available StatsBomb sample, compare player metrics, and open a player’s full FootyScout profile.",
      ),
    ).toBeTruthy();
    expect(screen.queryByText(/Find passers/i)).toBeNull();
    expect(screen.queryByText(/full passing profile/i)).toBeNull();
  });

  it("keeps long values inside a responsive primary grid and compact action row", async () => {
    mocks.getPlayers.mockResolvedValue({ total: 0, limit: 25, offset: 0, items: [] });
    mocks.getMeta.mockResolvedValue({
      teams: ["Borussia Mönchengladbach"],
      position_groups: ["MID"],
      player_count: 1,
      reliable_player_count: 1,
    });

    render(
      await PlayersPage({
        searchParams: Promise.resolve({ team: "Borussia Mönchengladbach" }),
      }),
    );

    const primary = screen.getByTestId("player-filter-primary");
    const actions = screen.getByTestId("player-filter-actions");
    const team = screen.getByRole("combobox", { name: "Team" });
    const sort = screen.getByRole("combobox", { name: "Sort by" });
    const order = screen.getByRole("combobox", { name: "Order" });
    const apply = screen.getByRole("button", { name: "Apply filters" });

    expect(primary.className).toContain("xl:grid-cols-[minmax(220px,2fr)");
    expect(primary.className).toContain("md:grid-cols-2");
    expect(primary.className).toContain("lg:grid-cols-3");
    expect(team.className).toContain("w-full");
    expect(team.className).toContain("min-w-0");
    expect((team as HTMLSelectElement).value).toBe("Borussia Mönchengladbach");
    expect(sort.className).toContain("w-full");
    expect(sort.className).toContain("min-w-0");
    expect(actions.className).toContain("sm:flex-row");
    expect(order.closest("label")?.className).toContain("sm:w-48");
    expect(apply.className).toContain("w-full");
    expect(apply.className).toContain("sm:w-auto");
  });
});
