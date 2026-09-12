// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { FootyScoutPassMap } from "./FootyScoutPassMap";
import { DIFFICULT_MAX } from "../lib/pass-filters";
import type { PassListResponse, PassResponse, PlayerIdentity } from "../lib/types";

const mocks = vi.hoisted(() => ({ getPlayerPasses: vi.fn() }));

vi.mock("@/lib/api", () => ({ api: { getPlayerPasses: mocks.getPlayerPasses } }));
vi.mock("@withqwerty/campos-react", () => ({
  DARK_THEME: {},
  ThemeProvider: ({ children }: { children: ReactNode }) => <>{children}</>,
  PassMap: ({ passes }: { passes: Array<{ id: string }> }) => (
    <div data-testid="campos-pass-map" data-pass-ids={passes.map((pass) => pass.id).join(",")}>
      {passes.length} plotted passes
    </div>
  ),
}));

const player: PlayerIdentity = {
  player_id: 3500,
  player_name: "Granit Xhaka",
  team_id: 904,
  team_name: "Bayer Leverkusen",
  position: "Left Defensive Midfield",
  position_group: "MID",
};

function pass(passIndex: number, expectedCompletion: number, completed = true): PassResponse {
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
    completed,
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

afterEach(() => {
  cleanup();
  mocks.getPlayerPasses.mockReset();
});

describe("FootyScoutPassMap interactions", () => {
  it.each([
    ["Completed", { completed: true }, 101],
    ["Incomplete", { completed: false }, 102],
    ["Progressive", { progressive: true }, 103],
    ["Under pressure", { under_pressure: true }, 104],
  ])("changes All to %s and replaces the displayed dataset", async (label, expectedFilter, passIndex) => {
    mocks.getPlayerPasses.mockResolvedValueOnce(response([pass(passIndex, 0.7)], passIndex));
    render(<FootyScoutPassMap player={player} initialData={response([pass(1, 0.8)], 500)} />);

    await userEvent.setup().click(screen.getByRole("button", { name: label }));

    await waitFor(() => expect(mocks.getPlayerPasses).toHaveBeenCalledTimes(1));
    expect(mocks.getPlayerPasses.mock.calls[0]?.[1]).toEqual({
      limit: 200,
      offset: 0,
      ...expectedFilter,
    });
    await waitFor(() => expect(screen.getByText(`Showing 1–1 of ${passIndex} passes`)).toBeTruthy());
    expect(screen.getByRole("button", { name: label }).getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByTestId("campos-pass-map").getAttribute("data-pass-ids")).toBe(`footyscout:${passIndex}`);
  });

  it("combines subset and difficulty and resets the offset", async () => {
    mocks.getPlayerPasses
      .mockResolvedValueOnce(response([pass(10, 0.5)], 300))
      .mockResolvedValueOnce(response([pass(11, 0.4)], 25));
    render(<FootyScoutPassMap player={player} initialData={response([pass(1, 0.8)], 500, 200)} />);
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: "Progressive" }));
    await waitFor(() => expect(mocks.getPlayerPasses).toHaveBeenCalledTimes(1));
    await user.selectOptions(screen.getByLabelText(/Pass difficulty/), "difficult");

    await waitFor(() => expect(mocks.getPlayerPasses).toHaveBeenCalledTimes(2));
    expect(mocks.getPlayerPasses.mock.calls[1]?.[1]).toEqual({
      limit: 200,
      offset: 0,
      progressive: true,
      max_expected_completion: DIFFICULT_MAX,
    });
    await waitFor(() => expect(screen.getByText("Showing 1–1 of 25 passes")).toBeTruthy());
  });

  it("preserves active filters while paginating", async () => {
    mocks.getPlayerPasses
      .mockResolvedValueOnce(response([pass(10, 0.7)], 450, 0))
      .mockResolvedValueOnce(response([pass(210, 0.65)], 450, 200));
    render(<FootyScoutPassMap player={player} initialData={response([pass(1, 0.8)], 500)} />);
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: "Completed" }));
    await waitFor(() => expect(mocks.getPlayerPasses).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole("button", { name: "Next 200" }));

    await waitFor(() => expect(mocks.getPlayerPasses).toHaveBeenCalledTimes(2));
    expect(mocks.getPlayerPasses.mock.calls[1]?.[1]).toEqual({
      limit: 200,
      offset: 200,
      completed: true,
    });
    await waitFor(() => expect(screen.getByText("Showing 201–201 of 450 passes")).toBeTruthy());
    expect((screen.getByRole("button", { name: "Previous 200" }) as HTMLButtonElement).disabled).toBe(false);
  });

  it("keeps selector and inspector synchronized by pass_index", async () => {
    render(<FootyScoutPassMap player={player} initialData={response([pass(14, 0.99), pass(27, 0.848)])} />);
    const selector = screen.getByLabelText(/Inspect a pass/) as HTMLSelectElement;

    await userEvent.setup().selectOptions(selector, "27");

    expect(selector.value).toBe("27");
    expect(selector.selectedOptions[0]?.textContent).toContain("Pass #27");
    expect(selector.selectedOptions[0]?.textContent).toContain("xPass 84.8%");
    expect(screen.getByText("Pass difficulty", { selector: "dt" }).parentElement?.textContent).toContain("84.8%");
  });

  it("clears the inspector and shows an empty state for an empty filtered response", async () => {
    mocks.getPlayerPasses.mockResolvedValueOnce(response([], 0));
    render(<FootyScoutPassMap player={player} initialData={response([pass(1, 0.8)], 500)} />);

    await userEvent.setup().click(screen.getByRole("button", { name: "Incomplete" }));

    await waitFor(() => expect(screen.getByText("No matching passes")).toBeTruthy());
    expect(screen.queryByLabelText(/Inspect a pass/)).toBeNull();
    expect(screen.getByText("Showing 0–0 of 0 passes")).toBeTruthy();
  });
});
