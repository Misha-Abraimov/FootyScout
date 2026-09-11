// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { PlayerAutocomplete } from "./PlayerAutocomplete";
import type { PlayerListResponse, PlayerSummary } from "../lib/types";

const mocks = vi.hoisted(() => ({ getPlayers: vi.fn() }));
vi.mock("@/lib/api", () => ({ api: { getPlayers: mocks.getPlayers } }));

function player(playerId: number, playerName: string): PlayerSummary {
  return {
    player_id: playerId,
    player_name: playerName,
    team_id: playerId,
    team_name: playerId === 3500 ? "Bayer Leverkusen" : "Test FC",
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

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((promiseResolve) => {
    resolve = promiseResolve;
  });
  return { promise, resolve };
}

afterEach(() => {
  cleanup();
  mocks.getPlayers.mockReset();
});

describe("PlayerAutocomplete", () => {
  it("allows the combobox to shrink within responsive grid tracks", () => {
    render(<PlayerAutocomplete label="Player" onSelect={vi.fn()} />);

    const input = screen.getByRole("combobox", { name: "Player" });
    expect(input.className).toContain("w-full");
    expect(input.className).toContain("min-w-0");
    expect(input.closest("label")?.className).toContain("min-w-0");
  });

  it("requests and displays suggestions for one character", async () => {
    mocks.getPlayers.mockResolvedValueOnce(response([player(3500, "Granit Xhaka")]));
    render(<PlayerAutocomplete label="Player" onSelect={vi.fn()} />);

    await userEvent.setup().type(screen.getByRole("combobox", { name: "Player" }), "x");

    await waitFor(() => expect(mocks.getPlayers).toHaveBeenCalledTimes(1));
    expect(mocks.getPlayers.mock.calls[0]?.[0]).toMatchObject({ search: "x" });
    expect(await screen.findByRole("option", { name: /Granit Xhaka/ })).toBeTruthy();
  });

  it("ignores a stale response after fast typing", async () => {
    const oldRequest = deferred<PlayerListResponse>();
    const currentRequest = deferred<PlayerListResponse>();
    mocks.getPlayers
      .mockReturnValueOnce(oldRequest.promise)
      .mockReturnValueOnce(currentRequest.promise);
    render(<PlayerAutocomplete label="Player" onSelect={vi.fn()} />);
    const input = screen.getByRole("combobox", { name: "Player" });
    const user = userEvent.setup();

    await user.type(input, "x");
    await waitFor(() => expect(mocks.getPlayers).toHaveBeenCalledTimes(1));
    await user.type(input, "h");
    await waitFor(() => expect(mocks.getPlayers).toHaveBeenCalledTimes(2));
    currentRequest.resolve(response([player(3500, "Granit Xhaka")]));
    expect(await screen.findByRole("option", { name: /Granit Xhaka/ })).toBeTruthy();
    oldRequest.resolve(response([player(99, "Xavi Simons")]));

    await Promise.resolve();
    expect(screen.queryByRole("option", { name: /Xavi Simons/ })).toBeNull();
  });

  it("supports Arrow Down and Enter selection", async () => {
    const onSelect = vi.fn();
    const xhaka = player(3500, "Granit Xhaka");
    mocks.getPlayers.mockResolvedValueOnce(response([xhaka]));
    render(<PlayerAutocomplete label="Player" onSelect={onSelect} />);
    const input = screen.getByRole("combobox", { name: "Player" });
    const user = userEvent.setup();

    await user.type(input, "x");
    await screen.findByRole("option", { name: /Granit Xhaka/ });
    await user.keyboard("{ArrowDown}{Enter}");

    expect(onSelect).toHaveBeenCalledWith(xhaka);
    expect(input.getAttribute("aria-expanded")).toBe("false");
  });

  it("supports mouse selection", async () => {
    const onSelect = vi.fn();
    const xhaka = player(3500, "Granit Xhaka");
    mocks.getPlayers.mockResolvedValueOnce(response([xhaka]));
    render(<PlayerAutocomplete label="Player" onSelect={onSelect} />);

    await userEvent.setup().type(screen.getByRole("combobox", { name: "Player" }), "aka");
    await userEvent.setup().click(await screen.findByRole("option", { name: /Granit Xhaka/ }));

    expect(onSelect).toHaveBeenCalledWith(xhaka);
  });

  it("clears suggestions immediately when the input is cleared", async () => {
    mocks.getPlayers.mockResolvedValueOnce(response([player(3500, "Granit Xhaka")]));
    render(<PlayerAutocomplete label="Player" onSelect={vi.fn()} />);
    const input = screen.getByRole("combobox", { name: "Player" });
    const user = userEvent.setup();

    await user.type(input, "x");
    await screen.findByRole("option", { name: /Granit Xhaka/ });
    await user.clear(input);

    expect(screen.queryByRole("listbox")).toBeNull();
    expect(input.getAttribute("aria-expanded")).toBe("false");
  });

  it("shows a clear no-results state", async () => {
    mocks.getPlayers.mockResolvedValueOnce(response([]));
    render(<PlayerAutocomplete label="Player" onSelect={vi.fn()} />);

    await userEvent.setup().type(screen.getByRole("combobox", { name: "Player" }), "zzz");

    expect((await screen.findByRole("status")).textContent).toContain("No players found");
  });

  it("closes on Escape and outside pointer input", async () => {
    mocks.getPlayers.mockResolvedValue(response([player(3500, "Granit Xhaka")]));
    render(<div><PlayerAutocomplete label="Player" onSelect={vi.fn()} /><button type="button">Outside</button></div>);
    const input = screen.getByRole("combobox", { name: "Player" });
    const user = userEvent.setup();

    await user.type(input, "x");
    await screen.findByRole("option", { name: /Granit Xhaka/ });
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("listbox")).toBeNull();
    fireEvent.focus(input);
    expect(await screen.findByRole("listbox")).toBeTruthy();
    fireEvent.pointerDown(screen.getByRole("button", { name: "Outside" }));
    expect(screen.queryByRole("listbox")).toBeNull();
  });
});
