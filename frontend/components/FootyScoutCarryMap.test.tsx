// @vitest-environment jsdom

import { cleanup, render } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { AttackingActionListResponse, PlayerIdentity } from "@/lib/types";

import { FootyScoutCarryMap } from "./FootyScoutCarryMap";

const mocks = vi.hoisted(() => ({ passMap: vi.fn() }));

vi.mock("@withqwerty/campos-react", () => ({
  DARK_THEME: {},
  ThemeProvider: ({ children }: { children: ReactNode }) => <>{children}</>,
  PassMap: (props: unknown) => {
    mocks.passMap(props);
    return <div data-testid="campos-carry-map" />;
  },
}));

const player: PlayerIdentity = {
  player_id: 3500,
  player_name: "Granit Xhaka",
  team_id: 904,
  team_name: "Bayer Leverkusen",
  position: "Left Defensive Midfield",
  position_group: "MID",
};

const carries: AttackingActionListResponse = {
  total: 1,
  limit: 200,
  offset: 0,
  items: [{
    action_id: "carry-1",
    pass_index: null,
    match_id: 1,
    possession_id: 2,
    event_index: 3,
    player_id: 3500,
    team_id: 904,
    action_type: "Carry",
    start_x: 50,
    start_y: 30,
    end_x: 65,
    end_y: 35,
    state_value_before: 0.02,
    state_value_after: 0.03,
    attacking_value: 0.01,
    success: true,
    under_pressure: false,
    progressive: true,
    expected_completion: null,
    pass_risk: null,
    risk_reward_category: null,
    fold: 1,
  }],
};

afterEach(() => {
  cleanup();
  mocks.passMap.mockReset();
});

describe("FootyScoutCarryMap sizing", () => {
  it("lets Campos fill the responsive card width without changing carry styling", () => {
    render(<FootyScoutCarryMap player={player} data={carries} />);

    const props = mocks.passMap.mock.calls.at(-1)?.[0] as {
      framePadding: number;
      maxWidth: number;
      lines: unknown;
      dots: unknown;
    };
    expect(props.framePadding).toBe(8);
    expect(props.maxWidth).toBe(1200);
    expect(props.lines).toBeDefined();
    expect(props.dots).toBeDefined();
  });
});
