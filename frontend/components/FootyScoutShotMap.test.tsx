// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { PlayerIdentity, ShotListResponse, ShotResponse } from "@/lib/types";

import { FootyScoutShotMap } from "./FootyScoutShotMap";

const mocks = vi.hoisted(() => ({ shotMap: vi.fn() }));

vi.mock("@withqwerty/campos-react", () => ({
  DARK_THEME: {},
  ThemeProvider: ({ children }: { children: ReactNode }) => <>{children}</>,
  ShotMap: (props: unknown) => {
    mocks.shotMap(props);
    return <div data-testid="campos-shot-map" />;
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

function shot(shotId: string, goal: boolean, expectedGoal: number): ShotResponse {
  return {
    shot_id: shotId,
    match_id: 1,
    player_id: 3500,
    period: 1,
    minute: 12,
    second: 34,
    start_x: 108,
    start_y: 40,
    distance: 12,
    angle: 0.64,
    goal,
    expected_goal: expectedGoal,
    body_part: "Right Foot",
    shot_type: "Open Play",
    technique: "Normal",
    play_pattern: "Regular Play",
    under_pressure: false,
    first_time: false,
    one_on_one: false,
    open_goal: false,
    penalty: false,
    penalty_shootout: false,
    model_eligible: true,
  };
}

function response(items: ShotResponse[]): ShotListResponse {
  return { total: items.length, limit: 200, offset: 0, items };
}

afterEach(() => {
  cleanup();
  mocks.shotMap.mockReset();
});

describe("FootyScoutShotMap outcome treatment", () => {
  it("adds the mint outline only to goals while preserving Campos xG encodings", () => {
    render(
      <FootyScoutShotMap
        player={player}
        shots={response([shot("goal", true, 0.42), shot("miss", false, 0.08)])}
      />,
    );

    const props = mocks.shotMap.mock.calls.at(-1)?.[0] as {
      shots: Array<{ id: string; outcome: string; xg: number | null }>;
      colorScale: string;
      markers: {
        stroke: (context: { shot: { outcome: string } }) => string | undefined;
        strokeWidth: (context: { shot: { outcome: string } }) => number | undefined;
      };
    };
    const goal = props.shots.find((item) => item.id === "footyscout:goal");
    const miss = props.shots.find((item) => item.id === "footyscout:miss");

    expect(goal).toMatchObject({ outcome: "goal", xg: 0.42 });
    expect(miss).toMatchObject({ outcome: "other", xg: 0.08 });
    expect(props.colorScale).toBe("magma");
    expect(Object.keys(props.markers).sort()).toEqual(["stroke", "strokeWidth"]);
    expect(props.markers.stroke({ shot: { outcome: "goal" } })).toBe("var(--accent)");
    expect(props.markers.strokeWidth({ shot: { outcome: "goal" } })).toBe(0.9);
    expect(props.markers.stroke({ shot: { outcome: "other" } })).toBeUndefined();
    expect(props.markers.strokeWidth({ shot: { outcome: "other" } })).toBeUndefined();
  });

  it("renders the goal and non-goal legend explanation", () => {
    render(<FootyScoutShotMap player={player} shots={response([shot("goal", true, 0.42)])} />);

    const legend = screen.getByLabelText("Shot outcome legend");
    expect(legend.textContent).toContain("Green outline = Goal");
    expect(legend.textContent).toContain("Normal outline = Non-goal");
  });
});
