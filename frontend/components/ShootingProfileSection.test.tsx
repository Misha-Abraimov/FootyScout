// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { ShootingProfileSection } from "./ShootingProfileSection";

afterEach(cleanup);

const profile = {
  player_id: 1,
  shots: 8,
  goals: 2,
  total_xg: 1.4,
  xg_per_shot: 0.175,
  goals_minus_xg: 0.6,
  goals_per_shot: 0.25,
  matches_observed: 5,
  shooting_reliable: false,
};

describe("ShootingProfileSection", () => {
  it("renders shooting aggregates and limited-sample status", () => {
    render(<ShootingProfileSection profile={profile} />);
    expect(screen.getByRole("heading", { name: "Non-penalty shooting" })).toBeTruthy();
    expect(screen.getByText("Limited sample: fewer than 20 non-penalty shots.")).toBeTruthy();
    expect(screen.getByText("+0.6")).toBeTruthy();
  });

  it("renders nothing when shooting data is unavailable", () => {
    const { container } = render(<ShootingProfileSection profile={null} />);
    expect(container.innerHTML).toBe("");
  });
});
