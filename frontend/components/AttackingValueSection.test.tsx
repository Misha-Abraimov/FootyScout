// @vitest-environment jsdom

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { AttackingValueSection } from "./AttackingValueSection";
import type { AttackingProfileResponse } from "@/lib/types";

const profile: AttackingProfileResponse = {
  player_id: 3500, matches_observed: 25, actions: 100, passes: 70, carries: 30,
  total_attacking_value: 0.5, attacking_value_per_100_actions: 0.5,
  total_pass_value: 0.4, pass_value_per_100_passes: 0.571,
  total_carry_value: 0.1, carry_value_per_100_carries: 0.333,
  positive_value_actions: 55, positive_value_action_rate: 0.55,
  progressive_action_value: 0.25, progressive_value_per_100_actions: 0.25,
  pressure_action_value: 0.15, pressure_value_per_100_actions: 0.15,
  attacking_value_reliable: true, pass_value_reliable: true, carry_value_reliable: true,
};

describe("AttackingValueSection", () => {
  it("keeps passing, carrying, and total action value distinct", () => {
    render(<AttackingValueSection profile={profile} />);
    expect(screen.getByText("Possession value added")).toBeTruthy();
    expect(screen.getByText("Action value / 100")).toBeTruthy();
    expect(screen.getByText("Pass value / 100")).toBeTruthy();
    expect(screen.getByText("Carry value / 100")).toBeTruthy();
    expect(screen.getByText(/not causal impact/)).toBeTruthy();
  });

  it("omits the section when no profile exists", () => {
    const { container } = render(<AttackingValueSection profile={null} />);
    expect(container.innerHTML).toBe("");
  });
});
