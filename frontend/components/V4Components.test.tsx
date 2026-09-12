// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type {
  PlayerRoleFitResponse,
  ScoutingRecommendation,
  TeamRoleResponse,
} from "@/lib/types";

import { PlayerRoleFitSection } from "./PlayerRoleFitSection";
import { ScoutingRecommendationCard } from "./ScoutingRecommendationCard";
import { TeamRoleProfile } from "./TeamRoleProfile";

const player = {
  player_id: 10,
  player_name: "Test Midfielder",
  team_id: 20,
  team_name: "Other Club",
  position: "Center Midfield",
  position_group: "MID" as const,
};

function role(positionGroup: "DEF" | "MID" | "FWD" = "MID"): TeamRoleResponse {
  return {
    team_id: 904,
    team_name: "Bayer Leverkusen",
    position_group: positionGroup,
    methodology_version: "V4.2",
    aggregation_method: "pooled_events_actions",
    matches_observed: positionGroup === "FWD" ? 33 : 34,
    contributor_count: positionGroup === "FWD" ? 3 : 10,
    contributors: [{ player_id: 1, player_name: "Contributor", actions: 100, action_share: 0.25 }],
    passes: 1000,
    carries: 800,
    actions: 1800,
    shots: 20,
    support_level: positionGroup === "FWD" ? "limited_contributor_diversity" : "established",
    support_message: positionGroup === "FWD" ? "Limited contributor diversity across three forwards." : "Observed role support.",
    dimensions: [
      { feature_name: "expected_completion_rate", label: "Expected completion", raw_value: 0.86, position_z: 0.42 },
      { feature_name: "progressive_pass_rate", label: "Progressive-pass rate", raw_value: 0.12, position_z: -0.31 },
    ],
    position_context: `Relative to eligible ${positionGroup} players; values describe style, not quality.`,
  };
}

function fit(overrides: Partial<PlayerRoleFitResponse> = {}): PlayerRoleFitResponse {
  return {
    player,
    available: true,
    unavailable_reason: null,
    target_team_id: 904,
    target_team_name: "Bayer Leverkusen",
    position_group: "MID",
    is_target_team_player: false,
    calculation_scope: "full_target_role",
    role_distance: 0.39,
    closest_dimensions: ["progressive_pass_rate", "carry_share_of_actions", "expected_completion_rate"],
    largest_difference: "long_pass_rate",
    feature_gaps: {},
    distance_contributions: {},
    player_matches_observed: 2,
    sample_support: "limited",
    sample_support_message: "Limited sample: two matches.",
    role_matches_observed: 34,
    role_contributor_count: 10,
    role_actions: 19867,
    role_support_message: "Observed role support.",
    methodology_version: "V4.3",
    interpretation: "Role Fit measures observed playing style and does not predict transfer success.",
    ...overrides,
  };
}

afterEach(cleanup);

describe("TeamRoleProfile", () => {
  it("renders position-relative style dimensions and support counts", () => {
    render(<TeamRoleProfile role={role()} />);
    expect(screen.getByText("MID")).toBeTruthy();
    expect(screen.getByLabelText(/Pass difficulty: 0.42 position-relative/)).toBeTruthy();
    expect(screen.getByText("10")).toBeTruthy();
    expect(screen.getByText(/style, not quality/)).toBeTruthy();
  });

  it("keeps the forward role visible with its contributor warning", () => {
    render(<TeamRoleProfile role={role("FWD")} />);
    expect(screen.getByText("FWD")).toBeTruthy();
    expect(screen.getByText(/Limited contributor diversity/)).toBeTruthy();
  });
});

describe("PlayerRoleFitSection", () => {
  it("renders raw distance, alignment, and limited sample separately", () => {
    render(<PlayerRoleFitSection fit={fit()} />);
    expect(screen.getByText("0.390")).toBeTruthy();
    expect(screen.getByText(/lower is closer/i)).toBeTruthy();
    expect(screen.getByText(/How closely a player's playing style matches this role/)).toBeTruthy();
    expect(screen.getByText("limited sample")).toBeTruthy();
    expect(screen.getByText(/does not predict transfer success/)).toBeTruthy();
  });

  it("uses leave-self-out copy for a current Leverkusen player", () => {
    render(<PlayerRoleFitSection fit={fit({ is_target_team_player: true, calculation_scope: "leave_self_out_target_role" })} />);
    expect(screen.getByText(/Leverkusen's other MID contributors/)).toBeTruthy();
  });

  it("renders a clean unavailable state", () => {
    render(<PlayerRoleFitSection fit={fit({ available: false, role_distance: null, unavailable_reason: "Requires 50 passes." })} />);
    expect(screen.getByText("Role Fit unavailable")).toBeTruthy();
    expect(screen.getByText("Requires 50 passes.")).toBeTruthy();
  });
});

describe("ScoutingRecommendationCard", () => {
  it("shows role distance, rank, metadata, and sample support", () => {
    const recommendation: ScoutingRecommendation = {
      rank: 1,
      player,
      role_distance: 0.39,
      closest_dimensions: ["progressive_pass_rate", "carry_share_of_actions"],
      largest_difference: "long_pass_rate",
      sample_support: "limited",
      sample_support_message: "Limited sample.",
      player_matches_observed: 2,
      archetype_name: "Safe Circulator",
    };
    render(<ScoutingRecommendationCard recommendation={recommendation} />);
    expect(screen.getByText("#1")).toBeTruthy();
    expect(screen.getByText("0.390")).toBeTruthy();
    expect(screen.getByText("limited")).toBeTruthy();
    expect(screen.getByText(/2 observed matches/)).toBeTruthy();
  });
});
