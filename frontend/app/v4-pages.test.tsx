// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type {
  ScoutingRecommendationsResponse,
  TeamIntelligenceResponse,
  TeamRoleResponse,
} from "@/lib/types";

import ScoutingPage from "./scouting/page";
import TeamIntelligencePage from "./teams/[id]/page";

const mocks = vi.hoisted(() => ({
  getScoutingRecommendations: vi.fn(),
  getTeamIntelligence: vi.fn(),
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return { ...actual, api: mocks };
});

function role(positionGroup: "DEF" | "MID" | "FWD"): TeamRoleResponse {
  return {
    team_id: 904,
    team_name: "Bayer Leverkusen",
    position_group: positionGroup,
    methodology_version: "V4.2",
    aggregation_method: "pooled_events_actions",
    matches_observed: positionGroup === "FWD" ? 33 : 34,
    contributor_count: positionGroup === "FWD" ? 3 : 9,
    contributors: [],
    passes: 1000,
    carries: 800,
    actions: 1800,
    shots: 20,
    support_level:
      positionGroup === "FWD" ? "limited_contributor_diversity" : "established",
    support_message:
      positionGroup === "FWD"
        ? "Limited contributor diversity across three forwards."
        : "Observed role support.",
    dimensions: [
      {
        feature_name: "expected_completion_rate",
        label: "Expected completion",
        raw_value: 0.86,
        position_z: 0.42,
      },
    ],
    position_context: `Relative to eligible ${positionGroup} players; values describe style, not quality.`,
  };
}

const intelligence: TeamIntelligenceResponse = {
  team: {
    team_id: 904,
    team_name: "Bayer Leverkusen",
    methodology_version: "V4.1",
    sample_scope: "Observed across the 34 Bundesliga matches in the product sample.",
    matches_observed: 34,
    contributors: 24,
    passes: 24_244,
    carries: 20_141,
    actions: 43_697,
    shots: 623,
    metrics: {
      expected_completion_rate: 0.872,
      pressure_pass_rate: 0.137,
      progressive_pass_rate: 0.127,
      long_pass_rate: 0.113,
      positive_forward_distance_per_100_passes: 677.6,
      carry_share_of_actions: 0.461,
      average_forward_distance: 3.34,
      final_third_entries_per_100_passes: 8.71,
      progressive_carry_rate: 0.046,
      progressive_action_rate: 0.088,
      pressure_action_rate: 0.191,
      shots_per_match: 18.32,
      xg_per_shot: 0.1,
      xg_per_match: 1.81,
      attacking_value_per_100_actions: 0.026,
    },
  },
  roles: [role("DEF"), role("MID"), role("FWD")],
  fit_definition: "Role Fit measures observed style resemblance.",
};

const recommendations: ScoutingRecommendationsResponse = {
  target_team_id: 904,
  target_team_name: "Bayer Leverkusen",
  position_group: "FWD",
  methodology_version: "V4.4",
  definition: "Scouting Recommendations rank players by Role Fit.",
  disclaimer: "They are not predictions of transfer success.",
  role_support_message: "Limited contributor diversity across three forwards.",
  total: 1,
  limit: 6,
  items: [
    {
      rank: 1,
      player: {
        player_id: 5623,
        player_name: "Jae-Sung Lee",
        team_id: 177,
        team_name: "FSV Mainz 05",
        position: "Left Wing",
        position_group: "FWD",
      },
      role_distance: 0.65,
      closest_dimensions: ["expected_completion_rate", "long_pass_rate"],
      largest_difference: "carry_share_of_actions",
      sample_support: "limited",
      sample_support_message: "Limited sample.",
      player_matches_observed: 2,
      archetype_name: "Safe Circulator",
    },
  ],
};

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("V4 production pages", () => {
  it("maps Team Intelligence data into descriptive metrics and all role profiles", async () => {
    mocks.getTeamIntelligence.mockResolvedValue(intelligence);
    render(await TeamIntelligencePage({ params: Promise.resolve({ id: "904" }) }));

    expect(screen.getByRole("heading", { name: "Bayer Leverkusen" })).toBeTruthy();
    expect(screen.getByText("87.2%")).toBeTruthy();
    expect(screen.getByText("Limited contributor diversity across three forwards.")).toBeTruthy();
    expect(mocks.getTeamIntelligence).toHaveBeenCalledWith(904);
  });

  it("maps the selected role into the FWD shortlist and limitation message", async () => {
    mocks.getScoutingRecommendations.mockResolvedValue(recommendations);
    render(await ScoutingPage({ searchParams: Promise.resolve({ role: "FWD" }) }));

    expect(screen.getByRole("heading", { name: "Scouting Recommendation Engine" })).toBeTruthy();
    expect(screen.getByText("Jae-Sung Lee")).toBeTruthy();
    expect(screen.getByText("0.650")).toBeTruthy();
    expect(screen.getByText("Limited contributor diversity across three forwards.")).toBeTruthy();
    expect(mocks.getScoutingRecommendations).toHaveBeenCalledWith(904, "FWD", 6);
  });
});
