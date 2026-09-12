// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { SimilarPlayerCard } from "@/components/SimilarPlayerCard";
import type { SimilarPlayerResponse } from "@/lib/types";

const player: SimilarPlayerResponse = {
  similar_player_id: 2,
  similar_player_name: "Example Midfielder",
  similar_team_name: "Example FC",
  similar_position: "Center Midfield",
  similar_position_group: "MID",
  rank: 1,
  rms_distance: 0.42,
  similarity_score: 79.7,
  same_position_group: true,
  closest_feature_1: "progressive_pass_rate",
  closest_feature_2: "carry_share_of_actions",
  closest_feature_3: "expected_completion_rate",
  closest_style_dimensions: [
    "progressive_pass_rate",
    "carry_share_of_actions",
    "expected_completion_rate",
  ],
  query_matches_observed: 33,
  candidate_matches_observed: 2,
  pair_support_matches: 2,
  sample_support: "limited",
  sample_support_explanation: "Limited sample for one profile.",
  methodology_version: "V3.3B",
};

afterEach(cleanup);

describe("SimilarPlayerCard", () => {
  it("keeps style similarity separate from limited sample support", () => {
    render(<SimilarPlayerCard player={player} />);

    expect(screen.getByText("Example Midfielder")).toBeTruthy();
    expect(screen.getByText("Style similarity")).toBeTruthy();
    expect(screen.getByText("79.7 / 100")).toBeTruthy();
    expect(screen.getByText(/Sample support: Limited/)).toBeTruthy();
    expect(screen.getByText("Observed: 33 vs 2 matches")).toBeTruthy();
    expect(screen.getByText("Progressive Pass Rate")).toBeTruthy();
    expect(screen.getByText("Carry Share Of Actions")).toBeTruthy();
  });

  it("renders higher support without altering the score", () => {
    render(
      <SimilarPlayerCard
        player={{
          ...player,
          query_matches_observed: 20,
          candidate_matches_observed: 8,
          pair_support_matches: 8,
          sample_support: "higher",
        }}
      />,
    );

    expect(screen.getByText("79.7 / 100")).toBeTruthy();
    expect(screen.getByText(/Sample support: Higher/)).toBeTruthy();
    expect(screen.getByText("Observed: 20 vs 8 matches")).toBeTruthy();
  });
});
