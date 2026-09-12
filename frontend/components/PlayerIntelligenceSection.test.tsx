// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { PlayerIntelligenceMetric, PlayerIntelligenceResponse } from "@/lib/types";

import { ArchetypeComparison, PositionPercentiles } from "./CompareClient";
import { PlayerIntelligenceSection } from "./PlayerIntelligenceSection";

const mocks = vi.hoisted(() => ({ radar: vi.fn() }));

vi.mock("@withqwerty/campos-react", () => ({
  DARK_THEME: {},
  ThemeProvider: ({ children }: { children: ReactNode }) => <>{children}</>,
  RadarChart: (props: unknown) => {
    mocks.radar(props);
    return <div data-testid="percentile-radar" />;
  },
}));

function metric(
  metricName: string,
  label: string,
  family: "style" | "performance",
  percentile: number | null,
  rawValue = 0.187,
): PlayerIntelligenceMetric {
  return {
    metric_name: metricName,
    label,
    family,
    raw_value: rawValue,
    unit: family === "style" ? "rate" : "percentage_points",
    percentile,
    peer_position_group: "MID",
    peer_count: 24,
    sample_count: 140,
    minimum_sample: 100,
    eligible: percentile !== null,
    eligibility_reason: percentile === null ? "requires_100_passes" : "eligible",
    directionality: "higher means more",
    stability_note: null,
  };
}

function intelligence(positionGroup: "GK" | "DEF" | "MID" | "FWD" = "MID"): PlayerIntelligenceResponse {
  const style = metric("progressive_pass_rate", "Progressive pass rate", "style", 87);
  const pressure = metric("pressure_pass_rate", "Under-pressure pass rate", "style", 62);
  const performance = metric("completion_above_expected_pp", "Completion above expected", "performance", 73, 2.4);
  return {
    player: {
      player_id: 1, player_name: "Player One", team_id: 10, team_name: "Team",
      position: "Center Midfield", position_group: positionGroup,
    },
    matches_observed: 12,
    position_group: positionGroup,
    style_metrics: [style, pressure, metric("long_pass_rate", "Long-pass rate", "style", null)],
    performance_metrics: [performance],
    radar_metrics: [style, pressure, performance],
    radar_status: positionGroup === "GK" ? "Limited goalkeeper profile" : "Available",
    percentile_context: "Percentiles are not ratings.",
    archetype: positionGroup === "GK" ? {
      id: null,
      name: null,
      eligible: false,
      eligibility_reason: "Goalkeepers are excluded from the outfield archetype model.",
      position_group: positionGroup,
      centroid_distance: null,
      second_centroid_distance: null,
      separation_margin: null,
      separation_interpretation: "Distance-based style separation, not a probability.",
      style_dimensions: [],
      distinguishing_features: [],
      methodology_version: "V3.2C",
    } : {
      id: "direct_progressor",
      name: "Direct Progressor",
      eligible: true,
      eligibility_reason: null,
      position_group: positionGroup,
      centroid_distance: 0.7,
      second_centroid_distance: 2.1,
      separation_margin: 0.667,
      separation_interpretation: "Distance-based style separation, not a probability.",
      style_dimensions: [
        { feature_name: "progressive_pass_rate", label: "Progressive-pass rate", position_z: 0.8 },
        { feature_name: "expected_completion_rate", label: "Expected completion", position_z: -0.5 },
      ],
      distinguishing_features: [
        { feature_name: "progressive_pass_rate", label: "Progressive-pass rate", position_z: 0.8, direction: "higher" },
      ],
      methodology_version: "V3.2C",
    },
  };
}

afterEach(() => {
  cleanup();
  mocks.radar.mockReset();
});

describe("PlayerIntelligenceSection", () => {
  it("renders an accessible percentile radar and separate style/performance details", () => {
    render(<PlayerIntelligenceSection profile={intelligence()} />);
    expect(screen.getByTestId("percentile-radar")).toBeTruthy();
    expect(screen.getByTestId("position-profile-radar").className).toContain("max-w-[640px]");
    expect(screen.getByTestId("position-profile-radar").className).toContain("w-full");
    expect(screen.getByLabelText("Style percentiles")).toBeTruthy();
    expect(screen.getByLabelText("Performance percentiles")).toBeTruthy();
    expect(screen.getByText(/87th percentile among MID players/)).toBeTruthy();
    expect(screen.getAllByText(/n = 24 eligible peers/).length).toBeGreaterThan(0);
  });

  it("shows unavailable metrics instead of assigning zero", () => {
    render(<PlayerIntelligenceSection profile={intelligence()} />);
    expect(screen.getByText(/Unavailable among MID players/)).toBeTruthy();
    expect(screen.getByText("requires 100 passes")).toBeTruthy();
  });

  it("uses a limited-profile state for goalkeepers", () => {
    const profile = intelligence("GK");
    profile.radar_metrics = [];
    render(<PlayerIntelligenceSection profile={profile} />);
    expect(screen.getByText("Limited goalkeeper profile")).toBeTruthy();
    expect(mocks.radar).not.toHaveBeenCalled();
    expect(screen.getByText("Archetype unavailable")).toBeTruthy();
    expect(screen.getByText(/Goalkeepers are excluded/)).toBeTruthy();
  });

  it("renders the archetype and position-relative z-score bars without rating language", () => {
    render(<PlayerIntelligenceSection profile={intelligence()} />);
    expect(screen.getByRole("heading", { name: "Direct Progressor" })).toBeTruthy();
    expect(screen.getByLabelText(/Progressive-pass rate: 0.80 standard deviations/)).toBeTruthy();
    expect(screen.getByLabelText(/Pass difficulty: -0.50 standard deviations/)).toBeTruthy();
    expect(screen.getByText("Passing vs. expected")).toBeTruthy();
    expect(screen.getByText(/not percentiles or ratings/)).toBeTruthy();
  });
});

describe("PositionPercentiles", () => {
  it("states that different-position percentiles use different peer distributions", () => {
    const left = intelligence("MID");
    const right = intelligence("DEF");
    right.player = { ...right.player, player_id: 2, player_name: "Player Two", position_group: "DEF" };
    render(<PositionPercentiles left={left} right={right} sameGroup={false} />);
    expect(screen.getByText(/These are not the same peer distribution/)).toBeTruthy();
    expect(screen.getByText(/Percentiles are not ratings/)).toBeTruthy();
  });
});

describe("ArchetypeComparison", () => {
  it("shows eligible and limited-sample archetypes without an ability inference", () => {
    const left = intelligence("MID");
    const right = intelligence("GK");
    right.player = { ...right.player, player_id: 2, player_name: "Keeper", position_group: "GK" };
    render(<ArchetypeComparison left={left} right={right} />);
    expect(screen.getByText("Direct Progressor")).toBeTruthy();
    expect(screen.getByText("Archetype unavailable")).toBeTruthy();
    expect(screen.getByText(/not player quality or overall similarity/)).toBeTruthy();
  });
});
