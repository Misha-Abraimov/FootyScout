// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ArchetypeCatalogueResponse } from "@/lib/types";

import { ArchetypeCatalogue } from "./ArchetypeCatalogue";

vi.mock("next/link", () => ({
  default: ({ children, href }: { children: React.ReactNode; href: string }) => <a href={href}>{children}</a>,
}));

const catalogue: ArchetypeCatalogueResponse = {
  methodology_version: "V3.2C",
  purpose: "Position-relative playing-style archetypes; not quality, ability, or rank.",
  methodology: { method: "K-Means", k: 2 },
  definitions: [
    {
      id: "direct_progressor",
      name: "Direct Progressor",
      description: "A direct and progressive position-relative passing style.",
      centroid: { progressive_pass_rate: 0.76 },
      distinguishing_features: [
        { feature_name: "progressive_pass_rate", label: "Progressive-pass rate", position_z: 0.76, direction: "higher" },
      ],
      player_count: 60,
      position_composition: {
        DEF: { count: 35, percentage: 58.3 },
        MID: { count: 20, percentage: 33.3 },
        FWD: { count: 5, percentage: 8.3 },
      },
      representative_players: [
        { player_id: 15797, player_name: "Matthias Bader", team_name: "Darmstadt 98", position_group: "DEF", centroid_distance: 0.7 },
      ],
      separation_distribution: { median: 0.35 },
    },
    {
      id: "safe_circulator",
      name: "Safe Circulator",
      description: "A safer position-relative circulation style.",
      centroid: { expected_completion_rate: 0.57 },
      distinguishing_features: [
        { feature_name: "expected_completion_rate", label: "Expected completion", position_z: 0.57, direction: "higher" },
      ],
      player_count: 73,
      position_composition: {
        DEF: { count: 40, percentage: 54.8 },
        MID: { count: 27, percentage: 37.0 },
        FWD: { count: 6, percentage: 8.2 },
      },
      representative_players: [],
      separation_distribution: { median: 0.4 },
    },
  ],
};

afterEach(cleanup);

describe("ArchetypeCatalogue", () => {
  it("renders descriptions, composition, and representative-player labeling", () => {
    render(<ArchetypeCatalogue catalogue={catalogue} />);
    expect(screen.getByRole("heading", { name: "Direct Progressor" })).toBeTruthy();
    expect(screen.getByText(/direct and progressive position-relative/)).toBeTruthy();
    expect(screen.getAllByRole("heading", { name: "Representative players" })).toHaveLength(2);
    expect(screen.getByRole("link", { name: "Matthias Bader" }).getAttribute("href")).toBe("/players/15797");
    expect(screen.getAllByText(/not the best players/)).toHaveLength(2);
    expect(screen.getByText("DEF 35 · MID 20 · FWD 5")).toBeTruthy();
    expect(screen.getByText("Pass difficulty")).toBeTruthy();
  });
});
