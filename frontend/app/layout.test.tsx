import { describe, expect, it } from "vitest";

import { metadata } from "./layout";

describe("application metadata", () => {
  it("uses the FootyScout brand in browser titles and descriptions", () => {
    expect(metadata.title).toEqual({
      default: "FootyScout — football scouting intelligence",
      template: "%s | FootyScout",
    });
    expect(metadata.description).toContain("FootyScout");
  });
});
