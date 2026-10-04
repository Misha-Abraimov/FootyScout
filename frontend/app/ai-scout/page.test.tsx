// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import AIScoutPage from "./page";

vi.mock("@/components/AIScoutExperience", () => ({
  AIScoutExperience: () => (
    <section aria-label="AI Scout card" className="rounded-2xl border border-[#242424]">
      AI Scout card
    </section>
  ),
}));

afterEach(cleanup);

describe("AI Scout page layout", () => {
  it("removes the header divider while retaining the rounded card border", () => {
    render(<AIScoutPage />);

    const header = screen.getByRole("banner");
    expect(header.className).not.toContain("border-b");
    expect(header.className).toContain("pb-8");

    const card = screen.getByRole("region", { name: "AI Scout card" });
    expect(card.className).toContain("rounded-2xl");
    expect(card.className).toContain("border");
    expect(card.className).toContain("border-[#242424]");
  });
});
