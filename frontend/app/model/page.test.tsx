// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { modelInfoFixture } from "@/test-fixtures/model-info";

import ModelPage from "./page";

const mocks = vi.hoisted(() => ({ getModel: vi.fn() }));

vi.mock("@/lib/api", () => ({ api: mocks }));
vi.mock("@/components/ModelEvaluation", () => ({
  ModelEvaluation: () => <div data-testid="model-evaluation" />,
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("model methodology navigation", () => {
  it("uses current model names while preserving the existing routes", async () => {
    mocks.getModel.mockResolvedValue(modelInfoFixture);

    render(await ModelPage());

    expect(screen.getByRole("link", { name: "Expected Goals model →" }).getAttribute("href")).toBe(
      "/model/xg",
    );
    expect(
      screen.getByRole("link", { name: "Attacking Value model →" }).getAttribute("href"),
    ).toBe("/model/action-value");
    expect(screen.getByText("Player Intelligence")).toBeTruthy();
    expect(screen.getByText("Team Intelligence")).toBeTruthy();
    expect(screen.queryByText(/V2\.1|V2\.2|V3\.3B methodology|V4 methodology/)).toBeNull();
  });
});
