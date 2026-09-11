// @vitest-environment jsdom

import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { xgModelInfoFixture } from "@/test-fixtures/xg-model-info";
import { XGModelEvaluation } from "./XGModelEvaluation";

afterEach(cleanup);

describe("XGModelEvaluation", () => {
  it("renders validation, untouched test, OOF, and feature importance", () => {
    render(<XGModelEvaluation model={xgModelInfoFixture} />);
    const validation = screen.getByRole("heading", { name: "Validation · selected form" }).closest("article") as HTMLElement;
    const test = screen.getByRole("heading", { name: "Untouched test" }).closest("article") as HTMLElement;
    const oof = screen.getByRole("heading", { name: "Grouped out-of-fold" }).closest("article") as HTMLElement;
    expect(within(validation).getByText("0.2859")).toBeTruthy();
    expect(within(test).getByText("0.2507")).toBeTruthy();
    expect(within(oof).getByText("0.2700")).toBeTruthy();
    expect(screen.getByText("Angle To Goal")).toBeTruthy();
  });

  it("does not crash when an optional selected metric group is missing", () => {
    render(<XGModelEvaluation model={{ ...xgModelInfoFixture, validation_metrics: {} }} />);
    const validation = screen.getByRole("heading", { name: "Validation · selected form" }).closest("article") as HTMLElement;
    expect(within(validation).getAllByText("—")).toHaveLength(5);
  });
});
