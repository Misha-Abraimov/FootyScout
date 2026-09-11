// @vitest-environment jsdom

import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { modelInfoFixture } from "@/test-fixtures/model-info";
import { ModelEvaluation } from "./ModelEvaluation";
import { ModelMetricCard } from "./ModelMetricCard";

afterEach(cleanup);

function card(title: string): HTMLElement {
  return screen.getByRole("heading", { name: title }).closest("article") as HTMLElement;
}

describe("ModelEvaluation", () => {
  it("renders validation, untouched-test, and OOF values from the real API shape", () => {
    render(<ModelEvaluation model={modelInfoFixture} />);

    expect(within(card("Validation · selected form")).getByText("0.9210")).toBeTruthy();
    expect(within(card("Validation · selected form")).getByText("0.2615")).toBeTruthy();
    expect(within(card("Untouched test")).getByText("0.9158")).toBeTruthy();
    expect(within(card("Untouched test")).getByText("0.2354")).toBeTruthy();
    expect(within(card("Grouped out-of-fold")).getByText("0.9221")).toBeTruthy();
    expect(within(card("Grouped out-of-fold")).getByText("0.2338")).toBeTruthy();
  });

  it("renders the Logistic Regression baseline and effective PyTorch metrics", () => {
    render(<ModelEvaluation model={modelInfoFixture} />);

    expect(screen.getAllByRole("row", { name: /Logistic Regression/ })).toHaveLength(2);
    expect(screen.getAllByRole("row", { name: /PyTorch MLP/ })).toHaveLength(2);
    expect(screen.getAllByRole("row", { name: /XGBoost · Selected/ })).toHaveLength(2);
  });

  it("does not crash when the selected test metric group is unavailable", () => {
    render(
      <ModelEvaluation
        model={{ ...modelInfoFixture, untouched_test_metrics: {} }}
      />,
    );

    expect(within(card("Untouched test")).getAllByText("—")).toHaveLength(5);
  });
});

describe("ModelMetricCard", () => {
  it("formats every supplied metric to four decimal places", () => {
    render(
      <ModelMetricCard
        title="Formatting"
        metrics={{
          roc_auc: 0.9,
          log_loss: 0.123456,
          brier_score: 0.08,
          accuracy: 1,
          expected_calibration_error: 0.004873,
        }}
      />,
    );

    const metricCard = card("Formatting");
    for (const value of ["0.9000", "0.1235", "0.0800", "1.0000", "0.0049"]) {
      expect(within(metricCard).getByText(value)).toBeTruthy();
    }
  });
});
