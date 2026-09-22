// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import ActionValueModelPage from "./page";

const mocks = vi.hoisted(() => ({ getActionValueModel: vi.fn() }));

vi.mock("@/lib/api", () => ({ api: mocks }));

const metrics = { rmse: 0.1, mae: 0.08, r2: 0.4, spearman: 0.5, positive_target_rmse: 0.12 };

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("Attacking Impact methodology copy", () => {
  it("presents clean model names and removes an internal version from limitations", async () => {
    mocks.getActionValueModel.mockResolvedValue({
      selection: {
        model: "xgboost_regressor",
        reason: "lowest validation RMSE",
        objective: "future possession value",
        primary_metric: "rmse",
        rounds: { regressor: 100 },
      },
      target_interpretation: "Change in possession value.",
      state_convention: "Pre-action and post-action states.",
      training_corpus: {
        matches: 10,
        events: 100,
        states: 80,
        possessions: 20,
        competitions: [],
      },
      split_methodology: {
        method: "Grouped by match.",
        train_states: 60,
        validation_states: 10,
        test_states: 10,
      },
      nested_cross_fitting: {
        validation: "training folds only",
        heldout_outcomes_used_for_training: false,
      },
      preprocessing: { encoded_feature_count: 12 },
      validation_metrics: { xgboost_regressor: metrics },
      untouched_test_metrics: { selected_model: metrics },
      out_of_fold_metrics: metrics,
      leakage_protection: "Outcome-safe inputs only.",
      feature_columns: ["start_x"],
      limitations: ["defensive value is outside V2.2"],
    });

    render(await ActionValueModelPage());

    expect(screen.getByRole("heading", { name: "Attacking Impact model" })).toBeTruthy();
    expect(screen.getByRole("link", { name: "← Expected Pass model" }).getAttribute("href")).toBe(
      "/model",
    );
    expect(screen.getByRole("link", { name: "Expected Goals model" }).getAttribute("href")).toBe(
      "/model/xg",
    );
    expect(screen.getByText("defensive value is outside this model")).toBeTruthy();
    expect(screen.queryByText(/V2\.2/)).toBeNull();
    expect(screen.getByRole("heading", { name: "Transformer results coming soon" })).toBeTruthy();
    expect(screen.getByText("We recently benchmarked a PyTorch causal Transformer for possession-value prediction using possession history across 667,000+ game states. Full evaluation results and model comparisons will be added here soon.")).toBeTruthy();
    expect(screen.getByText("The current production possession-value model remains XGBoost.")).toBeTruthy();
  });
});
