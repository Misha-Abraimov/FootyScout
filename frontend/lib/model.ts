import type { ModelInfoResponse, ModelMetrics } from "@/lib/types";

export interface ModelPageMetrics {
  selectedValidationKey: string | null;
  selectedValidation: ModelMetrics | undefined;
  selectedTest: ModelMetrics | undefined;
  outOfFold: ModelMetrics;
}

/**
 * Maps the public API's named metric groups to the three summary cards.
 *
 * `selected_model` identifies an architecture (`pytorch_mlp`), while the API's
 * final test result is intentionally published under the `selected_model`
 * metric-group key. They are separate namespaces and must not be indexed as if
 * their keys were interchangeable.
 */
export function mapModelPageMetrics(model: ModelInfoResponse): ModelPageMetrics {
  const effectiveKey = `${model.selected_model}_effective`;
  const selectedValidation =
    model.validation_metrics[effectiveKey] ??
    model.validation_metrics[model.selected_model] ??
    (model.selected_model === "pytorch_mlp"
      ? model.calibration.retained
        ? model.calibration.calibrated_validation_metrics
        : model.calibration.uncalibrated_validation_metrics
      : undefined);

  const selectedValidationKey = model.validation_metrics[effectiveKey]
    ? effectiveKey
    : model.validation_metrics[model.selected_model]
      ? model.selected_model
      : null;

  return {
    selectedValidationKey,
    selectedValidation,
    selectedTest: model.untouched_test_metrics.selected_model,
    outOfFold: model.out_of_fold_metrics,
  };
}
