import { describe, expect, it } from "vitest";

import { modelInfoFixture } from "@/test-fixtures/model-info";
import { mapModelPageMetrics } from "./model";

describe("model page API mapping", () => {
  it("maps the architecture identifier to the API's effective validation group", () => {
    const mapped = mapModelPageMetrics(modelInfoFixture);

    expect(mapped.selectedValidationKey).toBe("xgboost_effective");
    expect(mapped.selectedValidation).toEqual(
      modelInfoFixture.validation_metrics.xgboost_effective,
    );
  });

  it("reads final test metrics from the selected_model metric group", () => {
    const mapped = mapModelPageMetrics(modelInfoFixture);

    expect(mapped.selectedTest).toEqual(
      modelInfoFixture.untouched_test_metrics.selected_model,
    );
    expect(mapped.selectedTest?.roc_auc).toBe(0.9157716266104342);
  });

  it("maps the top-level out-of-fold metrics", () => {
    expect(mapModelPageMetrics(modelInfoFixture).outOfFold).toEqual(
      modelInfoFixture.out_of_fold_metrics,
    );
  });
});
