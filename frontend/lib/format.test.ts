import { describe, expect, it } from "vitest";

import { formatDecimal, formatPercent, formatPercentagePoints, formatSignedDecimal, formatSimilarity, metricLabel } from "./format";

describe("analytics formatting", () => {
  it("formats rates, percentage points, and similarity distinctly", () => {
    expect(formatPercent(0.923)).toBe("92.3%");
    expect(formatPercentagePoints(2.003)).toBe("+2.0 pp");
    expect(formatPercentagePoints(-1.26)).toBe("-1.3 pp");
    expect(formatSimilarity(91.76)).toBe("91.8 / 100");
  });

  it("uses an em dash for nullable analytics", () => {
    expect(formatPercent(null)).toBe("—");
    expect(formatPercentagePoints(null)).toBe("—");
    expect(formatDecimal(null)).toBe("—");
    expect(formatSignedDecimal(null)).toBe("—");
  });

  it("makes positive decimal direction explicit without changing zero or negative values", () => {
    expect(formatSignedDecimal(0.093, 3)).toBe("+0.093");
    expect(formatSignedDecimal(-0.02, 3)).toBe("-0.020");
    expect(formatSignedDecimal(0, 3)).toBe("0.000");
  });

  it("maps internal style fields to product labels", () => {
    expect(metricLabel("role_distance")).toBe("Role Fit distance");
    expect(metricLabel("positive_forward_distance_per_100_passes")).toBe("Forward distance per 100 passes");
    expect(metricLabel("carry_share_of_actions")).toBe("Carry involvement");
    expect(metricLabel("expected_completion_rate")).toBe("Expected completion rate");
    expect(metricLabel("pressure_pass_rate")).toBe("Passes under pressure");
  });
});
