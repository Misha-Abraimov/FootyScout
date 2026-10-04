export function formatPercent(value: number | null, digits = 1): string {
  return value === null ? "—" : `${(value * 100).toFixed(digits)}%`;
}

export function formatPercentagePoints(value: number | null, digits = 1): string {
  if (value === null) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)} pp`;
}

export function formatSimilarity(value: number): string {
  return `${value.toFixed(1)} / 100`;
}

export function formatCount(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

export function formatDecimal(value: number | null, digits = 1): string {
  return value === null ? "—" : value.toFixed(digits);
}

export function formatSignedDecimal(value: number | null, digits = 1): string {
  if (value === null) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}`;
}

export function formatPercentile(value: number | null): string {
  if (value === null) return "Unavailable";
  const rounded = Math.round(value);
  const remainder = rounded % 100;
  const suffix = remainder >= 11 && remainder <= 13
    ? "th"
    : ({ 1: "st", 2: "nd", 3: "rd" }[rounded % 10] ?? "th");
  return `${rounded}${suffix} percentile`;
}

export function humanizeField(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

const metricLabels: Record<string, string> = {
  role_distance: "Role Fit distance",
  positive_forward_distance_per_100_passes: "Forward distance per 100 passes",
  average_forward_distance: "Average forward distance per pass",
  carry_share_of_actions: "Carry involvement",
  actual_completion_rate: "Actual pass completion",
  expected_completion_rate: "Expected completion rate",
  completion_above_expected_pp: "Actual vs. expected passing",
  pressure_pass_rate: "Passes under pressure",
  progressive_pass_rate: "Progressive passing rate",
  long_pass_rate: "Long-pass rate",
  attacking_value: "Attacking impact",
  similarity_score: "Style similarity",
};

export function metricLabel(value: string): string {
  return metricLabels[value] ?? humanizeField(value);
}

export function cx(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}
