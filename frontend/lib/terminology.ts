const DISPLAY_LABELS: Readonly<Record<string, string>> = {
  "Expected completion": "Pass difficulty",
  "Expected completion rate": "Pass difficulty",
  "Completion above expected": "Actual vs. expected passing",
  "Completion above expectation": "Actual vs. expected passing",
  "Attacking value": "Attacking impact",
  "Attacking value / 100": "Attacking impact / 100",
  "Attacking value / 100 actions": "Attacking impact / 100 actions",
};

const TABLE_DISPLAY_LABELS: Readonly<Record<string, string>> = {
  "Completion above expected": "Actual vs. expected",
  "Completion above expectation": "Actual vs. expected",
};

export function displayMetricLabel(label: string, context: "detail" | "table" = "detail"): string {
  if (context === "table") return TABLE_DISPLAY_LABELS[label] ?? DISPLAY_LABELS[label] ?? label;
  return DISPLAY_LABELS[label] ?? label;
}
