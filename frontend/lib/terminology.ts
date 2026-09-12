const DISPLAY_LABELS: Readonly<Record<string, string>> = {
  "Expected completion": "Pass difficulty",
  "Expected completion rate": "Pass difficulty",
  "Completion above expected": "Actual vs. expected passing",
  "Completion above expectation": "Actual vs. expected passing",
  "Attacking value": "Attacking impact",
  "Attacking value / 100": "Attacking impact / 100",
  "Attacking value / 100 actions": "Attacking impact / 100 actions",
};

export function displayMetricLabel(label: string): string {
  return DISPLAY_LABELS[label] ?? label;
}
