const DISPLAY_LABELS: Readonly<Record<string, string>> = {
  "Expected completion": "Pass difficulty",
  "Expected completion rate": "Pass difficulty",
  "Completion above expected": "Passing vs. expected",
  "Completion above expectation": "Passing vs. expected",
  "Attacking value": "Attacking impact",
  "Attacking value / 100": "Attacking impact / 100",
  "Attacking value / 100 actions": "Attacking impact / 100 actions",
};

export function displayMetricLabel(label: string): string {
  return DISPLAY_LABELS[label] ?? label;
}
