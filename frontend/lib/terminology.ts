const DISPLAY_LABELS: Readonly<Record<string, string>> = {
  "Expected completion": "Pass difficulty",
  "Expected completion rate": "Pass difficulty",
  "Completion above expected": "Actual vs. expected passing",
  "Completion above expectation": "Actual vs. expected passing",
  "Attacking value": "Attacking impact",
  "Attacking value / 100": "Overall impact / 100 actions",
  "Attacking value / 100 actions": "Overall impact / 100 actions",
  "Pass value / 100 passes": "Passing impact / 100 passes",
  "Carry value / 100 carries": "Carrying impact / 100 carries",
  "Progressive value / 100 actions": "Progressive-action impact / 100 actions",
  "Under-pressure value / 100 actions": "Under-pressure impact / 100 actions",
};

export function displayMetricLabel(label: string): string {
  return DISPLAY_LABELS[label] ?? label;
}
