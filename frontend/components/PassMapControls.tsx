import type { DifficultyFilter, PassViewFilter } from "@/lib/pass-filters";

const views: Array<[PassViewFilter, string]> = [
  ["all", "All"],
  ["completed", "Completed"],
  ["incomplete", "Incomplete"],
  ["progressive", "Progressive"],
  ["pressure", "Under pressure"],
];

export function PassMapControls({
  view,
  difficulty,
  disabled,
  onViewChange,
  onDifficultyChange,
}: {
  view: PassViewFilter;
  difficulty: DifficultyFilter;
  disabled: boolean;
  onViewChange: (view: PassViewFilter) => void;
  onDifficultyChange: (difficulty: DifficultyFilter) => void;
}) {
  return (
    <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
      <fieldset>
        <legend className="mb-2 text-xs font-medium text-[var(--muted)]">Pass subset</legend>
        <div className="flex flex-wrap gap-2">
          {views.map(([value, label]) => (
            <button key={value} type="button" disabled={disabled} aria-pressed={view === value} onClick={() => onViewChange(value)} className="rounded-lg border border-[var(--border)] px-3 py-2 text-sm aria-pressed:border-emerald-300/40 aria-pressed:bg-emerald-300/10 aria-pressed:text-emerald-100 disabled:opacity-50">{label}</button>
          ))}
        </div>
      </fieldset>
      <label className="grid gap-2 text-xs text-[var(--muted)]">
        Expected-completion difficulty <span className="sr-only">filter</span>
        <select value={difficulty} disabled={disabled} onChange={(event) => onDifficultyChange(event.target.value as DifficultyFilter)} className="min-w-56 rounded-lg border border-[var(--border)] bg-[#0a100e] px-3 py-2.5 text-sm text-white">
          <option value="all">All difficulty</option>
          <option value="difficult">Difficult (&lt; 60%)</option>
          <option value="medium">Medium (60–85%)</option>
          <option value="routine">Routine (&gt; 85%)</option>
        </select>
      </label>
    </div>
  );
}
