import { ReliabilityBadge } from "@/components/ReliabilityBadge";
import { formatCount, formatPercent, formatPercentagePoints } from "@/lib/format";

export function PassingSubsetCard({
  title,
  attempts,
  actual,
  expected,
  aboveExpected,
  reliable,
}: {
  title: string;
  attempts: number;
  actual: number | null;
  expected: number | null;
  aboveExpected: number | null;
  reliable: boolean;
}) {
  return (
    <article className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-5">
      <div className="flex items-start justify-between gap-3">
        <div><h3 className="font-semibold">{title}</h3><p className="mt-1 text-xs text-[var(--muted)]">{formatCount(attempts)} attempts</p></div>
        <ReliabilityBadge reliable={reliable} />
      </div>
      <dl className="mt-7 grid grid-cols-3 gap-3">
        <div><dt className="text-xs text-[var(--muted)]">Actual</dt><dd className="metric-tabular mt-1 font-semibold">{formatPercent(actual)}</dd></div>
        <div><dt className="text-xs text-[var(--muted)]">Pass difficulty</dt><dd className="metric-tabular mt-1 font-semibold">{formatPercent(expected)}</dd></div>
        <div><dt className="text-xs text-[var(--muted)]">Passing vs. expected</dt><dd className="metric-tabular mt-1 font-semibold text-[var(--accent-strong)]">{formatPercentagePoints(aboveExpected)}</dd></div>
      </dl>
      {!reliable ? <p className="mt-5 border-t border-[var(--border)] pt-4 text-xs text-amber-100/70">Limited sample — interpret this split cautiously.</p> : null}
    </article>
  );
}
