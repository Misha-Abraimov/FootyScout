import { cx } from "@/lib/format";

export function ComparisonMetric({ label, left, right, format }: { label: string; left: number | null; right: number | null; format: (value: number | null) => string }) {
  const scale = Math.max(Math.abs(left ?? 0), Math.abs(right ?? 0), 0.0001);
  return (
    <div className="rounded-xl border border-[var(--border)] bg-black/10 p-4">
      <p className="mb-4 text-center text-xs font-medium tracking-wide text-[var(--muted)] uppercase">{label}</p>
      <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-3">
        <div className="text-right"><strong className={cx("metric-tabular", (left ?? 0) < 0 && "text-[var(--negative)]")}>{format(left)}</strong><div className="mt-2 ml-auto h-1.5 rounded-full bg-[var(--accent)]" style={{ width: `${Math.max(4, Math.abs(left ?? 0) / scale * 100)}%` }} /></div>
        <span className="text-xs text-[var(--muted)]">vs</span>
        <div><strong className={cx("metric-tabular", (right ?? 0) < 0 && "text-[var(--negative)]")}>{format(right)}</strong><div className="mt-2 h-1.5 rounded-full bg-sky-300" style={{ width: `${Math.max(4, Math.abs(right ?? 0) / scale * 100)}%` }} /></div>
      </div>
    </div>
  );
}
