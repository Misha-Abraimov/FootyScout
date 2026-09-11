import type { ReactNode } from "react";

export function MetricCard({
  label,
  value,
  note,
  eyebrow,
}: {
  label: string;
  value: ReactNode;
  note?: string;
  eyebrow?: string;
}) {
  return (
    <article className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-5">
      {eyebrow ? (
        <p className="mb-4 text-[11px] font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">
          {eyebrow}
        </p>
      ) : null}
      <p className="text-sm text-[var(--muted)]">{label}</p>
      <p className="metric-tabular mt-2 text-2xl font-semibold tracking-tight">{value}</p>
      {note ? <p className="mt-2 text-xs leading-5 text-[var(--muted)]">{note}</p> : null}
    </article>
  );
}
