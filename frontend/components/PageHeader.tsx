import type { ReactNode } from "react";

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
  neutral = false,
}: {
  eyebrow: string;
  title: string;
  description: string;
  actions?: ReactNode;
  neutral?: boolean;
}) {
  return (
    <header className={`pb-8 ${neutral ? "" : "border-b border-[var(--border)]"}`}>
      <p className={`text-xs font-semibold tracking-[0.18em] uppercase ${neutral ? "text-[var(--muted)]" : "text-[var(--accent)]"}`}>
        {eyebrow}
      </p>
      <div className="mt-3 flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h1 className="max-w-4xl text-3xl font-semibold tracking-[-0.035em] sm:text-5xl">
            {title}
          </h1>
          <p className="mt-4 max-w-3xl text-base leading-7 text-[var(--muted)]">{description}</p>
        </div>
        {actions ? <div className="shrink-0">{actions}</div> : null}
      </div>
    </header>
  );
}
