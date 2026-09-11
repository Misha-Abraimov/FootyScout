import Link from "next/link";

export function ErrorState({
  title = "Data unavailable",
  message = "FootyScout could not reach the analytics API. Check that FastAPI is running and try again.",
}: {
  title?: string;
  message?: string;
}) {
  return (
    <section role="alert" className="rounded-2xl border border-red-300/20 bg-red-300/5 p-6">
      <h2 className="font-semibold text-red-100">{title}</h2>
      <p className="mt-2 max-w-2xl text-sm leading-6 text-red-100/70">{message}</p>
    </section>
  );
}

export function EmptyState({
  title,
  message,
  href,
  action,
}: {
  title: string;
  message: string;
  href?: string;
  action?: string;
}) {
  return (
    <section className="rounded-2xl border border-dashed border-[var(--border)] bg-[var(--panel)] p-8 text-center">
      <h2 className="font-semibold">{title}</h2>
      <p className="mx-auto mt-2 max-w-xl text-sm leading-6 text-[var(--muted)]">{message}</p>
      {href && action ? (
        <Link
          href={href}
          className="mt-5 inline-flex rounded-lg bg-[var(--accent)] px-4 py-2 text-sm font-semibold text-[#07110d]"
        >
          {action}
        </Link>
      ) : null}
    </section>
  );
}

export function LoadingSkeleton() {
  return (
    <div aria-label="Loading FootyScout data" className="mx-auto max-w-7xl animate-pulse px-5 py-10 sm:px-8">
      <div className="h-8 w-56 rounded bg-white/8" />
      <div className="mt-5 h-28 rounded-2xl bg-white/5" />
      <div className="mt-5 grid gap-4 sm:grid-cols-3">
        <div className="h-32 rounded-2xl bg-white/5" />
        <div className="h-32 rounded-2xl bg-white/5" />
        <div className="h-32 rounded-2xl bg-white/5" />
      </div>
    </div>
  );
}
