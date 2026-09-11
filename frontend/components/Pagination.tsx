import Link from "next/link";

export function Pagination({
  pathname,
  query,
  total,
  limit,
  offset,
}: {
  pathname: string;
  query: Record<string, string | undefined>;
  total: number;
  limit: number;
  offset: number;
}) {
  const makeHref = (nextOffset: number) => {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(query)) {
      if (value) params.set(key, value);
    }
    params.set("offset", String(nextOffset));
    return `${pathname}?${params.toString()}`;
  };
  const start = total === 0 ? 0 : offset + 1;
  const end = Math.min(total, offset + limit);
  return (
    <nav aria-label="Player results pagination" className="flex items-center justify-between gap-4 text-sm">
      <p className="text-[var(--muted)]">Showing {start}–{end} of {total}</p>
      <div className="flex gap-2">
        {offset > 0 ? (
          <Link className="rounded-lg border border-[var(--border)] px-4 py-2 hover:bg-white/5" href={makeHref(Math.max(0, offset - limit))}>
            Previous
          </Link>
        ) : null}
        {offset + limit < total ? (
          <Link className="rounded-lg border border-[var(--border)] px-4 py-2 hover:bg-white/5" href={makeHref(offset + limit)}>
            Next
          </Link>
        ) : null}
      </div>
    </nav>
  );
}
