import { cx } from "@/lib/format";

export function ReliabilityBadge({ reliable }: { reliable: boolean }) {
  return (
    <span
      className={cx(
        "inline-flex rounded-full border px-2 py-1 text-[11px] font-semibold tracking-wide uppercase",
        reliable
          ? "border-emerald-300/25 bg-emerald-300/10 text-emerald-200"
          : "border-amber-300/25 bg-amber-300/10 text-amber-200",
      )}
    >
      {reliable ? "Reliable sample" : "Limited sample"}
    </span>
  );
}
