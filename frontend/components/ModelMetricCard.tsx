import type { ModelMetrics } from "@/lib/types";

export function ModelMetricCard({ title, metrics, highlighted = false, labels = {} }: { title: string; metrics?: ModelMetrics | null; highlighted?: boolean; labels?: Partial<Record<keyof ModelMetrics, string>> }) {
  return (
    <article className={highlighted ? "rounded-2xl border border-emerald-300/30 bg-emerald-300/[0.04] p-5" : "rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-5"}>
      <h3 className="font-semibold">{title}</h3>
      <dl className="mt-5 grid grid-cols-2 gap-4 text-sm">
        <Metric label={labels.roc_auc ?? "ROC-AUC"} value={metrics?.roc_auc} />
        <Metric label={labels.log_loss ?? "Log loss"} value={metrics?.log_loss} />
        <Metric label={labels.brier_score ?? "Brier score"} value={metrics?.brier_score} />
        <Metric label={labels.accuracy ?? "Accuracy"} value={metrics?.accuracy} />
        <Metric label={labels.expected_calibration_error ?? "ECE (10-bin)"} value={metrics?.expected_calibration_error} />
      </dl>
    </article>
  );
}

function Metric({ label, value }: { label: string; value?: number }) {
  return <div><dt className="text-xs text-[var(--muted)]">{label}</dt><dd className="metric-tabular mt-1 font-semibold">{value === undefined ? "—" : value.toFixed(4)}</dd></div>;
}
