import { ModelMetricCard } from "@/components/ModelMetricCard";
import { humanizeField } from "@/lib/format";
import type { XGModelInfoResponse } from "@/lib/types";

export function XGModelEvaluation({ model }: { model: XGModelInfoResponse }) {
  const validation = model.validation_metrics.selected_effective ?? model.validation_metrics[model.selected_model];
  const test = model.untouched_test_metrics.selected_effective ?? model.untouched_test_metrics[model.selected_model];
  return (
    <>
      <section aria-labelledby="xg-performance-title">
        <h2 id="xg-performance-title" className="text-2xl font-semibold">Evaluation performance</h2>
        <p className="mt-2 text-sm text-[var(--muted)]">Selection used validation log loss. The test split remained untouched until the model and calibration decision were frozen.</p>
        <div className="mt-5 grid gap-4 lg:grid-cols-3">
          <ModelMetricCard title="Validation · selected form" metrics={validation} />
          <ModelMetricCard title="Untouched test" metrics={test} />
          <ModelMetricCard title="Grouped out-of-fold" metrics={model.out_of_fold_metrics} highlighted />
        </div>
      </section>
      <section aria-labelledby="xg-importance-title">
        <h2 id="xg-importance-title" className="text-2xl font-semibold">Model feature importance</h2>
        <p className="mt-2 text-sm text-[var(--muted)]">Gain-based model feature importance; these are not causal effects.</p>
        <div className="mt-5 grid gap-2 sm:grid-cols-2">
          {model.feature_importance.map((item) => (
            <div key={item.feature} className="flex justify-between rounded-xl border border-[var(--border)] px-4 py-3 text-sm">
              <code>{humanizeField(item.feature)}</code>
              <span className="metric-tabular text-[var(--muted)]">{(item.normalized_gain * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}
