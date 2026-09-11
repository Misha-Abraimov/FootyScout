import { ModelMetricCard } from "@/components/ModelMetricCard";
import { humanizeField } from "@/lib/format";
import { mapModelPageMetrics } from "@/lib/model";
import type { ModelInfoResponse } from "@/lib/types";

const comparisonModels = [
  ["logistic_regression", "Logistic Regression"],
  ["pytorch_mlp_effective", "PyTorch MLP"],
  ["xgboost_effective", "XGBoost"],
] as const;

function ComparisonTable({
  title,
  metrics,
  selectedModel,
}: {
  title: string;
  metrics: ModelInfoResponse["validation_metrics"];
  selectedModel: string;
}) {
  return (
    <div className="overflow-x-auto rounded-2xl border border-[var(--border)] bg-[var(--panel)]">
      <h3 className="px-5 pt-5 font-semibold">{title}</h3>
      <table className="mt-3 w-full min-w-[620px] text-left text-sm">
        <thead className="border-y border-[var(--border)] text-xs text-[var(--muted)]">
          <tr><th className="px-5 py-3">Model</th><th className="px-3 py-3">ROC-AUC</th><th className="px-3 py-3">Log loss</th><th className="px-3 py-3">Brier</th><th className="px-3 py-3">ECE</th></tr>
        </thead>
        <tbody>
          {comparisonModels.map(([key, label]) => {
            const values = metrics[key];
            if (!values) return null;
            const selected = key.startsWith(selectedModel);
            return (
              <tr key={key} className={selected ? "bg-emerald-300/[0.04]" : "border-t border-[var(--border)]"}>
                <th className="px-5 py-3 font-medium">{label}{selected ? " · Selected" : ""}</th>
                <td className="metric-tabular px-3 py-3">{values.roc_auc.toFixed(4)}</td>
                <td className="metric-tabular px-3 py-3">{values.log_loss.toFixed(4)}</td>
                <td className="metric-tabular px-3 py-3">{values.brier_score.toFixed(4)}</td>
                <td className="metric-tabular px-3 py-3">{values.expected_calibration_error.toFixed(4)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const metricsExplained = [
  [
    "ROC-AUC",
    "How well the model ranks completed passes above incomplete ones across thresholds; higher is better.",
  ],
  [
    "Log loss",
    "Scores probability confidence and heavily penalizes confident mistakes; lower is better.",
  ],
  [
    "Brier score",
    "Mean squared error of predicted probabilities; lower is better.",
  ],
  [
    "ECE",
    "The gap between predicted probability and observed completion across probability bins; lower suggests better calibration.",
  ],
] as const;

export function ModelEvaluation({ model }: { model: ModelInfoResponse }) {
  const mapped = mapModelPageMetrics(model);

  return (
    <>
      <section aria-labelledby="performance-title">
        <h2 id="performance-title" className="text-2xl font-semibold">
          Evaluation performance
        </h2>
        <p className="mt-2 text-sm text-[var(--muted)]">
          OOF probabilities drive player analytics and are highlighted below. The
          untouched test split was evaluated only after validation selected the model.
        </p>
        <div className="mt-5 grid gap-4 lg:grid-cols-3">
          <ModelMetricCard
            title="Validation · selected form"
            metrics={mapped.selectedValidation}
          />
          <ModelMetricCard title="Untouched test" metrics={mapped.selectedTest} />
          <ModelMetricCard
            title="Grouped out-of-fold"
            metrics={mapped.outOfFold}
            highlighted
          />
        </div>
        <div className="mt-5 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {metricsExplained.map(([name, explanation]) => (
            <article key={name} className="rounded-xl border border-[var(--border)] p-4">
              <h3 className="font-semibold">{name}</h3>
              <p className="mt-2 text-xs leading-5 text-[var(--muted)]">
                {explanation}
              </p>
            </article>
          ))}
        </div>
      </section>

      <section aria-labelledby="baseline-title">
        <h2 id="baseline-title" className="text-2xl font-semibold">
          Validation model selection
        </h2>
        <div className="mt-5 grid gap-4 xl:grid-cols-2">
          <ComparisonTable title="Validation model comparison" metrics={model.validation_metrics} selectedModel={model.selected_model} />
          <ComparisonTable title="Untouched-test model comparison" metrics={model.untouched_test_metrics} selectedModel={model.selected_model} />
        </div>
        <p className="mt-4 text-sm leading-6 text-[var(--muted)]">
          Primary: {humanizeField(model.selection.primary_metric)} · tie-breaker:{" "}
          {humanizeField(model.selection.tie_breaker)} · supporting:{" "}
          {humanizeField(model.selection.supporting_metric)}. Test metrics used for
          selection: {model.selection.test_metrics_used ? "yes" : "no"}.
        </p>
      </section>

      <section aria-labelledby="importance-title">
        <h2 id="importance-title" className="text-2xl font-semibold">XGBoost feature importance</h2>
        <p className="mt-2 text-sm text-[var(--muted)]">Gain-based model feature importance; this is not a causal claim.</p>
        <div className="mt-5 grid gap-2 sm:grid-cols-2">
          {model.xgboost.feature_importance.map((item) => (
            <div key={item.feature} className="flex justify-between rounded-xl border border-[var(--border)] px-4 py-3 text-sm">
              <code>{item.feature}</code><span className="metric-tabular text-[var(--muted)]">{(item.normalized_gain * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}
