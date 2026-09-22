import type { Metadata } from "next";
import Link from "next/link";

import { ModelMetricCard } from "@/components/ModelMetricCard";
import { PageHeader } from "@/components/PageHeader";
import { ErrorState } from "@/components/States";
import { api } from "@/lib/api";
import { formatCount, humanizeField } from "@/lib/format";
import type { ValueModelMetrics } from "@/lib/types";

export const dynamic = "force-dynamic";
export const metadata: Metadata = { title: "Attacking impact model" };

function ValueMetrics({ title, metrics }: { title: string; metrics: ValueModelMetrics | undefined }) {
  const values = metrics ? { roc_auc: metrics.rmse, log_loss: metrics.mae, brier_score: metrics.r2, accuracy: metrics.spearman, expected_calibration_error: metrics.positive_target_rmse } : undefined;
  return <ModelMetricCard title={title} metrics={values} labels={{ roc_auc: "RMSE", log_loss: "MAE", brier_score: "R²", accuracy: "Spearman", expected_calibration_error: "Positive RMSE" }} />;
}

function publicLimitation(limitation: string): string {
  return limitation.replace(/\bV\d+(?:\.\d+)*[A-Z]?\b/g, "this model");
}

export default async function ActionValueModelPage() {
  const model = await api.getActionValueModel().catch(() => null);
  if (!model) return <main className="mx-auto min-h-[70vh] max-w-7xl px-5 py-16 sm:px-8"><ErrorState /></main>;
  const selectedValidation = model.validation_metrics[model.selection.model];
  return (
    <main className="mx-auto min-h-screen max-w-7xl space-y-12 px-5 py-10 sm:px-8 sm:py-14">
      <PageHeader eyebrow="Possession-value methodology" title="Attacking Impact model" description="Estimate how much a pass or carry changes the future expected-goal value of the current possession." />
      <p className="-mt-8 text-sm"><Link href="/model" className="text-[var(--accent)]">← Expected Pass model</Link><span className="mx-3 text-[var(--muted)]">·</span><Link href="/model/xg" className="text-[var(--accent)]">Expected Goals model</Link></p>
      <section className="grid gap-4 lg:grid-cols-2">
        <article className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-6"><p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">Selected model</p><h2 className="mt-3 text-2xl font-semibold">{humanizeField(model.selection.model)}</h2><p className="mt-3 text-sm leading-6 text-[var(--muted)]">{model.selection.reason}</p><p className="mt-4 text-xs text-[var(--muted)]">Objective: {model.selection.objective} · selected using {humanizeField(model.selection.primary_metric)}</p></article>
        <article className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-6"><h2 className="text-xl font-semibold">Target and state</h2><p className="mt-3 text-sm leading-6 text-[var(--muted)]">{model.target_interpretation}</p><p className="mt-3 text-sm leading-6 text-[var(--muted)]">{model.state_convention}</p></article>
      </section>
      <aside className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-5">
        <h2 className="text-lg font-semibold">Transformer results coming soon</h2>
        <p className="mt-2 text-sm leading-6 text-[var(--muted)]">We recently benchmarked a PyTorch causal Transformer for possession-value prediction using possession history across 667,000+ game states. Full evaluation results and model comparisons will be added here soon.</p>
        <p className="mt-2 text-xs leading-5 text-[var(--muted)]">The current production possession-value model remains XGBoost.</p>
      </aside>
      <section className="grid gap-4 lg:grid-cols-2">
        <article className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-6"><h2 className="text-xl font-semibold">Training corpus</h2><p className="mt-3 text-sm text-[var(--muted)]">{formatCount(model.training_corpus.matches)} matches · {formatCount(model.training_corpus.events)} events · {formatCount(model.training_corpus.states)} states · {formatCount(model.training_corpus.possessions)} possessions</p><ul className="mt-5 grid gap-2 text-sm">{model.training_corpus.competitions.map((item) => <li key={`${item.competition_id}-${item.season_id}`} className="rounded-lg border border-[var(--border)] px-3 py-2">{item.name} · {item.matches} matches · {formatCount(item.events)} events</li>)}</ul></article>
        <article className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-6"><h2 className="text-xl font-semibold">Grouped methodology</h2><p className="mt-3 text-sm leading-6 text-[var(--muted)]">{model.split_methodology.method}</p><p className="mt-3 text-sm leading-6 text-[var(--muted)]">Nested upstream labels: {model.nested_cross_fitting.validation}. Held-out outcomes used for training: {model.nested_cross_fitting.heldout_outcomes_used_for_training ? "yes" : "no"}.</p><dl className="mt-5 grid grid-cols-3 gap-3 text-sm"><div><dt className="text-xs text-[var(--muted)]">Train</dt><dd className="metric-tabular mt-1 font-semibold">{formatCount(Number(model.split_methodology.train_states))}</dd></div><div><dt className="text-xs text-[var(--muted)]">Validation</dt><dd className="metric-tabular mt-1 font-semibold">{formatCount(Number(model.split_methodology.validation_states))}</dd></div><div><dt className="text-xs text-[var(--muted)]">Test</dt><dd className="metric-tabular mt-1 font-semibold">{formatCount(Number(model.split_methodology.test_states))}</dd></div></dl><p className="mt-5 text-sm text-[var(--muted)]">{model.selection.objective} · {String(model.selection.rounds.regressor ?? "—")} boosting rounds · {String(model.preprocessing.encoded_feature_count ?? "—")} encoded features</p></article>
      </section>
      <section><h2 className="text-2xl font-semibold">Validation comparison</h2><p className="mt-2 text-sm text-[var(--muted)]">The selected model minimizes validation RMSE; the untouched test set is reported only after selection.</p><div className="mt-5 grid gap-4 lg:grid-cols-3">{Object.entries(model.validation_metrics).map(([name, metrics]) => <ValueMetrics key={name} title={humanizeField(name)} metrics={metrics} />)}</div></section>
      <section><h2 className="text-2xl font-semibold">Frozen evaluation</h2><div className="mt-5 grid gap-4 lg:grid-cols-3"><ValueMetrics title="Validation · selected" metrics={selectedValidation} /><ValueMetrics title="Untouched test" metrics={model.untouched_test_metrics.selected_model} /><ValueMetrics title="Grouped OOF" metrics={model.out_of_fold_metrics} /></div></section>
      <section><h2 className="text-2xl font-semibold">Leakage-safe features</h2><p className="mt-2 max-w-3xl text-sm text-[var(--muted)]">{model.leakage_protection}</p><div className="mt-4 flex flex-wrap gap-2">{model.feature_columns.map((feature) => <code key={feature} className="rounded-md bg-white/5 px-2.5 py-1 text-xs text-[var(--muted)]">{feature}</code>)}</div></section>
      <section className="grid gap-4 lg:grid-cols-2"><article className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-6"><h2 className="text-xl font-semibold">Definitions</h2><dl className="mt-4 space-y-3 text-sm"><div><dt className="font-semibold">State value</dt><dd className="text-[var(--muted)]">Future FootyScout OOF xG remaining in the same possession.</dd></div><div><dt className="font-semibold">Action value</dt><dd className="text-[var(--muted)]">V(after) − V(before); not the probability that an action succeeds.</dd></div></dl></article><article className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-6"><h2 className="text-xl font-semibold">Limitations</h2><ul className="mt-4 list-disc space-y-2 pl-5 text-sm text-[var(--muted)]">{model.limitations.map((item) => { const copy = publicLimitation(item); return <li key={copy}>{copy}</li>; })}</ul></article></section>
    </main>
  );
}
