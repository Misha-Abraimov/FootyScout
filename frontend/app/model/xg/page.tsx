import type { Metadata } from "next";
import Link from "next/link";

import { PageHeader } from "@/components/PageHeader";
import { ErrorState } from "@/components/States";
import { XGModelEvaluation } from "@/components/XGModelEvaluation";
import { api } from "@/lib/api";
import { formatCount, humanizeField } from "@/lib/format";

export const dynamic = "force-dynamic";
export const metadata: Metadata = { title: "Expected-goals methodology" };

export default async function XGModelPage() {
  const model = await api.getXGModel().catch(() => null);
  if (!model) return <main className="mx-auto min-h-[70vh] max-w-7xl px-5 py-16 sm:px-8"><ErrorState /></main>;
  return (
    <main className="mx-auto min-h-screen max-w-7xl space-y-14 px-5 py-10 sm:px-8 sm:py-14">
      <PageHeader eyebrow="Shooting methodology" title="Expected Goals model" description="FootyScout estimates the probability that a non-penalty shot becomes a goal using only information available before its outcome." />
      <p className="-mt-10 text-sm"><Link href="/model" className="text-[var(--accent)] hover:text-[var(--accent-strong)]">← Expected Pass model</Link></p>
      <section className="grid gap-4 lg:grid-cols-2">
        <article className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-6"><p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">Selected model</p><h2 className="mt-3 text-3xl font-semibold">{humanizeField(model.selected_model)}</h2><p className="mt-4 text-sm leading-6 text-[var(--muted)]">{model.selection.reason}. Test metrics used for selection: {model.selection.test_metrics_used ? "yes" : "no"}.</p><dl className="mt-6 grid grid-cols-2 gap-4"><div><dt className="text-xs text-[var(--muted)]">Modeled shots</dt><dd className="metric-tabular mt-1 font-semibold">{formatCount(model.dataset.eligible_non_penalty_shots)}</dd></div><div><dt className="text-xs text-[var(--muted)]">Grouped folds</dt><dd className="metric-tabular mt-1 font-semibold">{model.oof.fold_count}</dd></div></dl></article>
        <article className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-6"><h2 className="text-xl font-semibold">Training corpus</h2><p className="mt-3 text-sm text-[var(--muted)]">{formatCount(model.dataset.matches)} matches · {formatCount(model.dataset.shots)} shots · {formatCount(model.dataset.goals)} goals</p><p className="mt-4 text-sm leading-6 text-[var(--muted)]">{model.penalty_policy}</p><ul className="mt-5 grid gap-2 text-sm">{model.dataset.corpus.map((item) => <li key={`${item.competition_id}-${item.season_id}`} className="rounded-lg border border-[var(--border)] px-3 py-2">Competition {item.competition_id} · season {item.season_id} · {item.matches} matches</li>)}</ul></article>
      </section>
      <section aria-labelledby="xg-features-title"><h2 id="xg-features-title" className="text-2xl font-semibold">Pre-outcome features</h2><p className="mt-2 text-sm text-[var(--muted)]">Player/team identity, shot outcome, goalkeeper outcome, end location, future events, and StatsBomb’s provider xG are excluded.</p><div className="mt-5 flex flex-wrap gap-2">{model.feature_columns.map((feature) => <code key={feature} className="rounded-md bg-white/5 px-2.5 py-1 text-xs text-[var(--muted)]">{feature}</code>)}</div></section>
      <XGModelEvaluation model={model} />
      <section className="grid gap-4 lg:grid-cols-2"><article className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-6"><h2 className="text-xl font-semibold">Calibration</h2><p className="mt-3 text-sm leading-6 text-[var(--muted)]">{model.calibration.reason}. Temperature: {model.calibration.temperature.toFixed(4)}; retained: {model.calibration.retained ? "yes" : "no"}.</p></article><article className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-6"><h2 className="text-xl font-semibold">Grouped methodology</h2><p className="mt-3 text-sm leading-6 text-[var(--muted)]">{model.split_methodology.method}. Preprocessing is fitted only on training matches, including independently inside each OOF fold.</p></article></section>
    </main>
  );
}
