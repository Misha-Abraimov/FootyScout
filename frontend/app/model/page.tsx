import type { Metadata } from "next";
import Link from "next/link";

import { ModelEvaluation } from "@/components/ModelEvaluation";
import { PageHeader } from "@/components/PageHeader";
import { ErrorState } from "@/components/States";
import { api } from "@/lib/api";
import { formatCount, humanizeField } from "@/lib/format";

export const dynamic = "force-dynamic";
export const metadata: Metadata = { title: "Model methodology" };

const featureGroups = {
  "Pitch position": ["start_x", "start_y", "end_x", "end_y", "start_zone", "end_zone", "distance_to_goal_before", "distance_to_goal_after", "distance_toward_goal"],
  "Pass geometry": ["pass_length", "pass_angle", "forward_distance", "lateral_distance"],
  "Pressure / context": ["under_pressure", "progressive"],
  "Pass characteristics": ["pass_height", "body_part", "pass_type"],
};

export default async function ModelPage() {
  const model = await api.getModel().catch(() => null);
  if (!model) {
    return <main className="mx-auto min-h-[70vh] max-w-7xl px-5 py-16 sm:px-8"><ErrorState /></main>;
  }
  return (
      <main className="mx-auto min-h-screen max-w-7xl space-y-14 px-5 py-10 sm:px-8 sm:py-14">
        <PageHeader eyebrow="Passing methodology" title="Expected Pass model" description="FootyScout predicts the probability that a pass will be completed using information available about the pass before its outcome." />

        <p className="-mt-10 flex flex-wrap gap-4 text-sm"><Link href="/model/xg" className="text-[var(--accent)] hover:text-[var(--accent-strong)]">Expected Goals model →</Link><Link href="/model/action-value" className="text-[var(--accent)] hover:text-[var(--accent-strong)]">Attacking Impact model →</Link></p>

        <section className="grid gap-4 lg:grid-cols-[0.85fr_1.15fr]">
          <article className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-6"><p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">Selected model</p><h2 className="mt-3 text-3xl font-semibold">{humanizeField(model.selected_model)}</h2><p className="mt-4 text-sm leading-6 text-[var(--muted)]">{model.selection.reason}</p><dl className="mt-6 grid grid-cols-2 gap-4"><div><dt className="text-xs text-[var(--muted)]">Training passes</dt><dd className="metric-tabular mt-1 font-semibold">{formatCount(model.dataset.pass_count)}</dd></div><div><dt className="text-xs text-[var(--muted)]">Grouped OOF folds</dt><dd className="metric-tabular mt-1 font-semibold">{model.oof_fold_count}</dd></div></dl></article>
          <article className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-6"><h2 className="text-xl font-semibold">Architecture</h2><ol className="mt-5 grid gap-2">{model.architecture.map((layer, index) => <li key={`${layer}-${index}`} className="rounded-lg border border-[var(--border)] bg-black/10 px-4 py-3 font-mono text-sm"><span className="mr-3 text-[var(--muted)]">{String(index + 1).padStart(2, "0")}</span>{layer}</li>)}</ol></article>
        </section>

        <section aria-labelledby="features-title"><h2 id="features-title" className="text-2xl font-semibold">Pre-outcome features</h2><p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--muted)]">Identifiers, player/team names, pass outcomes, recipients, and future events are excluded from model inputs.</p><div className="mt-5 grid gap-4 md:grid-cols-2">{Object.entries(featureGroups).map(([group, candidates]) => { const active = candidates.filter((feature) => model.feature_columns.includes(feature)); return active.length ? <article key={group} className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-5"><h3 className="font-semibold">{group}</h3><div className="mt-4 flex flex-wrap gap-2">{active.map((feature) => <code key={feature} className="rounded-md bg-white/5 px-2.5 py-1 text-xs text-[var(--muted)]">{feature}</code>)}</div></article> : null; })}</div></section>

        <ModelEvaluation model={model} />

        <section className="grid gap-4 lg:grid-cols-2">
          <article className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-6"><h2 className="text-xl font-semibold">Calibration</h2><p className="mt-3 text-sm leading-6 text-[var(--muted)]">Temperature scaling was fit on the validation split only. It was {model.calibration.retained ? "retained" : "not retained"}: {model.calibration.reason}</p><dl className="mt-5 grid grid-cols-2 gap-4"><div><dt className="text-xs text-[var(--muted)]">Temperature</dt><dd className="metric-tabular mt-1 font-semibold">{model.calibration.temperature.toFixed(4)}</dd></div><div><dt className="text-xs text-[var(--muted)]">Fit split</dt><dd className="mt-1 font-semibold">{model.calibration.fit_split}</dd></div></dl><p className="mt-5 text-xs leading-5 text-[var(--muted)]">Calibration can improve probability quality without improving ranking; FootyScout does not claim it improves every held-out metric.</p></article>
          <article className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-6"><h2 className="text-xl font-semibold">Evaluation sequence</h2><ol className="mt-5 space-y-4 text-sm leading-6"><li><strong>1. Match-grouped split.</strong> <span className="text-[var(--muted)]">{model.split_methodology.method}</span></li><li><strong>2. Validation selection.</strong> <span className="text-[var(--muted)]">{model.methodology.model_selection}</span></li><li><strong>3. Untouched test.</strong> <span className="text-[var(--muted)]">{model.methodology.test_set}</span></li><li><strong>4. Grouped OOF predictions.</strong> <span className="text-[var(--muted)]">{model.methodology.player_profiles}</span></li><li><strong>5. Production fit.</strong> <span className="text-[var(--muted)]">{model.methodology.production_model}</span></li></ol></article>
        </section>

        <section aria-labelledby="similarity-method-title" className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-6">
          <p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">Player Intelligence</p>
          <h2 id="similarity-method-title" className="mt-3 text-2xl font-semibold">Player-style similarity</h2>
          <p className="mt-3 max-w-4xl text-sm leading-6 text-[var(--muted)]">FootyScout compares eligible outfield players within the same broad position using six style-only passing and carrying tendencies. Each feature is converted to a population z-score within DEF, MID, or FWD, then profiles are ranked by root-mean-square Euclidean distance. The 0–100 index is cohort-calibrated: 100 is identical and 50 is approximately the median same-position pair distance.</p>
          <div className="mt-5 grid gap-4 md:grid-cols-2">
            <div className="rounded-xl border border-[var(--border)] bg-black/10 p-4"><h3 className="font-semibold">Sample support</h3><p className="mt-2 text-sm leading-6 text-[var(--muted)]">Support uses the lower observed-match count of the two players. Pairs below three matches are marked limited. Coverage is uneven, so low-sample neighbor ranks may move as more matches are observed; support never changes the score.</p></div>
            <div className="rounded-xl border border-[var(--border)] bg-black/10 p-4"><h3 className="font-semibold">What it is not</h3><p className="mt-2 text-sm leading-6 text-[var(--muted)]">Similarity describes observed playing style. It is not player quality, future performance, probability of success, tactical fit, or transfer success. Performance metrics, outcomes, team identity, and archetype labels are excluded.</p></div>
          </div>
        </section>

        <section aria-labelledby="role-fit-method-title" className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-6">
          <p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">Team Intelligence</p>
          <h2 id="role-fit-method-title" className="mt-3 text-2xl font-semibold">Team Intelligence and Role Fit</h2>
          <div className="mt-4 grid gap-4 text-sm leading-6 text-[var(--muted)] md:grid-cols-2">
            <p><strong className="text-[var(--foreground)]">Team style and positional roles.</strong> Team style describes Leverkusen&apos;s 34 observed Bundesliga matches. DEF, MID and FWD roles pool the observed events and actions of players assigned to each frozen broad position.</p>
            <p><strong className="text-[var(--foreground)]">Role Fit.</strong> Eligible players and roles share the same six-feature, position-relative playing-style coordinate system. Root-mean-square distance measures observed style resemblance; lower is closer.</p>
            <p><strong className="text-[var(--foreground)]">Current players.</strong> A Leverkusen player is compared with a leave-self-out role so their own actions do not contribute to the target profile.</p>
            <p><strong className="text-[var(--foreground)]">Scouting Recommendations.</strong> External same-position players are ranked only by Role Fit. This is not transfer-success probability, a player-quality score, causal tactical compatibility, or a forecast.</p>
          </div>
        </section>
      </main>
  );
}
