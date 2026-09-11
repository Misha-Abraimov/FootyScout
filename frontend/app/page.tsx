import Link from "next/link";

import { LeaderboardTable } from "@/components/LeaderboardTable";
import { MetricCard } from "@/components/MetricCard";
import { ErrorState } from "@/components/States";
import { api } from "@/lib/api";
import { formatCount } from "@/lib/format";

export const dynamic = "force-dynamic";

const flow = [
  ["01", "Pass event", "Location, geometry, pressure and pass characteristics"],
  ["02", "XGBoost xPass model", "The validation-selected production model, trained without outcome leakage"],
  ["03", "Expected completion", "The probability that the attempt should be completed"],
  ["04", "Actual vs expected", "Execution assessed relative to pass difficulty"],
  ["05", "Player profile", "Season-level signals built from grouped OOF predictions"],
];

export default async function Home() {
  const result = await Promise.all([
      api.getMeta(),
      api.getModel(),
      api.getLeaderboard({ limit: 5 }),
    ]).catch(() => null);
  if (!result) {
    return <main className="mx-auto min-h-[70vh] max-w-7xl px-5 py-16 sm:px-8"><ErrorState /></main>;
  }
  const [meta, model, leaderboard] = result;
  return (
      <main>
        <section className="data-grid border-b border-[var(--border)] px-5 py-16 sm:px-8 sm:py-24">
          <div className="mx-auto max-w-7xl">
            <p className="text-xs font-semibold tracking-[0.2em] text-[var(--accent)] uppercase">Player intelligence for scouting</p>
            <h1 className="mt-5 max-w-5xl text-4xl leading-[1.03] font-semibold tracking-[-0.05em] text-balance sm:text-6xl lg:text-7xl">Scout players beyond traditional statistics.</h1>
            <p className="mt-7 max-w-3xl text-lg leading-8 text-[var(--muted)]">FootyScout models pass difficulty, expected goals, attacking value, and player style to build profiles, identify archetypes and similar players, analyze Team Intelligence, and surface Role Fit scouting recommendations.</p>
            <div className="mt-9 flex flex-wrap gap-3">
              <Link href="/players" className="rounded-lg bg-[var(--accent)] px-5 py-3 text-sm font-semibold text-[#07110d] hover:bg-[var(--accent-strong)]">Explore players</Link>
              <Link href="/archetypes" className="rounded-lg border border-[var(--border)] bg-[var(--panel)] px-5 py-3 text-sm font-semibold hover:bg-[var(--panel-raised)]">View archetypes</Link>
            </div>
            <p className="mt-6 text-xs text-[var(--muted)]">Based on available StatsBomb 2023/24 Bundesliga event data.</p>
          </div>
        </section>

        <div className="mx-auto max-w-7xl space-y-20 px-5 py-14 sm:px-8 sm:py-20">
          <section aria-labelledby="overview-heading">
            <div className="mb-6 flex items-end justify-between gap-4">
              <div><p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">Current dataset</p><h2 id="overview-heading" className="mt-2 text-2xl font-semibold">A transparent analytical base</h2></div>
              <Link href="/model" className="text-sm font-semibold text-[var(--accent-strong)] hover:underline">Read methodology →</Link>
            </div>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <MetricCard label="Players" value={formatCount(meta.player_count)} note={`${formatCount(meta.reliable_player_count)} meet the overall sample threshold`} />
              <MetricCard label="Passes" value={formatCount(model.dataset.pass_count)} note="Every pass has an out-of-fold expected probability" />
              <MetricCard label="Teams" value={formatCount(meta.teams.length)} note="Across the available competition matches" />
              <MetricCard label="OOF ROC-AUC" value={model.out_of_fold_metrics.roc_auc.toFixed(3)} note={`${model.oof_fold_count} grouped folds by match`} />
            </div>
          </section>

          <section aria-labelledby="pipeline-heading">
            <p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">How xPass becomes a profile</p>
            <h2 id="pipeline-heading" className="mt-2 text-2xl font-semibold">From one decision to a player signal</h2>
            <div className="mt-7 grid gap-px overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--border)] lg:grid-cols-5">
              {flow.map(([number, title, description], index) => (
                <article key={title} className="relative bg-[var(--panel)] p-5">
                  <span className="text-xs font-semibold text-[var(--accent)]">{number}</span>
                  <h3 className="mt-8 font-semibold">{title}</h3>
                  <p className="mt-2 text-sm leading-6 text-[var(--muted)]">{description}</p>
                  {index < flow.length - 1 ? <span aria-hidden="true" className="absolute top-5 right-5 text-[var(--muted)] lg:hidden">↓</span> : null}
                </article>
              ))}
            </div>
          </section>

          <section aria-labelledby="team-intelligence-heading">
            <div>
              <p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">Team Intelligence & Scouting</p>
              <h2 id="team-intelligence-heading" className="mt-2 text-2xl font-semibold">From observed roles to a focused shortlist</h2>
            </div>
            <div className="mt-7 grid gap-4 md:grid-cols-2">
              <Link href="/teams/904" className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-6 transition-colors hover:bg-[var(--panel-raised)]">
                <p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">Bayer Leverkusen</p>
                <h3 className="mt-3 text-xl font-semibold">Explore Team Intelligence</h3>
                <p className="mt-2 text-sm leading-6 text-[var(--muted)]">Review the 34-match team profile and pooled DEF, MID, and FWD positional roles.</p>
              </Link>
              <Link href="/scouting" className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-6 transition-colors hover:bg-[var(--panel-raised)]">
                <p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">Role Fit</p>
                <h3 className="mt-3 text-xl font-semibold">Open Scouting Recommendations</h3>
                <p className="mt-2 text-sm leading-6 text-[var(--muted)]">Rank eligible external players by observed style distance to Leverkusen&apos;s positional roles.</p>
              </Link>
            </div>
          </section>

          <section aria-labelledby="leaderboard-preview-heading">
            <div className="mb-6 flex items-end justify-between gap-4">
              <div><p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">Reliable samples</p><h2 id="leaderboard-preview-heading" className="mt-2 text-2xl font-semibold">Completion above expectation</h2></div>
              <Link href="/leaderboard" className="text-sm font-semibold text-[var(--accent-strong)] hover:underline">Full leaderboard →</Link>
            </div>
            <LeaderboardTable entries={leaderboard.items} />
          </section>
        </div>
      </main>
  );
}
