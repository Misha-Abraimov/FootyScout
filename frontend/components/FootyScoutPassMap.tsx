"use client";

import { DARK_THEME, PassMap, ThemeProvider } from "@withqwerty/campos-react";
import { useEffect, useMemo, useReducer, useRef, useState } from "react";

import { PassMapControls } from "@/components/PassMapControls";
import { EmptyState, ErrorState } from "@/components/States";
import { api } from "@/lib/api";
import { camposMeta, toCamposPasses } from "@/lib/campos";
import { formatCount, formatPercent } from "@/lib/format";
import {
  createPassMapState,
  formatPassOption,
  passMapReducer,
  selectedPassForState,
  type PassMapFilterState,
} from "@/lib/pass-map-state";
import {
  buildPassFilterQuery,
  type DifficultyFilter,
  type PassViewFilter,
} from "@/lib/pass-filters";
import type { PassListResponse, PlayerIdentity } from "@/lib/types";

const PAGE_SIZE = 200;

export function FootyScoutPassMap({
  player,
  initialData,
}: {
  player: PlayerIdentity;
  initialData: PassListResponse;
}) {
  const [state, dispatch] = useReducer(passMapReducer, initialData, createPassMapState);
  const requestIdRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const { data, filter, error, selectedPassIndex } = state;
  const loading = state.status === "loading";
  const [analysisView, setAnalysisView] = useState<"difficulty" | "value">("difficulty");

  useEffect(() => () => abortRef.current?.abort(), []);

  const camposPasses = useMemo(
    () => toCamposPasses(data.items, player),
    [data.items, player],
  );
  const selected = selectedPassForState(state);

  async function load(nextFilter: PassMapFilterState) {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const requestId = ++requestIdRef.current;
    dispatch({ type: "request", requestId, filter: nextFilter });
    try {
      const response = await api.getPlayerPasses(
        player.player_id,
        buildPassFilterQuery(
          nextFilter.view,
          nextFilter.difficulty,
          nextFilter.offset,
          PAGE_SIZE,
        ),
        controller.signal,
      );
      dispatch({ type: "success", requestId, response });
    } catch (requestError) {
      if (isAbortError(requestError)) return;
      dispatch({
        type: "failure",
        requestId,
        message: requestError instanceof Error
          ? requestError.message
          : "Passes could not be loaded.",
      });
    }
  }

  function changeView(next: PassViewFilter) {
    void load({ view: next, difficulty: filter.difficulty, offset: 0 });
  }

  function changeDifficulty(next: DifficultyFilter) {
    void load({ view: filter.view, difficulty: next, offset: 0 });
  }

  const start = data.total === 0 ? 0 : data.offset + 1;
  const end = Math.min(data.offset + data.items.length, data.total);

  return (
    <section aria-labelledby="pass-map-title" className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-4 sm:p-6">
      <div className="mb-6 flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="text-xs font-semibold tracking-[0.16em] text-[var(--accent)] uppercase">Campos visualization</p>
          <h2 id="pass-map-title" className="mt-2 text-xl font-semibold">Pass map</h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-[var(--muted)]">
            Attacking left to right. Switch between xPass execution difficulty and possession attacking impact.
          </p>
        </div>
        <p aria-live="polite" className="metric-tabular text-sm text-[var(--muted)]">
          Showing {formatCount(start)}–{formatCount(end)} of {formatCount(data.total)} passes
        </p>
      </div>

      <PassMapControls
        view={filter.view}
        difficulty={filter.difficulty}
        disabled={loading}
        onViewChange={changeView}
        onDifficultyChange={changeDifficulty}
      />
      <label className="mt-4 grid max-w-xs gap-2 text-xs text-[var(--muted)]">Analytical view
        <select value={analysisView} onChange={(event) => setAnalysisView(event.target.value as "difficulty" | "value")} className="rounded-lg border border-[var(--border)] bg-[#0a100e] px-3 py-2.5 text-sm text-white">
          <option value="difficulty">xPass difficulty</option>
          <option value="value">Attacking impact</option>
        </select>
      </label>

      <p aria-live="polite" className="mt-3 min-h-5 text-xs text-[var(--muted)]">
        {loading ? "Updating filtered pass data…" : ""}
      </p>

      {error ? <div className="mt-5"><ErrorState title="Pass map unavailable" message={error} /></div> : null}
      {!error && data.items.length === 0 ? (
        <div className="mt-5"><EmptyState title="No matching passes" message="No passes match this subset and difficulty combination." /></div>
      ) : null}

      {!error && data.items.length > 0 ? (
        <div className={loading ? "mt-6 opacity-45" : "mt-6"} aria-busy={loading}>
          <div className="campos-frame overflow-hidden rounded-xl border border-white/10 bg-[#0b1210]">
            <ThemeProvider value={DARK_THEME}>
              <PassMap
                passes={camposPasses}
                attackingDirection="right"
                pitchPreset="dark"
                framePadding={8}
                maxWidth={1200}
                showHeaderStats={false}
                showLegend={false}
                lines={{
                  stroke: ({ pass }) => camposMeta(pass).completed ? "#80e4a8" : "#ff8f86",
                  strokeWidth: ({ pass, active }) => {
                    const meta = camposMeta(pass);
                    if (active || meta.passIndex === selectedPassIndex) return 1.6;
                    if (meta.progressive || meta.underPressure) return 0.9;
                    return 0.58;
                  },
                  strokeLinecap: ({ pass }) => camposMeta(pass).underPressure ? "square" : "round",
                  strokeDasharray: ({ pass }) => camposMeta(pass).completed ? undefined : "2 1.4",
                  opacity: ({ pass, active }) => {
                    const meta = camposMeta(pass);
                    if (active || meta.passIndex === selectedPassIndex) return 1;
                    if (analysisView === "value") return Math.min(1, 0.3 + Math.abs(data.items.find((item) => item.pass_index === meta.passIndex)?.attacking_value ?? 0) * 8);
                    return 0.34 + (1 - meta.expectedCompletion) * 0.54;
                  },
                }}
                dots={{
                  fill: ({ pass }) => camposMeta(pass).completed ? "#80e4a8" : "#ff8f86",
                  radius: ({ pass, active }) => active || camposMeta(pass).passIndex === selectedPassIndex ? 1.05 : 0.55,
                  opacity: ({ pass }) => camposMeta(pass).progressive ? 0.9 : 0.52,
                }}
              />
            </ThemeProvider>
          </div>

          <div className="mt-4 flex flex-wrap gap-x-5 gap-y-2 text-xs text-[var(--muted)]" aria-label="Pass map legend">
            <span><i className="mr-2 inline-block h-0.5 w-6 bg-[#80e4a8] align-middle" />Completed</span>
            <span><i className="mr-2 inline-block w-6 border-t-2 border-dashed border-[#ff8f86] align-middle" />Incomplete</span>
            <span>{analysisView === "difficulty" ? "Stronger line = greater pass difficulty" : "Stronger line = larger absolute attacking impact"}</span>
            <span>Thicker line = progressive or under pressure</span>
          </div>

          <div className="mt-5 grid gap-4 rounded-xl border border-[var(--border)] bg-black/15 p-4 lg:grid-cols-[minmax(220px,0.7fr)_1.3fr]">
            <label className="grid content-start gap-2 text-xs text-[var(--muted)]">
              Inspect a pass in this view
              <select
                value={selectedPassIndex ?? ""}
                disabled={loading}
                onChange={(event) => dispatch({ type: "select", passIndex: Number(event.target.value) })}
                className="rounded-lg border border-[var(--border)] bg-[#0a100e] px-3 py-2.5 text-sm text-white"
              >
                {data.items.map((pass) => (
                  <option key={pass.pass_index} value={pass.pass_index}>
                    {formatPassOption(pass)}
                  </option>
                ))}
              </select>
            </label>
            {selected ? (
              <dl className="grid grid-cols-2 gap-x-5 gap-y-3 text-sm sm:grid-cols-4">
                <PassDetail label="Pass difficulty" value={formatPercent(selected.expected_completion)} />
                <PassDetail label="Outcome" value={selected.completed ? "Completed" : "Incomplete"} />
                <PassDetail label="Attacking impact" value={selected.attacking_value == null ? "—" : selected.attacking_value.toFixed(4)} />
                <PassDetail label="Value before" value={selected.state_value_before == null ? "—" : selected.state_value_before.toFixed(4)} />
                <PassDetail label="Value after" value={selected.state_value_after == null ? "—" : selected.state_value_after.toFixed(4)} />
                <PassDetail label="Length" value={`${selected.pass_length.toFixed(1)} m`} />
                <PassDetail label="Context" value={[selected.progressive && "Progressive", selected.under_pressure && "Under pressure"].filter(Boolean).join(" · ") || "Standard"} />
              </dl>
            ) : null}
          </div>

          <div className="mt-5 flex items-center justify-between gap-4">
            <button type="button" disabled={loading || data.offset === 0} onClick={() => void load({ ...filter, offset: Math.max(0, data.offset - PAGE_SIZE) })} className="rounded-lg border border-[var(--border)] px-4 py-2 text-sm disabled:opacity-35">Previous 200</button>
            <span className="text-center text-xs text-[var(--muted)]">Event clocks are not retained in the pass API.</span>
            <button type="button" disabled={loading || end >= data.total} onClick={() => void load({ ...filter, offset: data.offset + PAGE_SIZE })} className="rounded-lg border border-[var(--border)] px-4 py-2 text-sm disabled:opacity-35">Next 200</button>
          </div>
        </div>
      ) : null}
    </section>
  );
}

function isAbortError(error: unknown): boolean {
  return typeof error === "object" && error !== null && "name" in error && error.name === "AbortError";
}

function PassDetail({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs text-[var(--muted)]">{label}</dt>
      <dd className="metric-tabular mt-1 font-medium">{value}</dd>
    </div>
  );
}
