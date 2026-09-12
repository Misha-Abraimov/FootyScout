import type { Metadata } from "next";

import { PageHeader } from "@/components/PageHeader";
import { Pagination } from "@/components/Pagination";
import { PlayerExplorerAutocomplete } from "@/components/PlayerExplorerAutocomplete";
import { PlayerTable } from "@/components/PlayerTable";
import { ErrorState } from "@/components/States";
import { api } from "@/lib/api";
import type { PlayerQuery, PlayerSortField, PositionGroup, SortOrder } from "@/lib/types";

export const dynamic = "force-dynamic";
export const metadata: Metadata = { title: "Players" };

const sortOptions: Array<[PlayerSortField, string]> = [
  ["player_name", "Player name"],
  ["pass_attempts", "Pass attempts"],
  ["actual_completion_rate", "Actual completion"],
  ["expected_completion_rate", "Pass difficulty"],
  ["completion_above_expected_pp", "Actual vs. expected passing"],
  ["progressive_pass_rate", "Progressive-pass rate"],
  ["pressure_above_expected_pp", "Under-pressure above expected"],
  ["final_third_entries_per_100_passes", "Final-third entries / 100"],
];

type SearchParams = Promise<Record<string, string | string[] | undefined>>;
const one = (value: string | string[] | undefined) =>
  Array.isArray(value) ? value[0] : value;

function numberParam(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

export default async function PlayersPage({ searchParams }: { searchParams: SearchParams }) {
  const raw = await searchParams;
  const query: PlayerQuery = {
    search: one(raw.search),
    team: one(raw.team),
    position_group: one(raw.position_group) as PositionGroup | undefined,
    min_pass_attempts: numberParam(one(raw.min_pass_attempts), 0),
    sort_by: (one(raw.sort_by) as PlayerSortField | undefined) ?? "player_name",
    sort_order: (one(raw.sort_order) as SortOrder | undefined) ?? "asc",
    limit: 25,
    offset: numberParam(one(raw.offset), 0),
  };
  const result = await Promise.all([api.getPlayers(query), api.getMeta()]).catch(() => null);
  if (!result) {
    return <main className="mx-auto min-h-[70vh] max-w-7xl px-5 py-16 sm:px-8"><ErrorState /></main>;
  }
  const [players, meta] = result;
  const preservedQuery = {
      search: query.search,
      team: query.team,
      position_group: query.position_group,
      min_pass_attempts: String(query.min_pass_attempts ?? 0),
      sort_by: query.sort_by,
      sort_order: query.sort_order,
  };
  return (
      <main className="mx-auto min-h-screen max-w-7xl px-5 py-10 sm:px-8 sm:py-14">
        <PageHeader
          eyebrow="Player explorer"
          title="Find players by role, team, and performance."
          description="Explore the available StatsBomb sample, compare player metrics, and open a player’s full FootyScout profile."
        />
        <form action="/players" className="mt-8 w-full min-w-0 max-w-full rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-5">
          <div
            data-testid="player-filter-primary"
            className="grid min-w-0 gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-[minmax(220px,2fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.25fr)]"
          >
            <PlayerExplorerAutocomplete initialValue={query.search ?? ""} />
            <label className="grid min-w-0 gap-2 text-sm">
              <span className="text-[var(--muted)]">Team</span>
              <select name="team" defaultValue={query.team ?? ""} className="w-full min-w-0 max-w-full truncate rounded-lg border border-[var(--border)] bg-[#0a100e] px-3 py-2.5">
                <option value="">All teams</option>
                {meta.teams.map((team) => <option key={team}>{team}</option>)}
              </select>
            </label>
            <label className="grid min-w-0 gap-2 text-sm">
              <span className="text-[var(--muted)]">Position group</span>
              <select name="position_group" defaultValue={query.position_group ?? ""} className="w-full min-w-0 max-w-full rounded-lg border border-[var(--border)] bg-[#0a100e] px-3 py-2.5">
                <option value="">All positions</option>
                {meta.position_groups.map((group) => <option key={group}>{group}</option>)}
              </select>
            </label>
            <label className="grid min-w-0 gap-2 text-sm">
              <span className="text-[var(--muted)]">Minimum passes</span>
              <input type="number" min="0" name="min_pass_attempts" defaultValue={query.min_pass_attempts} className="w-full min-w-0 max-w-full rounded-lg border border-[var(--border)] bg-[#0a100e] px-3 py-2.5" />
            </label>
            <label className="grid min-w-0 gap-2 text-sm">
              <span className="text-[var(--muted)]">Sort by</span>
              <select name="sort_by" defaultValue={query.sort_by} className="w-full min-w-0 max-w-full truncate rounded-lg border border-[var(--border)] bg-[#0a100e] px-3 py-2.5">
                {sortOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select>
            </label>
          </div>
          <div
            data-testid="player-filter-actions"
            className="mt-4 flex min-w-0 flex-col gap-4 sm:flex-row sm:items-end"
          >
            <label className="grid w-full min-w-0 gap-2 text-sm sm:w-48">
              <span className="text-[var(--muted)]">Order</span>
              <select name="sort_order" defaultValue={query.sort_order} className="w-full min-w-0 max-w-full rounded-lg border border-[var(--border)] bg-[#0a100e] px-3 py-2.5">
                <option value="asc">Ascending</option>
                <option value="desc">Descending</option>
              </select>
            </label>
            <button type="submit" className="w-full self-end rounded-lg bg-[var(--accent)] px-4 py-2.5 text-sm font-semibold text-[#07110d] hover:bg-[var(--accent-strong)] sm:w-auto sm:min-w-36">Apply filters</button>
          </div>
        </form>
        <div className="mt-6"><PlayerTable players={players.items} /></div>
        <div className="mt-5"><Pagination pathname="/players" query={preservedQuery} total={players.total} limit={players.limit} offset={players.offset} /></div>
      </main>
  );
}
