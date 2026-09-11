import type { PlayerSummary } from "./types";

function normalize(value: string): string {
  return value.trim().toLocaleLowerCase("en");
}

export function rankPlayerSuggestions(
  players: PlayerSummary[],
  query: string,
  excludedPlayerIds: ReadonlySet<number> = new Set(),
): PlayerSummary[] {
  const needle = normalize(query);
  if (!needle) return [];

  return players
    .filter((player) => {
      const name = normalize(player.player_name);
      return !excludedPlayerIds.has(player.player_id) && name.includes(needle);
    })
    .map((player) => {
      const name = normalize(player.player_name);
      const words = name.split(/[\s\-']+/);
      const relevance = name.startsWith(needle)
        ? 0
        : words.some((word) => word.startsWith(needle))
          ? 1
          : 2;
      return { player, relevance };
    })
    .sort(
      (left, right) =>
        left.relevance - right.relevance ||
        left.player.player_name.localeCompare(right.player.player_name, "en", {
          sensitivity: "base",
        }) ||
        left.player.player_id - right.player.player_id,
    )
    .map(({ player }) => player);
}
