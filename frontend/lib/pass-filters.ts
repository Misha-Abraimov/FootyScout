import type { PassQuery } from "@/lib/types";

export type PassViewFilter = "all" | "completed" | "incomplete" | "progressive" | "pressure";
export type DifficultyFilter = "all" | "difficult" | "medium" | "routine";

// Backend bounds are inclusive. Adjacent IEEE-754 values encode the two
// exclusive UI boundaries without overlapping categories:
// difficult < 0.60; medium 0.60–0.85; routine > 0.85.
export const DIFFICULT_MAX = 0.5999999999999999;
export const MEDIUM_MIN = 0.6;
export const MEDIUM_MAX = 0.85;
export const ROUTINE_MIN = 0.8500000000000001;

export function buildPassFilterQuery(
  view: PassViewFilter,
  difficulty: DifficultyFilter,
  offset: number,
  limit = 200,
): PassQuery {
  const query: PassQuery = { limit, offset };
  if (view === "completed") query.completed = true;
  if (view === "incomplete") query.completed = false;
  if (view === "progressive") query.progressive = true;
  if (view === "pressure") query.under_pressure = true;
  if (difficulty === "difficult") query.max_expected_completion = DIFFICULT_MAX;
  if (difficulty === "medium") {
    query.min_expected_completion = MEDIUM_MIN;
    query.max_expected_completion = MEDIUM_MAX;
  }
  if (difficulty === "routine") query.min_expected_completion = ROUTINE_MIN;
  return query;
}
