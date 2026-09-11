import { describe, expect, it } from "vitest";

import {
  DIFFICULT_MAX,
  MEDIUM_MAX,
  MEDIUM_MIN,
  ROUTINE_MIN,
  buildPassFilterQuery,
} from "./pass-filters";

describe("pass filtering state", () => {
  it("maps outcome and pressure subsets to backend filters", () => {
    expect(buildPassFilterQuery("all", "all", 0)).toEqual({ limit: 200, offset: 0 });
    expect(buildPassFilterQuery("completed", "all", 0)).toEqual({ limit: 200, offset: 0, completed: true });
    expect(buildPassFilterQuery("incomplete", "all", 200)).toEqual({ limit: 200, offset: 200, completed: false });
    expect(buildPassFilterQuery("progressive", "all", 0)).toMatchObject({ progressive: true });
    expect(buildPassFilterQuery("pressure", "all", 0)).toMatchObject({ under_pressure: true });
  });

  it("maps the explicitly labeled UI difficulty bands", () => {
    expect(buildPassFilterQuery("all", "difficult", 0)).toMatchObject({ max_expected_completion: DIFFICULT_MAX });
    expect(buildPassFilterQuery("all", "medium", 0)).toMatchObject({ min_expected_completion: MEDIUM_MIN, max_expected_completion: MEDIUM_MAX });
    expect(buildPassFilterQuery("all", "routine", 0)).toMatchObject({ min_expected_completion: ROUTINE_MIN });
  });

  it("combines subset, difficulty, and pagination without leaking other subset flags", () => {
    expect(buildPassFilterQuery("progressive", "difficult", 400)).toEqual({
      limit: 200,
      offset: 400,
      progressive: true,
      max_expected_completion: DIFFICULT_MAX,
    });
  });
});
