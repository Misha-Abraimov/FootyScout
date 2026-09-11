import { describe, expect, it } from "vitest";

import { buildApiUrl } from "./api";
import { buildPassFilterQuery } from "./pass-filters";

describe("API URL construction", () => {
  it("encodes query values and skips missing ones", () => {
    const url = new URL(buildApiUrl("/api/players", { search: "Granit Xhaka", team: undefined, limit: 25, reliable: true }));
    expect(url.origin).toBe("http://localhost:8000");
    expect(url.pathname).toBe("/api/players");
    expect(url.searchParams.get("search")).toBe("Granit Xhaka");
    expect(url.searchParams.get("team")).toBeNull();
    expect(url.searchParams.get("limit")).toBe("25");
    expect(url.searchParams.get("reliable")).toBe("true");
  });

  it.each([
    ["completed", "completed=true"],
    ["incomplete", "completed=false"],
    ["progressive", "progressive=true"],
    ["pressure", "under_pressure=true"],
  ] as const)("builds the player-pass URL for the %s subset", (view, expectedParameter) => {
    const url = buildApiUrl(
      "/api/players/3500/passes",
      buildPassFilterQuery(view, "all", 0),
    );
    expect(url).toContain(`/api/players/3500/passes?`);
    expect(url).toContain(expectedParameter);
    expect(url).toContain("limit=200");
    expect(url).toContain("offset=0");
  });
});
