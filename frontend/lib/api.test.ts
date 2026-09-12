import { describe, expect, it } from "vitest";

import { buildApiUrl, resolveApiBaseUrl } from "./api";
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

  it("uses relative same-origin API paths in an unconfigured browser deployment", () => {
    const url = buildApiUrl(
      "/api/players",
      { search: "Xhaka" },
      { browserOrigin: "https://footyscout.example" },
    );
    expect(url).toBe("/api/players?search=Xhaka");
  });

  it("uses the Vercel deployment URL for server-rendered API requests", () => {
    expect(resolveApiBaseUrl({ vercelUrl: "footyscout-git-main.example.vercel.app" })).toBe(
      "https://footyscout-git-main.example.vercel.app",
    );
  });

  it("prefers an explicit local or external API URL", () => {
    expect(
      resolveApiBaseUrl({
        publicApiUrl: "http://localhost:8000/",
        vercelUrl: "footyscout.example.vercel.app",
      }),
    ).toBe("http://localhost:8000");
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
