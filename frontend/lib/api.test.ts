import { afterEach, describe, expect, it, vi } from "vitest";

import { api, buildApiUrl, resolveApiBaseUrl } from "./api";
import { buildPassFilterQuery } from "./pass-filters";

afterEach(() => vi.unstubAllGlobals());

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

  it("prefers the canonical project URL for production server-rendered requests", () => {
    expect(
      resolveApiBaseUrl({
        vercelEnv: "production",
        vercelProjectProductionUrl: "footy-scout.vercel.app",
        vercelUrl: "footy-scout-deployment-id-mishka3.vercel.app",
      }),
    ).toBe("https://footy-scout.vercel.app");
  });

  it("uses the deployment URL as the server-rendered fallback", () => {
    expect(resolveApiBaseUrl({ vercelUrl: "footyscout-git-main.example.vercel.app" })).toBe(
      "https://footyscout-git-main.example.vercel.app",
    );
  });

  it("prefers an explicit local or external API URL", () => {
    expect(
      resolveApiBaseUrl({
        publicApiUrl: "http://localhost:8000/",
        vercelEnv: "production",
        vercelProjectProductionUrl: "footy-scout.vercel.app",
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

describe("AI Scout client", () => {
  it("posts one JSON question through the typed API client", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        run_id: "run",
        status: "answered",
        answer_markdown: "Answer",
        evidence_ids: [],
        methodology_sources: [],
        web_sources: [],
        sources: [],
        limitations: [],
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await api.askAIScout("Explain Role Fit.");

    expect(result.status).toBe("answered");
    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:8000/api/ai-scout");
    expect(init).toMatchObject({
      method: "POST",
      body: JSON.stringify({ question: "Explain Role Fit." }),
    });
  });
});
