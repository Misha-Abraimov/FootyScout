// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "@/lib/api";
import type { AIScoutResponse, AIScoutStatus } from "@/lib/types";

import {
  AIScoutExperience,
  AI_SCOUT_LOADING_STAGES,
  readableCitations,
  safeTerminalMessage,
} from "./AIScoutExperience";

vi.mock("@/lib/api", () => ({ api: { askAIScout: vi.fn() } }));

const askAIScout = vi.mocked(api.askAIScout);

function response(overrides: Partial<AIScoutResponse> = {}): AIScoutResponse {
  return {
    run_id: "run-1",
    status: "answered",
    answer_markdown:
      "## Role Fit\n\nDistance is 0.389 [run:evidence-1; run:evidence-2].\n\nCurrent report [run:evidence-3].",
    evidence_ids: ["run:evidence-1", "run:evidence-2", "run:evidence-3"],
    methodology_sources: ["role_fit:primary"],
    web_sources: [
      {
        evidence_id: "run:evidence-3",
        title: "Official update",
        url: "https://example.com/update",
        domain: "example.com",
        published_at: "2026-09-29T12:00:00Z",
        source_quality: "official",
      },
    ],
    sources: [
      {
        evidence_id: "run:evidence-1",
        category: "analytics",
        label: "Player analytics — Granit Xhaka",
        url: null,
        domain: null,
        published_at: null,
      },
      {
        evidence_id: "run:evidence-2",
        category: "methodology",
        label: "Role Fit methodology",
        url: null,
        domain: null,
        published_at: null,
      },
      {
        evidence_id: "run:evidence-3",
        category: "web",
        label: "Official update",
        url: "https://example.com/update",
        domain: "example.com",
        published_at: "2026-09-29T12:00:00Z",
      },
    ],
    limitations: ["Role Fit does not predict future performance."],
    ...overrides,
  };
}

async function submitQuestion(question = "Explain Role Fit.") {
  const user = userEvent.setup();
  await user.type(screen.getByLabelText("Ask a scouting question"), question);
  await user.click(screen.getByRole("button", { name: "Ask AI Scout" }));
  return user;
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("AIScoutExperience", () => {
  it("renders a successful or repaired answer as normal grounded Markdown", async () => {
    askAIScout.mockResolvedValue(response());
    render(<AIScoutExperience />);
    await submitQuestion();
    expect(await screen.findByText("Grounded answer")).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Role Fit" })).toBeTruthy();
    expect(screen.getByText(/Distance is 0.389/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "1" }).getAttribute("href")).toBe(
      "#ai-source-1",
    );
    expect(screen.queryByText("Important context")).toBeNull();
    expect(
      screen.queryByText("Role Fit does not predict future performance."),
    ).toBeNull();
    expect(screen.queryByText(/repair/i)).toBeNull();
  });

  it("renders readable source categories and a clickable current web URL", async () => {
    askAIScout.mockResolvedValue(response());
    render(<AIScoutExperience />);
    await submitQuestion();
    expect(await screen.findByText(/FootyScout Analytics/)).toBeTruthy();
    expect(screen.getByText("Methodology · 2")).toBeTruthy();
    expect(screen.getByText("Role Fit methodology")).toBeTruthy();
    expect(screen.getByText(/Current Web Source/)).toBeTruthy();
    const link = screen.getByRole("link", { name: "Official update" });
    expect(link.getAttribute("href")).toBe("https://example.com/update");
    expect(link.getAttribute("target")).toBe("_blank");
    expect(link.getAttribute("rel")).toContain("noopener");
    expect(link.getAttribute("rel")).toContain("noreferrer");
    expect(screen.queryByText(/run:evidence/)).toBeNull();
  });

  it.each([
    ["clarification_required", "A little more detail is needed"],
    ["insufficient_evidence", "Not enough evidence"],
    ["unsupported", "Request not supported"],
    ["error", "AI Scout could not answer"],
  ] as Array<[AIScoutStatus, string]>)(
    "renders the %s application state",
    async (status, heading) => {
      askAIScout.mockResolvedValue(
        response({
          status,
          answer_markdown: "Safe application message.",
          evidence_ids: [],
          methodology_sources: [],
          web_sources: [],
          sources: [],
          limitations: [],
        }),
      );
      render(<AIScoutExperience />);
      await submitQuestion();
      expect(await screen.findByText(heading)).toBeTruthy();
      expect(screen.getByText("Safe application message.")).toBeTruthy();
      expect(screen.queryByText("Sources")).toBeNull();
      expect(screen.queryByText("Important context")).toBeNull();
    },
  );

  it("keeps the Smith clarification clean and available for follow-up", async () => {
    const clarification = (
      "Which 'Smith' do you mean? Please provide a first name, team, league, or the "
      + "player ID so I can load the correct profile."
    );
    askAIScout.mockResolvedValue(response({
      status: "clarification_required",
      answer_markdown: clarification,
      sources: [],
      evidence_ids: [],
      methodology_sources: [],
      web_sources: [],
      limitations: [],
    }));
    render(<AIScoutExperience />);
    await submitQuestion("Show me Smith's player profile.");
    expect(await screen.findByText(clarification)).toBeTruthy();
    expect(screen.queryByText("Sources")).toBeNull();
    expect(screen.queryByText("Important context")).toBeNull();
    expect(screen.getByLabelText("Ask a scouting question")).toHaveProperty("disabled", false);
    const result = screen.getByText("A little more detail is needed").closest("section");
    expect(result?.getAttribute("data-surface")).toBe("neutral");
  });

  it.each(["clarification_required", "unsupported", "insufficient_evidence", "error"] as const)(
    "fails closed for grounded content mislabeled as %s",
    async (status) => {
      const rawId = "9869a67c-f4eb-4511-8efb-4de76133dd99:evidence-2";
      askAIScout.mockResolvedValue(response({
        status,
        answer_markdown: `\\## Style overview\n\nComplete analytics [${rawId}]`,
        evidence_ids: [rawId],
      }));
      render(<AIScoutExperience />);
      await submitQuestion();
      await screen.findByText(statusHeadingForTest(status));
      expect(document.body.textContent).not.toContain(rawId);
      expect(document.body.textContent).not.toContain("Complete analytics");
      expect(screen.queryByText("Sources")).toBeNull();
      expect(screen.queryByText("Important context")).toBeNull();
    },
  );

  it("shows a safe request error state", async () => {
    askAIScout.mockRejectedValue(new Error("AI Scout could not complete this request."));
    render(<AIScoutExperience />);
    await submitQuestion();
    expect(await screen.findByText("AI Scout request failed")).toBeTruthy();
    expect(screen.getByText("AI Scout could not complete this request.")).toBeTruthy();
  });

  it("disables input and submission while one long request is pending", async () => {
    let resolveRequest: (value: AIScoutResponse) => void = () => undefined;
    askAIScout.mockImplementation(
      () => new Promise((resolve) => { resolveRequest = resolve; }),
    );
    render(<AIScoutExperience />);
    await submitQuestion();
    expect(screen.getByRole("button", { name: "Working..." })).toHaveProperty("disabled", true);
    expect(screen.getByLabelText("Ask a scouting question")).toHaveProperty("disabled", true);
    expect(screen.getByRole("status").textContent).toContain("Understanding your question...");
    expect(screen.getByRole("status").textContent).not.toMatch(/\b(?:40|50)\s*(?:seconds?|secs?)\b/i);
    expect(askAIScout).toHaveBeenCalledTimes(1);
    resolveRequest(response());
    await waitFor(() => expect(screen.getByText("Grounded answer")).toBeTruthy());
  });

  it("fills and submits a real example question", async () => {
    askAIScout.mockResolvedValue(response());
    const user = userEvent.setup();
    render(<AIScoutExperience />);
    await user.click(screen.getByRole("button", { name: "What does Role Fit measure?" }));
    expect(screen.getByLabelText("Ask a scouting question")).toHaveProperty(
      "value",
      "What does Role Fit measure?",
    );
    await user.click(screen.getByRole("button", { name: "Ask AI Scout" }));
    await waitFor(() => expect(askAIScout).toHaveBeenCalledWith("What does Role Fit measure?"));
  });

  it("does not render internal diagnostics or a raw evidence ledger", async () => {
    askAIScout.mockResolvedValue(response());
    render(<AIScoutExperience />);
    await submitQuestion();
    await screen.findByText("Grounded answer");
    expect(screen.queryByText(/validation_outcome/)).toBeNull();
    expect(screen.queryByText(/evidence ledger/i)).toBeNull();
    expect(screen.queryByText(/token/i)).toBeNull();
  });

  it("uses neutral surfaces for prompt, response, and source cards", async () => {
    askAIScout.mockResolvedValue(response());
    render(<AIScoutExperience />);
    await submitQuestion();
    await screen.findByText("Grounded answer");
    const neutralSurfaces = document.querySelectorAll('[data-surface="neutral"]');
    expect(neutralSurfaces.length).toBeGreaterThanOrEqual(3);
    for (const surface of neutralSurfaces) {
      expect(surface.className).not.toContain("bg-[var(--panel)]");
      expect(surface.className).not.toMatch(/green|emerald/);
    }
  });

  it("uses progress wording without an ETA or internal current-context language", () => {
    expect(AI_SCOUT_LOADING_STAGES).toEqual([
      "Understanding your question...",
      "Reviewing available evidence...",
      "Grounding the response...",
    ]);
    const loadingCopy = AI_SCOUT_LOADING_STAGES.join(" ");
    expect(loadingCopy).not.toMatch(/40|50|current context|Exa|provider/i);
  });
});

describe("readableCitations", () => {
  it("maps single and grouped run IDs to numbered source anchors", () => {
    const sources = response().sources;
    expect(
      readableCitations(
        "Claim [run:evidence-1; run:evidence-2].",
        sources,
      ),
    ).toBe("Claim [1](#ai-source-1) [2](#ai-source-2).");
  });
});

describe("safeTerminalMessage", () => {
  it("preserves a clean clarification but rejects grounded Markdown", () => {
    expect(safeTerminalMessage(response({
      status: "clarification_required",
      answer_markdown: "Which Smith do you mean?",
      evidence_ids: [],
      methodology_sources: [],
      web_sources: [],
      sources: [],
      limitations: [],
    }))).toBe("Which Smith do you mean?");
    expect(safeTerminalMessage(response({
      status: "clarification_required",
      answer_markdown: "\\## Style overview",
    }))).not.toContain("\\##");
  });
});

function statusHeadingForTest(status: Exclude<AIScoutStatus, "answered">): string {
  return {
    clarification_required: "A little more detail is needed",
    unsupported: "Request not supported",
    insufficient_evidence: "Not enough evidence",
    error: "AI Scout could not answer",
  }[status];
}
