"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";

import { api } from "@/lib/api";
import type { AIScoutResponse, AIScoutSource } from "@/lib/types";

const examples = [
  "Explain Granit Xhaka's passing profile.",
  "How well does Nicolas Seiwald fit Bayer Leverkusen and what is his current situation?",
  "Which players have a similar playing style to Granit Xhaka?",
  "What does Role Fit measure?",
  "Who are Bayer Leverkusen's closest midfield role matches?",
] as const;

export const AI_SCOUT_LOADING_STAGES = [
  "Understanding your question...",
  "Reviewing available evidence...",
  "Grounding the response...",
] as const;

const statusHeading = {
  answered: "Grounded answer",
  clarification_required: "A little more detail is needed",
  unsupported: "Request not supported",
  insufficient_evidence: "Not enough evidence",
  error: "AI Scout could not answer",
} as const;

export function AIScoutExperience() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState<AIScoutResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [loadingStage, setLoadingStage] = useState(0);

  useEffect(() => {
    if (!pending) return;
    const timer = window.setInterval(() => {
      setLoadingStage((stage) => Math.min(stage + 1, AI_SCOUT_LOADING_STAGES.length - 1));
    }, 8000);
    return () => window.clearInterval(timer);
  }, [pending]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalized = question.trim();
    if (!normalized || pending) return;
    setPending(true);
    setLoadingStage(0);
    setError(null);
    setResponse(null);
    try {
      setResponse(await api.askAIScout(normalized));
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "AI Scout could not complete this request.",
      );
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="space-y-8">
      <section data-surface="neutral" className="rounded-2xl border border-[#242424] bg-[#111111] p-5 sm:p-7">
        <form onSubmit={submit}>
          <label htmlFor="ai-scout-question" className="text-sm font-semibold">
            Ask a scouting question
          </label>
          <textarea
            id="ai-scout-question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            rows={4}
            disabled={pending}
            placeholder="Ask about a player profile, similar players, Role Fit, methodology, or current context."
            className="mt-3 w-full resize-y rounded-xl border border-[#242424] bg-[#0a0a0a] px-4 py-3 text-sm leading-6 text-[var(--foreground)] placeholder:text-[var(--muted)] disabled:cursor-wait disabled:opacity-70"
          />
          <div className="mt-4 flex flex-wrap items-center justify-between gap-4">
            <p className="text-xs text-[var(--muted)]">Grounded in FootyScout analytics, methodology, and current public reporting when requested.</p>
            <button
              type="submit"
              disabled={pending || !question.trim()}
              className="rounded-lg bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-black transition-opacity disabled:cursor-not-allowed disabled:opacity-45"
            >
              {pending ? "Working..." : "Ask AI Scout"}
            </button>
          </div>
        </form>
        {pending ? (
          <div data-surface="neutral" role="status" className="mt-5 rounded-xl border border-[#242424] bg-[#0a0a0a] px-4 py-4 text-sm text-[var(--muted)]">
            <span className="mr-3 inline-block size-2 animate-pulse rounded-full bg-white/70" />
            {AI_SCOUT_LOADING_STAGES[loadingStage]}
          </div>
        ) : null}
      </section>

      <section aria-labelledby="examples-title">
        <h2 id="examples-title" className="text-sm font-semibold">Try an example</h2>
        <div className="mt-3 flex flex-wrap gap-2">
          {examples.map((example) => (
            <button
              key={example}
              type="button"
              disabled={pending}
              onClick={() => setQuestion(example)}
              className="rounded-full border border-[#242424] bg-[#111111] px-3 py-2 text-left text-xs text-[var(--muted)] transition-colors hover:text-[var(--foreground)] disabled:opacity-50"
            >
              {example}
            </button>
          ))}
        </div>
      </section>

      {error ? <ResultState title="AI Scout request failed" message={error} /> : null}
      {response ? <AIScoutResult response={response} /> : null}
    </div>
  );
}

function AIScoutResult({ response }: { response: AIScoutResponse }) {
  const markdown = useMemo(
    () => readableCitations(response.answer_markdown, response.sources),
    [response.answer_markdown, response.sources],
  );
  const isAnswered = response.status === "answered";
  const terminalMessage = safeTerminalMessage(response);
  return (
    <section data-surface="neutral" className="rounded-2xl border border-[#242424] bg-[#111111] p-5 sm:p-7" aria-live="polite">
      <p className="text-xs font-semibold tracking-[0.16em] text-[var(--muted)] uppercase">AI Scout</p>
      <h2 className="mt-2 text-2xl font-semibold">{statusHeading[response.status]}</h2>
      <div className="mt-5 text-sm leading-7 text-[var(--foreground)]">
        {isAnswered ? (
          <ReactMarkdown
            components={{
              h1: ({ children }) => <h3 className="mt-6 text-xl font-semibold first:mt-0">{children}</h3>,
              h2: ({ children }) => <h3 className="mt-6 text-xl font-semibold first:mt-0">{children}</h3>,
              h3: ({ children }) => <h4 className="mt-5 text-lg font-semibold">{children}</h4>,
              p: ({ children }) => <p className="mt-3 first:mt-0">{children}</p>,
              ul: ({ children }) => <ul className="mt-3 list-disc space-y-2 pl-5">{children}</ul>,
              ol: ({ children }) => <ol className="mt-3 list-decimal space-y-2 pl-5">{children}</ol>,
              a: ({ href, children }) => <a href={href} className="text-[var(--accent-strong)] underline decoration-white/20 underline-offset-4">{children}</a>,
              strong: ({ children }) => <strong className="font-semibold text-white">{children}</strong>,
            }}
          >
            {markdown}
          </ReactMarkdown>
        ) : (
          <p>{terminalMessage}</p>
        )}
      </div>
      {isAnswered && response.sources.length ? <SourceList sources={response.sources} /> : null}
    </section>
  );
}

const terminalFallbacks: Record<Exclude<AIScoutResponse["status"], "answered">, string> = {
  clarification_required:
    "Please provide a little more detail so AI Scout can identify the correct player or request.",
  unsupported: "That request is not supported by AI Scout.",
  insufficient_evidence: "FootyScout does not have sufficient evidence to answer that request.",
  error: "AI Scout could not complete that request safely.",
};

export function safeTerminalMessage(response: AIScoutResponse): string {
  if (response.status === "answered") return response.answer_markdown;
  const hasGroundedPayload = Boolean(
    response.evidence_ids.length
      || response.methodology_sources.length
      || response.web_sources.length
      || response.sources.length
      || response.limitations.length,
  );
  const hasUnsafeMarkup = /(?:[0-9a-f-]{8,}:)?evidence-\d+|^\s*\\?#{1,6}\s+/im.test(
    response.answer_markdown,
  );
  return hasGroundedPayload || hasUnsafeMarkup
    ? terminalFallbacks[response.status]
    : response.answer_markdown;
}

function SourceList({ sources }: { sources: AIScoutSource[] }) {
  return (
    <div className="mt-8 border-t border-[#242424] pt-5">
      <h3 className="text-sm font-semibold">Sources</h3>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        {sources.map((source, index) => (
          <article data-surface="neutral" id={`ai-source-${index + 1}`} key={source.evidence_id} className="rounded-xl border border-[#242424] bg-[#0a0a0a] p-4">
            <p className="text-[11px] font-semibold tracking-[0.12em] text-[var(--muted)] uppercase">{sourceCategoryLabel(source.category)} · {index + 1}</p>
            {source.url ? (
              <a href={source.url} target="_blank" rel="noopener noreferrer" className="mt-2 block text-sm font-semibold text-[var(--foreground)] underline decoration-white/20 underline-offset-4 hover:text-[var(--accent-strong)]">{source.label}</a>
            ) : <p className="mt-2 text-sm font-semibold">{source.label}</p>}
            {source.domain || source.published_at ? <p className="mt-2 text-xs text-[var(--muted)]">{[source.domain, formatSourceDate(source.published_at)].filter(Boolean).join(" · ")}</p> : null}
          </article>
        ))}
      </div>
    </div>
  );
}

function ResultState({ title, message }: { title: string; message: string }) {
  return <section data-surface="neutral" className="rounded-2xl border border-[#242424] bg-[#111111] p-5"><h2 className="font-semibold">{title}</h2><p className="mt-2 text-sm text-[var(--muted)]">{message}</p></section>;
}

export function readableCitations(markdown: string, sources: AIScoutSource[]): string {
  const indexById = new Map(sources.map((source, index) => [source.evidence_id, index + 1]));
  return markdown.replace(/\[([^\[\]]*evidence-\d+(?:\s*;\s*[^\[\]]*evidence-\d+)*)\]/gi, (_group, value: string) => {
    const citations = value
      .split(";")
      .map((item) => item.trim())
      .map((id) => indexById.get(id))
      .filter((index): index is number => index !== undefined)
      .map((index) => `[${index}](#ai-source-${index})`);
    return citations.length ? citations.join(" ") : "";
  });
}

function sourceCategoryLabel(category: AIScoutSource["category"]): string {
  if (category === "web") return "Current Web Source";
  if (category === "methodology") return "Methodology";
  return "FootyScout Analytics";
}

function formatSourceDate(value: string | null): string | null {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date.toLocaleDateString("en-GB", { year: "numeric", month: "short", day: "numeric" });
}
