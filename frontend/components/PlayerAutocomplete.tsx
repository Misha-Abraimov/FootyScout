"use client";

import {
  useEffect,
  useId,
  useRef,
  useState,
  type ChangeEvent,
  type KeyboardEvent,
} from "react";

import { api } from "@/lib/api";
import { cx } from "@/lib/format";
import { rankPlayerSuggestions } from "@/lib/player-search";
import type { PlayerIdentity, PlayerSummary } from "@/lib/types";

const DEBOUNCE_MS = 200;
const RESULT_LIMIT = 10;

export function PlayerAutocomplete({
  label,
  name,
  initialValue = "",
  selected = null,
  excludedPlayerIds = [],
  onSelect,
  className,
  placeholder = "Search player name",
}: {
  label: string;
  name?: string;
  initialValue?: string;
  selected?: PlayerIdentity | null;
  excludedPlayerIds?: number[];
  onSelect: (player: PlayerSummary) => void;
  className?: string;
  placeholder?: string;
}) {
  const listboxId = useId();
  const rootRef = useRef<HTMLDivElement>(null);
  const debounceRef = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef(0);
  const [query, setQuery] = useState(initialValue || selected?.player_name || "");
  const [suggestions, setSuggestions] = useState<PlayerSummary[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const [completedQuery, setCompletedQuery] = useState<string | null>(null);

  useEffect(() => {
    function closeOnOutsidePointer(event: PointerEvent) {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    }
    document.addEventListener("pointerdown", closeOnOutsidePointer);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsidePointer);
      if (debounceRef.current !== null) window.clearTimeout(debounceRef.current);
      abortRef.current?.abort();
    };
  }, []);

  function cancelPendingRequest() {
    requestIdRef.current += 1;
    if (debounceRef.current !== null) window.clearTimeout(debounceRef.current);
    debounceRef.current = null;
    abortRef.current?.abort();
    abortRef.current = null;
  }

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    const nextQuery = event.target.value;
    const trimmedQuery = nextQuery.trim();
    setQuery(nextQuery);
    setActiveIndex(-1);
    setCompletedQuery(null);
    setSuggestions([]);
    cancelPendingRequest();

    if (!trimmedQuery) {
      setOpen(false);
      setLoading(false);
      return;
    }

    const requestId = requestIdRef.current;
    setOpen(true);
    setLoading(true);
    debounceRef.current = window.setTimeout(async () => {
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const response = await api.getPlayers(
          {
            search: trimmedQuery,
            sort_by: "player_name",
            sort_order: "asc",
            limit: 50,
            offset: 0,
          },
          controller.signal,
        );
        if (requestId !== requestIdRef.current) return;
        const ranked = rankPlayerSuggestions(
          response.items,
          trimmedQuery,
          new Set(excludedPlayerIds),
        ).slice(0, RESULT_LIMIT);
        setSuggestions(ranked);
        setCompletedQuery(trimmedQuery);
        setLoading(false);
        setOpen(true);
      } catch (error) {
        if (isAbortError(error) || requestId !== requestIdRef.current) return;
        setSuggestions([]);
        setCompletedQuery(trimmedQuery);
        setLoading(false);
        setOpen(true);
      }
    }, DEBOUNCE_MS);
  }

  function choose(player: PlayerSummary) {
    if (excludedPlayerIds.includes(player.player_id)) return;
    cancelPendingRequest();
    setQuery(player.player_name);
    setSuggestions([]);
    setOpen(false);
    setLoading(false);
    setActiveIndex(-1);
    setCompletedQuery(null);
    onSelect(player);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Escape") {
      setOpen(false);
      setActiveIndex(-1);
      return;
    }
    if (!open || suggestions.length === 0) return;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveIndex((current) => (current + 1) % suggestions.length);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIndex((current) =>
        current <= 0 ? suggestions.length - 1 : current - 1,
      );
    } else if (event.key === "Enter" && activeIndex >= 0) {
      event.preventDefault();
      const activePlayer = suggestions[activeIndex];
      if (activePlayer) choose(activePlayer);
    }
  }

  const showNoResults =
    open && !loading && completedQuery !== null && suggestions.length === 0;

  return (
    <div
      ref={rootRef}
      className={cx("relative min-w-0", className)}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false);
      }}
    >
      <label className="grid min-w-0 gap-2 text-sm">
        <span className="text-[var(--muted)]">{label}</span>
        <input
          name={name}
          value={query}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onFocus={() => {
            if (query.trim() && (loading || suggestions.length > 0 || completedQuery)) {
              setOpen(true);
            }
          }}
          placeholder={placeholder}
          autoComplete="off"
          role="combobox"
          aria-autocomplete="list"
          aria-controls={listboxId}
          aria-expanded={open}
          aria-activedescendant={activeIndex >= 0 ? `${listboxId}-${activeIndex}` : undefined}
          className="w-full min-w-0 max-w-full rounded-lg border border-[var(--border)] bg-[#0a100e] px-3 py-2.5 text-white"
        />
      </label>

      {open ? (
        <div
          id={listboxId}
          role="listbox"
          aria-label={`${label} suggestions`}
          className="absolute z-30 mt-2 max-h-80 w-full overflow-auto rounded-xl border border-[var(--border)] bg-[#111916] p-2 shadow-2xl"
        >
          {loading ? <p className="p-3 text-sm text-[var(--muted)]">Searching…</p> : null}
          {!loading
            ? suggestions.map((player, index) => (
                <button
                  key={player.player_id}
                  id={`${listboxId}-${index}`}
                  type="button"
                  role="option"
                  aria-selected={activeIndex === index}
                  onMouseEnter={() => setActiveIndex(index)}
                  onMouseDown={(event) => event.preventDefault()}
                  onClick={() => choose(player)}
                  className={cx(
                    "block w-full rounded-lg p-3 text-left",
                    activeIndex === index ? "bg-emerald-300/10" : "hover:bg-white/5",
                  )}
                >
                  <strong className="block text-sm">{player.player_name}</strong>
                  <span className="mt-1 block text-xs text-[var(--muted)]">
                    {player.team_name} · {player.position}
                  </span>
                </button>
              ))
            : null}
          {showNoResults ? (
            <p role="status" className="p-3 text-sm text-[var(--muted)]">
              No players found for “{completedQuery}”.
            </p>
          ) : null}
        </div>
      ) : null}

      {selected && query === selected.player_name ? (
        <p className="mt-2 text-xs text-[var(--accent)]">
          Selected: {selected.team_name} · {selected.position}
        </p>
      ) : null}
    </div>
  );
}

function isAbortError(error: unknown): boolean {
  return typeof error === "object" && error !== null && "name" in error && error.name === "AbortError";
}
