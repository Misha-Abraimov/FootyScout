import { formatPercent } from "./format";
import type { DifficultyFilter, PassViewFilter } from "./pass-filters";
import type { PassListResponse, PassResponse } from "./types";

export interface PassMapFilterState {
  view: PassViewFilter;
  difficulty: DifficultyFilter;
  offset: number;
}

export interface PassMapState {
  data: PassListResponse;
  filter: PassMapFilterState;
  selectedPassIndex: number | null;
  status: "idle" | "loading";
  error: string | null;
  activeRequestId: number;
}

export type PassMapAction =
  | { type: "request"; requestId: number; filter: PassMapFilterState }
  | { type: "success"; requestId: number; response: PassListResponse }
  | { type: "failure"; requestId: number; message: string }
  | { type: "select"; passIndex: number };

export function createPassMapState(initialData: PassListResponse): PassMapState {
  return {
    data: initialData,
    filter: { view: "all", difficulty: "all", offset: initialData.offset },
    selectedPassIndex: initialData.items[0]?.pass_index ?? null,
    status: "idle",
    error: null,
    activeRequestId: 0,
  };
}

export function passMapReducer(state: PassMapState, action: PassMapAction): PassMapState {
  if (action.type === "request") {
    return {
      ...state,
      filter: action.filter,
      status: "loading",
      error: null,
      activeRequestId: action.requestId,
    };
  }
  if (action.type === "success") {
    if (action.requestId !== state.activeRequestId) return state;
    const selectedStillExists = action.response.items.some(
      (pass) => pass.pass_index === state.selectedPassIndex,
    );
    return {
      ...state,
      data: action.response,
      selectedPassIndex: selectedStillExists
        ? state.selectedPassIndex
        : (action.response.items[0]?.pass_index ?? null),
      status: "idle",
      error: null,
    };
  }
  if (action.type === "failure") {
    if (action.requestId !== state.activeRequestId) return state;
    return { ...state, status: "idle", error: action.message };
  }
  if (!state.data.items.some((pass) => pass.pass_index === action.passIndex)) {
    return state;
  }
  return { ...state, selectedPassIndex: action.passIndex };
}

export function selectedPassForState(state: PassMapState): PassResponse | null {
  return state.data.items.find((pass) => pass.pass_index === state.selectedPassIndex) ?? null;
}

export function formatPassOption(pass: PassResponse): string {
  return `Pass #${pass.pass_index} · ${pass.completed ? "Completed" : "Incomplete"} · xPass ${formatPercent(pass.expected_completion)}`;
}
