import type {
  ActionValueModelInfoResponse,
  ArchetypeCatalogueResponse,
  AttackingActionListResponse,
  AttackingActionQuery,
  AttackingProfileResponse,
  ComparisonResponse,
  LeaderboardQuery,
  LeaderboardResponse,
  MetaResponse,
  ModelInfoResponse,
  PassListResponse,
  PassQuery,
  PlayerListResponse,
  PlayerIntelligenceResponse,
  PlayerProfileResponse,
  PlayerQuery,
  SimilarPlayersResponse,
  ShootingProfileResponse,
  ScoutingRecommendationsResponse,
  ShotListResponse,
  ShotQuery,
  XGModelInfoResponse,
  TeamIntelligenceResponse,
  TeamRoleResponse,
  PlayerRoleFitResponse,
} from "@/lib/types";

const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(
  /\/$/,
  "",
);

type QueryValue = string | number | boolean | null | undefined;

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export function buildApiUrl(
  path: string,
  query?: object,
): string {
  const url = new URL(path, `${API_URL}/`);
  if (query) {
    const entries = Object.entries(query) as Array<[string, QueryValue]>;
    for (const [key, value] of entries) {
      if (value !== undefined && value !== null && value !== "") {
        url.searchParams.set(key, String(value));
      }
    }
  }
  return url.toString();
}

async function apiFetch<T>(
  path: string,
  query?: object,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(buildApiUrl(path, query), { cache: "no-store", signal });
  if (!response.ok) {
    let detail = `FootyScout API request failed (${response.status}).`;
    try {
      const body: unknown = await response.json();
      if (
        typeof body === "object" &&
        body !== null &&
        "detail" in body &&
        typeof body.detail === "string"
      ) {
        detail = body.detail;
      }
    } catch {
      // Keep the safe status-based message when an upstream body is not JSON.
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

export const api = {
  getMeta: () => apiFetch<MetaResponse>("/api/meta"),
  getModel: () => apiFetch<ModelInfoResponse>("/api/model"),
  getXGModel: () => apiFetch<XGModelInfoResponse>("/api/models/xg"),
  getActionValueModel: () => apiFetch<ActionValueModelInfoResponse>("/api/models/action-value"),
  getArchetypes: () => apiFetch<ArchetypeCatalogueResponse>("/api/archetypes"),
  getPlayers: (query: PlayerQuery = {}, signal?: AbortSignal) =>
    apiFetch<PlayerListResponse>("/api/players", query, signal),
  getPlayer: (playerId: number) =>
    apiFetch<PlayerProfileResponse>(`/api/players/${playerId}`),
  getPlayerIntelligence: (playerId: number) =>
    apiFetch<PlayerIntelligenceResponse>(`/api/players/${playerId}/intelligence`),
  getPlayerRoleFit: (playerId: number, targetTeamId = 904) =>
    apiFetch<PlayerRoleFitResponse>(`/api/players/${playerId}/role-fit`, {
      target_team_id: targetTeamId,
    }),
  getSimilarPlayers: (playerId: number, limit = 6) =>
    apiFetch<SimilarPlayersResponse>(`/api/players/${playerId}/similar`, { limit }),
  getPlayerPasses: (playerId: number, query: PassQuery = {}, signal?: AbortSignal) =>
    apiFetch<PassListResponse>(`/api/players/${playerId}/passes`, query, signal),
  getPlayerShooting: (playerId: number) =>
    apiFetch<ShootingProfileResponse>(`/api/players/${playerId}/shooting`),
  getPlayerShots: (playerId: number, query: ShotQuery = {}, signal?: AbortSignal) =>
    apiFetch<ShotListResponse>(`/api/players/${playerId}/shots`, query, signal),
  getPlayerAttacking: (playerId: number) =>
    apiFetch<AttackingProfileResponse>(`/api/players/${playerId}/attacking`),
  getPlayerActions: (playerId: number, query: AttackingActionQuery = {}, signal?: AbortSignal) =>
    apiFetch<AttackingActionListResponse>(`/api/players/${playerId}/actions`, query, signal),
  comparePlayers: (playerIds: readonly [number, number]) =>
    apiFetch<ComparisonResponse>("/api/compare", {
      player_ids: playerIds.join(","),
    }),
  getLeaderboard: (query: LeaderboardQuery = {}) =>
    apiFetch<LeaderboardResponse>("/api/leaderboard", query),
  getTeamIntelligence: (teamId: number) =>
    apiFetch<TeamIntelligenceResponse>(`/api/teams/${teamId}/intelligence`),
  getTeamRole: (teamId: number, positionGroup: "DEF" | "MID" | "FWD") =>
    apiFetch<TeamRoleResponse>(`/api/teams/${teamId}/roles/${positionGroup}`),
  getScoutingRecommendations: (
    teamId: number,
    positionGroup: "DEF" | "MID" | "FWD",
    limit = 6,
  ) =>
    apiFetch<ScoutingRecommendationsResponse>(
      `/api/teams/${teamId}/roles/${positionGroup}/recommendations`,
      { limit },
    ),
};
