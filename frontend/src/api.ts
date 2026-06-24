import type {
  ExpansionMethod,
  PlaylistTrack,
  Recommendation,
  SongSearchResult,
} from "./types";

const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined) ??
  "https://pyo-backend.up.railway.app";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText} on ${path}`);
  }
  return res.json() as Promise<T>;
}

export function searchSongs(q: string) {
  const params = new URLSearchParams({ q });
  return request<SongSearchResult[]>(`/songs/search?${params}`);
}

export function getSongStatus(track_id: string) {
  return request<{ exists: boolean; cached: boolean }>(
    `/songs/${track_id}/status`,
  );
}

export type TrackFeatures = {
  track_id: string;
  name: string;
  artist: string;
  listeners: number;
  tags: string[];
  embedding: number[];
};

// Cache hit (already embedded — true for any node on the graph) returns the
// stored tags without hitting Last.fm.
export function getSongFeatures(track_id: string) {
  return request<TrackFeatures>(`/songs/${track_id}/features`);
}

export function getSpotifyLink(track_id: string) {
  // checked=false means the backend couldn't reach Spotify (not a definitive
  // "not on Spotify") — callers should not cache that result.
  return request<{ url: string | null; checked: boolean }>(
    `/songs/${track_id}/spotify`,
  );
}

export function seedSong(track_id: string) {
  return request<{ track_id: string; name: string; artist: string }>(
    `/graph/seed`,
    { method: "POST", body: JSON.stringify({ track_id }) },
  );
}

export function getRecommendations(
  track_id: string,
  k: number,
  lambdaParam: number,
  excludeIds: string[],
  excludeArtists: string[] = [],
) {
  const params = new URLSearchParams({ k: String(k), lambda: String(lambdaParam) });
  for (const id of excludeIds) params.append("exclude", id);
  for (const a of excludeArtists) params.append("exclude_artists", a);
  return request<{ recommendations: Recommendation[] }>(
    `/recommendations/${track_id}?${params}`,
  );
}

export function linearPlaylist(
  track_id: string,
  n: number,
  niche: boolean,
  exclude_ids: string[],
) {
  return request<{ seed_track_id: string; tracks: PlaylistTrack[] }>(
    `/playlists/linear`,
    { method: "POST", body: JSON.stringify({ track_id, n, niche, exclude_ids }) },
  );
}

export function treePlaylist(
  track_id: string,
  n: number,
  max_depth: number,
  niche: boolean,
  exclude_ids: string[],
) {
  return request<{ seed_track_id: string; tracks: PlaylistTrack[] }>(
    `/playlists/tree`,
    {
      method: "POST",
      body: JSON.stringify({ track_id, n, max_depth, niche, exclude_ids }),
    },
  );
}

export type ExpandedTrack = {
  track_id: string;
  name: string;
  artist: string;
  similarity: number;
  listeners: number;
  image: string | null;
  // Top tags carried through from the recommendation/playlist response (issue #22).
  tags: string[];
};

export function getDominantTags(track_ids: string[], top_n = 5) {
  return request<{ tags: Array<{ tag: string; weight: number; count: number; share: number }> }>(
    `/graph/tags`,
    { method: "POST", body: JSON.stringify({ track_ids, top_n }) },
  );
}

export async function expandFromTrack(
  track_id: string,
  method: ExpansionMethod,
  opts: {
    k: number;
    lambda: number;
    niche: boolean;
    maxDepth: number;
    excludeIds: string[];
    excludeArtists?: string[];
  },
): Promise<ExpandedTrack[]> {
  // Older backends may omit `tags` on each item — default to [] so callers can
  // rely on tags always being an array (issue #22).
  const withTags = (items: ExpandedTrack[]): ExpandedTrack[] =>
    items.map((t) => ({ ...t, tags: t.tags ?? [] }));

  if (method === "recommendations") {
    const data = await getRecommendations(
      track_id,
      opts.k,
      opts.lambda,
      opts.excludeIds,
      opts.excludeArtists ?? [],
    );
    return withTags(data.recommendations);
  }
  if (method === "linear") {
    const data = await linearPlaylist(
      track_id,
      opts.k,
      opts.niche,
      opts.excludeIds,
    );
    return withTags(data.tracks);
  }
  const data = await treePlaylist(
    track_id,
    opts.k,
    opts.maxDepth,
    opts.niche,
    opts.excludeIds,
  );
  return withTags(data.tracks);
}
