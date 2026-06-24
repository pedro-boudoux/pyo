export type SongSearchResult = {
  track_id: string;
  name: string;
  artist: string;
  image: string | null;
};

export type Recommendation = {
  track_id: string;
  name: string;
  artist: string;
  similarity: number;
  listeners: number;
  image: string | null;
  // Top tags delivered with the response so the popover shows them without a
  // second /features round-trip (issue #22). Older backends omit it → default [].
  tags: string[];
};

export type PlaylistTrack = {
  track_id: string;
  name: string;
  artist: string;
  similarity: number;
  listeners: number;
  image: string | null;
  tags: string[];
};

export type ExpansionMethod = "recommendations" | "linear" | "tree";

export type ExpansionParams = {
  method: ExpansionMethod;
  k: number;
  lambda: number;
  niche: boolean;
  maxDepth: number;
  allowDuplicates: boolean;
  minSimilarity: number; // 0..1 — drop expansion results below this cosine similarity
};

export const DEFAULT_EXPANSION: ExpansionParams = {
  method: "recommendations",
  k: 5,
  lambda: 0.7,
  niche: false,
  maxDepth: 3,
  allowDuplicates: false,
  minSimilarity: 0,
};
