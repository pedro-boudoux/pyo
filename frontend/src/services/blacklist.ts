// Per-session artist blacklist. Layered ON TOP of the server's mandatory
// blacklist (BLACKLIST_ARTISTS): these artists are sent to /recommendations as
// `exclude_artists` so they're never recommended again from this point on.
//
// Intentionally in-memory only — a fresh page load starts with an empty list, so
// a blocked artist doesn't follow the user across sessions.
//
// The matching here mirrors the backend (app/services/blacklist.py): case-
// insensitive and credit-aware, so blocking "Drake" also matches "Drake, 21
// Savage" and "X feat. Drake".

import { useSyncExternalStore } from "react";

// Same separators as the backend's _ARTIST_SPLIT.
const CREDIT_SPLIT =
  /\s*(?:,|&|\/| x |\bfeat\.?\b|\bft\.?\b|\bfeaturing\b|\bvs\.?\b|\bwith\b)\s*/i;
// Trim leading/trailing punctuation (e.g. the ". " left by splitting "feat.").
const PUNCT_EDGES = /^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu;

function norm(name: string): string {
  return name.trim().replace(PUNCT_EDGES, "").trim().toLowerCase();
}

function credits(artist: string): string[] {
  return artist
    .split(CREDIT_SPLIT)
    .map((p) => p.trim())
    .filter(Boolean);
}

// In-memory source of truth (this module is the only writer), so snapshots are
// referentially stable for useSyncExternalStore. Starts empty on every page
// load — blocked artists don't persist between sessions.
let current: string[] = [];
const listeners = new Set<() => void>();

function set(next: string[]): string[] {
  current = next;
  listeners.forEach((l) => l());
  return next;
}

export function getBlockedArtists(): string[] {
  return current;
}

export function isArtistBlocked(artist: string, list: string[] = current): boolean {
  if (list.length === 0 || !artist) return false;
  const blocked = new Set(list.map(norm));
  if (blocked.has(norm(artist))) return true;
  return credits(artist).some((c) => blocked.has(norm(c)));
}

export function blockArtist(name: string): string[] {
  const trimmed = name.trim();
  if (!trimmed) return current;
  if (current.some((a) => norm(a) === norm(trimmed))) return current;
  return set([...current, trimmed]);
}

export function unblockArtist(name: string): string[] {
  return set(current.filter((a) => norm(a) !== norm(name)));
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** React binding — re-renders on any block/unblock. */
export function useBlockedArtists(): string[] {
  return useSyncExternalStore(subscribe, getBlockedArtists, getBlockedArtists);
}
