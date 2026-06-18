// Per-user artist blacklist, kept in localStorage. Layered ON TOP of the
// server's mandatory blacklist (BLACKLIST_ARTISTS): these artists are sent to
// /recommendations as `exclude_artists` so they're never recommended, and their
// nodes are pulled from the graph the moment they're blocked.
//
// The matching here mirrors the backend (app/services/blacklist.py): case-
// insensitive and credit-aware, so blocking "Drake" also matches "Drake, 21
// Savage" and "X feat. Drake". It's used to filter nodes already on the graph.

import { useSyncExternalStore } from "react";

const KEY = "pyo:blocked-artists";

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

function readInitial(): string[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((x) => typeof x === "string") : [];
  } catch {
    return [];
  }
}

// In-memory source of truth (this module is the only writer), so snapshots are
// referentially stable for useSyncExternalStore.
let current: string[] = readInitial();
const listeners = new Set<() => void>();

function set(next: string[]): string[] {
  current = next;
  try {
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    // ignore quota / privacy-mode errors — in-memory still works for the session
  }
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
