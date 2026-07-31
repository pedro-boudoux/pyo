"""Build independent Stage B ground truth from public Deezer playlists.

Phase 2 is trained from Last.fm ``getSimilar`` edges, so Last.fm cannot also be
the judge. This builder uses adjacency in independently curated public Deezer
playlists as a lightweight co-listening proxy. It never calls Last.fm and only
keeps tracks that already have Stage A embeddings in the selected database.

The committed JSON is the reproducible fixture; playlist search results may
change later, so regeneration should be an explicit eval-data update.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
import sys
import time

from dotenv import load_dotenv
import requests


def _preload_env_file(argv: list[str]) -> None:
    for index, arg in enumerate(argv):
        if arg == "--env-file" and index + 1 < len(argv):
            load_dotenv(argv[index + 1], override=True)
            return
        if arg.startswith("--env-file="):
            load_dotenv(arg.split("=", 1)[1], override=True)
            return


_preload_env_file(sys.argv[1:])

from app.db import get_cursor
from app.services.embeddings import make_canonical_key, make_track_id


DEEZER_API = "https://api.deezer.com"
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "ground_truth_colisten.json")
DEFAULT_QUERIES = (
    "ambient electronic",
    "brazilian indie",
    "cloud rap",
    "darkwave",
    "dream pop",
    "drum and bass",
    "experimental electronic",
    "indie folk",
    "jazz rap",
    "math rock",
    "mpb",
    "neo soul",
    "noise rock",
    "post punk",
    "psychedelic rock",
    "shoegaze",
    "slowcore",
    "trip hop",
    "underground hip hop",
)


def _request_json(session, url: str, *, params: dict | None = None, retries: int = 3) -> dict:
    for attempt in range(retries):
        response = session.get(url, params=params, timeout=30)
        if response.status_code == 429 and attempt + 1 < retries:
            time.sleep(float(response.headers.get("Retry-After", "1")))
            continue
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(f"Deezer API error: {payload['error']}")
        return payload
    raise RuntimeError(f"Deezer request failed after {retries} attempts: {url}")


def discover_playlists(session, queries: list[str], per_query: int) -> list[dict]:
    playlists: dict[str, dict] = {}
    for query in queries:
        payload = _request_json(
            session,
            f"{DEEZER_API}/search/playlist",
            params={"q": query, "limit": per_query},
        )
        for item in payload.get("data", []):
            if int(item.get("nb_tracks") or 0) < 10:
                continue
            playlist_id = str(item["id"])
            record = playlists.setdefault(
                playlist_id,
                {
                    "id": playlist_id,
                    "title": item.get("title") or "",
                    "track_count": int(item.get("nb_tracks") or 0),
                    "queries": [],
                },
            )
            if query not in record["queries"]:
                record["queries"].append(query)
    return sorted(playlists.values(), key=lambda item: item["id"])


def fetch_playlist_tracks(session, playlist_id: str, max_tracks: int) -> list[dict]:
    tracks = []
    index = 0
    while len(tracks) < max_tracks:
        limit = min(100, max_tracks - len(tracks))
        payload = _request_json(
            session,
            f"{DEEZER_API}/playlist/{playlist_id}/tracks",
            params={"index": index, "limit": limit},
        )
        page = payload.get("data", [])
        if not page:
            break
        for item in page:
            artist = (item.get("artist") or {}).get("name")
            title = item.get("title")
            if artist and title:
                tracks.append({"artist": artist, "name": title})
        index += len(page)
        if len(page) < limit:
            break
    return tracks[:max_tracks]


def load_embedded_songs() -> list[dict]:
    with get_cursor() as cursor:
        cursor.execute(
            """SELECT track_id, name, artist, canonical_key
               FROM songs
               WHERE embedding IS NOT NULL
               ORDER BY track_id"""
        )
        return [dict(row) for row in cursor.fetchall()]


def song_lookups(songs: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    exact = {song["track_id"]: song for song in songs}
    canonical = {}
    for song in songs:
        key = song.get("canonical_key") or make_canonical_key(song["artist"], song["name"])
        canonical.setdefault(key, song)
    return exact, canonical


def match_track(track: dict, exact: dict[str, dict], canonical: dict[str, dict]) -> dict | None:
    track_id = make_track_id(track["artist"], track["name"])
    if track_id in exact:
        return exact[track_id]
    return canonical.get(make_canonical_key(track["artist"], track["name"]))


def build_entries(
    playlist_rows: list[tuple[dict, list[dict]]],
    songs: list[dict],
    *,
    window: int,
    target_limit: int,
    min_targets: int,
    seed_limit: int,
    max_per_artist: int = 2,
) -> tuple[list[dict], list[dict]]:
    """Turn playlist-local adjacency into deterministic seed/target entries."""
    exact, canonical = song_lookups(songs)
    song_by_id = {song["track_id"]: song for song in songs}
    scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    evidence: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    playlist_meta = []

    for playlist, raw_tracks in playlist_rows:
        matched = [match_track(track, exact, canonical) for track in raw_tracks]
        matched_count = sum(track is not None for track in matched)
        if matched_count < min_targets + 1:
            continue
        playlist_meta.append({**playlist, "matched_tracks": matched_count})
        playlist_id = str(playlist["id"])
        for left_index, left in enumerate(matched):
            if left is None:
                continue
            stop = min(len(matched), left_index + window + 1)
            for right_index in range(left_index + 1, stop):
                right = matched[right_index]
                if right is None or right["track_id"] == left["track_id"]:
                    continue
                if right["artist"].casefold() == left["artist"].casefold():
                    continue
                proximity = 1.0 / (right_index - left_index)
                for source, target in ((left, right), (right, left)):
                    scores[source["track_id"]][target["track_id"]] += proximity
                    evidence[source["track_id"]][target["track_id"]].add(playlist_id)

    candidates = []
    for seed_id, target_scores in scores.items():
        ranked = sorted(
            target_scores,
            key=lambda target_id: (
                -len(evidence[seed_id][target_id]),
                -target_scores[target_id],
                target_id,
            ),
        )[:target_limit]
        if len(ranked) < min_targets:
            continue
        candidates.append(
            (
                -sum(len(evidence[seed_id][target_id]) for target_id in ranked),
                -sum(target_scores[target_id] for target_id in ranked),
                seed_id,
                ranked,
            )
        )

    entries = []
    artist_counts: dict[str, int] = defaultdict(int)
    for _, _, seed_id, targets in sorted(candidates):
        song = song_by_id[seed_id]
        artist_key = song["artist"].casefold()
        if artist_counts[artist_key] >= max_per_artist:
            continue
        artist_counts[artist_key] += 1
        entries.append(
            {
                "seed_track_id": seed_id,
                "name": song["name"],
                "artist": song["artist"],
                "targets": targets,
            }
        )
        if len(entries) >= seed_limit:
            break
    return entries, playlist_meta


def validate_fixture(data: dict, *, min_seeds: int = 50) -> None:
    if data.get("independent_from_lastfm") is not True:
        raise ValueError("Stage B fixture must explicitly be independent from Last.fm")
    if "lastfm" in str(data.get("source", "")).casefold():
        raise ValueError("Last.fm-derived data cannot grade the co-listening model")
    seeds = data.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < min_seeds:
        raise ValueError(f"Stage B fixture needs at least {min_seeds} seeds")
    seen = set()
    for entry in seeds:
        seed_id = entry.get("seed_track_id")
        targets = entry.get("targets")
        if not seed_id or seed_id in seen:
            raise ValueError("fixture seed IDs must be present and unique")
        if not isinstance(targets, list) or not targets:
            raise ValueError(f"fixture seed {seed_id} has no targets")
        if seed_id in targets or len(targets) != len(set(targets)):
            raise ValueError(f"fixture seed {seed_id} has invalid targets")
        seen.add(seed_id)


def build(
    *,
    out_path: str,
    queries: list[str],
    playlists_per_query: int,
    max_tracks: int,
    window: int,
    target_limit: int,
    min_targets: int,
    seed_limit: int,
    min_seeds: int,
) -> dict:
    session = requests.Session()
    playlists = discover_playlists(session, queries, playlists_per_query)
    playlist_rows = []
    for index, playlist in enumerate(playlists, start=1):
        tracks = fetch_playlist_tracks(session, playlist["id"], max_tracks)
        playlist_rows.append((playlist, tracks))
        if index % 10 == 0 or index == len(playlists):
            print(f"fetched {index}/{len(playlists)} playlists", flush=True)

    entries, used_playlists = build_entries(
        playlist_rows,
        load_embedded_songs(),
        window=window,
        target_limit=target_limit,
        min_targets=min_targets,
        seed_limit=seed_limit,
    )
    data = {
        "source": "deezer_public_playlist_adjacency",
        "independent_from_lastfm": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "description": "nearby cross-artist tracks in independently curated public playlists",
            "window": window,
            "target_limit": target_limit,
            "min_targets": min_targets,
            "playlists_per_query": playlists_per_query,
            "max_tracks_per_playlist": max_tracks,
        },
        "queries": queries,
        "playlists": used_playlists,
        "seeds": entries,
    }
    validate_fixture(data, min_seeds=min_seeds)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    print(
        f"wrote {len(entries)} seeds from {len(used_playlists)} usable playlists to {out_path}"
    )
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Build independent Stage B ground truth.")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--queries", nargs="+", default=list(DEFAULT_QUERIES))
    parser.add_argument("--playlists-per-query", type=int, default=4)
    parser.add_argument("--max-tracks", type=int, default=250)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--targets", type=int, default=10)
    parser.add_argument("--min-targets", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--min-seeds", type=int, default=50)
    args = parser.parse_args()
    if args.window < 1 or args.targets < 1 or args.seeds < 1:
        parser.error("window, targets, and seeds must be positive")
    build(
        out_path=args.out,
        queries=args.queries,
        playlists_per_query=args.playlists_per_query,
        max_tracks=args.max_tracks,
        window=args.window,
        target_limit=args.targets,
        min_targets=args.min_targets,
        seed_limit=args.seeds,
        min_seeds=args.min_seeds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
