"""Candidate discovery for graph seeding, with reproducible ablation controls.

Issue #35 showed that hybrid ANN plus the seed's direct ``getSimilar`` results
preserve quality and cold-seed coverage without recursive expansion or the
similar-artist fallback. The public API therefore uses the minimal defaults. The
offline harness opts mechanisms back in explicitly to keep the committed
ablation reproducible without putting experimental switches on an HTTP endpoint.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from time import perf_counter

from app.config import DEFAULT_K
from app.db import get_cursor
from app.services import blacklist, colisten, embeddings, ingest, lastfm


SEED_SIMILAR_LIMIT = 25
EXPANSION_DEPTH = 3
EXPANSION_LIMIT = 10
ARTIST_TOPTRACKS_LIMIT = 10


@dataclass(frozen=True)
class SeedDiscoveryOptions:
    """Ablation controls; defaults are the evidence-backed production behavior."""

    recursive_expansion: bool = False
    artist_fallback: bool = False
    record_colisten: bool = True


@dataclass
class SeedDiscoveryResult:
    candidates: list[dict]
    discovered_by_source: dict[str, int]
    selected_by_source: dict[str, int]
    discovery_calls: dict[str, int]
    timing_ms: dict[str, float]
    fallback_attempted: bool
    total_discovered: int

    def measurements(self) -> dict:
        artists = {
            str(candidate.get("artist", "")).strip().casefold()
            for candidate in self.candidates
            if candidate.get("artist")
        }
        return {
            "candidate_count": len(self.candidates),
            "unique_artist_count": len(artists),
            "total_discovered": self.total_discovered,
            "discovered_by_source": self.discovered_by_source,
            "selected_by_source": self.selected_by_source,
            "discovery_calls": self.discovery_calls,
            "timing_ms": self.timing_ms,
            "fallback_attempted": self.fallback_attempted,
        }


def discover_seed_candidates(
    *,
    track_id: str,
    artist: str,
    name: str,
    vector: list[float],
    options: SeedDiscoveryOptions | None = None,
    limit: int = DEFAULT_K,
) -> SeedDiscoveryResult:
    """Build the candidate pool used by ``POST /graph/seed``.

    This function intentionally does not write graph nodes or edges. Candidate
    embedding still uses the shared ingest pipeline, so an offline ablation must
    run against a disposable database snapshot (the CLI enforces an explicit
    write acknowledgement).
    """

    options = options or SeedDiscoveryOptions()
    started = perf_counter()
    timings: defaultdict[str, float] = defaultdict(float)
    calls: Counter[str] = Counter()
    discovered: Counter[str] = Counter()

    ann_started = perf_counter()
    candidates = embeddings.ann_search(
        vector,
        exclude_ids=[track_id],
        limit=limit,
    )
    timings["ann"] += (perf_counter() - ann_started) * 1000
    candidates = [
        {**candidate, "source": "ann"}
        for candidate in candidates
        if not blacklist.is_blocked(candidate["artist"])
    ]
    discovered["ann"] = len(candidates)

    seen_ids = {candidate["track_id"] for candidate in candidates} | {track_id}
    seen_keys = {embeddings.make_canonical_key(artist, name)}
    seen_keys |= {
        embeddings.make_canonical_key(candidate["artist"], candidate["name"])
        for candidate in candidates
    }
    with get_cursor() as cursor:
        cursor.execute(
            """SELECT s.canonical_key FROM graph_nodes gn
               JOIN songs s ON gn.track_id = s.track_id
               WHERE s.canonical_key IS NOT NULL"""
        )
        seen_keys |= {row["canonical_key"] for row in cursor.fetchall()}

    def record_edges(
        source_artist: str,
        source_name: str,
        tracks: list[dict],
        *,
        source: str,
        weight: float | None = None,
    ) -> None:
        if options.record_colisten:
            colisten.record_edges(
                source_artist,
                source_name,
                tracks,
                source=source,
                weight=weight,
            )

    def absorb(tracks: list[dict], source: str) -> int:
        added = 0
        for candidate in tracks:
            if blacklist.is_blocked(candidate["artist"]):
                continue
            try:
                candidate_id = embeddings.make_track_id(
                    candidate["artist"], candidate["name"]
                )
                candidate_key = embeddings.make_canonical_key(
                    candidate["artist"], candidate["name"]
                )
                if candidate_id in seen_ids or candidate_key in seen_keys:
                    continue
                song = ingest.embed_and_store_track(
                    candidate["artist"], candidate["name"]
                )
                if song is None:
                    continue
                candidates.append(
                    {
                        **song,
                        "similarity": embeddings.cosine_similarity(
                            vector, song["embedding"]
                        ),
                        "source": source,
                    }
                )
                seen_ids.add(song["track_id"])
                seen_keys.add(candidate_key)
                added += 1
            except Exception:
                # A single malformed/unavailable Last.fm track must not break
                # graph seeding. This matches the previous router behavior.
                continue
        discovered[source] += added
        return added

    direct_started = perf_counter()
    calls["track.getSimilar"] += 1
    similar = lastfm.get_similar_tracks(
        artist, name, limit=SEED_SIMILAR_LIMIT
    )
    record_edges(artist, name, similar, source="track_similar")
    absorb(similar, "seed_similar")
    timings["seed_similar"] += (perf_counter() - direct_started) * 1000

    if options.recursive_expansion:
        expansion_started = perf_counter()
        expansion_seeds = sorted(
            candidates,
            key=lambda candidate: candidate["similarity"],
            reverse=True,
        )[:EXPANSION_DEPTH]
        for candidate in expansion_seeds:
            try:
                calls["track.getSimilar"] += 1
                expanded = lastfm.get_similar_tracks(
                    candidate["artist"],
                    candidate["name"],
                    limit=EXPANSION_LIMIT,
                )
                record_edges(
                    candidate["artist"],
                    candidate["name"],
                    expanded,
                    source="track_similar",
                )
                absorb(expanded, "recursive_expansion")
            except Exception:
                continue
        timings["recursive_expansion"] += (
            perf_counter() - expansion_started
        ) * 1000

    fallback_attempted = False
    if not candidates and options.artist_fallback:
        fallback_attempted = True
        fallback_started = perf_counter()
        calls["artist.getSimilar"] += 1
        for similar_artist in lastfm.get_similar_artists(artist):
            calls["artist.getTopTracks"] += 1
            top_tracks = lastfm.get_artist_top_tracks(
                similar_artist["artist"], limit=ARTIST_TOPTRACKS_LIMIT
            )
            record_edges(
                artist,
                name,
                top_tracks,
                source="artist_similar",
                weight=similar_artist["match"],
            )
            absorb(top_tracks, "artist_fallback")
        timings["artist_fallback"] += (
            perf_counter() - fallback_started
        ) * 1000

    total_discovered = len(candidates)
    selected = sorted(
        candidates,
        key=lambda candidate: candidate["similarity"],
        reverse=True,
    )[:limit]
    selected_sources = Counter(
        candidate.get("source", "unknown") for candidate in selected
    )
    timings["total"] = (perf_counter() - started) * 1000

    return SeedDiscoveryResult(
        candidates=selected,
        discovered_by_source=dict(sorted(discovered.items())),
        selected_by_source=dict(sorted(selected_sources.items())),
        discovery_calls=dict(sorted(calls.items())),
        timing_ms={key: round(value, 3) for key, value in sorted(timings.items())},
        fallback_attempted=fallback_attempted,
        total_discovered=total_discovered,
    )
