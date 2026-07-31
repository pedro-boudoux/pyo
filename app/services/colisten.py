"""
Co-listening edge collection (algorithm 2.0, Stage B).

Every time the app asks Last.fm "what's similar to this?" we already have a set
of crowd-sourced co-listening relationships in hand — we just throw them away
after building the graph. This module persists them as weighted edges in
`colisten_edges` so that, weeks down the line, there's a dense graph to train
node2vec on. It is pure data collection: append-only, idempotent, and best-effort
(a failure here must never break seeding or recommendations), with zero new
Last.fm calls.
"""
from app.db import get_cursor
from app.services.embeddings import make_track_id


def edge_rows(source_artist, source_track, targets, source, weight=None):
    """
    Shape co-listening edge rows from one source track to a list of target tracks.

    `targets` is a list of {"artist", "name", ...} dicts. The edge weight is each
    target's own "match" score when present (track.getSimilar carries one),
    otherwise the shared `weight` argument (used for artist.getSimilar, where the
    match score lives on the artist, not the track).
    """
    source_id = make_track_id(source_artist, source_track)
    rows = []
    for t in targets:
        try:
            target_id = make_track_id(t["artist"], t["name"])
        except (KeyError, TypeError):
            continue
        if target_id == source_id:
            continue
        w = t.get("match") if isinstance(t, dict) else None
        if w is None:
            w = weight
        rows.append((source_id, target_id, w, source))
    return rows


def record_edge_rows(rows):
    """
    Persist pre-shaped co-listening edge rows.

    Idempotent on (source, target, provenance): re-recording refreshes the weight.
    Swallows all errors — this is opportunistic collection, not part of any
    request's contract.
    """
    try:
        if not rows:
            return 0

        with get_cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO colisten_edges (source_track_id, target_track_id, weight, source)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (source_track_id, target_track_id, source)
                DO UPDATE SET weight = EXCLUDED.weight
                """,
                rows,
            )
        return len(rows)
    except Exception:
        return 0


def record_edges(source_artist, source_track, targets, source, weight=None):
    """
    Persist co-listening edges from one source track to a list of target tracks.

    Kept as the request-path API; batch crawls use edge_rows + record_edge_rows
    so many source tracks can be flushed through fewer DB connections.
    """
    try:
        return record_edge_rows(edge_rows(source_artist, source_track, targets, source, weight))
    except Exception:
        return 0


def record_crawl_states(track_ids, result_count=0):
    """Mark successful crawler calls complete, including zero-result tracks.

    API failures are deliberately not written here so a later run can retry.
    Edge-producing sources remain resumable through ``colisten_edges`` itself;
    this state primarily prevents empty successful calls from looping forever.
    """
    try:
        rows = [(track_id, result_count) for track_id in dict.fromkeys(track_ids) if track_id]
        if not rows:
            return 0
        with get_cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO colisten_crawl_state (track_id, result_count, crawled_at)
                VALUES (%s, %s, now())
                ON CONFLICT (track_id) DO UPDATE SET
                    result_count = EXCLUDED.result_count,
                    crawled_at = EXCLUDED.crawled_at
                """,
                rows,
            )
        return len(rows)
    except Exception:
        return 0


def graph_stats() -> dict:
    """
    Node/edge counts + average degree for the density gate (Phase 2 task 13).
    `nodes` counts distinct track_ids appearing on either end of any edge.
    """
    with get_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) AS edges FROM colisten_edges")
        edges = cursor.fetchone()["edges"]
        cursor.execute("""
            SELECT COUNT(*) AS nodes FROM (
                SELECT source_track_id AS t FROM colisten_edges
                UNION
                SELECT target_track_id FROM colisten_edges
            ) q
        """)
        nodes = cursor.fetchone()["nodes"]

    avg_degree = round((2 * edges) / nodes, 2) if nodes else 0.0
    return {"nodes": nodes, "edges": edges, "avg_degree": avg_degree}
