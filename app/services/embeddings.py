import hashlib
import re
import numpy as np
from app.db import get_cursor
from app.config import EMBEDDING_DIM, TAG_EMBEDDING_DIM, DEFAULT_K
from app.services import tag_encoder
from app.services.vector_utils import to_float_list


def make_track_id(artist: str, track: str) -> str:
    key = f"{artist.strip().lower()}|||{track.strip().lower()}"
    return hashlib.sha1(key.encode()).hexdigest()[:20]


# Two classes of trailing qualifier, handled differently:
#
# 1. COSMETIC (_CANONICAL_STRIP) — same recording, just a label. Stripped away
#    entirely so "Song", "Song (Clean)", "Song (Explicit)" and
#    "Song - Remastered 2011" collapse to one identity.
#
# 2. VARIANT (_VARIANT_MARKER) — a genuinely different recording (live/acoustic/
#    remix/demo/instrumental). NOT folded into the studio cut, but normalized to a
#    canonical token so different *spellings* of the same variant merge:
#    "Song (Live)", "Song - live" and "Song (Live Version)" → "Song (live)".
#    Marker must be bare: "Song (Tiësto Remix)" (named remixer) and
#    "Song (Live at Wembley)" (specific recording) keep their full title and stay
#    distinct — only generic markers are normalized.
_CANONICAL_STRIP = re.compile(
    r"\s*[\(\[\-]\s*("
    r"clean|explicit|dirty|"
    r"remaster(?:ed)?(?:\s*\d{4})?|"
    r"single version|album version|radio edit|"
    r"mono|stereo|bonus track|"
    r"feat\.?.*|ft\.?.*"
    r")\s*[\)\]]?\s*$",
    re.IGNORECASE,
)

_VARIANT_MARKER = re.compile(
    r"\s*[\(\[\-]\s*(live|acoustic|remix|demo|instrumental)(?:\s+version)?\s*[\)\]]?\s*$",
    re.IGNORECASE,
)


def canonical_title(track: str) -> str:
    """
    Normalize a track title to a canonical identity:
      1. Strip cosmetic, same-recording suffixes (clean/explicit/remastered/...),
         repeatedly so stacked ones ("Song (Remastered) (Bonus Track)") collapse.
      2. Normalize a trailing variant marker (live/acoustic/remix/demo/
         instrumental) to a "(marker)" token so spellings of the same variant
         merge, while staying distinct from the base recording.
    Never returns empty — if a strip would erase the whole title, the previous
    value is kept (guards against a title that is itself just "(Remastered)").
    """
    t = track.strip()
    while True:
        stripped = _CANONICAL_STRIP.sub("", t).strip()
        if stripped == t or not stripped:
            break
        t = stripped

    m = _VARIANT_MARKER.search(t)
    if m:
        base = t[: m.start()].strip()
        if base:
            t = f"{base} ({m.group(1).lower()})"
    return t


# Separators in a multi-artist credit string, in order of how a track is usually
# credited: "MC A, MC B, DJ C", "X feat. Y", "A & B", "A x B", "A vs B". Used only
# to recover a *primary* artist when the full credit doesn't resolve on Last.fm.
_ARTIST_SPLIT = re.compile(
    r"\s*(?:,|&|/| x |\bfeat\.?\b|\bft\.?\b|\bfeaturing\b|\bvs\.?\b|\bwith\b)\s*",
    re.IGNORECASE,
)


def split_credits(artist: str) -> list[str]:
    """
    Every individually-credited artist in a combined credit string, split on the
    usual separators (",", "&", "/", " x ", feat./ft./featuring/vs/with). Used by
    the embedding fallback (primary credit) and by blacklist matching (any credit).
    """
    return [part.strip() for part in _ARTIST_SPLIT.split(artist.strip()) if part.strip()]


def primary_artist(artist: str) -> str:
    """
    The first credited artist from a multi-artist string. Last.fm doesn't resolve
    combined credits like "GORDÃO DO PC, Mc Ag, Dj Wesley" (common in baile funk),
    returning no tags and so a zero embedding — but the primary artist resolves.
    Returns the original string unchanged when there's no separator. Identity keys
    (make_track_id / make_canonical_key) still use the full credit; this only feeds
    a tag-lookup fallback in the embedding pipeline.
    """
    parts = split_credits(artist)
    return parts[0] if parts else artist.strip()


def make_canonical_key(artist: str, track: str) -> str:
    """
    Identity that folds cosmetic variants of one recording together. Mirrors
    make_track_id but normalizes the title first, so clean/explicit/remastered
    editions of a song share a key. Used to dedupe search results and to keep
    variants out of a single recommendation/graph candidate pool — track_id stays
    the exact-string key (no FK churn), this is the looser "same song?" key.
    """
    key = f"{artist.strip().lower()}|||{canonical_title(track).strip().lower()}"
    return hashlib.sha1(key.encode()).hexdigest()[:20]


"""
    takes a list of tag strings and upserts them into the tag_vocab table, this is how our vocab grows over time
"""
def get_or_create_tag_ids(tags: list[str]) -> dict[str, int]:
    tag_ids = {}
    with get_cursor() as cursor:
        for tag in tags:
            # SELECT first so existing tags don't consume a SERIAL value. The old
            # `INSERT ... ON CONFLICT DO UPDATE` burned a sequence id on EVERY call
            # (Postgres allocates one even when the insert conflicts), so ids raced
            # far past the row count and past EMBEDDING_DIM — and build_tag_vector
            # silently drops any tag whose id >= EMBEDDING_DIM. Only genuinely-new
            # tags should claim an id.
            cursor.execute("SELECT id FROM tag_vocab WHERE tag = %s", (tag,))
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    "INSERT INTO tag_vocab (tag) VALUES (%s) ON CONFLICT (tag) DO NOTHING RETURNING id",
                    (tag,),
                )
                row = cursor.fetchone()
                if row is None:
                    # lost a race to a concurrent insert — read the winner's id
                    cursor.execute("SELECT id FROM tag_vocab WHERE tag = %s", (tag,))
                    row = cursor.fetchone()
            tag_ids[tag] = row["id"]
    return tag_ids


def build_tag_vector(tag_counts: dict[str, float]) -> list[float]:
    """
    Build a track's dense semantic embedding from its blended {tag: count} dict
    (algorithm 2.0, Phase 1). Each tag is encoded to a 384-dim MiniLM vector
    (cached per unique tag in tag_vocab.embedding), and the track vector is the
    count-weighted average of those tag vectors, then L2-normalized.

    Weighting by count means a track's dominant tags pull its vector hardest, the
    same intent as the old normalized-slot model — but tags now live in a shared
    semantic space, so "hip hop" and "rap" land near each other instead of in
    unrelated slots. Returns a zero vector (length TAG_EMBEDDING_DIM) when there
    are no tags or none could be encoded; the all-zero guard downstream treats that
    as "no usable tags".
    """
    if not tag_counts:
        return [0.0] * TAG_EMBEDDING_DIM

    emb_map = tag_encoder.get_tag_embeddings(list(tag_counts.keys()))
    if not emb_map:
        return [0.0] * TAG_EMBEDDING_DIM

    # Infer the dimension from the encoder output rather than assuming it, so the
    # vector matches whatever model tag_encoder is configured with.
    dim = len(next(iter(emb_map.values())))
    acc = np.zeros(dim, dtype=float)
    total_weight = 0.0
    for tag, count in tag_counts.items():
        vec = emb_map.get(tag)
        if vec is None:
            continue
        weight = float(count)
        acc += weight * vec
        total_weight += weight

    if total_weight == 0:
        return [0.0] * TAG_EMBEDDING_DIM

    norm = np.linalg.norm(acc)
    if norm == 0:
        return [0.0] * TAG_EMBEDDING_DIM

    return (acc / norm).tolist()


def ann_search(
    embedding: list,
    *,
    listeners_cap: float = float("inf"),
    exclude_ids=(),
    allowed_ids=None,
    limit: int = DEFAULT_K,
    cursor=None,
) -> list[dict]:
    """
    Approximate-nearest-neighbor search over the songs table — the single source
    of truth for vector lookups used by seeding, recommendations and playlists.

    Returns songs ordered by cosine distance to `embedding`, excluding zero-
    similarity results (no shared tags), never including `exclude_ids`, and (if
    `allowed_ids` is a non-empty set) restricted to that set. Pass an optional
    `listeners_cap` to filter by listener count (used by niche playlist mode).
    Pass an open `cursor` to reuse a transaction; otherwise one is opened.
    """
    use_allowed = bool(allowed_ids)
    use_cap = listeners_cap < float("inf")
    cap_clause = "AND listeners < %s" if use_cap else ""
    allowed_clause = "AND track_id = ANY(%s)" if use_allowed else ""
    sql = f"""
        SELECT track_id, name, artist, listeners, image, embedding,
               1 - (embedding <=> %s::vector) AS similarity
        FROM songs
        WHERE embedding IS NOT NULL
          {cap_clause}
          AND track_id != ALL(%s)
          {allowed_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    params = [embedding]
    if use_cap:
        params.append(listeners_cap)
    params.append(list(exclude_ids))
    if use_allowed:
        params.append(list(allowed_ids))
    params += [embedding, limit]

    def _run(cur) -> list[dict]:
        cur.execute(sql, params)
        return [
            {
                "track_id": r["track_id"],
                "name": r["name"],
                "artist": r["artist"],
                "listeners": r["listeners"],
                "image": r["image"],
                "embedding": to_float_list(r["embedding"]),
                "similarity": round(r["similarity"], 3),
            }
            for r in cur.fetchall()
            if r["similarity"] > 0
        ]

    if cursor is not None:
        return _run(cursor)
    with get_cursor() as cur:
        return _run(cur)


"""
    manual cos similarity between two vectors, used for in-memory similarity checks rather than hitting the DB
"""
def cosine_similarity(a: list, b: list) -> float:
    a = np.array(a)
    b = np.array(b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def sorted_tags(tag_counts: dict, limit: int | None = None) -> list[str]:
    """
    A single track's tags as a flat list, most-applied first, read from its raw
    blended {tag: count} dict (songs.tags). The dense averaged embedding can't be
    inverted back to discrete tags, so this is the one place that turns a stored
    tag dict into the ordered list the UI shows under a node title. Reused by
    /songs/{id}/features and the recommendation/playlist responses (issue #22) so
    the sort isn't re-inlined. An empty/None dict yields [].
    """
    if not tag_counts:
        return []
    ordered = [t for t, _ in sorted(tag_counts.items(), key=lambda kv: kv[1], reverse=True)]
    return ordered[:limit] if limit is not None else ordered


def dominant_tags(tag_dicts: list[dict], top_n: int = 15) -> list[dict]:
    """
    Aggregate the dominant tags across a set of songs (e.g. every node in a graph)
    so the UI can show which genres are taking over.

    Reads each song's raw blended {tag: count} dict (stored in songs.tags). The
    dense averaged embedding can't be inverted back to discrete tags, so this works
    from the persisted tag dicts instead of embedding slots. We sum each tag's
    count across all songs (its total presence in the graph); `count` is how many
    songs carry the tag, and `share` is the tag's fraction of the summed weight (a
    rough "% of the vibe"). Returns the top `top_n` by weight, descending.
    """
    if not tag_dicts:
        return []

    weight: dict[str, float] = {}
    count: dict[str, int] = {}
    for tags in tag_dicts:
        if not tags:
            continue
        for tag, c in tags.items():
            weight[tag] = weight.get(tag, 0.0) + float(c)
            count[tag] = count.get(tag, 0) + 1

    total = sum(weight.values()) or 1.0
    rows = [
        {
            "tag": tag,
            "weight": round(w, 4),
            "count": count[tag],
            "share": round(w / total, 4),
        }
        for tag, w in weight.items()
    ]
    rows.sort(key=lambda r: r["weight"], reverse=True)
    return rows[:top_n]


def mmr_rerank(query_embedding: list, candidates: list[dict], k: int, lambda_param: float) -> list[dict]:
    """
    Maximal Marginal Relevance re-ranking. Each candidate dict must have an 'embedding' key.
    Balances similarity to the query (relevance) against similarity to already-selected items (diversity).
    """
    if not candidates:
        return []

    selected = []
    remaining = candidates[:]

    while len(selected) < k and remaining:
        best = None
        best_score = float("-inf")

        for candidate in remaining:
            relevance = cosine_similarity(query_embedding, candidate["embedding"])
            redundancy = max(
                (cosine_similarity(candidate["embedding"], s["embedding"]) for s in selected),
                default=0.0,
            )
            score = lambda_param * relevance - (1 - lambda_param) * redundancy

            if score > best_score:
                best_score = score
                best = candidate

        selected.append(best)
        remaining.remove(best)

    return selected
