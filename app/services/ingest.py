from psycopg2.extras import Json

from app.db import get_cursor
from app.services import lastfm, embeddings, hybrid
from app.services.covers import get_cover_url
from app.services.vector_utils import to_float_list


def embed_and_store_track(artist: str, name: str) -> dict | None:
    """
    Ensure a track is embedded and stored in the songs table, fetching tags from
    Last.fm when we haven't seen it before. Returns the full song row
    (track_id, name, artist, listeners, image, embedding) or None if the track
    can't be fetched.

    This is the shared building block behind seed bootstrapping and the
    recommendation top-up: both need to turn a (artist, name) pair into a stored
    embedding without duplicating the Last.fm/embedding pipeline.
    """
    track_id = embeddings.make_track_id(artist, name)
    canonical_key = embeddings.make_canonical_key(artist, name)

    with get_cursor() as cursor:
        cursor.execute(
            """SELECT track_id, name, artist, listeners, image, embedding,
                      colisten_embedding, hybrid_embedding, tags
               FROM songs WHERE track_id = %s""",
            (track_id,),
        )
        row = cursor.fetchone()

    if row and row["embedding"] is not None:
        active_embedding = hybrid.embedding_from_row(row)
        if (
            hybrid.active_embedding_column() == "hybrid_embedding"
            and row.get("hybrid_embedding") is None
        ):
            with get_cursor() as cursor:
                cursor.execute(
                    "UPDATE songs SET hybrid_embedding = %s WHERE track_id = %s",
                    (active_embedding, track_id),
                )
        return {
            "track_id": track_id,
            "name": row["name"],
            "artist": row["artist"],
            "listeners": row["listeners"],
            "image": row["image"],
            "embedding": active_embedding,
            "tags": row["tags"] or {},
        }

    info = lastfm.get_track_info(artist, name)

    artist_tags = lastfm.get_artist_top_tags(artist)
    track_tags = lastfm.get_track_top_tags(artist, name)
    similar_artists = lastfm.get_similar_artists(artist)

    # Multi-artist credit strings ("MC A, MC B, DJ C" — rampant in baile funk) don't
    # resolve to TAGS on Last.fm, so the track gets a zero embedding and zero
    # recommendations. (They often DO return similar artists — but those are other
    # unresolvable mashed strings, equally tagless, so they can't be the trigger.)
    # When the full credit yields no tags, fall back to the primary (first) credited
    # artist, which does resolve, and take its similar artists too. Gated on "no tags
    # at all", so legit comma/&-containing band names (which DO have tags) are never
    # touched. Identity (track_id / canonical_key) still keys the full credit.
    primary = embeddings.primary_artist(artist)
    if primary != artist and not artist_tags and not track_tags:
        artist_tags = lastfm.get_artist_top_tags(primary)
        track_tags = lastfm.get_track_top_tags(primary, name)
        similar_artists = lastfm.get_similar_artists(primary)

    similar_tags = [(lastfm.get_artist_top_tags(a["artist"]), a["match"]) for a in similar_artists]
    tag_counts = lastfm.blend_tags(artist_tags, track_tags, similar_tags)
    # build_tag_vector encodes + caches each tag's MiniLM vector in tag_vocab, so the
    # old get_or_create_tag_ids slot-allocation step is no longer needed. We persist
    # the raw {tag: count} dict in songs.tags because a dense averaged vector can't be
    # inverted back to discrete tags (dominant_tags / /features read it from there).
    vector = embeddings.build_tag_vector(tag_counts)
    hybrid_vector = hybrid.compose(vector, row.get("colisten_embedding") if row else None)
    image = get_cover_url(artist, name)

    with get_cursor() as cursor:
        cursor.execute("""
            INSERT INTO songs (
                track_id, name, artist, listeners, embedding, hybrid_embedding,
                image, canonical_key, tags
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (track_id) DO UPDATE SET
                listeners = EXCLUDED.listeners,
                embedding = EXCLUDED.embedding,
                hybrid_embedding = EXCLUDED.hybrid_embedding,
                image = COALESCE(EXCLUDED.image, songs.image),
                canonical_key = EXCLUDED.canonical_key,
                tags = EXCLUDED.tags
        """, (
            track_id, name, artist, info["listeners"], vector, hybrid_vector,
            image, canonical_key, Json(tag_counts),
        ))

    return {
        "track_id": track_id,
        "name": name,
        "artist": artist,
        "listeners": info["listeners"],
        "image": image,
        "embedding": hybrid_vector if hybrid.active_embedding_column() == "hybrid_embedding" else vector,
        "tags": tag_counts,
    }
