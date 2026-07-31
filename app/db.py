import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from pgvector.psycopg2 import register_vector
from app.config import (
    DATABASE_URL,
    EMBEDDING_DIM,
    TAG_EMBEDDING_DIM,
    LEGACY_EMBEDDING_DIM,
    COLISTEN_EMBEDDING_DIM,
    HYBRID_EMBEDDING_DIM,
)


def get_connection():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    register_vector(conn)
    return conn


@contextmanager
def get_cursor():
    conn = get_connection()
    try:
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def _try(sql):
    """Run a DDL statement in its own transaction, silently ignoring errors."""
    try:
        with get_cursor() as cursor:
            cursor.execute(sql)
    except Exception:
        pass


def _column_type(table: str, column: str) -> str | None:
    """Return the formatted type of a column (e.g. 'vector(300)'), or None if the
    column doesn't exist. Whitespace is stripped so callers can compare to
    'vector(384)' without worrying about pg_catalog's spacing."""
    try:
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT format_type(a.atttypid, a.atttypmod) AS coltype
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                WHERE c.relname = %s AND a.attname = %s
                  AND a.attnum > 0 AND NOT a.attisdropped
                """,
                (table, column),
            )
            row = cursor.fetchone()
    except Exception:
        return None
    if not row or not row["coltype"]:
        return None
    return row["coltype"].replace(" ", "")


def _migrate_embedding_to_384():
    """Algorithm 2.0, Phase 1: move songs.embedding from the sparse vector(300) to
    the dense vector(384) semantic tag vector, keeping the old column as
    embedding_legacy_300 for rollback. Idempotent and resumable:

      - already vector(384)           → no-op.
      - legacy vector(300), no legacy → drop the stale HNSW index, rename the old
        column aside, add the new 384 column (NULL until backfill).
      - partial (renamed but new col   → just (re)add the 384 column.
        missing)
    """
    legacy_col = f"embedding_legacy_{LEGACY_EMBEDDING_DIM}"
    coltype = _column_type("songs", "embedding")
    new_type = f"vector({EMBEDDING_DIM})"

    if coltype == new_type:
        return  # already migrated

    if coltype == f"vector({LEGACY_EMBEDDING_DIM})" and _column_type("songs", legacy_col) is None:
        # The HNSW index would otherwise follow the renamed column; drop it here and
        # let the recreate at the bottom of init_db rebuild it on the new 384 column.
        _try("DROP INDEX IF EXISTS idx_songs_embedding")
        _try(f"ALTER TABLE songs RENAME COLUMN embedding TO {legacy_col}")

    # Add the new dense column (no-op if it already exists). NULL for every row
    # until the backfill (/songs/reembed) repopulates it through the new pipeline.
    _try(f"ALTER TABLE songs ADD COLUMN IF NOT EXISTS embedding vector({EMBEDDING_DIM})")


def init_db():
    with get_cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS songs (
                id         SERIAL PRIMARY KEY,
                track_id   TEXT UNIQUE NOT NULL,
                name       TEXT NOT NULL,
                artist     TEXT NOT NULL,
                listeners  INTEGER,
                image      TEXT,
                embedding  vector({EMBEDDING_DIM}),
                colisten_embedding vector({COLISTEN_EMBEDDING_DIM}),
                hybrid_embedding   vector({HYBRID_EMBEDDING_DIM}),
                tags       jsonb,
                spotify_url        TEXT,
                spotify_checked_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT now()
            )
        """)

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS tag_vocab (
                id        SERIAL PRIMARY KEY,
                tag       TEXT UNIQUE NOT NULL,
                embedding vector({TAG_EMBEDDING_DIM})
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id         SERIAL PRIMARY KEY,
                track_id   TEXT UNIQUE REFERENCES songs(track_id),
                is_seed    BOOLEAN DEFAULT false,
                created_at TIMESTAMPTZ DEFAULT now()
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS graph_edges (
                id         SERIAL PRIMARY KEY,
                source_id  TEXT REFERENCES songs(track_id),
                target_id  TEXT REFERENCES songs(track_id),
                similarity FLOAT,
                created_at TIMESTAMPTZ DEFAULT now()
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id         SERIAL PRIMARY KEY,
                track_id   TEXT REFERENCES songs(track_id),
                action     TEXT CHECK (action IN ('accept', 'reject')),
                created_at TIMESTAMPTZ DEFAULT now()
            )
        """)

        # Co-listening graph (algorithm 2.0, Stage B data collection — issue: hybrid
        # embedding refactor). Append-only weighted edges harvested from Last.fm
        # getSimilar at every call site. Deliberately NO foreign key to songs:
        # getSimilar targets usually aren't embedded yet, and we still want their
        # edges so the graph densifies ahead of node2vec training. source marks
        # provenance: 'track_similar' (track.getSimilar) or 'artist_similar'
        # (artist.getSimilar → top tracks, weighted by the artist match).
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS colisten_edges (
                id              SERIAL PRIMARY KEY,
                source_track_id TEXT NOT NULL,
                target_track_id TEXT NOT NULL,
                weight          FLOAT,
                source          TEXT,
                created_at      TIMESTAMPTZ DEFAULT now()
            )
        """)

        # Successful getSimilar calls that returned no edges still need durable
        # crawl state. Without this table an offline crawl retries the same empty
        # tracks on every run and never advances through the remaining catalog.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS colisten_crawl_state (
                track_id     TEXT PRIMARY KEY,
                result_count INTEGER NOT NULL DEFAULT 0,
                crawled_at   TIMESTAMPTZ DEFAULT now()
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_runs (
                id            BIGSERIAL PRIMARY KEY,
                model         TEXT NOT NULL,
                trained_at    TIMESTAMPTZ DEFAULT now(),
                node_count    BIGINT NOT NULL,
                edge_count    BIGINT NOT NULL,
                dimension     INTEGER,
                songs_updated INTEGER,
                params        jsonb
            )
        """)

    # Migrations — each runs in its own transaction so one failure doesn't block the rest
    _try("ALTER TABLE songs RENAME COLUMN spotify_id TO track_id")
    _try("ALTER TABLE graph_nodes RENAME COLUMN spotify_id TO track_id")
    _try("ALTER TABLE feedback RENAME COLUMN spotify_id TO track_id")
    _try("ALTER TABLE songs ADD COLUMN IF NOT EXISTS image TEXT")
    # Cached Spotify "listen" link. spotify_checked_at distinguishes "never looked
    # up" (NULL) from "looked up, not on Spotify" (set, with spotify_url NULL).
    _try("ALTER TABLE songs ADD COLUMN IF NOT EXISTS spotify_url TEXT")
    _try("ALTER TABLE songs ADD COLUMN IF NOT EXISTS spotify_checked_at TIMESTAMPTZ")
    # Canonical identity: sha1(artist|||canonical_title) — folds cosmetic variants
    # (clean/explicit/remastered) of one recording together so they don't dedupe
    # only against their exact track_id. Nullable: an unset value just behaves like
    # today (no folding). Backfill existing rows via POST /songs/backfill-canonical.
    _try("ALTER TABLE songs ADD COLUMN IF NOT EXISTS canonical_key TEXT")

    # Algorithm 2.0, Phase 1: dense semantic tag embeddings.
    #   - songs.embedding goes from sparse vector(300) to dense vector(384). The old
    #     column is preserved as embedding_legacy_300 for rollback (dropped only after
    #     Phase 1 sign-off).
    #   - songs.tags (jsonb) stores the raw blended {tag: count} dict: a dense averaged
    #     vector can't be inverted to discrete tags, so dominant_tags / /features read
    #     this instead of embedding slots.
    #   - tag_vocab.embedding caches each tag's MiniLM vector (encode once, reuse).
    _migrate_embedding_to_384()
    _try("ALTER TABLE songs ADD COLUMN IF NOT EXISTS tags jsonb")
    _try(f"ALTER TABLE tag_vocab ADD COLUMN IF NOT EXISTS embedding vector({TAG_EMBEDDING_DIM})")

    # Algorithm 2.0, Phase 2. Keep the signed-off Stage A vector in `embedding`
    # and build the candidate hybrid in a separate column. RECOMMENDATION_MODEL
    # controls the read path, so rollout and rollback are configuration-only.
    _try(f"ALTER TABLE songs ADD COLUMN IF NOT EXISTS colisten_embedding vector({COLISTEN_EMBEDDING_DIM})")
    _try(f"ALTER TABLE songs ADD COLUMN IF NOT EXISTS hybrid_embedding vector({HYBRID_EMBEDDING_DIM})")

    # Ensure unique constraints and indexes exist regardless of how the table was created
    _try("CREATE UNIQUE INDEX IF NOT EXISTS songs_track_id_unique ON songs(track_id)")
    _try("CREATE UNIQUE INDEX IF NOT EXISTS graph_nodes_track_id_unique ON graph_nodes(track_id)")
    _try("CREATE INDEX IF NOT EXISTS idx_songs_embedding ON songs USING hnsw (embedding vector_cosine_ops)")
    _try("CREATE INDEX IF NOT EXISTS idx_songs_hybrid_embedding ON songs USING hnsw (hybrid_embedding vector_cosine_ops)")
    _try("CREATE INDEX IF NOT EXISTS idx_songs_canonical_key ON songs(canonical_key)")
    _try("CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_edges_source_target ON graph_edges(source_id, target_id)")

    # Co-listening edges: unique per (source, target, provenance) so recording the
    # same getSimilar result twice just refreshes the weight (idempotent append).
    # Lookup index on source for the graph crawl / node2vec walk generation.
    _try("CREATE UNIQUE INDEX IF NOT EXISTS idx_colisten_edges_unique ON colisten_edges(source_track_id, target_track_id, source)")
    _try("CREATE INDEX IF NOT EXISTS idx_colisten_edges_source ON colisten_edges(source_track_id)")

    # Trigram indexes for fast substring search on the songs cache.
    # Silently skipped if pg_trgm isn't available on the host.
    _try("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    _try("CREATE INDEX IF NOT EXISTS idx_songs_name_trgm ON songs USING gin (name gin_trgm_ops)")
    _try("CREATE INDEX IF NOT EXISTS idx_songs_artist_trgm ON songs USING gin (artist gin_trgm_ops)")
