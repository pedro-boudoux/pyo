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
                id                     BIGSERIAL PRIMARY KEY,
                model                  TEXT NOT NULL,
                status                 TEXT NOT NULL DEFAULT 'running',
                started_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
                trained_at             TIMESTAMPTZ,
                validated_at           TIMESTAMPTZ,
                published_at           TIMESTAMPTZ,
                finished_at            TIMESTAMPTZ,
                edge_cutoff            TIMESTAMPTZ,
                node_count             BIGINT NOT NULL DEFAULT 0,
                edge_count             BIGINT NOT NULL DEFAULT 0,
                dimension              INTEGER,
                hybrid_dimension       INTEGER,
                song_count             INTEGER,
                songs_updated          INTEGER,
                fallback_count         INTEGER,
                params                 jsonb,
                validation             jsonb,
                failure_details        TEXT,
                previous_active_run_id BIGINT REFERENCES model_runs(id)
            )
        """)

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS model_run_vectors (
                model_run_id       BIGINT NOT NULL REFERENCES model_runs(id) ON DELETE CASCADE,
                track_id           TEXT NOT NULL REFERENCES songs(track_id) ON DELETE CASCADE,
                colisten_embedding vector({COLISTEN_EMBEDDING_DIM}),
                hybrid_embedding   vector({HYBRID_EMBEDDING_DIM}) NOT NULL,
                tag_only_fallback  BOOLEAN NOT NULL DEFAULT false,
                created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (model_run_id, track_id)
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

    # Safe Phase 2 candidate staging. Old successful trainer rows predate
    # lifecycle metadata; the newest one describes the vectors currently stored
    # on songs, so migrate it to `active` and mark older rows superseded.
    _try("ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'running'")
    _try("ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ NOT NULL DEFAULT now()")
    _try("ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS validated_at TIMESTAMPTZ")
    _try("ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ")
    _try("ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS finished_at TIMESTAMPTZ")
    _try("ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS edge_cutoff TIMESTAMPTZ")
    _try("ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS hybrid_dimension INTEGER")
    _try("ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS song_count INTEGER")
    _try("ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS fallback_count INTEGER")
    _try("ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS validation jsonb")
    _try("ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS failure_details TEXT")
    _try("ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS previous_active_run_id BIGINT REFERENCES model_runs(id)")
    _try("ALTER TABLE model_runs ALTER COLUMN node_count SET DEFAULT 0")
    _try("ALTER TABLE model_runs ALTER COLUMN edge_count SET DEFAULT 0")
    _try("UPDATE model_runs SET started_at = COALESCE(started_at, trained_at, now())")
    _try("UPDATE model_runs SET status = 'superseded' WHERE status = 'running' AND trained_at IS NOT NULL")
    _try("""
        UPDATE model_runs SET status = 'active',
            validated_at = COALESCE(validated_at, trained_at),
            published_at = COALESCE(published_at, trained_at),
            finished_at = COALESCE(finished_at, trained_at)
        WHERE id = (
            SELECT id FROM model_runs
            WHERE trained_at IS NOT NULL
            ORDER BY trained_at DESC, id DESC LIMIT 1
        ) AND NOT EXISTS (SELECT 1 FROM model_runs WHERE status = 'active')
    """)
    _try("""
        ALTER TABLE model_runs ADD CONSTRAINT model_runs_status_check
        CHECK (status IN ('running', 'candidate', 'validated', 'active', 'failed', 'superseded'))
    """)
    # Preserve a pre-lifecycle active run as a rollback-ready candidate. Never
    # append to a real immutable candidate on later startups: new songs may have
    # appeared since its training snapshot.
    _try("""
        INSERT INTO model_run_vectors
            (model_run_id, track_id, colisten_embedding, hybrid_embedding,
             tag_only_fallback)
        SELECT run.id, song.track_id, song.colisten_embedding,
               song.hybrid_embedding, song.colisten_embedding IS NULL
        FROM model_runs run
        CROSS JOIN songs song
        WHERE run.status = 'active' AND song.hybrid_embedding IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM model_run_vectors existing
              WHERE existing.model_run_id = run.id
          )
        ON CONFLICT (model_run_id, track_id) DO NOTHING
    """)
    _try("""
        UPDATE model_runs run SET
            song_count = candidate.count,
            songs_updated = candidate.count,
            fallback_count = candidate.fallback_count
        FROM (
            SELECT model_run_id, COUNT(*)::integer AS count,
                   COUNT(*) FILTER (WHERE tag_only_fallback)::integer AS fallback_count
            FROM model_run_vectors GROUP BY model_run_id
        ) candidate
        WHERE run.id = candidate.model_run_id AND run.status = 'active'
          AND run.song_count IS NULL
    """)

    # Ensure unique constraints and indexes exist regardless of how the table was created
    _try("CREATE UNIQUE INDEX IF NOT EXISTS songs_track_id_unique ON songs(track_id)")
    _try("CREATE UNIQUE INDEX IF NOT EXISTS graph_nodes_track_id_unique ON graph_nodes(track_id)")
    _try("CREATE INDEX IF NOT EXISTS idx_songs_embedding ON songs USING hnsw (embedding vector_cosine_ops)")
    _try("CREATE INDEX IF NOT EXISTS idx_songs_hybrid_embedding ON songs USING hnsw (hybrid_embedding vector_cosine_ops)")
    _try("CREATE INDEX IF NOT EXISTS idx_songs_canonical_key ON songs(canonical_key)")
    _try("CREATE UNIQUE INDEX IF NOT EXISTS idx_model_runs_one_active ON model_runs ((status)) WHERE status = 'active'")
    _try("CREATE INDEX IF NOT EXISTS idx_model_run_vectors_track ON model_run_vectors(track_id)")
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
