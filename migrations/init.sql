CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS songs (
    id                   SERIAL PRIMARY KEY,
    track_id             TEXT UNIQUE NOT NULL,
    name                 TEXT NOT NULL,
    artist               TEXT NOT NULL,
    listeners            INTEGER,
    image                TEXT,
    embedding            vector(384),
    colisten_embedding   vector(128),
    hybrid_embedding     vector(512),
    tags                 jsonb,
    spotify_url          TEXT,
    spotify_checked_at   TIMESTAMPTZ,
    canonical_key        TEXT,
    created_at           TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_songs_embedding
    ON songs USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_songs_hybrid_embedding
    ON songs USING hnsw (hybrid_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_songs_canonical_key ON songs(canonical_key);
CREATE INDEX IF NOT EXISTS idx_songs_name_trgm ON songs USING gin (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_songs_artist_trgm ON songs USING gin (artist gin_trgm_ops);

CREATE TABLE IF NOT EXISTS tag_vocab (
    id        SERIAL PRIMARY KEY,
    tag       TEXT UNIQUE NOT NULL,
    embedding vector(384)
);

CREATE TABLE IF NOT EXISTS graph_nodes (
    id         SERIAL PRIMARY KEY,
    track_id   TEXT UNIQUE REFERENCES songs(track_id),
    is_seed    BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS graph_edges (
    id         SERIAL PRIMARY KEY,
    source_id  TEXT REFERENCES songs(track_id),
    target_id  TEXT REFERENCES songs(track_id),
    similarity FLOAT,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_edges_source_target
    ON graph_edges(source_id, target_id);

CREATE TABLE IF NOT EXISTS feedback (
    id         SERIAL PRIMARY KEY,
    track_id   TEXT REFERENCES songs(track_id),
    action     TEXT CHECK (action IN ('accept', 'reject')),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS colisten_edges (
    id              SERIAL PRIMARY KEY,
    source_track_id TEXT NOT NULL,
    target_track_id TEXT NOT NULL,
    weight          FLOAT,
    source          TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_colisten_edges_unique
    ON colisten_edges(source_track_id, target_track_id, source);
CREATE INDEX IF NOT EXISTS idx_colisten_edges_source
    ON colisten_edges(source_track_id);

-- Durable completion state for successful getSimilar calls, including tracks
-- that return zero results and therefore cannot create a colisten_edges source.
CREATE TABLE IF NOT EXISTS colisten_crawl_state (
    track_id     TEXT PRIMARY KEY,
    result_count INTEGER NOT NULL DEFAULT 0,
    crawled_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_runs (
    id                     BIGSERIAL PRIMARY KEY,
    model                  TEXT NOT NULL,
    status                 TEXT NOT NULL DEFAULT 'running'
                           CHECK (status IN ('running', 'candidate', 'validated', 'active', 'failed', 'superseded')),
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
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_model_runs_one_active
    ON model_runs ((status)) WHERE status = 'active';

-- Immutable candidates. Training writes only here; publication copies a whole
-- validated run into songs inside one transaction.
CREATE TABLE IF NOT EXISTS model_run_vectors (
    model_run_id       BIGINT NOT NULL REFERENCES model_runs(id) ON DELETE CASCADE,
    track_id           TEXT NOT NULL REFERENCES songs(track_id) ON DELETE CASCADE,
    colisten_embedding vector(128),
    hybrid_embedding   vector(512) NOT NULL,
    tag_only_fallback  BOOLEAN NOT NULL DEFAULT false,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model_run_id, track_id)
);

CREATE INDEX IF NOT EXISTS idx_model_run_vectors_track
    ON model_run_vectors(track_id);
