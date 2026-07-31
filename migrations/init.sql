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
    embedding_legacy_300 vector(300),
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

CREATE TABLE IF NOT EXISTS model_runs (
    id            BIGSERIAL PRIMARY KEY,
    model         TEXT NOT NULL,
    trained_at    TIMESTAMPTZ DEFAULT now(),
    node_count    BIGINT NOT NULL,
    edge_count    BIGINT NOT NULL,
    dimension     INTEGER,
    songs_updated INTEGER,
    params        jsonb
);
