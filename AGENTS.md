# Underground Music Discovery — Backend

## What we're building

A backend that powers a graph-based music discovery app. The user drops a seed
song onto a graph, the backend finds sonically similar underground tracks via
vector similarity search, and the graph expands based on user feedback. The user
can also export branches of the graph as linear or tree-shaped playlists.

"Underground" = low listener count from Last.fm (proxy for popularity, since
Spotify's popularity/audio-feature APIs are deprecated). The default underground
ceiling is `listeners < 500_000` (`MAX_LISTENERS`).

> See `ARCHITECTURE.md` for Mermaid diagrams of every flow described below.

---

## Stack

| Layer | Tool | Why |
|---|---|---|
| API | FastAPI (Python) | Async, fast, pairs well with ML tooling |
| Song search | Last.fm `track.search` + local Postgres cache | No login, and we reuse songs we've already embedded |
| Tags + listener counts | Last.fm API | Tag-based embeddings, listener counts, similar tracks — all free |
| Album covers | Deezer + iTunes | Last.fm stopped serving real artist/album images |
| Embeddings | fastembed (all-MiniLM-L6-v2, ONNX/CPU) + numpy | Encode Last.fm tags into a shared semantic space, then count-weighted average |
| Vector DB | Postgres + pgvector | ANN search + graph state in one DB |
| Hosting | Railway (API) + Neon (Postgres) + GitHub Pages (frontend) | Railway runs the FastAPI service via the `Procfile`, Neon is managed Postgres with pgvector, the Vite build is published to GitHub Pages |

### Dependencies (`requirements.txt`)

```
fastapi
slowapi            # per-IP rate limiting on the public API (issue #20)
uvicorn
requests
numpy
scikit-learn
fastembed          # ONNX all-MiniLM-L6-v2 tag encoder (no torch — keeps the build light)
psycopg2-binary
pgvector
python-dotenv
pydantic
```

> **Almost no Spotify.** `spotipy` is gone and Spotify plays no part in search,
> embeddings, or recommendations. The one narrow exception is resolving a public
> "listen on Spotify" link for a track: `app/services/spotify.py` uses the
> **client-credentials** flow (`SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET`) to
> hit `track.search` and return the `open.spotify.com` URL, exposed via
> `GET /songs/{track_id}/spotify`. It's optional — unset the creds and the
> endpoint just returns `{ "url": null }`. (The `spotify_id → track_id` rename
> migrations in `db.py` remain, to migrate older deployed databases.)

---

## Why not Spotify?

Spotify deprecated `GET /audio-features` and `GET /recommendations` in November
2024 (apps created after that date get a 403). Without audio features there was
no compelling reason to keep Spotify in the loop at all, so song search was moved
to **Last.fm `track.search`** merged with our **local Postgres cache**, and
album art is fetched from **Deezer/iTunes**. Everything embeddings-related comes
from Last.fm.

---

## Track identity

There is no Spotify ID. Every track is keyed by a **`track_id`**: a 20-char SHA1
of `"{artist}|||{track}"`, lowercased and stripped.

```python
# app/services/embeddings.py
def make_track_id(artist: str, track: str) -> str:
    key = f"{artist.strip().lower()}|||{track.strip().lower()}"
    return hashlib.sha1(key.encode()).hexdigest()[:20]
```

This means the same song always resolves to the same id regardless of where it
entered the system (search, seed bootstrap, recommendation top-up, playlist
expansion), which is what lets all those paths dedupe against each other.

### Canonical key (cosmetic-variant folding)

`track_id` keys the *exact* title string, so "Song", "Song (Clean)", "Song
(Explicit)" and "Song - Remastered 2011" each get a **different** id and slip
past every track_id-based dedupe — and because variants share tags, their vectors
are near-identical, so the duplicate ranks at the very top of recommendations.
To collapse them there's a second, looser identity (issue #11):

```python
# app/services/embeddings.py
def make_canonical_key(artist: str, track: str) -> str:
    # sha1(artist|||canonical_title(track)) — canonical_title strips ONLY
    # same-recording cosmetic suffixes: clean/explicit/dirty, remastered[ YYYY],
    # single/album version, radio edit, mono/stereo, bonus track, trailing feat.
```

Two classes of qualifier, handled differently in `canonical_title`:
- **Cosmetic** (clean/explicit/remastered/…) → **stripped**, so variants fold into
  the base recording.
- **Variant** (`live`, `acoustic`, `remix`, `demo`, `instrumental`) → **kept
  distinct** from the studio cut, but **normalized to a `(marker)` token** so
  different *spellings* of the same variant merge: `Tin Man (Live)`,
  `Tin man - live` and `Tin Man (Live Version)` all → `tin man (live)`, while
  `Tin Man` stays separate. Only *generic* markers normalize — a named remixer
  (`Song (Tiësto Remix)`) or a specific recording (`Song (Live at Wembley)`) keeps
  its full title and stays distinct. Numbered/multi-part tracks (`Untitled 02` vs
  `Untitled 03`) are untouched.

`canonical_key` is stored on `songs` and used
to dedupe at four surfaces: search merge, the recommendation pool + exclusion +
top-up, and the seed bootstrap pool. `track_id` stays the FK/cache key (no churn);
`canonical_key` is the "is this the same song?" key. It's nullable — an unset
value simply means "no folding" for that row (graceful), backfilled via
`POST /songs/backfill-canonical`.

---

## How it works

### The core pipeline

1. User searches → `GET /songs/search` (local DB + Last.fm in parallel, covers from Deezer/iTunes)
2. User drops a song on the graph → `POST /graph/seed`
3. Backend builds the seed's tag embedding (cache-aware), runs ANN search, then
   bootstraps + recursively expands the candidate pool from Last.fm `getSimilar`
4. Candidates become graph nodes/edges
5. User accepts or rejects nodes → `POST /feedback`
   - **Accept** → song is promoted to a seed and recommendations rerun from it
   - **Reject** → stored as a negative; future queries from the parent seed steer away
6. User exports a branch → `POST /playlists/linear` or `POST /playlists/tree`

### Embedding strategy (semantic blended tags — algorithm 2.0, Phase 1)

A single embedding blends **three** Last.fm tag sources into one
`{tag: count}` dict (`lastfm.blend_tags`), so the vector reflects both the
specific track and its broader stylistic context:

| Source | Last.fm method | Weight |
|---|---|---|
| Track tags (dominant) | `track.getTopTags` | `1.0` |
| Artist tags (context) | `artist.getTopTags` | `0.3` |
| Similar-artist tags | `artist.getSimilar` → `artist.getTopTags` each | `0.1 × match` |

> **This replaced the old sparse-slot model.** Until algorithm 2.0 the embedding
> was a sparse `vector(300)` where each `tag_vocab.id` *was* a dimension and the
> value was the normalized blended count. The problem: unrelated spellings/synonyms
> ("hip hop" vs "rap") landed in unrelated slots, so semantically close tracks
> weren't close in vector space. The dense semantic model below fixes that. The old
> 300-dim vector is preserved per row as `embedding_legacy_300` for rollback.

The pipeline (`embeddings.build_tag_vector` → `tag_encoder`):

1. Clean each tag: `tag.lower().strip()` — collapses "Hip-Hop"/"hip hop"/"hip-hop".
2. **Encode each tag string to a 384-dim vector** with `all-MiniLM-L6-v2`
   (`fastembed`, ONNX on CPU, no torch). Each unique tag is encoded **once** and the
   vector is cached in `tag_vocab.embedding` — the same handful of tags ("rock",
   "electronic", …) recur across thousands of songs, so the backfill stays cheap.
3. The track vector is the **count-weighted average** of its tag vectors
   (`Σ count·tag_vec`), then **L2-normalized**.
4. Store the dense `vector(384)` on `songs.embedding`, and the raw `{tag: count}`
   dict on `songs.tags` (jsonb) — a dense averaged vector can't be inverted back to
   discrete tags, so `dominant_tags` / `/features` read tags from there now.

Weighting by count means a track's dominant tags pull its vector hardest (same
intent as the old normalized-slot model), but tags now live in a shared semantic
space, so "hip hop" and "rap" land near each other. A track with no usable tags
yields an all-zero vector, which downstream code guards against. `tag_vocab.id` is
no longer a vector slot — it's just the row PK; the meaningful column is
`tag_vocab.embedding` (the cached per-tag MiniLM vector).

> **Phase 2 (in progress) — co-listening half.** Every `getSimilar` result is also
> persisted as a weighted edge in `colisten_edges` (see `services/colisten.py`), pure
> append-only data collection with zero new Last.fm calls, so a dense co-listening
> graph accumulates for future node2vec training. It is **not yet blended into the
> live vector** — `EMBEDDING_DIM` is still `384`. The plan (`NEW_ALGORITHM_IMPLEMENTATION.md`)
> widens it to `384 + 128 = 512` (`normalize(concat(tag_vec, β·colisten_vec))`) once
> the graph is dense enough.

### ANN search + diversity (recommendations)

`GET /recommendations/{track_id}` does more than raw nearest-neighbor:

1. **Steering** — `query = base − α·Σ(rejected neighbors)`, then normalized (`α = 0.3`).
2. **Over-fetch** — pull `k × MMR_POOL_MULTIPLIER` (3×) candidates with `listeners < 500k`.
3. **Per-artist cap** — at most `MMR_MAX_PER_ARTIST` (2) per artist in the pool; the rest go to an overflow list.
4. **MMR re-rank** — `score = λ·relevance − (1−λ)·redundancy` (`λ = 0.7`) for relevance/diversity balance.
5. **Backfill** — if still short of `k`, refill from the capped-out overflow (most similar first).
6. **Top-up** — if *still* short, fetch the seed's Last.fm `getSimilar`, embed+store, and score against the steered query. If the seed has no `getSimilar` at all, fall back to its **similar artists' top tracks** (same cold-start escape hatch as seeding).

### Vector steering on reject

When a song is rejected, future ANN queries from its parent seed are nudged away:

```
query_vector = seed_embedding − α · Σ rejected_embeddings   # then L2-normalized
```

`α` = `STEERING_ALPHA` = `0.3`. Rejections are scoped to a seed via
`graph_edges` (only rejected *neighbors of that seed* steer it) — see
`steering.get_rejected_embeddings`.

### Seed bootstrapping & recursive expansion

A fresh DB is sparse, so `POST /graph/seed` doesn't rely on ANN alone:

- Pull the seed's `getSimilar` (limit 25), embed+store each, score against the seed.
- If nothing lands under the underground cap, **escalate** the listener cap:
  `500k → 1M → 2M → 10M` until at least one candidate is added.
- Then **recursively expand**: take the top 3 candidates, pull *their* `getSimilar`
  (limit 10) and embed those too — this thickens the local neighborhood so BFS
  playlists don't drift into unrelated music once direct edges run out.
- **Cold-start fallback**: if the pool is *still* empty (instrumental / soundtrack /
  very obscure seeds often have no `track.getSimilar` at all), mine the seed's
  **similar artists' top tracks** (`artist.getSimilar` → `artist.getTopTracks`) and
  embed those instead. Without this such a seed yields an empty graph — nothing to
  embed means `/recommendations` returns nothing and the UI shows a lone node.
- Merge ANN + getSimilar + expansion, keep the top `DEFAULT_K` (10) by similarity, write edges.

### Playlists

Two strategies over a seed's graph neighborhood (`app/routers/playlists.py`):

- **Linear** (`/playlists/linear`) — flat list of the seed's neighbors. `niche=true`
  walks listener thresholds `100 → 1k → 10k → 100k → 500k`, collecting the most
  underground matches first, then sorts ascending by listener count.
- **Tree** (`/playlists/tree`) — BFS from the seed (`max_depth`, default 3),
  taking 2 neighbors per node and growing the allowed set with each visited
  node's own edges. Produces a branching path through the graph.

Both call `embed_missing` first to fill any null embeddings in the neighborhood.

---

## Data sources

### Last.fm (tags + listeners + similar — the core)

`app/services/lastfm.py`. Auth: just an API key (free, no OAuth). Methods used:

- **`track.search`** — song search (returns name, artist, listeners, image).
- **`track.getInfo`** — listener count (underground filter) + basic track tags.
- **`track.getTopTags`** — track-level tags (dominant embedding source).
- **`artist.getTopTags`** — artist-level tags with confidence counts (context).
- **`artist.getSimilar`** — similar artists (kept if `match > 0.5`) to widen tag context.
- **`track.getSimilar`** — candidate bootstrapping and recommendation top-up.

The `count` field on tags is how many users applied that tag — it acts as a
confidence score and drives normalization.

### Album covers (Deezer → iTunes → Deezer artist photo)

`app/services/covers.py`. Last.fm serves a single broken placeholder
(`2a96cbd8b46e442fc41c2b86b821562f`) for every artist, so covers are resolved by:

1. Deezer track search → `album.cover_xl` (best for underground)
2. iTunes search → `artworkUrl100` upscaled to `600x600`
3. Deezer artist photo (`picture_xl`) as a last resort

`is_broken_image()` detects the Last.fm placeholder so we never persist it, and
upserts use `COALESCE` so a known-good cover is never regressed to NULL.

---

## Database schema

Schema lives in **`migrations/init.sql`** and is also created/migrated at startup
by **`app/db.py:init_db()`** (idempotent — `CREATE TABLE IF NOT EXISTS` plus
best-effort `ALTER`/index migrations, each in its own transaction).

> The live schema is `migrations/init.sql` + `init_db()`. Connections use
> `RealDictCursor` and `pgvector.psycopg2.register_vector`.

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- for fast substring search on the cache

CREATE TABLE songs (
    id         SERIAL PRIMARY KEY,
    track_id   TEXT UNIQUE NOT NULL,   -- sha1(artist|||track)[:20]
    name       TEXT NOT NULL,
    artist     TEXT NOT NULL,
    listeners  INTEGER,
    image      TEXT,                   -- resolved album/artist cover URL
    embedding  vector(384),            -- dense semantic tag vector (algorithm 2.0, Phase 1)
    embedding_legacy_300 vector(300),  -- old sparse slot vector, kept for rollback
    tags       jsonb,                  -- raw blended {tag: count} dict (dominant_tags / /features read this)
    spotify_url        TEXT,           -- cached open.spotify.com link (NULL = none)
    spotify_checked_at TIMESTAMPTZ,    -- when we resolved it (NULL = never looked up)
    canonical_key      TEXT,           -- sha1(artist|||canonical_title): folds cosmetic variants (issue #11)
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON songs USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON songs USING gin (name gin_trgm_ops);
CREATE INDEX ON songs USING gin (artist gin_trgm_ops);
CREATE INDEX ON songs (canonical_key);   -- variant-folding dedupe lookups

CREATE TABLE tag_vocab (
    id        SERIAL PRIMARY KEY,       -- row PK only (NOT a vector dimension anymore)
    tag       TEXT UNIQUE NOT NULL,
    embedding vector(384)               -- cached MiniLM vector for this tag (encode once, reuse)
);

CREATE TABLE graph_nodes (
    id         SERIAL PRIMARY KEY,
    track_id   TEXT UNIQUE REFERENCES songs(track_id),
    is_seed    BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE graph_edges (
    id         SERIAL PRIMARY KEY,
    source_id  TEXT REFERENCES songs(track_id),
    target_id  TEXT REFERENCES songs(track_id),
    similarity FLOAT,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE UNIQUE INDEX ON graph_edges(source_id, target_id);

CREATE TABLE feedback (
    id         SERIAL PRIMARY KEY,
    track_id   TEXT REFERENCES songs(track_id),
    action     TEXT CHECK (action IN ('accept', 'reject')),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Co-listening graph (algorithm 2.0, Phase 2 data collection). Weighted edges
-- harvested from every Last.fm getSimilar call (zero extra API calls). NO FK to
-- songs on purpose — getSimilar targets usually aren't embedded yet, but we still
-- want their edges so the graph densifies ahead of node2vec training. `source`
-- marks provenance: 'track_similar' or 'artist_similar'. Append-only, idempotent
-- on (source_track_id, target_track_id, source).
CREATE TABLE colisten_edges (
    id              SERIAL PRIMARY KEY,
    source_track_id TEXT NOT NULL,
    target_track_id TEXT NOT NULL,
    weight          FLOAT,
    source          TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE UNIQUE INDEX ON colisten_edges(source_track_id, target_track_id, source);
CREATE INDEX ON colisten_edges(source_track_id);
```

A row in `songs` can exist with `embedding IS NULL` — that's a search-cache hit
we haven't embedded yet. Embedding happens lazily on seed / features / playlist.

---

## API Endpoints

### Songs

```
GET /songs/search?q={query}
```
Local DB ILIKE search + Last.fm `track.search` run in parallel, merged
(Last.fm first, then local-only), capped at 15. Reuses cached covers; only calls
Deezer/iTunes for tracks missing a real one. Upserts everything into `songs`.

```
GET /songs/{track_id}/status
```
Lightweight check: `{ exists, cached }` — `cached` means an embedding is already
stored. The frontend uses this to warn about "cold" seeds (multiple Last.fm calls, slow).

```
GET /songs/{track_id}/spotify
```
Resolves a public "listen on Spotify" link via the Spotify client-credentials
search. Returns `{ url, checked }` — `url` is the `open.spotify.com` track URL or
`null`; `checked` is `false` only when Spotify couldn't be reached (creds unset /
network error), which is *not* a definitive "not on Spotify". 404 if the track
isn't in `songs`.

The answer is **persisted on `songs`** (`spotify_url` + `spotify_checked_at`):
the first call resolves and stores it, later calls serve the stored value without
hitting Spotify. A non-definitive result (`checked=false`) is not stored, so it's
retried next time. The frontend mirrors this with a `localStorage` cache
(`spotifyCache.ts`) and prefetches links as nodes appear on the graph, so opening
a node popover is instant; it only caches `checked=true` answers. The popover
renders the link when `url` is non-null, a muted "Not on Spotify" when it's a
definitive `null`, and a brief loading state on the rare un-prefetched miss.

```
GET /songs/{track_id}/features
```
Returns `{ track_id, name, artist, listeners, tags, embedding }`. Cache hit →
returns stored embedding + tags. Cache miss → runs the full blended-tag embedding
pipeline, stores it, and returns it.

```
POST /songs/backfill-covers?limit={n}
```
Maintenance: re-resolve covers for songs with NULL or placeholder images.

```
POST /songs/backfill-canonical?limit={n}&force={bool}
```
Maintenance: fill `canonical_key` for rows written before the column existed
(`canonical_key IS NULL`). Pure string transform, no Last.fm calls — fast. Call
repeatedly until `remaining` is 0. Until a row is backfilled its NULL key just
means "no variant folding" for that song (graceful degradation). Pass
`force=true` (with a `limit` ≥ the song count) to **re-key every row** after the
`canonical_title` rules change — existing keys were computed under the old rules.

```
POST /songs/repack-vocab
```
Maintenance: re-packs `tag_vocab.id` to be dense (1..N) and NULLs all
`songs.embedding` values so `/reembed` rebuilds them. **Largely vestigial under
algorithm 2.0** — `tag_vocab.id` is no longer a vector dimension, so id density no
longer affects embeddings. It survives mainly as the "null every embedding so they
get rebuilt" trigger; the meaningful per-tag data now lives in `tag_vocab.embedding`.

```
POST /songs/reembed?limit={n}
```
Maintenance: re-embeds up to `n` songs with `embedding IS NULL`, re-fetching tags
from Last.fm and rebuilding the dense 384-dim semantic vector (`build_tag_vector`,
reusing cached per-tag MiniLM vectors from `tag_vocab.embedding`). Call repeatedly
until `remaining` is 0 — this is how a DB is migrated onto the algorithm 2.0
embedding after `_migrate_embedding_to_384()` adds the new (NULL) column.

### Graph

```
GET /graph
```
All nodes + edges for the frontend.
```json
{
  "nodes": [{ "track_id": "abc", "name": "...", "artist": "...", "is_seed": true, "listeners": 18200 }],
  "edges": [{ "source": "abc", "target": "xyz", "similarity": 0.91 }]
}
```

```
POST /graph/tags
body: { "track_ids": ["abc", ...], "top_n": 15 }
```
Dominant tags across a graph (issue #2) — "which genres are taking over". Sums
each song's blended tag weights (read from `songs.tags`, since the dense averaged
embedding can't be inverted back to discrete tags) across the node set and
returns the top `top_n` as `{ tag, weight, count, share }` (`share` ≈ % of the
vibe). Pass `track_ids` to scope to the UI's current node set; omit it to
aggregate the whole persisted graph (nodes + both ends of every edge). Empty set
or no stored tags → `{ "tags": [] }`.

```
POST /graph/seed
body: { "track_id": "abc123" }
```
Builds the embedding (cache-aware), promotes to seed node, runs ANN + getSimilar
bootstrap + recursive expansion, writes edges. Returns `{ track_id, name, artist }`.
**The track must already exist in `songs` (i.e. have been returned by search) — 404 otherwise.**

### Recommendations

```
GET /recommendations/{track_id}?k=10&lambda=0.7&exclude=...
```
Steering → ANN over-fetch → per-artist cap → MMR re-rank → overflow backfill →
Last.fm top-up. Returns `k` neighbors with `listeners < 500k`.
```json
{ "recommendations": [{ "track_id": "xyz", "name": "...", "artist": "...", "similarity": 0.94, "listeners": 18200, "image": "..." }] }
```
A cold seed (row exists but `embedding IS NULL`) is **embedded on demand** before
ranking. An unknown `track_id` → **404**. An empty list means the seed genuinely
has no underground neighbors locally or on Last.fm (or it has no usable tags at
all, yielding an all-zero vector, which is guarded against).

### Feedback

```
POST /feedback
body: { "track_id": "xyz", "action": "accept" | "reject" }
```
- `accept` → promote to seed node, copy the parent edge, rerun ANN from the
  accepted node (with its own steering) and write new edges.
- `reject` → log it; it becomes a negative that steers future recs from the parent seed.

### Playlists

```
POST /playlists/linear
body: { "track_id": "abc", "n": 10, "niche": false, "exclude_ids": [] }

POST /playlists/tree
body: { "track_id": "abc", "n": 10, "max_depth": 3, "niche": false, "exclude_ids": [] }
```
Both return `{ "seed_track_id": "abc", "tracks": [PlaylistTrack...] }`.

---

## Project structure

```
discover/
├── CLAUDE.md
├── ARCHITECTURE.md           # Mermaid diagrams of every flow
├── README.md
├── requirements.txt
├── requirements-dev.txt      # test/dev-only deps (pytest, etc.)
├── pyproject.toml
├── Procfile                  # Railway start command (uvicorn)
├── .python-version           # pins Python 3.12 for the Railway build
├── docker-compose.yml        # local Postgres + pgvector
├── Makefile
├── .env.example
├── tests/
├── migrations/
│   └── init.sql              # canonical DDL (track_id schema)
│
├── app/
│   ├── main.py               # FastAPI app, CORS, router registration, startup init_db()
│   ├── config.py             # env vars + tunables (MAX_LISTENERS, MMR_*, STEERING_ALPHA, ...)
│   ├── db.py                 # psycopg2 connection, pgvector register, init_db() + migrations
│   ├── models.py             # pydantic request/response models
│   │
│   ├── routers/
│   │   ├── songs.py          # search, status, features, backfill-covers
│   │   ├── graph.py          # GET /graph, POST /graph/seed (+ bootstrap/expansion)
│   │   ├── recommendations.py# ANN + steering + MMR + backfill + top-up
│   │   ├── feedback.py       # accept/reject
│   │   └── playlists.py      # linear + tree (BFS)
│   │
│   └── services/
│       ├── lastfm.py         # Last.fm API + blend_tags
│       ├── embeddings.py     # track_id, build_tag_vector (count-weighted avg), cosine, MMR, ann_search
│       ├── tag_encoder.py    # all-MiniLM-L6-v2 tag→384-dim encoder (fastembed), cached in tag_vocab.embedding
│       ├── steering.py       # reject vector math
│       ├── ingest.py         # embed_and_store_track — the one tag→vector pipeline, called everywhere
│       ├── colisten.py       # co-listening edge collection (Phase 2 data, append-only)
│       ├── blacklist.py      # never-recommend artist filtering (BLACKLIST_ARTISTS + per-request)
│       ├── spotify.py        # "listen on Spotify" link (client-credentials search)
│       └── covers.py         # Deezer/iTunes cover resolution
│
├── jobs/
│   └── crawl_colisten.py     # offline co-listening graph crawl (algorithm 2.0, Phase 2)
├── eval/                     # offline eval harness (recall@k, MRR, diversity, listener health)
└── frontend/                 # React + Vite + ReactFlow graph UI
```

---

## Configuration (`app/config.py`)

```python
STEERING_ALPHA      = 0.3      # reject steering strength
MAX_LISTENERS       = 500000   # underground ceiling
DEFAULT_K           = 10       # default neighbors per query
EMBEDDING_DIM       = 384      # live songs.embedding dim (Phase 2 widens to 384+128=512)
TAG_EMBEDDING_DIM   = 384      # semantic tag half (all-MiniLM-L6-v2 output)
LEGACY_EMBEDDING_DIM = 300     # old sparse vector, kept as embedding_legacy_300 for rollback
TAG_ENCODER_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"  # fastembed ONNX, CPU, no torch
MMR_LAMBDA          = 0.7      # relevance vs diversity (1.0 = pure relevance)
MMR_POOL_MULTIPLIER = 3        # over-fetch k × this before re-ranking
MMR_MAX_PER_ARTIST  = 2        # per-artist cap in the candidate pool
BLACKLIST_ARTISTS   = [...]    # parsed from env CSV — never-recommend list (credit-aware)
RATE_LIMIT_ENABLED  = True     # slowapi per-IP limiting; falsey disables it (tests turn it off)
RATE_LIMIT_DEFAULT  = "100/minute"  # applied to every route
RATE_LIMIT_HEAVY    = "20/minute"   # stricter cap on routes that call upstream APIs/embed
RATE_LIMIT_MAINTENANCE = "2/minute" # authenticated bulk-maintenance cap
MAINTENANCE_API_KEY = None     # unset disables public access to maintenance routes
```

### Rate limiting (`app/ratelimit.py`, issue #20)

A single shared slowapi `Limiter` (keyed by Railway's validated `X-Real-IP`,
falling back to the socket peer locally) guards the public API.
`app.main` registers it (`app.state.limiter`, the `RateLimitExceeded` → 429
handler, and `SlowAPIMiddleware`) so `RATE_LIMIT_DEFAULT` applies to **every**
route. The endpoints that fan out to Last.fm and run the ONNX embedder
(`/songs/search`, `/songs/{id}/features`, `/graph/seed`,
`/recommendations/{id}`, `/playlists/*`, `/feedback`) add a stricter
`@limiter.limit(RATE_LIMIT_HEAVY)`. Decorated handlers take both
`request: Request` and `response: Response`; SlowAPI needs the former to identify
the client and the latter to attach rate-limit headers to dict/Pydantic
responses. `/health` is `@limiter.exempt` (uptime pollers). The limiter is
disabled when `RATE_LIMIT_ENABLED` is falsey; the test suite sets that in
`conftest.py` and `test_ratelimit.py` builds its own enabled limiter to exercise
the production header + 429 path.

Bulk maintenance routes (`/songs/reembed`, `/songs/repack-vocab`, and both
`/songs/backfill-*` routes) use `RATE_LIMIT_MAINTENANCE` and also require
`X-Maintenance-Key` to match `MAINTENANCE_API_KEY`. They return 503 when no key
is configured, keeping them disabled on a public deployment by default.

### Environment variables

```
LASTFM_API_KEY=
LASTFM_SHARED_SECRET=          # present in .env.example; not currently required
SPOTIFY_CLIENT_ID=             # optional — only for the "listen on Spotify" link
SPOTIFY_CLIENT_SECRET=         # optional — only for the "listen on Spotify" link
BLACKLIST_ARTISTS=             # optional CSV — artists never recommended (e.g. "Drake, ...")
RATE_LIMIT_ENABLED=true        # optional — false/0/no/off disables per-IP rate limiting
RATE_LIMIT_DEFAULT=100/minute  # optional — global per-IP limit on every route
RATE_LIMIT_HEAVY=20/minute     # optional — stricter limit on upstream/API-heavy routes
RATE_LIMIT_MAINTENANCE=2/minute # optional — authenticated bulk-maintenance cap
MAINTENANCE_API_KEY=           # optional — unset disables maintenance endpoints
DATABASE_URL=postgresql://user:password@localhost:5432/music_db
```

Get a Last.fm key at [last.fm/api](https://www.last.fm/api).

---

## Local development

```bash
pip install -r requirements.txt

# Postgres with pgvector via Docker
docker run -e POSTGRES_PASSWORD=password -p 5432:5432 ankane/pgvector

# Schema is auto-created on startup by init_db(), but you can also run it manually:
psql $DATABASE_URL -f migrations/init.sql

uvicorn app.main:app --reload
```

### Testing

- **Unit suite** (`tests/`, mocked seams, no live server/DB) — `make test` / `pytest -q`.
- **Stress / load** (`tests/stress.py`, drives a *live* server over HTTP) — issue #12.
  Surfaces latency tails, error rates and DB connection-pool exhaustion under
  concurrency. Not collected by pytest. Run it:
  ```bash
  make stress                                              # local journey (search→seed→recs→…)
  make stress STRESS_URL=https://pyo-backend.up.railway.app # prod, read-only & safe
  python -m tests.stress --scenario search --concurrency 25 --duration 15
  ```
  **Prod safety:** `POST /graph/seed` and `POST /feedback` mutate graph state, so
  the harness self-skips them against a non-localhost host unless `--allow-writes`
  is passed; the default remote scenario is `read-only`. Exits non-zero if any
  endpoint's error rate exceeds `--fail-error-rate` (default 5%). The
  `.github/workflows/stress.yml` workflow runs it on demand (Actions tab, defaults
  to a read-only prod sweep) plus a weekly read-only canary.

---

## Deployment (Railway + Neon + GitHub Pages)

The API runs on **Railway**, which starts the service from the `Procfile`
(`web: uvicorn app.main:app --host 0.0.0.0 --port $PORT`) and pins Python 3.12 via
`.python-version`. The database is **Neon** managed Postgres — enable pgvector once
(`CREATE EXTENSION IF NOT EXISTS vector;`), though `init_db()` also attempts this on
startup. Set `LASTFM_API_KEY` and `DATABASE_URL` (the Neon connection string) in the
Railway service variables.

The **frontend** is built with Vite and published to **GitHub Pages** (live at
`pedro-boudoux.github.io`); point its API base URL at the Railway service.

> Migrated off Render (the old `render.yaml` was removed). Free-tier instances may
> cold-start after idle, so the first request can be slow.

---

## Notes for future work

- **Two shared building blocks — don't re-inline them.** Every embedding goes
  through `ingest.embed_and_store_track` (the one tag→vector pipeline, cache-aware)
  and every nearest-neighbor lookup goes through `embeddings.ann_search`. Routers
  call these instead of repeating the Last.fm pipeline or the `embedding <=>`
  SQL. If you need a slightly different query/pipeline, extend the helper rather
  than copying it back into a router.
- **Tag encoding is cached, per unique tag, in `tag_vocab.embedding`.** Never encode
  per song — the same tags recur across thousands of songs, so `tag_encoder` encodes
  each once and reuses it. The whole backfill is only fast because of that cache.
- **Embedding dimension is `384` (`all-MiniLM-L6-v2`).** Unlike the old slot model,
  tags are no longer dropped at a vocab cutoff — every tag contributes to the
  count-weighted average. Phase 2 widens the stored vector to `512` by concatenating
  a 128-dim co-listening embedding; if you change the model/dim, update
  `EMBEDDING_DIM`, `TAG_EMBEDDING_DIM`, the `vector(...)` columns, and re-run
  `/reembed`. See `NEW_ALGORITHM_IMPLEMENTATION.md` for the full Phase 0–2 plan.
