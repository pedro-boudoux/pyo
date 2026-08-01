# Backend Architecture — Mermaid Diagrams

Visual reference for how the Underground Music Discovery backend actually works
(grounded in the code under `app/`, not just the original plan).

> **Heads up — a few things to know up front:**
> - **Production API is on Coolify, not Railway.** Public base:
>   `https://pyo-backend.pedroboudoux.com`; health:
>   `https://pyo-backend.pedroboudoux.com/health`. From Pedro's MacBook, use
>   `ssh pedro-homelab` to inspect the Coolify host. The Coolify app is
>   `pyo prod` (`id=1`, uuid `y120lq5jmtgobmx27s562roa`) built from
>   `pedro-boudoux/pyo` branch `main` with Nixpacks. Treat Coolify env values,
>   especially `DATABASE_URL`, as secrets.
> - **Spotify plays no part in search, embeddings, or recommendations.** Its one
>   narrow role is resolving a public "listen on Spotify" link
>   (`services/spotify.py`, client-credentials flow) via `GET /songs/{id}/spotify`
>   — optional, and the result is cached on the `songs` row.
> - **IDs are `track_id`**, a 20-char SHA1 of `"{artist}|||{track}"` (see `embeddings.make_track_id`), not `spotify_id`.
> - A second, looser **`canonical_key`** folds cosmetic variants (clean/explicit/
>   remastered/…) of the same recording so duplicates don't flood recs (issue #11, diagram 3a).
> - **Album covers** come from Deezer → iTunes → Deezer artist photo (`services/covers.py`).
> - Extra machinery beyond the original plan: **blended tags**, **MMR re-ranking**,
>   **reject steering**, **linear & tree playlists**, **recursive seed expansion**,
>   and **dominant-tag aggregation** (`POST /graph/tags`, issue #2).
> - **Embeddings are dense semantic vectors now (algorithm 2.0, Phase 1).** Each
>   Last.fm tag is encoded with `all-MiniLM-L6-v2` (`fastembed`, ONNX/CPU) and a
>   track's `vector(384)` is the count-weighted average of its tag vectors. This
>   replaced the old sparse `vector(300)` slot model (kept per row as
>   `embedding_legacy_300` for rollback). Phase 2 stores learned graph vectors in
>   `colisten_embedding` and builds a separately indexed `hybrid_embedding`
>   (`vector(512)`). Production serves the hybrid path at `COLISTEN_BETA=2`;
>   `RECOMMENDATION_MODEL` can switch back to intact Stage A for instant rollback.
>   See `PHASE2_TODAY_PLAN.md` for the rollout record and `PROJECT_PLAN.md` for
>   remaining work.

## Two shared building blocks

Most of the flows below are built from the same two helpers, so several diagrams
reference them instead of redrawing the steps:

- **`ingest.embed_and_store_track(artist, name, listener_cap)`** — the *one*
  tag→vector embedding pipeline (diagram 3). Cache-aware: returns the stored row
  if already embedded, otherwise runs the Last.fm calls + `blend_tags` + vector
  build and upserts. Used by seeding, `/songs/.../features`, recommendation
  top-up, and playlist backfill.
- **`embeddings.ann_search(embedding, *, listeners_cap, exclude_ids, allowed_ids, limit, cursor)`**
  — the *one* pgvector nearest-neighbor query. Used by seeding, recommendations,
  feedback (accept rerun), and both playlist strategies.

---

## 1. System overview

```mermaid
graph TB
    subgraph client["Frontend (React + ReactFlow)"]
        UI["Graph UI / SearchBar / NodePopover"]
    end

    subgraph api["FastAPI app (app/main.py)"]
        R_songs["/songs router"]
        R_graph["/graph router"]
        R_recs["/recommendations router"]
        R_fb["/feedback router"]
        R_pl["/playlists router"]
    end

    subgraph svc["Services (app/services)"]
        S_lastfm["lastfm.py"]
        S_emb["embeddings.py"]
        S_tagenc["tag_encoder.py (MiniLM)"]
        S_steer["steering.py"]
        S_ingest["ingest.py"]
        S_colisten["colisten.py"]
        S_blacklist["blacklist.py"]
        S_covers["covers.py"]
        S_spotify["spotify.py"]
    end

    subgraph data["Postgres + pgvector"]
        T_songs[("songs")]
        T_vocab[("tag_vocab")]
        T_nodes[("graph_nodes")]
        T_edges[("graph_edges")]
        T_fb[("feedback")]
        T_colisten[("colisten_edges")]
    end

    subgraph ext["External APIs"]
        E_lastfm["Last.fm API"]
        E_deezer["Deezer"]
        E_itunes["iTunes"]
        E_spotify["Spotify API<br/>(link resolution only)"]
    end

    UI --> R_songs & R_graph & R_recs & R_fb & R_pl

    R_songs --> S_lastfm & S_emb & S_covers & S_spotify
    R_graph --> S_lastfm & S_emb & S_ingest
    R_recs --> S_steer & S_ingest & S_emb & S_lastfm
    R_fb --> S_steer
    R_pl --> S_lastfm & S_emb & S_covers

    S_lastfm --> E_lastfm
    S_covers --> E_deezer & E_itunes
    S_spotify --> E_spotify
    S_ingest --> S_lastfm & S_emb & S_covers

    svc --> data
    api --> data
```

---

## 2. Database schema (ER)

```mermaid
erDiagram
    songs {
        serial id PK
        text track_id UK "sha1(artist|||track)"
        text name
        text artist
        int listeners "underground filter"
        vector embedding "vector(384) dense semantic tag vector (algo 2.0)"
        vector embedding_legacy_300 "old sparse slot vector (rollback)"
        jsonb tags "raw blended {tag: count} (dominant_tags / features read this)"
        text image "cover URL"
        text canonical_key "sha1(artist|||canonical_title): folds variants, indexed (issue #11)"
        text spotify_url "cached open.spotify.com link (NULL = none)"
        timestamptz spotify_checked_at "when resolved (NULL = never)"
    }
    tag_vocab {
        serial id PK "row PK only (no longer a vector dimension)"
        text tag UK "cleaned, lowercased"
        vector embedding "cached MiniLM vector(384) for this tag"
    }
    graph_nodes {
        serial id PK
        text track_id FK
        bool is_seed
    }
    graph_edges {
        serial id PK
        text source_id FK
        text target_id FK
        float similarity
    }
    feedback {
        serial id PK
        text track_id FK
        text action "accept | reject"
    }
    colisten_edges {
        serial id PK
        text source_track_id "no FK — targets often unembedded"
        text target_track_id
        float weight
        text source "track_similar | artist_similar"
    }

    songs ||--o| graph_nodes : "appears as"
    songs ||--o{ graph_edges : "source_id"
    songs ||--o{ graph_edges : "target_id"
    songs ||--o{ feedback : "rated in"
    tag_vocab ||..|| songs : "tag vectors averaged → songs.embedding"
```

---

## 3. Embedding pipeline (tags → vector) — `ingest.embed_and_store_track`

How any `(artist, track)` becomes a stored dense `vector(384)` semantic embedding
(algorithm 2.0, Phase 1). This *is* `ingest.embed_and_store_track` — the single
shared block every other flow calls when it needs an embedding
(`services/ingest.py`, `services/embeddings.py`, `services/tag_encoder.py`,
`lastfm.blend_tags`). A cache hit short-circuits before any Last.fm call.

```mermaid
flowchart TD
    start(["(artist, track)"]) --> info["lastfm.get_track_info<br/>→ listeners + basic tags"]
    info --> cap{"listeners ≥ cap?"}
    cap -->|yes| drop["return None<br/>(too popular)"]
    cap -->|no| tags

    subgraph tags["Gather tags (4+ Last.fm calls)"]
        a["artist.getTopTags<br/>× weight 0.3"]
        t["track.getTopTags<br/>× weight 1.0 (dominant)"]
        sim["artist.getSimilar →<br/>each artist's tags × 0.1 × match"]
    end

    tags --> blend["blend_tags()<br/>merge into one {tag: count} dict<br/>drop count ≤ 0"]
    blend --> enc["tag_encoder.get_tag_embeddings()<br/>each tag → 384-dim MiniLM vector<br/>cached per tag in tag_vocab.embedding"]
    enc --> build["build_tag_vector()<br/>Σ count·tag_vec → L2-normalize"]
    build --> norm["dense vector(384)<br/>(all-zero if no usable tags)"]
    norm --> store[("UPSERT into songs<br/>(embedding, tags jsonb, listeners, image)")]
    cover["covers.get_cover_url<br/>Deezer → iTunes → artist photo"] --> store
```

---

## 3a. Canonical key — folding cosmetic variants (issue #11)

`track_id` keys the *exact* title, so "Song", "Song (Clean)" and "Song -
Remastered 2011" get different ids and slip past track_id dedupe — yet their tags
(and vectors) are near-identical, so the duplicate ranks at the top of recs. A
second, looser identity collapses them: `canonical_key = sha1(artist|||canonical_title(track))`
(`embeddings.canonical_title` / `make_canonical_key`). It's stored on `songs`
(indexed, nullable → "no folding"), backfilled by `POST /songs/backfill-canonical`,
and used to dedupe at the search merge, the recommendation pool/exclusion/top-up,
and the seed bootstrap pool. `track_id` stays the FK/cache key.

```mermaid
flowchart TD
    in(["raw track title"]) --> low["lowercase + strip"]
    low --> kind{"qualifier in (…)/after -?"}

    kind -->|cosmetic<br/>clean·explicit·dirty·remastered[ YYYY]·<br/>single/album version·radio edit·mono/stereo·<br/>bonus track·trailing feat.| strip["STRIP it<br/>→ folds into base recording"]
    kind -->|variant<br/>live·acoustic·remix·demo·instrumental| keepgen{"generic marker?"}
    kind -->|none / numbered<br/>(Untitled 02)| asis["leave title as-is"]

    keepgen -->|generic<br/>'(Live)', '- live', '(Live Version)'| normtok["normalize to '(marker)' token<br/>→ spellings merge, stays distinct from studio cut"]
    keepgen -->|named/specific<br/>'(Tiësto Remix)', '(Live at Wembley)'| keepfull["keep full title<br/>→ stays distinct"]

    strip --> key
    normtok --> key
    keepfull --> key
    asis --> key["sha1(artist|||canonical_title)<br/>= canonical_key"]
```

---

## 4. Song search (`GET /songs/search`)

```mermaid
flowchart LR
    q(["q = user query"]) --> par

    subgraph par["Parallel (ThreadPoolExecutor)"]
        local["_search_local_songs<br/>ILIKE on songs table"]
        lfm["lastfm.search_tracks<br/>track.search"]
    end

    par --> merge["merge: Last.fm first,<br/>then local-only<br/>(dedupe by track_id)"]
    merge --> trim["trim to SEARCH_LIMIT (15)"]
    trim --> covercheck{"cover cached<br/>& not broken?"}
    covercheck -->|yes| usecached["reuse cached cover"]
    covercheck -->|no| fetchcover["get_cover_url in parallel<br/>(Deezer/iTunes)"]
    usecached --> upsert
    fetchcover --> upsert[("_upsert_songs<br/>COALESCE keeps good covers")]
    upsert --> resp(["SongSearchResult[]"])
```

---

## 5. Seeding the graph (`POST /graph/seed`)

The core loop. Builds the seed embedding (cache-aware), runs ANN search,
bootstraps from Last.fm `getSimilar`, and recursively expands so the
neighborhood is dense enough for playlists.

```mermaid
flowchart TD
    req(["POST /graph/seed {track_id}"]) --> lookup["SELECT song by track_id"]
    lookup --> exist{"row exists?"}
    exist -->|no| e404["404 — search first"]
    exist -->|yes| cached{"embedding cached?"}

    cached -->|yes| reuse["reuse stored vector<br/>(no API calls)"]
    cached -->|no| build["embed_and_store_track<br/>(diagram 3, cap = ∞)"]

    reuse --> node
    build --> node["UPSERT graph_nodes<br/>is_seed = true"]

    node --> ann["ann_search<br/>no listener cap, limit k"]
    ann --> simseed["lastfm.get_similar_tracks(seed)<br/>limit 25"]

    simseed --> expand["Recursive expansion<br/>top 3 candidates → getSimilar(10)<br/>embed+store those too"]

    expand --> coldcheck{"pool still empty?<br/>(no track.getSimilar)"}
    coldcheck -->|yes| fallback["Cold-start fallback:<br/>artist.getSimilar → artist.getTopTracks<br/>embed+store those"]
    coldcheck -->|no| rank
    fallback --> rank["merge ANN + getSimilar + expansion<br/>sort by similarity, keep top k"]
    rank --> edges[("INSERT graph_edges<br/>seed → each candidate")]
    edges --> done(["{track_id, name, artist}"])
```

---

## 5a. Dominant tags across a graph (`POST /graph/tags`, issue #2)

"Which genres are taking over." Sums each song's blended tag weights (read from the
`songs.tags` jsonb — the dense averaged embedding can't be inverted back to discrete
tags) across a node set and returns the top `top_n` as `{ tag, weight, count, share }`
(`embeddings.dominant_tags`). Pass `track_ids` to scope to exactly what the UI is
showing; omit it to aggregate the whole persisted graph.

```mermaid
flowchart TD
    req(["POST /graph/tags {track_ids?, top_n}"]) --> scope{"track_ids given?"}
    scope -->|yes| sel1["SELECT tags WHERE track_id = ANY(...)<br/>AND tags IS NOT NULL"]
    scope -->|no| sel2["SELECT tags for every song that is a<br/>graph_node OR an edge endpoint<br/>(nodes ∪ source ∪ target)"]
    sel1 --> agg
    sel2 --> agg
    agg["dominant_tags():<br/>sum {tag: count} weight per tag across songs,<br/>sort desc, take top_n, compute share"]
    agg --> out(["{tag, weight, count, share}[]<br/>(empty set / no stored tags → [])"])
```

---

## 6. Recommendations (`GET /recommendations/{track_id}`)

ANN + reject-steering + per-artist cap + MMR diversity, with two fallbacks to
honor the requested `k`.

```mermaid
flowchart TD
    req(["GET /recommendations/{id}?k&lambda&exclude"]) --> emb["load seed embedding"]
    emb --> none{"has embedding?"}
    none -->|no| cold["cold seed →<br/>embed_and_store_track (diagram 3)"]
    none -->|yes| steer
    cold --> zero{"all-zero vector?"}
    zero -->|yes| empty["return [] (no usable tags)"]
    zero -->|no| steer["steering.apply_steering<br/>base − α·Σ(rejected)<br/>then normalize"]

    steer --> pool["ann_search<br/>fetch k × 3 candidates<br/>no listener cap, exclude set"]
    pool --> cap["per-artist cap (max 2)<br/>→ capped_pool + overflow"]
    cap --> mmr["mmr_rerank<br/>λ·relevance − (1−λ)·redundancy<br/>(λ = 0.7)"]

    mmr --> short1{"len < k?"}
    short1 -->|yes| backfill["backfill from overflow<br/>(capped-out, most similar first)"]
    short1 -->|no| out
    backfill --> short2{"still < k?"}
    short2 -->|yes| topup["topup_from_lastfm<br/>seed getSimilar(30) → embed+store<br/>(empty? → similar-artist top tracks)<br/>score vs steered embedding"]
    short2 -->|no| out
    topup --> out(["Recommendation[]"])
```

### Reject steering (`services/steering.py`)

```mermaid
flowchart LR
    seed(["seed embedding"]) --> calc
    rej["rejected neighbors of this seed<br/>(feedback JOIN graph_edges)"] --> calc["query = seed − α·Σ rejected<br/>α = STEERING_ALPHA = 0.3"]
    calc --> nrm["normalize"] --> q(["steered query vector"])
```

---

## 7. Feedback loop (`POST /feedback`)

```mermaid
flowchart TD
    req(["POST /feedback {track_id, action}"]) --> valid{"action valid?<br/>exists in songs?"}
    valid -->|no| err["400 / 404"]
    valid -->|yes| log[("INSERT feedback")]

    log --> branch{"action?"}

    branch -->|reject| rdone["done — stored as negative.<br/>Future recs from the parent seed<br/>steer away (diagram 6)"]

    branch -->|accept| promote["UPSERT graph_nodes<br/>is_seed = true"]
    promote --> reparent["copy parent edge → accepted node<br/>(keeps it linked in graph)"]
    reparent --> rerun["ann_search from accepted node<br/>(with its own steering)<br/>INSERT new graph_edges"]
    rerun --> adone(["success"])
```

---

## 8. Playlist generation (`POST /playlists/*`)

Two strategies over the seed's graph neighborhood. `niche` mode walks listener
thresholds (100 → 1k → 10k → 100k → ∞) to favor the most underground tracks.

```mermaid
flowchart TD
    subgraph linear["/playlists/linear"]
        L1["load seed embedding"] --> L2["get_neighborhood (direct edges)"]
        L2 --> L3["embed_missing<br/>(embed_and_store_track per null, diagram 3)"]
        L3 --> L4["find_neighbors → ann_search<br/>niche → walk thresholds through ∞<br/>sort by listeners asc"]
        L4 --> L5(["flat track list"])
    end

    subgraph tree["/playlists/tree (BFS)"]
        T1["queue = [(seed, emb, depth 0)]"] --> T2{"queue &<br/>len < n?"}
        T2 -->|pop| T3["expand allowed set<br/>with this node's edges"]
        T3 --> T4["find_neighbors (ann_search) → take 2"]
        T4 --> T5["append to playlist,<br/>enqueue neighbors (depth+1)"]
        T5 --> T2
        T2 -->|done| T6(["branching track list"])
    end
```

---

## 9. End-to-end discovery journey

```mermaid
sequenceDiagram
    actor U as User
    participant FE as Frontend
    participant API as FastAPI
    participant DB as Postgres/pgvector
    participant LF as Last.fm

    U->>FE: type query
    FE->>API: GET /songs/search?q
    API->>DB: local ILIKE
    API->>LF: track.search
    API-->>FE: results (+covers)

    U->>FE: drop song on graph
    FE->>API: POST /graph/seed
    API->>LF: getInfo + tags + getSimilar
    API->>DB: store embedding, nodes, edges
    API-->>FE: seed node

    FE->>API: GET /recommendations/{id}
    API->>DB: ANN (steered) + MMR
    API-->>FE: k neighbors

    U->>FE: accept / reject node
    FE->>API: POST /feedback
    alt accept
        API->>DB: promote to seed + new edges
    else reject
        API->>DB: store negative (steers future recs)
    end

    Note over FE,API: as nodes appear, FE prefetches<br/>GET /songs/{id}/spotify (link)<br/>and may poll GET /songs/{id}/status

    U->>FE: build playlist
    FE->>API: POST /playlists/tree
    API->>DB: BFS over neighborhood
    API-->>FE: ordered playlist
```

---

## 10. Spotify "listen on" link (`GET /songs/{track_id}/spotify`)

The only place Spotify is touched. Resolves a public `open.spotify.com` URL via
the client-credentials search (`services/spotify.py`) and **persists it on the
`songs` row** so later calls are free. A non-definitive result (creds unset /
network error → `checked=false`) is *not* stored, so it's retried next time.

```mermaid
flowchart TD
    req(["GET /songs/{id}/spotify"]) --> row{"song in DB?"}
    row -->|no| e404["404"]
    row -->|yes| cached{"spotify_checked_at set?"}
    cached -->|yes| serve["return stored<br/>{url, checked: true}"]
    cached -->|no| resolve["spotify.find_track_url<br/>(cached client-credentials token)"]
    resolve -->|SpotifyUnavailable| nodef["return {url: null, checked: false}<br/>NOT persisted → retried later"]
    resolve -->|url or definitive null| persist[("UPDATE songs<br/>spotify_url, spotify_checked_at = now()")]
    persist --> serve2["return {url, checked: true}"]
```

> `GET /songs/{id}/status` is a sibling lightweight check — `{ exists, cached }`
> where `cached` means an embedding is already stored — used by the frontend to
> warn about slow "cold" seeds.
