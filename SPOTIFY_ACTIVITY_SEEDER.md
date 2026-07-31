# Spotify Activity Seeder Service

This document is for an agent building an always-running companion service that
observes one user's Spotify listening activity and uses it to keep Pyo's discovery
graph fresh. The service should treat Spotify as a **taste signal and trigger**,
not as Pyo's recommendation engine.

## Purpose

The service watches the user's real listening activity and periodically seeds Pyo
with tracks the user actually played. Seeding a track causes Pyo to:

- resolve/store the track in `songs`,
- build its Last.fm tag embedding if missing,
- call Last.fm `track.getSimilar`,
- record graph nodes/edges,
- record co-listening edges in `colisten_edges`,
- thicken local recommendation neighborhoods over time.

The expected effect is that Pyo's graph and future Phase 2 co-listening training
data drift toward the user's actual taste, without manually dropping every seed.

## Important Boundaries

- Pyo does **not** use Spotify IDs as track identity.
- Pyo's stable song key is `track_id = sha1(lower_artist || "|||" || lower_track)[:20]`.
- Spotify activity should be mapped to Pyo by artist/title, then resolved through
  Pyo's own `/songs/search` and `/graph/seed` APIs.
- Do not write directly to Pyo's database from the activity service for the MVP.
- Do not call Spotify audio features or Spotify recommendations; those are not part
  of the backend architecture.
- Do not use Spotify client-credentials for user activity. User playback data needs
  user OAuth.
- Do not use `--with-topup` or eval tooling from this service. It is not an eval job.

## Spotify API Requirements

Use Spotify Web API OAuth for user data.

Recommended auth flow:

- Authorization Code with PKCE for a local/desktop/user-owned service.
- Authorization Code with a client secret is also acceptable if this runs as a
  private server process and the secret is stored safely.
- Store refresh tokens securely and refresh access tokens before expiry.

Required scopes:

- `user-read-recently-played`: read recently played tracks.
- `user-read-currently-playing`: optional, useful for near-real-time detection.

Useful Spotify endpoints:

- `GET /v1/me/player/recently-played`
  - Primary source of completed plays.
  - Returns track objects and `played_at`.
  - Currently for tracks, not podcast episodes.
- `GET /v1/me/player/currently-playing`
  - Optional real-time source.
  - Use only for candidates; do not seed until the play is likely intentional.

Policy caution:

- The service should use Spotify data only as consented user activity metadata.
- It should not train a model on Spotify audio/content or copy Spotify catalog data
  into Pyo beyond the minimal artist/title/album metadata needed to resolve a play.
- Check Spotify's current developer policy before publishing beyond personal use.

Official docs checked while writing this:

- Spotify currently playing reference:
  https://developer.spotify.com/documentation/web-api/reference/get-the-users-currently-playing-track
- Spotify recently played reference:
  https://developer.spotify.com/documentation/web-api/reference/get-recently-played
- Spotify Authorization Code with PKCE:
  https://developer.spotify.com/documentation/web-api/tutorials/code-pkce-flow
- Spotify refreshing tokens:
  https://developer.spotify.com/documentation/web-api/tutorials/refreshing-tokens

## Pyo Backend Calls

Assume `PYO_API_BASE` points at the deployed API, for example:

```text
https://pyo-backend.pedroboudoux.com
```

Operational note for agents: this API is the Coolify app `pyo prod` on Pedro's
homelab. From Pedro's MacBook, use `ssh pedro-homelab` to inspect the host or
Coolify state. Do not paste Coolify environment values or full database URLs into
logs, docs, or chat; verify `DATABASE_URL` in Coolify before assuming where the
backend database lives.

### 1. Resolve Spotify play to a Pyo track candidate

Use the Spotify track's primary artist and name:

```text
query = "{primary_artist} {track_name}"
GET /songs/search?q={query}
```

Response shape:

```json
[
  {
    "track_id": "abc123...",
    "name": "Track Name",
    "artist": "Artist Name",
    "image": "https://..."
  }
]
```

What this does in Pyo:

- searches local `songs`,
- searches Last.fm `track.search`,
- merges and canonical-dedupes results,
- resolves covers where needed,
- upserts returned tracks into `songs`,
- does **not** necessarily embed the track yet.

Candidate selection rules:

- Prefer exact normalized title match.
- Prefer exact normalized primary artist match.
- If Spotify has multiple artists, accept a Pyo candidate whose artist contains the
  primary artist or whose split credits contain it.
- Strip obvious Spotify title decorations for matching only:
  - remaster labels,
  - explicit/clean labels,
  - radio edit/album version,
  - trailing `feat.` variants.
- Preserve meaningful variants:
  - live,
  - remix,
  - acoustic,
  - demo,
  - instrumental.
- If the top candidate is ambiguous, skip the play. Bad seeds are worse than missed
  seeds because seeding writes graph state.

### 2. Check whether the track is already warm

Optional but useful for logging and backpressure:

```text
GET /songs/{track_id}/status
```

Expected response:

```json
{
  "exists": true,
  "cached": true
}
```

`cached = true` means the embedding already exists. `cached = false` means seeding
will be slower because Pyo needs Last.fm tag calls and MiniLM tag embedding.

### 3. Seed the track

```text
POST /graph/seed
Content-Type: application/json

{
  "track_id": "abc123..."
}
```

Response:

```json
{
  "track_id": "abc123...",
  "name": "Track Name",
  "artist": "Artist Name"
}
```

Important behavior:

- The track must already exist in `songs`; call `/songs/search` first.
- If the track embedding is missing, `/graph/seed` embeds it on demand.
- It marks the track as a seed node.
- It performs ANN search and Last.fm bootstrapping.
- It records `graph_edges`.
- It records `colisten_edges` from every `getSimilar` call.
- It recursively expands from top candidates.

This is the main write operation the activity service should perform.

### 4. Optional: fetch recommendations after seeding

Usually not needed for the activity service, but useful for diagnostics:

```text
GET /recommendations/{track_id}?k=10
```

This may top up from Last.fm if the local DB is sparse, so it can write newly
embedded songs and `colisten_edges`. Use sparingly in the daemon.

## Recommended Service Loop

Use recently played as the source of truth:

1. Every 5-10 minutes, call Spotify recently played with `after=<last_seen_ms>`.
2. Normalize and dedupe returned plays by Spotify track ID plus `played_at`.
3. Filter out plays that should not become Pyo seeds.
4. Resolve each accepted play through Pyo `/songs/search`.
5. Seed the best matching Pyo `track_id` through `/graph/seed`.
6. Store the outcome locally so failed/ambiguous tracks are not retried forever.

Use currently playing only as an optional enhancement:

1. Poll every 30-60 seconds.
2. Track `progress_ms`, `duration_ms`, and Spotify track ID.
3. Do not seed immediately on first sight.
4. Promote to a seed candidate only after one of:
   - at least 60 seconds played,
   - at least 50% of the track played,
   - the same track later appears in recently played.

## Reasonable Depth / Rate Controls

`POST /graph/seed` is expensive. It can call Last.fm many times:

- seed tags,
- artist tags,
- similar artists,
- seed `track.getSimilar`,
- candidate embeddings,
- recursive expansion.

Do not seed every short play. Suggested initial policy:

- Poll recently played every 10 minutes.
- Seed at most 3 tracks per poll.
- Seed at most 30 tracks per day.
- Seed only tracks played for real, not previews/skips.
- Skip tracks already seeded recently.
- Skip repeat listens of the same canonical song within 7 days.
- Prefer tracks with full plays or multiple listens.

Initial "reasonable depth" means:

- let Pyo's existing `/graph/seed` do its built-in depth:
  - seed `getSimilar` limit 25,
  - recursive expansion from top 3 candidates,
  - expansion `getSimilar` limit 10,
  - cold-start fallback to similar artists' top tracks.
- do not add additional external recursive crawling in this service.

If more growth is needed, use `jobs/crawl_colisten.py` separately. Do not make the
activity daemon a general crawler.

## Local State Required by the Service

The service needs its own small database or durable state file. Recommended tables:

### `spotify_tokens`

- `user_id`
- `access_token`
- `refresh_token`
- `expires_at`
- `scope`
- `updated_at`

### `spotify_play_cursor`

- `user_id`
- `last_seen_played_at`
- `last_seen_spotify_track_id`
- `updated_at`

### `spotify_seed_events`

- `id`
- `spotify_track_id`
- `played_at`
- `spotify_name`
- `spotify_artists_json`
- `spotify_album`
- `duration_ms`
- `pyo_track_id`
- `pyo_name`
- `pyo_artist`
- `status`
- `reason`
- `attempt_count`
- `last_error`
- `created_at`
- `updated_at`

Suggested statuses:

- `seen`
- `skipped_short_play`
- `skipped_duplicate`
- `skipped_ambiguous`
- `resolved`
- `seeded`
- `failed_search`
- `failed_seed`

## Matching Details

The hardest bug class is mapping a Spotify track to the wrong Pyo/Last.fm song.

Implement these helpers:

- `normalize_artist(s)`
- `normalize_title(s)`
- `strip_cosmetic_title_suffixes(s)`
- `is_meaningful_variant(s)`
- `score_candidate(spotify_track, pyo_search_result)`

Scoring suggestion:

- title exact normalized match: strong positive
- title normalized after cosmetic stripping: positive
- primary artist exact match: strong positive
- primary artist appears in Pyo split credits: positive
- Pyo title keeps a meaningful variant absent from Spotify title: negative
- Pyo artist unrelated to Spotify artists: hard reject

Minimum policy:

- If confidence is not high, skip.
- Log the skipped event.
- Do not seed guessed matches.

## Backend Rate Limits and Failure Handling

The public API uses SlowAPI rate limits. Heavy endpoints include:

- `/songs/search`
- `/graph/seed`
- `/recommendations/{track_id}`
- playlist endpoints
- feedback

The activity daemon should be polite:

- Use a single worker for Pyo seeding initially.
- Add jitter to polling.
- Respect HTTP `429`; back off and retry later.
- Treat `404` from `/graph/seed` as a service bug because `/songs/search` should
  have upserted the track first.
- Treat `502` from `/graph/seed` as transient upstream Last.fm trouble.
- Use exponential backoff for 5xx.
- Do not retry ambiguous search results automatically.

## Configuration

Suggested environment variables:

```text
PYO_API_BASE=https://...
SPOTIFY_CLIENT_ID=
SPOTIFY_CLIENT_SECRET=          # only for confidential Authorization Code flow
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback
SPOTIFY_SCOPES=user-read-recently-played user-read-currently-playing
SEED_POLL_SECONDS=600
CURRENTLY_PLAYING_POLL_SECONDS=60
MAX_SEEDS_PER_POLL=3
MAX_SEEDS_PER_DAY=30
RESEED_COOLDOWN_DAYS=7
MIN_PLAY_MS=60000
MIN_PLAY_FRACTION=0.5
```

## MVP Implementation Shape

Suggested files if built inside this repo:

```text
services/spotify_activity_seeder/
  README.md
  pyproject.toml
  spotify_auth.py
  spotify_client.py
  pyo_client.py
  matcher.py
  state.py
  daemon.py
```

Suggested main loop:

```text
load config
load/refresh Spotify token
load cursor
while running:
    plays = spotify.recently_played(after=cursor.last_seen_played_at)
    for play in chronological_order(plays):
        if should_skip_play(play):
            record skip
            continue
        result = pyo.search(best_query(play))
        candidate = matcher.pick(play, result)
        if not candidate:
            record ambiguous/failed search
            continue
        if recently_seeded(candidate.track_id):
            record duplicate
            continue
        pyo.seed(candidate.track_id)
        record seeded
    update cursor
    sleep with jitter
```

## Recommended Backend Enhancements

The MVP can work with current endpoints, but an internal ingestion service would be
cleaner with two small backend additions:

### `POST /songs/resolve`

Request:

```json
{
  "artist": "Artist",
  "name": "Track"
}
```

Expected behavior:

- compute `track_id`,
- upsert a minimal `songs` row if absent,
- resolve cover if possible,
- return the exact Pyo identity.

This would avoid ambiguous `/songs/search` matching for known Spotify artist/title
pairs.

### `POST /activity/seed`

Request:

```json
{
  "source": "spotify",
  "source_track_id": "spotify:track:...",
  "artist": "Artist",
  "name": "Track",
  "played_at": "2026-07-11T00:00:00Z"
}
```

Expected behavior:

- resolve the song,
- dedupe by source event,
- apply backend-side seed cooldown,
- call the same seed pipeline as `/graph/seed`,
- return whether it seeded, skipped, or was already processed.

This is not required for the first service, but it is the safer long-term API.

## Acceptance Criteria

The agent building this service is done when:

- OAuth authorization and token refresh work unattended.
- Recently played polling resumes correctly after restart.
- The service resolves Spotify plays to Pyo tracks with conservative matching.
- It calls `/songs/search` before `/graph/seed`.
- It seeds only within configured daily/poll limits.
- It records every seen play and every seed/skip outcome.
- It respects Pyo `429` and upstream failures with backoff.
- It can run for 24 hours without duplicate seeding loops.
- Pyo `colisten_edges` and graph nodes grow gradually from real listening activity.
