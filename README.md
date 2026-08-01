# pyo

**pyo** stands for *putting you on* good music.

It is a music discovery tool built around one idea: an app should recommend music that sounds like what you already like. Yeah, that simple. You drop a song onto a graph, pyo finds tracks that fit its sound, and the graph grows as you accept the hits and reject the misses.

The recommendations are not popularity picks. pyo reads genre tags and listener counts from Last.fm, so every pick is judged on its sound rather than its popularity. Listener counts stay visible on the graph, and niche playlist mode sorts by them so the most obscure tracks surface first.

<p align="center">
  <img src="images/home.png" alt="pyo landing page, a single search box that says: Tell us what you like, and we'll find similar." width="100%">
</p>

Try it live at [pedro-boudoux.github.io/pyo](https://pedro-boudoux.github.io/pyo).

---

## How it works

### From song to vector

Every song becomes a vector. pyo builds that vector from three layers of Last.fm tags:

| Layer | What it captures | Weight |
|---|---|---|
| Track tags | The specific song | 1.0 |
| Artist tags | The artist's broader sound | 0.3 |
| Similar-artist tags | The surrounding scene | 0.1 × match |

Each tag is encoded into a shared semantic space with a small sentence-transformer model, `all-MiniLM-L6-v2`. The song's vector is the count-weighted average of its tag vectors, then L2-normalized (scaled so its length is exactly 1). This keeps comparisons fair between songs with many tags and songs with few. Tags that many listeners applied pull harder. Because the space is semantic, "hip hop" and "rap" land close together, and so do the songs that share them.

Two consequences are worth knowing:

- A song with no usable tags gets an all-zero vector. pyo treats that as "nothing to recommend" and does not guess.
- The stored embedding is dense, so it cannot be read back into discrete tags. pyo keeps the raw `{tag: count}` dict too, and that is what the Graph Info panel reads.

### The recommendation loop

Finding "songs like this one" starts as a nearest-neighbor search, but the loop does more:

- **Steering.** Reject a song and the next query leans away from it. The query becomes `seed − α·rejected`, with α = 0.3.
- **Over-fetch + MMR.** pyo pulls three times more candidates than it needs, then re-ranks them to balance relevance against diversity (λ = 0.7). It allows at most two tracks per artist.
- **Last.fm top-up.** If the local search still comes up short, pyo fetches the seed's `getSimilar` tracks, embeds them, scores them, and fills the gap.
- **Cold-start fallback.** If a seed has no usable `track.getSimilar` at all, pyo falls back to the similar *artists'* top tracks. Even the nichest seed gets a graph.

### Seeding the graph

Dropping a seed is not one query. pyo runs the local ANN search, then:

1. Pulls the seed's `track.getSimilar` (up to 25 tracks) and embeds them.
2. Expands recursively. It takes the top three candidates, pulls *their* similar tracks (up to 10 each), and embeds those too. This thickens the neighborhood so playlist branches do not drift into unrelated music.
3. Falls back to similar artists' top tracks when the seed is too obscure for step 1.
4. Keeps the top 10 by similarity and writes the edges.

The result is a graph that branches the way taste actually branches, not a flat "more like this" list.

<p align="center">
  <img src="images/tiffany-day.png" alt="A pyo graph grown from a pop seed. Album covers are connected by edges, and the Graph Info panel shows dominant tags like Electropop, Electronic, Dance-Pop, and Indie." width="100%">
</p>

It works just as well pointed somewhere completely different, even in a completely different language. Here is pyo helping me find Brazilian rock:

<p align="center">
  <img src="images/charlie-brown-jr.png" alt="A denser pyo graph grown from a Brazilian rock seed. The Graph Info panel shows Rock, Pop Rock, MPB, and Folk as dominant tags." width="100%">
</p>

The **Graph Info** panel sums the tag weights across every node on screen and tells you which genres dominate your graph.

*Note:* dominant-tag genre names are limited for non-English songs. Tracks in the same foreign language often get the same generic tag (e.g. "Brazilian") even when they are completely different genres.

### Track identity

The backend stores two keys per song:

- **`track_id`** is the SHA1 of `artist|||track`, lowercased, first 20 characters. It is exact: each search result gets its own id.
- **`canonical_key`** is the same hash over a normalized title. Cosmetic variants like `(Clean)`, `(Explicit)`, and `- Remastered 2011` fold into one identity, so the same recording cannot show up three times at the top of your recommendations.

`track_id` is the cache and foreign key. `canonical_key` is the "is this the same song?" key. It dedupes search results, the recommendation pool, and the seed bootstrap.

### Phase 2: the co-listening signal

Every `track.getSimilar` call also records a weighted edge in `colisten_edges`. This costs nothing extra in API calls. The graph is append-only.

A separate trainer learns a 128-dimension vector per track from that graph, using a weighted random walk. The hybrid representation is:

```
normalize(concat(tag_vector, beta × colisten_vector))
```

That is 512 dimensions. The beta sweep, run on a fixture built from public Deezer playlists rather than from Last.fm, selected `beta = 2.0`. Production serves this hybrid model. Stage A remains available as an instant config-only rollback:

```
RECOMMENDATION_MODEL=stage_a
```

The original `songs.embedding` column stays intact, so rolling back is instant.

---

## Stack

| Layer | Tool |
|---|---|
| API | FastAPI (Python, async) |
| Search | Last.fm `track.search` + local Postgres cache |
| Tags, listeners, similar tracks | Last.fm |
| Album covers | Deezer, then iTunes, then a Deezer artist photo |
| Embeddings | all-MiniLM-L6-v2 (fastembed, ONNX, CPU) over blended Last.fm tags |
| Vector DB + graph state | Postgres + pgvector |
| Frontend | React + Vite + ReactFlow |

There is no Spotify in the core loop. Spotify deprecated its audio-features and recommendations APIs in late 2024, so song search, tags, and embeddings all come from Last.fm. The only Spotify integration is an optional "listen on Spotify" link per track, resolved through the client-credentials flow.

---

## Current deployment

The frontend runs on GitHub Pages:

```text
https://pedro-boudoux.github.io/pyo
```

The production backend is the Coolify app `pyo prod` on Pedro's homelab:

```text
https://pyo-backend.pedroboudoux.com
```

Check it:

```bash
curl -fsS https://pyo-backend.pedroboudoux.com/health
```

From Pedro's MacBook you can inspect the homelab with:

```bash
ssh pedro-homelab
```

The Coolify app builds `pedro-boudoux/pyo` from `main` with Nixpacks and starts the API through the `Procfile`. Runtime secrets live in Coolify, not in the repo: `DATABASE_URL`, `LASTFM_API_KEY`, Spotify credentials, and `BLACKLIST_ARTISTS`. Treat them as secrets. Production now uses the private Coolify Postgres database; still verify the current `DATABASE_URL` in Coolify before any migration, restore, crawl, or training run.

---

## Run it locally

You need Python 3.12, Docker, and Node.

### Backend

Create a virtual environment and install the dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your key:

```
LASTFM_API_KEY=your_key_here
DATABASE_URL=postgresql://user:password@localhost:5432/music_db
```

Get a key at [last.fm/api](https://www.last.fm/api).

Start Postgres with pgvector:

```bash
docker run -e POSTGRES_PASSWORD=password -p 5432:5432 ankane/pgvector
```

`make db` does the same with docker compose. The schema is created automatically at startup. To run it by hand:

```bash
psql $DATABASE_URL -f migrations/init.sql
```

Run the API:

```bash
uvicorn app.main:app --reload
```

The Makefile also has `make api` and `make web` for the backend and frontend.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Tests

```bash
make test
```

The test suite uses mocked seams and needs no live server or database.

---

## Phase 2 training

Training has a separate dependency file, `requirements-jobs.txt`, so it does not inflate the API image. The trainer runs in its own virtual environment:

```bash
make train-install
make crawl-colisten
make train-colisten COLISTEN_ARGS="--check-density --env-file path/to/runtime.env"
make train-colisten COLISTEN_ARGS="--env-file path/to/runtime.env --workers 8 --beta 2.0"
```

The trainer is density-gated. It refuses to train until the graph reaches at least `COLISTEN_MIN_NODES` (20000) nodes and `COLISTEN_MIN_AVG_DEGREE` (8.0) average degree. It records every successful run in `model_runs`. The first production run passed, but the current writer updates active vectors in batches. Treat the command above as a supervised maintenance operation, and do not schedule it until candidate staging and atomic publication are implemented; that work is specified in [`PROJECT_PLAN.md`](PROJECT_PLAN.md).

Once atomic publication is implemented, production retraining should run as a separate Coolify scheduled job built from `Dockerfile.training`. Let the job inherit `DATABASE_URL` and `COLISTEN_BETA` from Coolify. Verify the database target before enabling the schedule. Its training command is:

```bash
python -m jobs.train_colisten_embeddings --workers 4 --beta "$COLISTEN_BETA"
```

The MacBook's `.env.prod` (git-ignored) targets the live Coolify Postgres through a localhost SSH tunnel. The tunnel must be listening before crawl, training, or eval commands use that file. The database itself stays private. The earlier Neon crawl was merged idempotently into production before the switch, so those edges were preserved.

### Independent Stage B evaluation

The Stage B fixture comes from cross-artist adjacency in public Deezer playlists, never from Last.fm `getSimilar`. Grading a getSimilar-trained model against getSimilar would be circular.

```bash
make build-colisten-ground-truth
```

The committed fixture is `eval/ground_truth_colisten.json`. Its Stage A reference result is `eval/baselines/stage_a_deezer_fixture.json`. `eval/sweep_beta.py` validates the fixture's provenance and rejects any circular Last.fm-derived input. The production sweep selected `COLISTEN_BETA=2.0`; the full metrics are in `eval/baselines/stage_b_beta_sweep.json`. The hybrid model is live, and [`PHASE2_TODAY_PLAN.md`](PHASE2_TODAY_PLAN.md) records the completed rollout.

## What comes next

The remaining work is ordered in [`PROJECT_PLAN.md`](PROJECT_PLAN.md): atomic recurring training, the Coolify schedule, country/language tag attenuation, cold-start ablations, steering evaluation, human A/B evaluation, observability, and eventual removal of only the obsolete 300-dimensional sparse model.

---

Built by [Pedro Boudoux](https://github.com/pedro-boudoux). The live frontend lives at [pedro-boudoux.github.io/pyo](https://pedro-boudoux.github.io/pyo).
