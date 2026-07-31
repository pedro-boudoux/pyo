# pyo

**pyo** stands for *putting you on* good music.

A music discovery tool built around one idea: music apps should recommend you music *similar to what you like* (yeah, that simple) and that's what pyo does.

You give it a song, pyo will find songs just like that one based on its genre tags and listener activity. pyo lays them out as a graph, and lets you grow the map by accepting songs that hit and rejecting the ones that don't. 

<p align="center">
  <img src="images/home.png" alt="pyo landing page — a single search box that says 'Tell us what you like, and we'll find similar.'" width="100%">
</p>

Try it out at: [pedro-boudoux.github.io/pyo](https://pedro-boudoux.github.io/pyo)

---

## How it actually works

Every song gets turned into a vector, and that vector is built from a blended set of Last.fm tags:

| Layer | What it captures | Weight |
|---|---|---|
| Track tags | the specific song | `1.0` |
| Artist tags | the artist's broader sound | `0.3` |
| Similar-artist tags | the surrounding scene | `0.1 × match` |

Each of those tags is encoded into a shared semantic space with a small sentence-transformer model (`all-MiniLM-L6-v2`), and a song's vector is the count-weighted average of its tag vectors. Thus related tags like "hip hop" and "rap" sit close together, so songs end up near each other when they *sound* alike.

Phase 2 adds a second, 128-dimensional signal learned from the accumulated
`track.getSimilar` graph. The candidate representation is
`normalize(concat(tag_vec, beta × colisten_vec))` (512 dimensions). Stage A remains
the production default until the graph density gate and independent beta evaluation
pass; `RECOMMENDATION_MODEL=hybrid` is the explicit rollout switch.

Track identity is also a bit stricter than it used to be. The backend still stores the exact song you searched for, but it also computes a looser canonical key so cosmetic variants like clean / explicit / remastered versions collapse together instead of clogging the top of the recs with the same recording three times.

So yes, finding "songs like this one" does start as a nearest-neighbour search. But the actual recommendation loop does more than that:

- **Steering**: reject a song and future suggestions lean *away* from it by querying with `seed − α·rejected`.
- **Over-fetch + MMR**: pyo pulls a bigger candidate pool than it needs, then re-ranks it to balance "close" against "not all the same."
- **Last.fm top-up**: if the local vector search still comes up short, pyo fetches the seed's `getSimilar` tracks, embeds them, scores them against the same query, and fills the gap that way.
- **Cold-start fallback**: if a seed basically has no usable `track.getSimilar` at all, pyo falls back to similar *artists'* top tracks so even the nichest seed gives you *something*.

Seeding the graph also does more than a one-shot recommendation query. When you drop in a seed, pyo merges local ANN results with Last.fm similar tracks, then recursively expands from the top few candidates so the local neighborhood gets thick enough to branch into playlists without immediately drifting off somewhere random.

The result is a graph that branches the way taste actually branches, not just a flat "more like this" list.

<p align="center">
  <img src="images/tiffany-day.png" alt="A pyo graph grown from a pop seed — album covers connected by edges, with a Graph Info panel showing dominant tags like Electropop, Electronic, Dance-Pop and Indie." width="100%">
</p>

And it works just as well when you point it somewhere completely different, even in a completely different language. Here's pyo helping me discover some Brazilian rock music:

<p align="center">
  <img src="images/charlie-brown-jr.png" alt="A denser pyo graph grown from a Brazilian rock seed, with the Graph Info panel showing Rock, Pop Rock, MPB and Folk as dominant tags." width="100%">
</p>

The **Graph Info** panel sums the tag weights across every node on screen and tells you which genres are the most dominant in your graph.

*NOTE*: Dominant tag genre descriptors are quite limited for non-English songs, this is because oftentimes songs in the same foreign language will be given the same tag (i.e. "Brazilian") despite being completely different genres. 

---

## Stack

| Layer | Tool |
|---|---|
| API | FastAPI (Python, async) |
| Search | Last.fm `track.search` + local Postgres cache |
| Tags + listeners + similar tracks | Last.fm |
| Album covers | Deezer → iTunes → Deezer artist photo |
| Embeddings | all-MiniLM-L6-v2 (fastembed, ONNX/CPU) over blended Last.fm tags |
| Vector DB + graph state | Postgres + pgvector |
| Frontend | React + Vite + ReactFlow |

---

## Current deployment

The live frontend is GitHub Pages:

```text
https://pedro-boudoux.github.io/pyo
```

The production backend is the Coolify app `pyo prod` on Pedro's homelab:

```text
https://pyo-backend.pedroboudoux.com
```

Quick check:

```bash
curl -fsS https://pyo-backend.pedroboudoux.com/health
```

From Pedro's MacBook, agents can inspect the homelab with:

```bash
ssh pedro-homelab
```

The Coolify app builds `pedro-boudoux/pyo` from `main` with Nixpacks and starts
the API through the `Procfile`. Runtime environment variables, including
`DATABASE_URL`, `LASTFM_API_KEY`, Spotify credentials, and blacklist settings,
live in Coolify and should be treated as secrets. Verify `DATABASE_URL` in
Coolify before any DB migration or restore; it may point at Neon until the
database is intentionally moved.

---

## Running it locally

```bash
pip install -r requirements.txt

# Postgres with pgvector, the easy way
docker run -e POSTGRES_PASSWORD=password -p 5432:5432 ankane/pgvector

# Schema auto-creates on startup, but you can run it by hand too:
psql $DATABASE_URL -f migrations/init.sql

uvicorn app.main:app --reload
```

Offline Phase 2 training has a separate dependency so it does not inflate the API
deployment:

```bash
make train-install
make train-colisten COLISTEN_ARGS="--check-density --env-file path/to/runtime.env"
make train-colisten COLISTEN_ARGS="--env-file path/to/runtime.env --workers 8 --beta 0.5"
```

For production, run this from a separate Coolify scheduled job/worker built from
`Dockerfile.training`. Let the job inherit `DATABASE_URL` and `COLISTEN_BETA`
from Coolify, verify the database target before enabling its schedule, and invoke:

```bash
python -m jobs.train_colisten_embeddings --workers 4 --beta "$COLISTEN_BETA"
```

The trainer remains density-gated and records every successful run in
`model_runs`. It is not part of the lean API process and should not be scheduled
until the first manual production training and independent beta evaluation pass.
The MacBook's ignored `.env.prod` now targets the live private Coolify Postgres
through a localhost SSH tunnel. The tunnel must be listening before local crawl,
training, or eval commands use that file; the database itself remains private.
The previous Neon crawl was merged idempotently into production before the file
was switched, so its collected edges were preserved.

The independent Stage B fixture is built from cross-artist adjacency in public
Deezer playlists, never Last.fm `getSimilar`:

```bash
make build-colisten-ground-truth
```

The committed fixture is `eval/ground_truth_colisten.json`; its Stage A reference
result is `eval/baselines/stage_a_deezer_fixture.json`. `eval/sweep_beta.py`
validates the fixture provenance and rejects circular Last.fm-derived input.

You'll need a free Last.fm API key from [last.fm/api](https://www.last.fm/api),
dropped into a `.env`:

```
LASTFM_API_KEY=your_key_here
DATABASE_URL=postgresql://user:password@localhost:5432/music_db
```

Then start the frontend:

```bash
cd frontend
npm install
npm run dev
```

---

Built by [Pedro Boudoux](https://github.com/pedro-boudoux). The live frontend
lives at [pedro-boudoux.github.io/pyo](https://pedro-boudoux.github.io/pyo).
