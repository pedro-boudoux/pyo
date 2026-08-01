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
- **Cold-start fallback.** If a seed has no usable `track.getSimilar` at all, pyo falls back to the similar *artists'* top tracks. So even the most pretencious song picks get recommendations (although your mileage may vary here). 

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

The **Graph Info** panel sums the tag weights across every node on screen, tells you which genres (from Last.fm) dominate your graph and what the average listener count is across all songs in the graph.

*Note:* dominant-tag genre names are limited for non-English songs. Tracks in the same foreign language often get the same generic tag (e.g. "Brazilian") even when they are completely different genres. This is quite a tricky problem to fix since the limitation here is Last.fm's genre tagging system 😔.

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

There is no Spotify in the core loop. Spotify deprecated its audio-features and recommendations APIs in late 2024, so song search, tags, and embeddings all come from Last.fm. The only Spotify integration is an optional "listen on Spotify" link per track, resolved through the client-credentials flow. Although if that ever changes the "Export Playlist to Spotify" button is already ready to go and waiting 😊.

---

## Run it locally

### For humans: let an agent set it up

You can just ask an agent to handle setup for you. Do these two things first, though:

1. Get a free Last.fm API key at [last.fm/api](https://www.last.fm/api). pyo reads its tags and listener counts from Last.fm, so there is no way around this.
2. Install Docker Desktop from [docker.com](https://www.docker.com). pyo stores its vectors in Postgres, and Docker is how you run that database on your own machine.

Then open your coding agent in this project and paste this prompt:

```
Set up the local development environment for this repo. Follow the "For agents" section in the README. Install the backend dependencies, start Postgres with pgvector, create a .env file with my Last.fm API key, and start the API and the frontend. I will give you the key. Do not commit anything.
```

Give the agent the key when it asks for it. When it finishes, open http://localhost:5173 and search for a song you like. If the graph comes up, everything works.

### For agents: the setup runbook

Follow these steps in order. This is a runbook, not a menu: do not skip steps.

Prerequisites:

- Python 3.12
- Docker with the `ankane/pgvector` image
- Node.js and npm

**1. Install the backend dependencies**

Create a virtual environment and install the requirements:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Configure the environment**

Copy the example file:

```bash
cp .env.example .env
```

Set these values in `.env`:

```
LASTFM_API_KEY=your_key_here
DATABASE_URL=postgresql://user:password@localhost:5432/music_db
```

Ask the user for their Last.fm API key. Put that value in `LASTFM_API_KEY`. Do not invent a key. Do not commit `.env`.

**3. Start the database**

Start Postgres with pgvector:

```bash
docker run -e POSTGRES_PASSWORD=password -p 5432:5432 ankane/pgvector
```

`make db` does the same with docker compose. The schema is created automatically when the API starts. To create it by hand, run:

```bash
psql $DATABASE_URL -f migrations/init.sql
```

**4. Start the API**

```bash
uvicorn app.main:app --reload
```

`make api` does the same. The API listens on http://localhost:8000. Confirm it with:

```bash
curl -fsS http://localhost:8000/health
```

The response must be `{"status":"ok"}`.

**5. Start the frontend**

```bash
cd frontend
npm install
npm run dev
```

`make web` does the same. The frontend listens on http://localhost:5173.

**6. Run the tests**

```bash
make test
```

All tests must pass. The tests use mocked seams, so they need no live server or database.

---

## Phase 2 training

Every song already has one vector that describes how it sounds, built from its tags. Phase 2 adds a second vector that describes what people play together. Every time the app asks Last.fm for similar tracks, it saves that pair as a link. Given enough links, a separate program learns a vector for each song from that web of "these songs get played together."

Recommendations then use both signals: what the song sounds like, and what people actually listen to alongside it. That combined model is what production serves today.

The trainer is not part of the web app. It is a separate program with its own dependencies and its own virtual environment, so the app stays small and fast. It is also picky about when it runs. It refuses to train until the web is big enough to learn from: at least 20,000 songs, with each song linked to about 8 others on average. Every successful run is recorded in the `model_runs` table.

The first production training run has already happened. Retraining still needs a person to start it and watch it, because it updates the live vectors in batches and a bad run could hurt recommendations. It is not safe to put on a schedule yet. Making it fully automatic is currently in the works.

### The Commands

If you run the backend locally and want to retrain a fresh model:

```bash
make train-install
make crawl-colisten
make train-colisten COLISTEN_ARGS="--check-density --env-file path/to/runtime.env"
make train-colisten COLISTEN_ARGS="--env-file path/to/runtime.env --workers 8 --beta 2.0"
```

Once retraining can run unattended, production should run it as a separate scheduled job built from `Dockerfile.training`. The job reads `DATABASE_URL` and `COLISTEN_BETA` from its environment. Verify the database target before you enable the schedule. The training command is:

```bash
python -m jobs.train_colisten_embeddings --workers 4 --beta "$COLISTEN_BETA"
```

### How the model is checked

A model is only as good as its test, and this one is tested properly. It is judged against a set of "these songs appear in the same playlist" pairs built from public Deezer playlists, never from Last.fm's own similar-track lists. That distinction matters: the model learns from Last.fm's similar tracks, so testing it on the same data would make it look better than it really is.

```bash
make build-colisten-ground-truth
```

The test set lives at `eval/ground_truth_colisten.json`, and the full results are in `eval/baselines/`. The tests chose `beta = 2.0` as the best balance between the sound signal and the co-listening signal. 

## What comes next

The remaining work, dependencies, and acceptance criteria are tracked in
[GitHub issues #32–#40](https://github.com/pedro-boudoux/pyo/issues). Atomic model
publication and safe recurring training come first.

---

Built by [Pedro Boudoux](https://github.com/pedro-boudoux) with coding agents. The live frontend lives at [pedro-boudoux.github.io/pyo](https://pedro-boudoux.github.io/pyo).
