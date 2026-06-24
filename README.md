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

Every song gets turned into a vector, which is built with the following info from Last.fm:

| Layer | What it captures | Weight |
|---|---|---|
| Track tags | the specific song | `1.0` |
| Artist tags | the artist's broader sound | `0.3` |
| Similar-artist tags | the surrounding scene | `0.1 × match` |

Each of those tags is encoded into a shared semantic space with a small sentence-transformer model (`all-MiniLM-L6-v2`), and a song's vector is the count-weighted average of its tag vectors. The upshot: related tags like "hip hop" and "rap" sit close together, so songs end up near each other when they *sound* alike — not just when they share the exact same tag spelling.

This means that finding "songs like this one" is just a nearest neighbour search. Which is cool and all but pyo offers more than just that, we offer:

- **Steering**: reject a song and future suggestions actively lean *away* from it.
- **Diversity (MMR)**: re-ranks results to balance "close" against "not all the same."
- **Per-artist caps**: no single artist is allowed to flood the recommendation pool (however depending on how you play with pyo's settings, this might still happen-- completely up to you though 🙏🏼)
- **Cold-start mining**: obscure seeds with no direct matches fall back to digging
  through similar *artists'* top tracks, so even the nichest seed gives you *something*.

The result is a graph that branches the way taste actually branches.

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
| Search + tags + listeners | Last.fm |
| Album covers | Deezer → iTunes → Deezer artist photo |
| Embeddings | all-MiniLM-L6-v2 (fastembed, ONNX/CPU) over blended Last.fm tags |
| Vector DB + graph state | Postgres + pgvector |
| Frontend | React + Vite + ReactFlow |

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
