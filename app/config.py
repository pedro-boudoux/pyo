import os
from dotenv import load_dotenv

load_dotenv()

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/music_db")

# Spotify client-credentials — used only to resolve a public "listen on Spotify"
# link for a track (no user OAuth, no audio features / recommendations). Optional:
# if unset, the link endpoint simply reports no match.
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


# Mandatory artist blacklist, managed from the deployment. Comma-separated artist
# names in BLACKLIST_ARTISTS are never recommended (they can still be searched and
# used as a seed directly). Matching is case-insensitive and credit-aware (blocking
# "Drake" also blocks "Drake, 21 Savage" and "X feat. Drake"). Users can layer
# their own per-client blocks on top at request time; this set is always enforced.
#   BLACKLIST_ARTISTS=Drake, Some Other Artist
BLACKLIST_ARTISTS = _parse_csv(os.getenv("BLACKLIST_ARTISTS", ""))

STEERING_ALPHA = 0.3
DEFAULT_K = 10

# Rate limiting (slowapi). Per-client-IP limits guard the public deployment from
# abuse / runaway cost before release (issue #20). Three tiers:
#   RATE_LIMIT_DEFAULT — applied to every route (cheap reads like search/status).
#   RATE_LIMIT_HEAVY   — stricter cap on the endpoints that fan out to Last.fm and
#                        run the ONNX embedder (seed, recommendations, playlists,
#                        feedback, search, features).
#   RATE_LIMIT_MAINTENANCE — stricter still for authenticated bulk maintenance.
# All tunable via env; the limiter is disabled outright when RATE_LIMIT_ENABLED is
# falsey (the test suite turns it off so fixtures aren't throttled).
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").strip().lower() not in ("false", "0", "no", "off")
RATE_LIMIT_DEFAULT = os.getenv("RATE_LIMIT_DEFAULT", "100/minute")
RATE_LIMIT_HEAVY = os.getenv("RATE_LIMIT_HEAVY", "20/minute")
RATE_LIMIT_MAINTENANCE = os.getenv("RATE_LIMIT_MAINTENANCE", "2/minute")

# Public traffic must not be able to invoke bulk database/API maintenance. When
# unset, maintenance routes are disabled; when set, callers must send the value
# in X-Maintenance-Key.
MAINTENANCE_API_KEY = os.getenv("MAINTENANCE_API_KEY")

# Embedding dimensions (algorithm 2.0).
#   EMBEDDING_DIM        — the live stored `songs.embedding` vector. Phase 1 = the
#                          dense semantic tag vector (384). Phase 2 widens this to
#                          384 + 128 = 512 once the co-listening half is blended in.
#   TAG_EMBEDDING_DIM    — the semantic tag half (all-MiniLM-L6-v2 output). Stays
#                          384 across phases; it's the Phase-1 vector and the first
#                          half of the Phase-2 hybrid.
EMBEDDING_DIM = 384
TAG_EMBEDDING_DIM = 384
COLISTEN_EMBEDDING_DIM = 128
HYBRID_EMBEDDING_DIM = TAG_EMBEDDING_DIM + COLISTEN_EMBEDDING_DIM

# Phase 2 model selection. Stage A remains the safe code/local default and instant
# rollback; Coolify production overrides this to hybrid. The independent
# Deezer-fixture sweep selected beta=2.0.
# Setting RECOMMENDATION_MODEL=hybrid switches ANN, MMR, steering and playlists to
# songs.hybrid_embedding; the original songs.embedding vector(384) stays intact as
# an instant rollback path.
RECOMMENDATION_MODEL = os.getenv("RECOMMENDATION_MODEL", "stage_a").strip().lower()
if RECOMMENDATION_MODEL not in ("stage_a", "hybrid"):
    RECOMMENDATION_MODEL = "stage_a"
COLISTEN_BETA = float(os.getenv("COLISTEN_BETA", "2"))
COLISTEN_MIN_NODES = int(os.getenv("COLISTEN_MIN_NODES", "20000"))
COLISTEN_MIN_AVG_DEGREE = float(os.getenv("COLISTEN_MIN_AVG_DEGREE", "8"))

# Semantic tag encoder. fastembed (ONNX) loads all-MiniLM-L6-v2 on CPU (~80MB),
# no torch — keeps the Coolify API image light. Output is 384-dim.
TAG_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MMR_LAMBDA = 0.7         # relevance vs diversity tradeoff (1.0 = pure relevance, 0.0 = pure diversity)
MMR_POOL_MULTIPLIER = 3  # fetch this many × k candidates before re-ranking
MMR_MAX_PER_ARTIST = 2   # max tracks per artist in the MMR candidate pool
