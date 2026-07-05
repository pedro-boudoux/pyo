from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from app.db import init_db
from app.ratelimit import limiter
from app.routers import songs, graph, recommendations, feedback, playlists

app = FastAPI(title="Underground Music Discovery")

# Per-IP rate limiting (issue #20). The limiter enforces RATE_LIMIT_DEFAULT on
# every route; heavy routes add a stricter @limiter.limit on top. A breach
# returns 429 via slowapi's handler. No-op when RATE_LIMIT_ENABLED is falsey.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "Retry-After",
    ],
)

app.include_router(songs.router)
app.include_router(graph.router)
app.include_router(recommendations.router)
app.include_router(feedback.router)
app.include_router(playlists.router)


@app.on_event("startup")
def startup():
    init_db()


# Exempt from the global limit: uptime/health checkers poll this frequently and
# it does no real work.
@limiter.exempt
@app.get("/health")
def health():
    return {"status": "ok"}
