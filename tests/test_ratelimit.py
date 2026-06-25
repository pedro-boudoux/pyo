"""
Rate limiting (issue #20).

The suite disables the real limiter (conftest sets RATE_LIMIT_ENABLED=false), so
here we build a throwaway app wired exactly like app.main — limiter + middleware +
exception handler + a heavy-limited route — with an *enabled* limiter and a tiny
limit, and assert the 429 once the limit is exceeded. This validates the slowapi
integration (key func, middleware, per-route decorator) against the installed
versions without depending on the app's production limits.
"""
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.ratelimit import client_ip


def _make_app(default="100/minute", heavy="2/minute"):
    limiter = Limiter(key_func=client_ip, default_limits=[default], enabled=True)
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    @app.get("/cheap")
    def cheap():
        return {"ok": True}

    @app.get("/heavy")
    @limiter.limit(heavy)
    def heavy_route(request: Request):
        return {"ok": True}

    @limiter.exempt
    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


class TestClientIp:
    def test_prefers_first_forwarded_for(self):
        req = type("R", (), {"headers": {"x-forwarded-for": "1.2.3.4, 10.0.0.1"}, "client": None})()
        assert client_ip(req) == "1.2.3.4"

    def test_falls_back_to_peer_when_no_header(self):
        req = type("R", (), {"headers": {}, "client": type("C", (), {"host": "9.9.9.9"})()})()
        assert client_ip(req) == "9.9.9.9"


class TestEnforcement:
    def test_heavy_route_429s_after_limit(self):
        client = TestClient(_make_app(heavy="2/minute"))
        assert client.get("/heavy").status_code == 200
        assert client.get("/heavy").status_code == 200
        assert client.get("/heavy").status_code == 429

    def test_default_limit_applies_to_undecorated_route(self):
        client = TestClient(_make_app(default="2/minute"))
        assert client.get("/cheap").status_code == 200
        assert client.get("/cheap").status_code == 200
        assert client.get("/cheap").status_code == 429

    def test_health_is_exempt(self):
        client = TestClient(_make_app(default="1/minute"))
        for _ in range(5):
            assert client.get("/health").status_code == 200

    def test_disabled_limiter_never_429s(self):
        limiter = Limiter(key_func=client_ip, default_limits=["1/minute"], enabled=False)
        app = FastAPI()
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        app.add_middleware(SlowAPIMiddleware)

        @app.get("/x")
        @limiter.limit("1/minute")
        def x(request: Request):
            return {"ok": True}

        client = TestClient(app)
        for _ in range(5):
            assert client.get("/x").status_code == 200