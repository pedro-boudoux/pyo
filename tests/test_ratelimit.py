"""
Rate limiting (issue #20).

The suite disables the real limiter (conftest sets RATE_LIMIT_ENABLED=false), so
here we build a throwaway app wired exactly like app.main — limiter + middleware +
exception handler + a heavy-limited route — with an *enabled* limiter and a tiny
limit, and assert the 429 once the limit is exceeded. This validates the slowapi
integration (key func, middleware, per-route decorator) against the installed
versions without depending on the app's production limits.
"""
import inspect
from contextlib import contextmanager

from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.ratelimit import client_ip, limiter as app_limiter


def _make_app(default="100/minute", heavy="2/minute"):
    limiter = Limiter(
        key_func=client_ip,
        default_limits=[default],
        enabled=True,
        headers_enabled=True,
    )
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    @app.get("/cheap")
    def cheap():
        return {"ok": True}

    @app.get("/heavy")
    @limiter.limit(heavy)
    def heavy_route(request: Request, response: Response):
        return {"ok": True}

    @limiter.exempt
    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


class TestClientIp:
    def test_prefers_proxy_real_ip(self):
        req = type("R", (), {"headers": {"x-real-ip": "2001:0db8::1"}, "client": None})()
        assert client_ip(req) == "2001:db8::1"

    def test_ignores_spoofable_forwarded_for(self):
        req = type(
            "R",
            (),
            {
                "headers": {"x-forwarded-for": "1.2.3.4, 10.0.0.1"},
                "client": type("C", (), {"host": "9.9.9.9"})(),
            },
        )()
        assert client_ip(req) == "9.9.9.9"

    def test_invalid_real_ip_falls_back_to_peer(self):
        req = type(
            "R",
            (),
            {
                "headers": {"x-real-ip": "not-an-ip"},
                "client": type("C", (), {"host": "9.9.9.9"})(),
            },
        )()
        assert client_ip(req) == "9.9.9.9"

    def test_ipv4_real_ip(self):
        req = type("R", (), {"headers": {"x-real-ip": "1.2.3.4"}, "client": None})()
        assert client_ip(req) == "1.2.3.4"

    def test_falls_back_to_peer_when_no_header(self):
        req = type("R", (), {"headers": {}, "client": type("C", (), {"host": "9.9.9.9"})()})()
        assert client_ip(req) == "9.9.9.9"


class TestEnforcement:
    def test_heavy_route_429s_after_limit(self):
        client = TestClient(_make_app(heavy="2/minute"))
        first = client.get("/heavy")
        assert first.status_code == 200
        assert first.headers["x-ratelimit-limit"] == "2"
        assert first.headers["x-ratelimit-remaining"] == "1"
        assert "retry-after" in first.headers
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
        def x(request: Request, response: Response):
            return {"ok": True}

        client = TestClient(app)
        for _ in range(5):
            assert client.get("/x").status_code == 200


def test_all_production_limited_routes_accept_response_for_header_injection():
    """SlowAPI returns 500 when headers are enabled and a dict/Pydantic route
    has no Response parameter. Exercise every decorated production endpoint."""
    for functions in app_limiter._Limiter__marked_for_limiting.values():
        for function in functions:
            parameters = inspect.signature(function).parameters
            assert "request" in parameters, function
            assert "response" in parameters, function


def test_real_app_decorated_route_emits_headers(monkeypatch):
    """Exercise a production router with the production limiter configured the
    way the production proxy runs it; the original throwaway test missed this 500 path."""
    from app.main import app
    from app.routers import songs

    class Cursor:
        def execute(self, *_args, **_kwargs):
            pass

        def fetchone(self):
            return {
                "name": "Archangel",
                "artist": "Burial",
                "listeners": 1000,
                "embedding": [0.0] * 384,
                "tags": {},
            }

    @contextmanager
    def fake_cursor():
        yield Cursor()

    monkeypatch.setattr(songs, "get_cursor", fake_cursor)
    app_limiter.enabled = True
    app_limiter.reset()
    try:
        response = TestClient(app).get(
            "/songs/test-track/features",
            headers={"x-real-ip": "203.0.113.10"},
        )
        assert response.status_code == 200
        assert response.headers["x-ratelimit-limit"] == "20"
        assert response.headers["x-ratelimit-remaining"] == "19"
    finally:
        app_limiter.enabled = False
        app_limiter.reset()
