"""
Per-client-IP rate limiting (issue #20), built on slowapi.

One shared `limiter` is created here so both `app.main` (wiring + global default)
and the routers (stricter per-route limits via `@limiter.limit(RATE_LIMIT_HEAVY)`)
import the same instance. The whole thing is a no-op when `RATE_LIMIT_ENABLED` is
falsey — the test suite turns it off so fixtures aren't throttled.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import RATE_LIMIT_ENABLED, RATE_LIMIT_DEFAULT


def client_ip(request) -> str:
    """
    Identify the caller for rate-limit bucketing.

    On Railway the app sits behind a proxy, so the socket peer is the proxy and
    every client would share one bucket. Prefer the left-most `X-Forwarded-For`
    entry (the original client) and fall back to the socket address locally.

    Caveat: `X-Forwarded-For` is client-settable, so a determined attacker can
    rotate it to dodge the limit. That's acceptable here — this is a release-gate
    guard against casual abuse and runaway upstream cost, not a security control.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return get_remote_address(request)


limiter = Limiter(
    key_func=client_ip,
    default_limits=[RATE_LIMIT_DEFAULT],
    enabled=RATE_LIMIT_ENABLED,
    headers_enabled=True,  # emit X-RateLimit-* so clients can back off
)