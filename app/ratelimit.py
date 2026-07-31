"""
Per-client-IP rate limiting (issue #20), built on slowapi.

One shared `limiter` is created here so both `app.main` (wiring + global default)
and the routers (stricter per-route limits via `@limiter.limit(RATE_LIMIT_HEAVY)`)
import the same instance. The whole thing is a no-op when `RATE_LIMIT_ENABLED` is
falsey — the test suite turns it off so fixtures aren't throttled.
"""
from ipaddress import ip_address

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import RATE_LIMIT_ENABLED, RATE_LIMIT_DEFAULT


def client_ip(request) -> str:
    """
    Identify the caller for rate-limit bucketing.

    The production reverse proxy sets X-Real-IP to the original client address. Do not trust
    X-Forwarded-For here: callers can supply it themselves and rotate the value
    to evade a per-IP limit. Invalid/missing proxy headers fall back to the
    socket peer for local development.
    """
    proxy_ip = request.headers.get("x-real-ip", "").strip()
    if proxy_ip:
        try:
            return str(ip_address(proxy_ip))
        except ValueError:
            pass
    return get_remote_address(request)


limiter = Limiter(
    key_func=client_ip,
    default_limits=[RATE_LIMIT_DEFAULT],
    enabled=RATE_LIMIT_ENABLED,
    headers_enabled=True,  # emit X-RateLimit-* so clients can back off
)
