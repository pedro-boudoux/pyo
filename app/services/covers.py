import requests
from urllib.parse import quote

DEEZER_URL = "https://api.deezer.com/search"
DEEZER_ARTIST_URL = "https://api.deezer.com/search/artist"
ITUNES_URL = "https://itunes.apple.com/search"

# One shared session → TCP/TLS keep-alive across calls (each request is ~100-200ms
# cheaper). Safe to share across threads here because we never mutate session
# state (headers/cookies) — every call passes its own params.
_session = requests.Session()

# Deezer/iTunes normally answer in well under a second; 5s just let a hung
# provider stall search/backfill paths.
TIMEOUT = 2

# Last.fm's "no image" placeholder hash — anything matching this is a stale
# Last.fm CDN URL that doesn't actually point at an album cover.
LASTFM_PLACEHOLDER_HASH = "2a96cbd8b46e442fc41c2b86b821562f"


class CoversUnavailable(Exception):
    """Every cover provider errored (network/HTTP) — the result is *unknown*, as
    opposed to a successful lookup that found no cover. Callers should not
    persist this outcome so the lookup is retried later."""


def is_broken_image(url: str | None) -> bool:
    if not url:
        return True
    return LASTFM_PLACEHOLDER_HASH in url


def _deezer_cover(artist: str, name: str) -> str | None:
    q = f'artist:"{artist}" track:"{name}"'
    resp = _session.get(DEEZER_URL, params={"q": q, "limit": 1}, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    for r in data.get("data", []):
        cover = r.get("album", {}).get("cover_xl")
        if cover:
            return cover
    return None


def _itunes_cover(artist: str, name: str) -> str | None:
    term = quote(f"{artist} {name}")
    resp = _session.get(
        ITUNES_URL,
        params={"term": term, "entity": "song", "limit": 5},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    artist_lower = artist.lower()
    name_lower = name.lower()
    for r in data.get("results", []):
        if (
            artist_lower in r.get("artistName", "").lower()
            and name_lower in r.get("trackName", "").lower()
        ):
            url = r.get("artworkUrl100")
            if url:
                return url.replace("100x100bb.jpg", "600x600bb.jpg")
    if data.get("results"):
        url = data["results"][0].get("artworkUrl100")
        if url:
            return url.replace("100x100bb.jpg", "600x600bb.jpg")
    return None


def _deezer_artist_image(artist: str) -> str | None:
    resp = _session.get(
        DEEZER_ARTIST_URL,
        params={"q": artist, "limit": 1},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    for r in data.get("data", []):
        pic = r.get("picture_xl")
        if pic and "dzcdn.net" in pic:
            return pic
    return None


def resolve_cover(artist: str, name: str) -> str | None:
    """
    Try Deezer album cover first (best for underground), then iTunes album cover,
    then fall back to a Deezer artist photo. Last.fm artist images are not used
    because Last.fm removed them years ago and serves the same broken placeholder
    for every artist.

    Returns None when providers answered but none had art (a definitive miss).
    Raises CoversUnavailable when EVERY provider errored — a transient outage,
    not a definitive answer, so callers must not negative-cache it.
    """
    errors = 0
    for provider in (_deezer_cover, _itunes_cover, _deezer_artist_image):
        try:
            url = provider(artist, name)
        except (requests.RequestException, ValueError):
            errors += 1
            continue
        if url:
            return url
    if errors == 3:
        raise CoversUnavailable("all cover providers failed")
    return None


def get_cover_url(artist: str, name: str) -> str | None:
    """Best-effort cover lookup — never raises; outages and misses both → None."""
    try:
        return resolve_cover(artist, name)
    except CoversUnavailable:
        return None
