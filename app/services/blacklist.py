"""
Artist blacklist enforcement.

Two layers, combined wherever song candidates are filtered:

  - MANDATORY — `BLACKLIST_ARTISTS` from the deployment env. Always enforced:
    blocked artists are never recommended (they can still be searched and used as
    a seed directly). This is the "never recommend Drake to anyone" knob.
  - per-request — extra artist names a client passes (the user's own block list,
    kept in their browser). Applied on top of the mandatory set for that request
    only via `extra=`.

Matching is case-insensitive and credit-aware: a name blocks a track if it equals
the full credit OR any individually-credited artist in it (`split_credits`), so
blocking "Drake" also blocks "Drake, 21 Savage" and "X feat. Drake".
"""
import re

from app.config import BLACKLIST_ARTISTS
from app.services.embeddings import split_credits

# Trim leading/trailing punctuation so a credit like ". Drake" (the "." is a
# leftover from splitting "feat.") still matches "drake".
_PUNCT_EDGES = re.compile(r"^[^\w]+|[^\w]+$")


def _norm(name: str) -> str:
    return _PUNCT_EDGES.sub("", name.strip()).strip().lower()


def normalize(names) -> set[str]:
    """Lowercased, stripped set of artist names (blanks dropped)."""
    return {_norm(n) for n in names if n and n.strip()}


# Computed once at import from the env. Empty set = blacklist disabled.
MANDATORY: set[str] = normalize(BLACKLIST_ARTISTS)


def is_blocked(artist: str, extra: set[str] = frozenset()) -> bool:
    """True if `artist` is blocked by the mandatory set or the per-request `extra`."""
    blocked = MANDATORY | extra
    if not blocked or not artist:
        return False
    if _norm(artist) in blocked:
        return True
    return any(_norm(part) in blocked for part in split_credits(artist))
