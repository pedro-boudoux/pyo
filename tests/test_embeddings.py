"""
Tier-1 unit tests for app/services/embeddings.py.
No database required — get_cursor is monkeypatched where needed.
"""

import hashlib
import math
import pytest
from contextlib import contextmanager

from app.services.embeddings import (
    make_track_id,
    primary_artist,
    cosine_similarity,
    mmr_rerank,
    build_tag_vector,
    dominant_tags,
    get_or_create_tag_ids,
)


# ---------------------------------------------------------------------------
# get_or_create_tag_ids — must NOT burn SERIAL ids for existing tags
# ---------------------------------------------------------------------------

class _VocabCursor:
    """Fake cursor: SELECT returns an existing id or None; INSERT assigns the
    next id and bumps an insert counter (a proxy for sequence consumption)."""

    def __init__(self, existing, next_id):
        self.existing = dict(existing)
        self.next_id = next_id
        self.insert_count = 0
        self._last = None

    def execute(self, sql, params=()):
        verb = sql.strip().split()[0].upper()
        tag = params[0]
        if verb == "SELECT":
            self._last = {"id": self.existing[tag]} if tag in self.existing else None
        elif verb == "INSERT":
            self.insert_count += 1
            self.existing[tag] = self.next_id
            self._last = {"id": self.next_id}
            self.next_id += 1

    def fetchone(self):
        return self._last


class TestGetOrCreateTagIds:
    def _patch(self, monkeypatch, cursor):
        @contextmanager
        def fake_get_cursor():
            yield cursor
        monkeypatch.setattr("app.services.embeddings.get_cursor", fake_get_cursor)

    def test_existing_tags_do_not_insert(self, monkeypatch):
        """The whole point of the fix: existing tags are pure SELECTs, no INSERT,
        so the SERIAL sequence never advances for them."""
        cur = _VocabCursor(existing={"jazz": 1, "funk": 2}, next_id=300)
        self._patch(monkeypatch, cur)
        ids = get_or_create_tag_ids(["jazz", "funk"])
        assert ids == {"jazz": 1, "funk": 2}
        assert cur.insert_count == 0

    def test_new_tag_inserted_once(self, monkeypatch):
        cur = _VocabCursor(existing={"jazz": 1}, next_id=5)
        self._patch(monkeypatch, cur)
        ids = get_or_create_tag_ids(["jazz", "hyperpop"])
        assert ids["jazz"] == 1
        assert ids["hyperpop"] == 5
        assert cur.insert_count == 1
from app.config import EMBEDDING_DIM, TAG_EMBEDDING_DIM
from tests.conftest import make_fake_get_cursor


# ---------------------------------------------------------------------------
# dominant_tags
# ---------------------------------------------------------------------------

class TestDominantTags:
    """dominant_tags now aggregates the raw {tag: count} dicts persisted in
    songs.tags (jsonb), not embedding slots."""

    def test_empty_input_returns_empty(self):
        assert dominant_tags([], top_n=5) == []

    def test_sums_weight_and_counts_songs(self):
        # jazz appears in both songs; funk in one
        tag_dicts = [{"jazz": 1.0, "funk": 0.5}, {"jazz": 0.5}]
        out = dominant_tags(tag_dicts, top_n=5)

        jazz = next(r for r in out if r["tag"] == "jazz")
        funk = next(r for r in out if r["tag"] == "funk")
        assert jazz["weight"] == 1.5 and jazz["count"] == 2
        assert funk["weight"] == 0.5 and funk["count"] == 1
        # jazz (1.5) outranks funk (0.5)
        assert [r["tag"] for r in out] == ["jazz", "funk"]

    def test_share_is_fraction_of_total_weight(self):
        out = dominant_tags([{"a": 3.0, "b": 1.0}], top_n=5)
        shares = {r["tag"]: r["share"] for r in out}
        assert shares["a"] == 0.75 and shares["b"] == 0.25

    def test_top_n_caps_results(self):
        out = dominant_tags([{"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6}], top_n=2)
        assert [r["tag"] for r in out] == ["a", "b"]

    def test_none_or_empty_dicts_are_skipped(self):
        # a null tags column (None) or empty dict contributes nothing
        out = dominant_tags([None, {}, {"a": 1.0}], top_n=5)
        assert [r["tag"] for r in out] == ["a"]


# ---------------------------------------------------------------------------
# make_track_id
# ---------------------------------------------------------------------------

class TestMakeTrackId:
    def test_deterministic(self):
        """Same input always produces the same id."""
        assert make_track_id("Artist", "Track") == make_track_id("Artist", "Track")

    def test_length_is_20(self):
        assert len(make_track_id("Burial", "Archangel")) == 20

    def test_case_insensitive(self):
        """Upper, lower, mixed — all produce the same id."""
        assert make_track_id("Burial", "Archangel") == make_track_id("BURIAL", "ARCHANGEL")
        assert make_track_id("Burial", "Archangel") == make_track_id("burial", "archangel")

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace is ignored."""
        assert make_track_id("  Burial  ", "  Archangel  ") == make_track_id("Burial", "Archangel")

    def test_different_songs_differ(self):
        id1 = make_track_id("Burial", "Archangel")
        id2 = make_track_id("Burial", "Shell of Light")
        assert id1 != id2


# ---------------------------------------------------------------------------
# primary_artist — recover the first credit from a multi-artist string
# ---------------------------------------------------------------------------

class TestPrimaryArtist:
    def test_comma_separated_takes_first(self):
        assert primary_artist("GORDÃO DO PC, Mc Ag, Dj Wesley") == "GORDÃO DO PC"

    def test_feat_variants(self):
        assert primary_artist("Drake feat. Rihanna") == "Drake"
        assert primary_artist("Drake ft Rihanna") == "Drake"
        assert primary_artist("Drake featuring Rihanna") == "Drake"

    def test_ampersand_slash_x_vs(self):
        assert primary_artist("Calvin Harris & Dua Lipa") == "Calvin Harris"
        assert primary_artist("Calvin Harris x Dua Lipa") == "Calvin Harris"
        assert primary_artist("Blur vs Oasis") == "Blur"
        assert primary_artist("A / B") == "A"

    def test_single_artist_unchanged(self):
        assert primary_artist("Burial") == "Burial"

    def test_no_false_split_inside_word(self):
        # 'x' only splits as a standalone word, not inside "Max" / "Foxygen"
        assert primary_artist("Max Cooper") == "Max Cooper"
        assert primary_artist("Foxygen") == "Foxygen"

    def test_whitespace_stripped(self):
        assert primary_artist("  MC A , MC B ") == "MC A"

    def test_different_artists_differ(self):
        id1 = make_track_id("Burial", "Archangel")
        id2 = make_track_id("Massive Attack", "Archangel")
        assert id1 != id2

    def test_matches_manual_sha1(self):
        """Verify the exact SHA1 construction."""
        key = "burial|||archangel"
        expected = hashlib.sha1(key.encode()).hexdigest()[:20]
        assert make_track_id("Burial", "Archangel") == expected

    def test_hex_string(self):
        """Result contains only lowercase hex chars."""
        tid = make_track_id("x", "y")
        assert all(c in "0123456789abcdef" for c in tid)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector_a(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_zero_vector_b(self):
        assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_both_zero_vectors(self):
        assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_opposite_vectors(self):
        """Anti-parallel vectors should give -1.0."""
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_known_value(self):
        """[1,1] vs [1,0]: cos = 1/sqrt(2)."""
        assert cosine_similarity([1.0, 1.0], [1.0, 0.0]) == pytest.approx(1.0 / math.sqrt(2))

    def test_symmetry(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))


# ---------------------------------------------------------------------------
# mmr_rerank
# ---------------------------------------------------------------------------

class TestMmrRerank:
    """
    We use 3-dimensional unit vectors so similarity calculations are easy
    to verify by hand.

    Vector set:
      q  = [1, 0, 0]   (query)
      c0 = [1, 0, 0]   cos(q, c0) = 1.0   (most relevant, identical to query)
      c1 = [0, 1, 0]   cos(q, c1) = 0.0   (orthogonal — diverse)
      c2 = [0.8, 0, 0.6] — less relevant than c0 but similar direction
    """

    @pytest.fixture
    def query(self):
        return [1.0, 0.0, 0.0]

    @pytest.fixture
    def candidates(self):
        return [
            {"id": "c0", "embedding": [1.0, 0.0, 0.0]},
            {"id": "c1", "embedding": [0.0, 1.0, 0.0]},
            {"id": "c2", "embedding": [0.8, 0.0, 0.6]},
        ]

    def test_empty_candidates(self, query):
        assert mmr_rerank(query, [], k=5, lambda_param=0.7) == []

    def test_k_larger_than_pool_returns_all(self, query, candidates):
        result = mmr_rerank(query, candidates, k=100, lambda_param=0.7)
        assert len(result) == len(candidates)

    def test_k_limits_results(self, query, candidates):
        result = mmr_rerank(query, candidates, k=2, lambda_param=0.7)
        assert len(result) == 2

    def test_lambda_1_pure_relevance_order(self, query, candidates):
        """
        With lambda=1.0 (pure relevance) the ranking is by cosine similarity to query:
          c0 (1.0) > c2 (0.8) > c1 (0.0).
        """
        result = mmr_rerank(query, candidates, k=3, lambda_param=1.0)
        ids = [r["id"] for r in result]
        assert ids == ["c0", "c2", "c1"]

    def test_lambda_0_pure_diversity(self, query, candidates):
        """
        With lambda=0.0 the MMR score is purely -redundancy, so after picking the
        first candidate (arbitrarily c0, highest relevance breaks the tie in the
        very first iteration when selected is empty and redundancy=0 for all),
        subsequent picks maximize distance from already-selected items.

        After c0=[1,0,0] is picked:
          c1: score = 0 * 0.0 - 1.0 * cos(c1, c0) = -cos([0,1,0],[1,0,0]) = 0.0
          c2: score = 0 * 0.0 - 1.0 * cos(c2, c0) = -0.8

        So c1 should be chosen second (less redundant with c0 than c2 is).
        """
        result = mmr_rerank(query, candidates, k=3, lambda_param=0.0)
        # c0 is first (all scores equal 0 on first iteration, but c0 has highest relevance
        # since tie-break still uses lambda*rel = 0 for all; in practice the loop finds the
        # first candidate in list order with max score — c0 has same score as others,
        # first-found wins, so c0 is first).
        # The key assertion is that c1 (diversity winner) comes before c2.
        ids = [r["id"] for r in result]
        assert ids[0] == "c0"
        assert ids[1] == "c1"
        assert ids[2] == "c2"

    def test_result_items_are_from_candidates(self, query, candidates):
        """All returned dicts are original candidate objects."""
        result = mmr_rerank(query, candidates, k=3, lambda_param=0.7)
        assert all(r in candidates for r in result)

    def test_k_equals_zero_returns_empty(self, query, candidates):
        result = mmr_rerank(query, candidates, k=0, lambda_param=0.7)
        assert result == []


# ---------------------------------------------------------------------------
# build_tag_vector
# ---------------------------------------------------------------------------

class TestBuildTagVector:
    """build_tag_vector is now a count-weighted average of dense MiniLM tag
    vectors, L2-normalized. The tag encoder is mocked so no model loads."""

    def _patch_encoder(self, monkeypatch, tag_to_vec):
        import numpy as np
        emb = {t: np.asarray(v, dtype=float) for t, v in tag_to_vec.items()}
        monkeypatch.setattr(
            "app.services.embeddings.tag_encoder.get_tag_embeddings",
            lambda tags: {t: emb[t] for t in tags if t in emb},
        )

    def test_empty_dict_returns_all_zeros(self):
        """No encode needed — empty guard is hit first."""
        result = build_tag_vector({})
        assert len(result) == TAG_EMBEDDING_DIM
        assert all(v == 0.0 for v in result)

    def test_result_is_l2_normalized(self, monkeypatch):
        self._patch_encoder(monkeypatch, {
            "rock": [1.0, 0.0, 0.0],
            "indie": [0.0, 1.0, 0.0],
        })
        result = build_tag_vector({"rock": 100, "indie": 50})
        assert len(result) == 3
        assert math.sqrt(sum(x * x for x in result)) == pytest.approx(1.0)
        # weighted toward rock (count 100 > 50) → first component larger
        assert result[0] > result[1] > 0

    def test_weighted_average_direction(self, monkeypatch):
        # equal, orthogonal tag vectors with equal weight → 45° between them,
        # each component = 1/sqrt(2) after normalization
        self._patch_encoder(monkeypatch, {
            "a": [1.0, 0.0],
            "b": [0.0, 1.0],
        })
        result = build_tag_vector({"a": 10, "b": 10})
        assert result[0] == pytest.approx(1 / math.sqrt(2))
        assert result[1] == pytest.approx(1 / math.sqrt(2))

    def test_unencodable_tags_skipped(self, monkeypatch):
        # only "ambient" has a vector; "unknown" is dropped (encoder returns nothing)
        self._patch_encoder(monkeypatch, {"ambient": [0.0, 3.0, 4.0]})
        result = build_tag_vector({"ambient": 80, "unknown": 200})
        # normalized [0,3,4] → [0, 0.6, 0.8]
        assert result == pytest.approx([0.0, 0.6, 0.8])

    def test_no_encodable_tags_returns_zeros(self, monkeypatch):
        self._patch_encoder(monkeypatch, {})  # nothing encodes
        result = build_tag_vector({"x": 1, "y": 2})
        assert len(result) == TAG_EMBEDDING_DIM
        assert all(v == 0.0 for v in result)
