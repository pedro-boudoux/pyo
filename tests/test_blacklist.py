"""
Tests for the artist blacklist (issue #19).

Covers the two layers — the mandatory env set and the per-request `extra` set —
and the credit-aware, case-insensitive matching that makes "Drake" also block
"Drake, 21 Savage" and "X feat. Drake".
"""
import pytest

from app.services import blacklist


@pytest.fixture
def drake_blocked(monkeypatch):
    """Pin the mandatory set to {"drake"} regardless of the real env."""
    monkeypatch.setattr(blacklist, "MANDATORY", {"drake"})


class TestNormalize:
    def test_lowercases_strips_and_drops_blanks(self):
        assert blacklist.normalize([" Drake ", "", "  ", "MF DOOM"]) == {"drake", "mf doom"}


class TestMandatory:
    def test_nothing_blocked_when_set_empty(self, monkeypatch):
        monkeypatch.setattr(blacklist, "MANDATORY", set())
        assert blacklist.is_blocked("Drake") is False

    def test_exact_match_case_insensitive(self, drake_blocked):
        assert blacklist.is_blocked("Drake") is True
        assert blacklist.is_blocked("drake") is True
        assert blacklist.is_blocked("DRAKE") is True

    def test_unrelated_artist_allowed(self, drake_blocked):
        assert blacklist.is_blocked("Kendrick Lamar") is False

    def test_blocks_any_credit_in_multi_artist(self, drake_blocked):
        assert blacklist.is_blocked("Drake, 21 Savage") is True       # primary
        assert blacklist.is_blocked("Future & Drake") is True          # secondary
        assert blacklist.is_blocked("Rihanna feat. Drake") is True     # feature
        assert blacklist.is_blocked("DJ Khaled ft Drake") is True

    def test_substring_is_not_a_match(self, drake_blocked):
        # "Drake" must not knock out "Drakeo the Ruler" or "Nick Drake".
        assert blacklist.is_blocked("Drakeo the Ruler") is False
        assert blacklist.is_blocked("Nick Drake") is False

    def test_whole_string_match_for_slash_names(self, monkeypatch):
        # A name containing a separator can still be blocked as the full credit.
        monkeypatch.setattr(blacklist, "MANDATORY", {"ac/dc"})
        assert blacklist.is_blocked("AC/DC") is True


class TestPerRequestExtra:
    def test_extra_blocks_on_top_of_mandatory(self, drake_blocked):
        extra = blacklist.normalize(["Coldplay"])
        assert blacklist.is_blocked("Coldplay", extra) is True   # user block
        assert blacklist.is_blocked("Drake", extra) is True      # still mandatory
        assert blacklist.is_blocked("Radiohead", extra) is False

    def test_extra_is_credit_aware(self, monkeypatch):
        monkeypatch.setattr(blacklist, "MANDATORY", set())
        extra = blacklist.normalize(["coldplay"])
        assert blacklist.is_blocked("Coldplay feat. Beyoncé", extra) is True

    def test_empty_inputs(self, monkeypatch):
        monkeypatch.setattr(blacklist, "MANDATORY", set())
        assert blacklist.is_blocked("Anyone") is False
        assert blacklist.is_blocked("", blacklist.normalize(["x"])) is False
