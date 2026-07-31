import math

import pytest

from app.config import COLISTEN_EMBEDDING_DIM, HYBRID_EMBEDDING_DIM, TAG_EMBEDDING_DIM
from app.services import hybrid


def unit_tag():
    return [1.0] + [0.0] * (TAG_EMBEDDING_DIM - 1)


def unit_colisten():
    return [1.0] + [0.0] * (COLISTEN_EMBEDDING_DIM - 1)


def test_missing_colisten_is_clean_tag_only_fallback():
    result = hybrid.compose(unit_tag(), None, beta=2.0)
    assert len(result) == HYBRID_EMBEDDING_DIM
    assert result[:TAG_EMBEDDING_DIM] == unit_tag()
    assert result[TAG_EMBEDDING_DIM:] == [0.0] * COLISTEN_EMBEDDING_DIM


def test_beta_zero_matches_tag_only_even_with_graph_vector():
    result = hybrid.compose(unit_tag(), unit_colisten(), beta=0.0)
    assert result[:TAG_EMBEDDING_DIM] == unit_tag()
    assert not any(result[TAG_EMBEDDING_DIM:])


def test_nonzero_beta_blends_and_normalizes():
    result = hybrid.compose(unit_tag(), unit_colisten(), beta=1.0)
    expected = 1 / math.sqrt(2)
    assert result[0] == pytest.approx(expected)
    assert result[TAG_EMBEDDING_DIM] == pytest.approx(expected)
    assert math.sqrt(sum(value * value for value in result)) == pytest.approx(1.0)


def test_colisten_magnitude_does_not_change_beta_meaning():
    graph = unit_colisten()
    scaled_graph = [value * 25 for value in graph]
    assert hybrid.compose(unit_tag(), graph, beta=0.5) == pytest.approx(
        hybrid.compose(unit_tag(), scaled_graph, beta=0.5)
    )


def test_dimension_mismatch_fails_before_pgvector():
    with pytest.raises(ValueError, match="tag embedding"):
        hybrid.compose([1.0, 0.0], None)


def test_active_column_is_closed_model_switch(monkeypatch):
    monkeypatch.setattr(hybrid, "RECOMMENDATION_MODEL", "hybrid")
    assert hybrid.active_embedding_column() == "hybrid_embedding"
    monkeypatch.setattr(hybrid, "RECOMMENDATION_MODEL", "stage_a")
    assert hybrid.active_embedding_column() == "embedding"
