"""Unit tests for the eval metrics (algorithm 2.0, Phase 0)."""
import math

import pytest

from eval import metrics


def test_recall_at_k_partial_hit():
    rec = ["a", "b", "c", "d"]
    target = {"b", "d", "z"}  # 2 of 3 retrievable within k
    # min(len(target)=3, k=4) = 3 → 2/3
    assert metrics.recall_at_k(rec, target, k=4) == pytest.approx(2 / 3)


def test_recall_at_k_capped_by_k():
    rec = ["a", "b"]
    target = {"a", "b", "c", "d"}  # 4 targets but only k=2 slots
    # both top-2 are hits → 2 / min(4, 2) = 1.0
    assert metrics.recall_at_k(rec, target, k=2) == pytest.approx(1.0)


def test_recall_at_k_empty_target():
    assert metrics.recall_at_k(["a"], set(), k=5) == 0.0


def test_mrr_first_and_second():
    assert metrics.mrr(["a", "b", "c"], {"a"}) == pytest.approx(1.0)
    assert metrics.mrr(["a", "b", "c"], {"b"}) == pytest.approx(0.5)
    assert metrics.mrr(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)


def test_mrr_no_hit():
    assert metrics.mrr(["a", "b"], {"z"}) == 0.0


def test_intra_list_distance_orthogonal_and_identical():
    # two orthogonal unit vectors → cosine 0 → distance 1.0
    assert metrics.intra_list_distance([[1, 0], [0, 1]]) == pytest.approx(1.0)
    # identical vectors → cosine 1 → distance 0.0
    assert metrics.intra_list_distance([[1, 1], [1, 1]]) == pytest.approx(0.0, abs=1e-9)


def test_intra_list_distance_too_few():
    assert metrics.intra_list_distance([[1, 0]]) == 0.0
    assert metrics.intra_list_distance([]) == 0.0


def test_intra_list_distance_zero_vector_is_max_distance():
    # a zero vector has no direction → treated as maximally distant
    assert metrics.intra_list_distance([[0, 0], [1, 1]]) == pytest.approx(1.0)


def test_median_listeners():
    assert metrics.median_listeners([100, 200, 300]) == pytest.approx(200.0)
    assert metrics.median_listeners([10, 20, 30, 40]) == pytest.approx(25.0)
    assert metrics.median_listeners([]) == 0.0