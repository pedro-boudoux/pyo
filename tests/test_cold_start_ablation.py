import copy

import pytest

from eval import cold_start_ablation as ablation


def _record(cohort, count, *, target=False, calls=2, latency=100.0):
    candidates = [
        {
            "track_id": f"candidate-{index}",
            "name": f"Candidate {index}",
            "artist": f"Artist {index}",
            "listeners": 1000,
            "source": "ann",
        }
        for index in range(count)
    ]
    return {
        "cohort": cohort,
        "targets": ["candidate-0"] if target else [],
        "recall": 1.0 if target else None,
        "mrr": 1.0 if target else None,
        "candidates": candidates,
        "lastfm_calls": {"track.getSimilar": calls},
        "latency_ms": latency,
        "unique_artists": count,
        "selected_by_source": {"ann": count},
    }


def test_aggregate_tracks_quality_cost_branching_and_cold_coverage():
    records = [
        _record("independent_quality", 10, target=True),
        _record("no_similar", 0, calls=6, latency=300.0),
        _record("obscure_warm", 5, calls=4, latency=200.0),
    ]

    result = ablation._aggregate(records, [], k=10)

    assert result["independent_quality"]["recall_at_10"] == 1.0
    assert result["cold_seed_coverage"]["any"] == 0.0
    assert result["obscure_seed_coverage"]["any"] == 0.5
    assert result["lastfm"]["total_calls"] == 12
    assert result["seed_latency_ms"]["mean"] == 200.0
    assert result["graph_branching"]["mean_degree"] == 5.0
    assert result["listener_policy"]["mode"] == "uncapped"


def _result(variant, *, recall=0.5, mrr_value=0.4, cold=1.0, calls=100):
    return {
        "variant": variant,
        "k": 10,
        "database_snapshot": {"songs": 100},
        "summary": {
            "independent_quality": {"recall_at_10": recall, "mrr": mrr_value},
            "cold_seed_coverage": {"any": cold},
            "lastfm": {"total_calls": calls},
            "seed_latency_ms": {"mean": 1000.0},
            "listener_policy": {"mode": "uncapped"},
        },
    }


def test_compare_gates_each_mechanism_independently():
    full = _result("full")
    no_expansion = _result("no_expansion", recall=0.49, calls=60)
    no_fallback = _result("no_artist_fallback", cold=0.5, calls=80)
    minimal = _result("minimal", recall=0.49, cold=0.5, calls=40)

    result = ablation.compare_results(
        [full, no_expansion, no_fallback, minimal]
    )

    assert result["recursive_expansion"]["eligible_for_removal"] is True
    assert result["recursive_expansion"]["lastfm_calls_delta"] == -40
    assert result["artist_fallback"]["eligible_for_removal"] is False


def test_compare_rejects_different_database_snapshots():
    results = [_result(variant) for variant in ablation.VARIANTS]
    results[-1] = copy.deepcopy(results[-1])
    results[-1]["database_snapshot"]["songs"] = 101

    with pytest.raises(ValueError, match="database snapshots differ"):
        ablation.compare_results(results)
