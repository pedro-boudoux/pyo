"""Tests for the offline eval runner's recommendation entry point."""
import json
from unittest.mock import MagicMock

from app.models import Recommendation, RecommendationsResponse
from eval import run_eval


def test_evaluate_calls_undecorated_recommendation_pipeline(monkeypatch, tmp_path):
    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(json.dumps({
        "similar_limit": 20,
        "seeds": [
            {
                "seed_track_id": "seed",
                "name": "Seed Track",
                "artist": "Seed Artist",
                "targets": ["target"],
            }
        ],
    }))

    response = RecommendationsResponse(recommendations=[
        Recommendation(
            track_id="target",
            name="Target Track",
            artist="Target Artist",
            similarity=0.9,
            listeners=1234,
            image=None,
            tags=[],
        )
    ])
    build = MagicMock(return_value=response)
    monkeypatch.setattr(run_eval, "build_recommendations", build)
    monkeypatch.setattr(run_eval, "_embeddings_for", lambda ids: {"target": [1.0, 0.0]})

    result = run_eval.evaluate("test", k=10, gt_path=str(gt_path), read_only=True)

    assert result["seeds_scored"] == 1
    assert result["coverage"] == 1.0
    assert result["recall_at_10"] == 1.0
    assert result["mrr"] == 1.0
    assert result["median_listeners"] == 1234.0
    build.assert_called_once_with(
        "seed",
        k=10,
        lambda_param=run_eval.MMR_LAMBDA,
        exclude=[],
        include_tags=False,
    )
