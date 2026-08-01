"""Tests for the reproducible blind human preference evaluator."""

import json

import pytest

from eval import human_preference


def _model_export(
    model_id: str,
    version: str,
    run_id: str,
    *,
    seed_ids=("seed-1", "seed-2", "seed-3"),
) -> dict:
    return {
        "schema_version": 1,
        "kind": "human_preference_model_export",
        "model": {
            "model_id": model_id,
            "model_version": version,
            "run_id": run_id,
            "metadata": {"beta": "2.0" if model_id == "hybrid" else "0.0"},
        },
        "automated_evaluation": {
            "fixture": "eval/ground_truth_colisten.json",
            "result": f"eval/baselines/{model_id}.json",
            "status": "passed",
        },
        "capture": {
            "k": 10,
            "lambda": 0.7,
            "read_only": True,
            "seed_digest": "fixture-seed-digest",
        },
        "recommendations": [
            {
                "seed": {
                    "track_id": seed_id,
                    "name": f"Seed {index}",
                    "artist": "Seed Artist",
                },
                "tracks": [
                    {
                        "track_id": (
                            f"track-{seed_id}-"
                            f"{track_index if model_id == 'stage_a' else track_index + 10}"
                        ),
                        "name": (
                            f"Track {track_index if model_id == 'stage_a' else track_index + 10}"
                        ),
                        "artist": f"Artist {track_index}",
                        "listeners": 100 * track_index,
                        # Model-specific ranking internals must never enter the
                        # public blind artifact.
                        "similarity": 0.99 - track_index / 100,
                    }
                    for track_index in range(1, 4)
                ],
            }
            for index, seed_id in enumerate(seed_ids, start=1)
        ],
    }


def _prepared():
    return human_preference.prepare_session(
        _model_export("stage_a", "git:abc123", "eval-stage-a-7"),
        _model_export("hybrid", "model-run:42", "42"),
        study_id="beta-2-v-stage-a",
        randomization_seed="listener-study-2026-08",
    )


def test_prepare_is_deterministic_balanced_and_blind():
    stage_a = _model_export("stage_a", "git:abc123", "eval-stage-a-7")
    hybrid = _model_export("hybrid", "model-run:42", "42")

    session, key = human_preference.prepare_session(
        stage_a,
        hybrid,
        study_id="beta-2-v-stage-a",
        randomization_seed="listener-study-2026-08",
    )
    reversed_session, reversed_key = human_preference.prepare_session(
        hybrid,
        stage_a,
        study_id="beta-2-v-stage-a",
        randomization_seed="listener-study-2026-08",
    )

    assert session == reversed_session
    assert key == reversed_key
    assert session["randomization"] == {
        "algorithm": human_preference.RANDOMIZATION_ALGORITHM
    }
    public_json = json.dumps(session)
    assert "stage_a" not in public_json
    assert "hybrid" not in public_json
    assert "model-run:42" not in public_json
    assert "similarity" not in public_json

    model_1_left = sum(
        placement["left"] == "model_1" for placement in key["placements"].values()
    )
    assert abs(model_1_left - (len(session["pairs"]) - model_1_left)) <= 1
    assert key["models"]["model_1"]["model"]["model_version"]
    assert key["models"]["model_2"]["model"]["run_id"]
    assert key["source_digest"]


def test_prepare_requires_automated_fixture_evidence():
    first = _model_export("stage_a", "v1", "run-1")
    second = _model_export("hybrid", "v2", "run-2")
    del second["automated_evaluation"]

    with pytest.raises(human_preference.HumanPreferenceError, match="do not replace fixtures"):
        human_preference.prepare_session(
            first,
            second,
            study_id="study",
            randomization_seed="seed",
        )


def test_prepare_requires_same_seed_set():
    first = _model_export("stage_a", "v1", "run-1")
    second = _model_export("hybrid", "v2", "run-2", seed_ids=("seed-1", "seed-2"))

    with pytest.raises(human_preference.HumanPreferenceError, match="identical seed IDs"):
        human_preference.prepare_session(
            first,
            second,
            study_id="study",
            randomization_seed="seed",
        )


def test_vote_records_ties_metadata_and_no_model_identity(tmp_path):
    session, _key = _prepared()
    answers = iter(["a", "tie", "b"])
    messages = []
    output_path = tmp_path / "listener-1.json"

    result = human_preference.vote_on_session(
        session,
        output_path=output_path,
        evaluator_id="listener-1",
        session_metadata={"headphones": "wired", "room": "quiet"},
        input_fn=lambda _prompt: next(answers),
        output=messages.append,
    )

    assert [vote["choice"] for vote in result["votes"]] == ["left", "tie", "right"]
    assert result["completed_at"] is not None
    assert result["session_metadata"] == {"headphones": "wired", "room": "quiet"}
    assert all(vote["seed_track_id"] for vote in result["votes"])
    persisted = output_path.read_text()
    assert "model_version" not in persisted
    assert "model-run:42" not in persisted
    assert any("model placement key is not loaded" in message for message in messages)


def test_vote_can_quit_and_resume(tmp_path):
    session, _key = _prepared()
    output_path = tmp_path / "listener-1.json"

    first_answers = iter(["a", "q"])
    partial = human_preference.vote_on_session(
        session,
        output_path=output_path,
        evaluator_id="listener-1",
        input_fn=lambda _prompt: next(first_answers),
        output=lambda _message: None,
    )
    assert len(partial["votes"]) == 1
    assert partial["completed_at"] is None

    answers = iter(["t", "b"])
    complete = human_preference.vote_on_session(
        session,
        output_path=output_path,
        evaluator_id="listener-1",
        input_fn=lambda _prompt: next(answers),
        output=lambda _message: None,
    )
    assert len(complete["votes"]) == 3
    assert complete["completed_at"] is not None


def test_aggregate_stays_blind_without_key_and_deblinds_with_versions(tmp_path):
    session, key = _prepared()
    answers = iter(["a", "tie", "b"])
    votes = human_preference.vote_on_session(
        session,
        output_path=tmp_path / "votes.json",
        evaluator_id="listener-1",
        input_fn=lambda _prompt: next(answers),
        output=lambda _message: None,
    )

    blind = human_preference.aggregate_votes(session, [votes])
    assert blind["blind"] is True
    assert "models" not in blind
    assert blind["anonymous_choices"] == {"list_a": 1, "list_b": 1, "ties": 1}
    assert blind["completion_rate"] == 1.0
    assert blind["deployment_gate"] == {
        "human_preference_is_supplemental": True,
        "automated_evaluation_required": True,
    }

    deblinded = human_preference.aggregate_votes(session, [votes], key=key)
    assert deblinded["blind"] is False
    assert deblinded["ties"] == 1
    assert sum(model["wins"] for model in deblinded["models"]) == 2
    assert {model["model"]["model_version"] for model in deblinded["models"]} == {
        "git:abc123",
        "model-run:42",
    }
    assert all(model["model"]["run_id"] for model in deblinded["models"])
    assert set(deblinded["deployment_gate"]["automated_evaluations"]) == {
        "model_1",
        "model_2",
    }
    assert all("model_wins" in seed for seed in deblinded["per_seed"])


def test_aggregate_rejects_votes_from_another_session():
    session, _key = _prepared()
    votes = {
        "schema_version": 1,
        "kind": "blind_human_preference_votes",
        "session_id": "hpe_other",
        "session_metadata": {},
        "votes": [],
    }

    with pytest.raises(human_preference.HumanPreferenceError, match="different blind session"):
        human_preference.aggregate_votes(session, [votes])


def test_aggregate_rejects_same_ballot_twice(tmp_path):
    session, _key = _prepared()
    answers = iter(["a", "t", "b"])
    votes = human_preference.vote_on_session(
        session,
        output_path=tmp_path / "votes.json",
        evaluator_id="listener-1",
        input_fn=lambda _prompt: next(answers),
        output=lambda _message: None,
    )

    with pytest.raises(human_preference.HumanPreferenceError, match="more than once"):
        human_preference.aggregate_votes(session, [votes, votes])


def test_capture_sanitizes_tracks_and_records_model_evidence(monkeypatch):
    from app.routers import recommendations

    def fake_recommendations(track_id, **kwargs):
        assert track_id == "seed-1"
        assert kwargs["include_tags"] is False
        return {
            "recommendations": [
                {
                    "track_id": "track-1",
                    "name": "Track One",
                    "artist": "Artist One",
                    "listeners": 55,
                    "similarity": 0.987,
                    "tags": ["secret-ranking-detail"],
                }
            ]
        }

    monkeypatch.setattr(recommendations, "build_recommendations", fake_recommendations)
    captured = human_preference.capture_recommendations(
        [{"track_id": "seed-1", "name": "Seed One", "artist": "Seed Artist"}],
        model={
            "model_id": "hybrid",
            "model_version": "model-run:42",
            "run_id": "42",
            "metadata": {"beta": "2"},
        },
        automated_evaluation={
            "fixture": "eval/ground_truth_colisten.json",
            "result": "eval/baselines/candidate.json",
            "status": "passed",
        },
        k=10,
    )

    assert captured["model"]["model_version"] == "model-run:42"
    assert captured["automated_evaluation"]["status"] == "passed"
    assert captured["capture"] == {
        "k": 10,
        "lambda": 0.7,
        "read_only": True,
        "seed_digest": human_preference._digest(
            [{"track_id": "seed-1", "name": "Seed One", "artist": "Seed Artist"}]
        ),
    }
    assert captured["recommendations"][0]["tracks"] == [
        {
            "track_id": "track-1",
            "name": "Track One",
            "artist": "Artist One",
            "listeners": 55,
        }
    ]


def test_read_only_capture_refuses_to_embed_a_cold_seed(monkeypatch):
    from app.routers import recommendations

    def fake_recommendations(_track_id, **_kwargs):
        recommendations.ingest.embed_and_store_track("Seed Artist", "Seed One")

    monkeypatch.setattr(recommendations, "build_recommendations", fake_recommendations)

    with pytest.raises(human_preference.HumanPreferenceError, match="read-only capture"):
        human_preference.capture_recommendations(
            [{"track_id": "seed-1", "name": "Seed One", "artist": "Seed Artist"}],
            model={
                "model_id": "hybrid",
                "model_version": "model-run:42",
                "run_id": "42",
                "metadata": {},
            },
            automated_evaluation={
                "fixture": "eval/ground_truth_colisten.json",
                "result": "eval/baselines/candidate.json",
                "status": "passed",
            },
            k=10,
            read_only=True,
        )


def test_metadata_parser_rejects_ambiguous_values():
    assert human_preference._parse_metadata(["device=laptop", "notes=a=b"]) == {
        "device": "laptop",
        "notes": "a=b",
    }
    with pytest.raises(human_preference.HumanPreferenceError, match="key=value"):
        human_preference._parse_metadata(["device"])
