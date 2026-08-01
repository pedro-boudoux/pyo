"""Reproducible blind human preference evaluation for recommendation lists.

The workflow deliberately separates the public session from its private key:

1. ``capture`` snapshots recommendations and immutable model/run metadata.
2. ``prepare`` deterministically randomizes two snapshots into a blind session
   and writes the model placement to a separate key file.
3. ``vote`` shows only anonymous List A/List B pairs and records A, B, or tie.
4. ``aggregate`` summarizes one or more vote files, optionally using the key to
   deblind model wins and versions after voting is complete.

Human preference is supplemental. Every captured model must name its automated
fixture/result/status, and aggregate output explicitly preserves that deployment
gate requirement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from unittest.mock import patch


SCHEMA_VERSION = 1
RANDOMIZATION_ALGORITHM = "sha256-seeded-balanced-v1"
CHOICES = {"left", "right", "tie"}
AUTOMATED_STATUSES = {"passed", "failed", "pending"}


class HumanPreferenceError(ValueError):
    """Raised when an evaluation artifact is invalid or inconsistent."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any, length: int = 64) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()[:length]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise HumanPreferenceError(f"{path} must contain a JSON object")
    return value


def _write_json(path: str | Path, value: dict, *, overwrite: bool = True) -> None:
    target = Path(path)
    if target.exists() and not overwrite:
        raise HumanPreferenceError(f"refusing to overwrite {target}; pass --force to replace it")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, target)


def _required_string(value: dict, field: str, context: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result.strip():
        raise HumanPreferenceError(f"{context}.{field} must be a non-empty string")
    return result.strip()


def _public_track(value: dict, context: str) -> dict:
    if not isinstance(value, dict):
        raise HumanPreferenceError(f"{context} must be an object")
    track = {
        "track_id": _required_string(value, "track_id", context),
        "name": _required_string(value, "name", context),
        "artist": _required_string(value, "artist", context),
    }
    listeners = value.get("listeners")
    if listeners is not None:
        if not isinstance(listeners, int) or isinstance(listeners, bool) or listeners < 0:
            raise HumanPreferenceError(f"{context}.listeners must be a non-negative integer or null")
        track["listeners"] = listeners
    return track


def _validate_model_export(value: dict, context: str) -> dict:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise HumanPreferenceError(f"{context}.schema_version must be {SCHEMA_VERSION}")
    if value.get("kind") != "human_preference_model_export":
        raise HumanPreferenceError(f"{context}.kind must be human_preference_model_export")

    raw_model = value.get("model")
    if not isinstance(raw_model, dict):
        raise HumanPreferenceError(f"{context}.model must be an object")
    model = {
        "model_id": _required_string(raw_model, "model_id", f"{context}.model"),
        "model_version": _required_string(raw_model, "model_version", f"{context}.model"),
        "run_id": _required_string(raw_model, "run_id", f"{context}.model"),
    }
    metadata = raw_model.get("metadata", {})
    if not isinstance(metadata, dict):
        raise HumanPreferenceError(f"{context}.model.metadata must be an object")
    model["metadata"] = metadata

    raw_automated = value.get("automated_evaluation")
    if not isinstance(raw_automated, dict):
        raise HumanPreferenceError(
            f"{context}.automated_evaluation is required; human votes do not replace fixtures"
        )
    status = _required_string(raw_automated, "status", f"{context}.automated_evaluation")
    if status not in AUTOMATED_STATUSES:
        raise HumanPreferenceError(
            f"{context}.automated_evaluation.status must be one of {sorted(AUTOMATED_STATUSES)}"
        )
    automated = {
        "fixture": _required_string(
            raw_automated, "fixture", f"{context}.automated_evaluation"
        ),
        "result": _required_string(raw_automated, "result", f"{context}.automated_evaluation"),
        "status": status,
    }

    raw_capture = value.get("capture")
    if not isinstance(raw_capture, dict):
        raise HumanPreferenceError(f"{context}.capture must be an object")
    k = raw_capture.get("k")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise HumanPreferenceError(f"{context}.capture.k must be a positive integer")
    lambda_param = raw_capture.get("lambda")
    if (
        not isinstance(lambda_param, (int, float))
        or isinstance(lambda_param, bool)
        or not 0 <= lambda_param <= 1
    ):
        raise HumanPreferenceError(f"{context}.capture.lambda must be between 0 and 1")
    read_only = raw_capture.get("read_only")
    if not isinstance(read_only, bool):
        raise HumanPreferenceError(f"{context}.capture.read_only must be a boolean")
    capture = {
        "k": k,
        "lambda": float(lambda_param),
        "read_only": read_only,
        "seed_digest": _required_string(raw_capture, "seed_digest", f"{context}.capture"),
    }

    raw_recommendations = value.get("recommendations")
    if not isinstance(raw_recommendations, list) or not raw_recommendations:
        raise HumanPreferenceError(f"{context}.recommendations must be a non-empty list")
    recommendations = []
    seen_seeds: set[str] = set()
    for index, raw_entry in enumerate(raw_recommendations):
        entry_context = f"{context}.recommendations[{index}]"
        if not isinstance(raw_entry, dict):
            raise HumanPreferenceError(f"{entry_context} must be an object")
        seed = _public_track(raw_entry.get("seed"), f"{entry_context}.seed")
        if seed["track_id"] in seen_seeds:
            raise HumanPreferenceError(f"{context} repeats seed {seed['track_id']}")
        seen_seeds.add(seed["track_id"])
        raw_tracks = raw_entry.get("tracks")
        if not isinstance(raw_tracks, list) or not raw_tracks:
            raise HumanPreferenceError(f"{entry_context}.tracks must be a non-empty list")
        tracks = [
            _public_track(track, f"{entry_context}.tracks[{track_index}]")
            for track_index, track in enumerate(raw_tracks)
        ]
        if len({track["track_id"] for track in tracks}) != len(tracks):
            raise HumanPreferenceError(f"{entry_context}.tracks contains duplicate track IDs")
        recommendations.append({"seed": seed, "tracks": tracks})

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "human_preference_model_export",
        "model": model,
        "automated_evaluation": automated,
        "capture": capture,
        "recommendations": recommendations,
    }


def _model_identity(model_export: dict) -> tuple[str, str, str]:
    model = model_export["model"]
    return model["model_id"], model["model_version"], model["run_id"]


def prepare_session(
    model_a: dict,
    model_b: dict,
    *,
    study_id: str,
    randomization_seed: str,
) -> tuple[dict, dict]:
    """Return a deterministic public blind session and its private placement key."""
    study_id = study_id.strip()
    randomization_seed = randomization_seed.strip()
    if not study_id:
        raise HumanPreferenceError("study_id must be a non-empty string")
    if not randomization_seed:
        raise HumanPreferenceError("randomization_seed must be a non-empty string")

    exports = sorted(
        (
            _validate_model_export(model_a, "model_a"),
            _validate_model_export(model_b, "model_b"),
        ),
        key=_model_identity,
    )
    if _model_identity(exports[0]) == _model_identity(exports[1]):
        raise HumanPreferenceError("the two exports must have different model/version/run identifiers")

    by_seed = []
    for model_export in exports:
        by_seed.append(
            {entry["seed"]["track_id"]: entry for entry in model_export["recommendations"]}
        )
    seed_ids = sorted(by_seed[0])
    if set(seed_ids) != set(by_seed[1]):
        missing_from_first = sorted(set(by_seed[1]) - set(by_seed[0]))
        missing_from_second = sorted(set(by_seed[0]) - set(by_seed[1]))
        raise HumanPreferenceError(
            "model exports must contain identical seed IDs; "
            f"only in first={missing_from_second}, only in second={missing_from_first}"
        )
    for seed_id in seed_ids:
        if by_seed[0][seed_id]["seed"] != by_seed[1][seed_id]["seed"]:
            raise HumanPreferenceError(f"seed metadata differs between exports for {seed_id}")

    identity_material = {
        "study_id": study_id,
        "randomization_seed": randomization_seed,
        "models": exports,
    }
    session_id = f"hpe_{_digest(identity_material, 20)}"
    rng_seed = int(
        _digest(
            {
                "randomization_seed": randomization_seed,
                "model_identifiers": [_model_identity(item) for item in exports],
                "seed_ids": seed_ids,
            }
        ),
        16,
    )
    rng = random.Random(rng_seed)
    # Keep exposure balanced (difference <= 1) while randomizing which model gets
    # the extra left-side placement when the number of seeds is odd.
    placement_start = rng.randrange(2)
    placements = [(placement_start + index) % 2 for index in range(len(seed_ids))]
    rng.shuffle(placements)

    pairs = []
    placement_key = {}
    for seed_id, first_side in zip(seed_ids, placements):
        pair_id = f"pair_{_digest({'session_id': session_id, 'seed_track_id': seed_id}, 16)}"
        second_side = 1 - first_side
        pairs.append(
            {
                "pair_id": pair_id,
                "seed": by_seed[0][seed_id]["seed"],
                "left": by_seed[first_side][seed_id]["tracks"],
                "right": by_seed[second_side][seed_id]["tracks"],
            }
        )
        placement_key[pair_id] = {
            "seed_track_id": seed_id,
            "left": f"model_{first_side + 1}",
            "right": f"model_{second_side + 1}",
        }

    public_session = {
        "schema_version": SCHEMA_VERSION,
        "kind": "blind_human_preference_session",
        "session_id": session_id,
        "study_id": study_id,
        "randomization": {"algorithm": RANDOMIZATION_ALGORITHM},
        "automated_evaluation_required_for_deployment": True,
        "pairs": pairs,
    }
    private_key = {
        "schema_version": SCHEMA_VERSION,
        "kind": "blind_human_preference_key",
        "session_id": session_id,
        "study_id": study_id,
        "randomization": {
            "algorithm": RANDOMIZATION_ALGORITHM,
            "seed": randomization_seed,
        },
        "session_digest": _digest(public_session),
        "source_digest": _digest(identity_material),
        "models": {
            f"model_{index + 1}": {
                "model": model_export["model"],
                "automated_evaluation": model_export["automated_evaluation"],
                "capture": model_export["capture"],
            }
            for index, model_export in enumerate(exports)
        },
        "placements": placement_key,
    }
    return public_session, private_key


def _validate_session(value: dict) -> dict:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise HumanPreferenceError(f"session.schema_version must be {SCHEMA_VERSION}")
    if value.get("kind") != "blind_human_preference_session":
        raise HumanPreferenceError("session.kind must be blind_human_preference_session")
    _required_string(value, "session_id", "session")
    pairs = value.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise HumanPreferenceError("session.pairs must be a non-empty list")
    seen: set[str] = set()
    for index, pair in enumerate(pairs):
        context = f"session.pairs[{index}]"
        if not isinstance(pair, dict):
            raise HumanPreferenceError(f"{context} must be an object")
        pair_id = _required_string(pair, "pair_id", context)
        if pair_id in seen:
            raise HumanPreferenceError(f"session repeats pair {pair_id}")
        seen.add(pair_id)
        _public_track(pair.get("seed"), f"{context}.seed")
        for side in ("left", "right"):
            tracks = pair.get(side)
            if not isinstance(tracks, list) or not tracks:
                raise HumanPreferenceError(f"{context}.{side} must be a non-empty list")
            for track_index, track in enumerate(tracks):
                _public_track(track, f"{context}.{side}[{track_index}]")
    return value


def _parse_metadata(items: Iterable[str]) -> dict[str, str]:
    result = {}
    for item in items:
        if "=" not in item:
            raise HumanPreferenceError(f"metadata must use key=value syntax: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise HumanPreferenceError("metadata keys cannot be empty")
        if key in result:
            raise HumanPreferenceError(f"duplicate metadata key: {key}")
        result[key] = value
    return result


def _display_pair(pair: dict, position: int, total: int, output: Callable[[str], None]) -> None:
    seed = pair["seed"]
    output("")
    output(f"[{position}/{total}] Seed: {seed['artist']} — {seed['name']}")
    for label, side in (("A", "left"), ("B", "right")):
        output(f"\nList {label}")
        for index, track in enumerate(pair[side], start=1):
            output(f"  {index:>2}. {track['artist']} — {track['name']}")


def _validate_vote_file(value: dict, session: dict, context: str = "votes") -> dict:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise HumanPreferenceError(f"{context}.schema_version must be {SCHEMA_VERSION}")
    if value.get("kind") != "blind_human_preference_votes":
        raise HumanPreferenceError(f"{context}.kind must be blind_human_preference_votes")
    if value.get("session_id") != session["session_id"]:
        raise HumanPreferenceError(f"{context} belongs to a different blind session")
    raw_votes = value.get("votes")
    if not isinstance(raw_votes, list):
        raise HumanPreferenceError(f"{context}.votes must be a list")
    known_pairs = {pair["pair_id"]: pair for pair in session["pairs"]}
    seen = set()
    for index, vote in enumerate(raw_votes):
        vote_context = f"{context}.votes[{index}]"
        if not isinstance(vote, dict):
            raise HumanPreferenceError(f"{vote_context} must be an object")
        pair_id = _required_string(vote, "pair_id", vote_context)
        if pair_id not in known_pairs:
            raise HumanPreferenceError(f"{vote_context} references unknown pair {pair_id}")
        if pair_id in seen:
            raise HumanPreferenceError(f"{context} contains duplicate vote for {pair_id}")
        seen.add(pair_id)
        if vote.get("seed_track_id") != known_pairs[pair_id]["seed"]["track_id"]:
            raise HumanPreferenceError(f"{vote_context}.seed_track_id does not match the session")
        if vote.get("choice") not in CHOICES:
            raise HumanPreferenceError(f"{vote_context}.choice must be left, right, or tie")
    metadata = value.get("session_metadata", {})
    if not isinstance(metadata, dict):
        raise HumanPreferenceError(f"{context}.session_metadata must be an object")
    return value


def vote_on_session(
    session: dict,
    *,
    output_path: str | Path,
    evaluator_id: str,
    session_metadata: dict[str, str] | None = None,
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
) -> dict:
    """Run or resume the interactive blind ballot, saving after every answer."""
    session = _validate_session(session)
    evaluator_id = evaluator_id.strip()
    if not evaluator_id:
        raise HumanPreferenceError("evaluator_id must be a non-empty string")
    metadata = session_metadata or {}
    output_path = Path(output_path)
    if output_path.exists():
        votes_file = _validate_vote_file(_load_json(output_path), session)
        if votes_file.get("evaluator_id") != evaluator_id:
            raise HumanPreferenceError("cannot resume with a different evaluator_id")
        if metadata and votes_file.get("session_metadata") != metadata:
            raise HumanPreferenceError("cannot resume with different session metadata")
    else:
        votes_file = {
            "schema_version": SCHEMA_VERSION,
            "kind": "blind_human_preference_votes",
            "session_id": session["session_id"],
            "evaluator_id": evaluator_id,
            "started_at": _utc_now(),
            "completed_at": None,
            "session_metadata": metadata,
            "votes": [],
        }

    answered = {vote["pair_id"] for vote in votes_file["votes"]}
    remaining = [pair for pair in session["pairs"] if pair["pair_id"] not in answered]
    output(
        f"Blind session {session['session_id']}: {len(answered)} answered, "
        f"{len(remaining)} remaining. The model placement key is not loaded."
    )
    for position, pair in enumerate(remaining, start=len(answered) + 1):
        _display_pair(pair, position, len(session["pairs"]), output)
        while True:
            try:
                answer = input_fn(
                    "Prefer [A], [B], [T]ie, [S]kip, or [Q]uit: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                _write_json(output_path, votes_file)
                output(f"Saved partial votes to {output_path}")
                return votes_file
            if answer in {"a", "left"}:
                choice = "left"
                break
            if answer in {"b", "right"}:
                choice = "right"
                break
            if answer in {"t", "tie"}:
                choice = "tie"
                break
            if answer in {"s", "skip"}:
                choice = None
                break
            if answer in {"q", "quit"}:
                _write_json(output_path, votes_file)
                output(f"Saved partial votes to {output_path}")
                return votes_file
            output("Enter A, B, T, S, or Q.")
        if choice is None:
            continue
        votes_file["votes"].append(
            {
                "pair_id": pair["pair_id"],
                "seed_track_id": pair["seed"]["track_id"],
                "choice": choice,
                "voted_at": _utc_now(),
            }
        )
        _write_json(output_path, votes_file)

    if len(votes_file["votes"]) == len(session["pairs"]) and not votes_file["completed_at"]:
        votes_file["completed_at"] = _utc_now()
    _write_json(output_path, votes_file)
    output(f"Saved {len(votes_file['votes'])} votes to {output_path}")
    return votes_file


def _validate_key(value: dict, session: dict) -> dict:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise HumanPreferenceError(f"key.schema_version must be {SCHEMA_VERSION}")
    if value.get("kind") != "blind_human_preference_key":
        raise HumanPreferenceError("key.kind must be blind_human_preference_key")
    if value.get("session_id") != session["session_id"]:
        raise HumanPreferenceError("key belongs to a different blind session")
    if value.get("session_digest") != _digest(session):
        raise HumanPreferenceError("session content does not match the private key")
    models = value.get("models")
    placements = value.get("placements")
    if not isinstance(models, dict) or set(models) != {"model_1", "model_2"}:
        raise HumanPreferenceError("key.models must contain model_1 and model_2")
    for model_key, details in models.items():
        if not isinstance(details, dict) or not isinstance(details.get("model"), dict):
            raise HumanPreferenceError(f"key.models.{model_key}.model must be an object")
        for field in ("model_id", "model_version", "run_id"):
            _required_string(details["model"], field, f"key.models.{model_key}.model")
        automated = details.get("automated_evaluation")
        if not isinstance(automated, dict):
            raise HumanPreferenceError(
                f"key.models.{model_key}.automated_evaluation must be an object"
            )
        for field in ("fixture", "result", "status"):
            _required_string(
                automated, field, f"key.models.{model_key}.automated_evaluation"
            )
        capture = details.get("capture")
        if not isinstance(capture, dict):
            raise HumanPreferenceError(f"key.models.{model_key}.capture must be an object")
    if not isinstance(placements, dict):
        raise HumanPreferenceError("key.placements must be an object")
    pair_ids = {pair["pair_id"] for pair in session["pairs"]}
    if set(placements) != pair_ids:
        raise HumanPreferenceError("key placements do not match session pairs")
    for pair_id, placement in placements.items():
        if not isinstance(placement, dict):
            raise HumanPreferenceError(f"key placement for {pair_id} must be an object")
        if {placement.get("left"), placement.get("right")} != {"model_1", "model_2"}:
            raise HumanPreferenceError(f"key placement for {pair_id} is invalid")
        session_seed = next(
            pair["seed"]["track_id"] for pair in session["pairs"] if pair["pair_id"] == pair_id
        )
        if placement.get("seed_track_id") != session_seed:
            raise HumanPreferenceError(f"key placement seed for {pair_id} is invalid")
    return value


def aggregate_votes(session: dict, vote_files: list[dict], key: dict | None = None) -> dict:
    """Aggregate ballots while remaining blind unless a private key is supplied."""
    session = _validate_session(session)
    votes = [
        _validate_vote_file(value, session, context=f"vote_files[{index}]")
        for index, value in enumerate(vote_files)
    ]
    if not votes:
        raise HumanPreferenceError("at least one vote file is required")
    fingerprints = [_digest(vote_file) for vote_file in votes]
    if len(fingerprints) != len(set(fingerprints)):
        raise HumanPreferenceError("the same ballot was supplied more than once")
    key = _validate_key(key, session) if key is not None else None

    totals = Counter()
    per_pair: dict[str, Counter] = {
        pair["pair_id"]: Counter() for pair in session["pairs"]
    }
    for vote_file in votes:
        for vote in vote_file["votes"]:
            totals[vote["choice"]] += 1
            per_pair[vote["pair_id"]][vote["choice"]] += 1

    recorded = sum(totals.values())
    possible = len(session["pairs"]) * len(votes)
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "blind_human_preference_aggregate",
        "session_id": session["session_id"],
        "study_id": session.get("study_id"),
        "blind": key is None,
        "ballots": len(votes),
        "pairs_per_ballot": len(session["pairs"]),
        "votes_recorded": recorded,
        "votes_possible": possible,
        "completion_rate": round(recorded / possible, 6) if possible else 0.0,
        "anonymous_choices": {
            "list_a": totals["left"],
            "list_b": totals["right"],
            "ties": totals["tie"],
        },
        "ballot_sessions": [
            {
                "evaluator_id": vote_file.get("evaluator_id"),
                "started_at": vote_file.get("started_at"),
                "completed_at": vote_file.get("completed_at"),
                "session_metadata": vote_file.get("session_metadata", {}),
                "votes_recorded": len(vote_file["votes"]),
            }
            for vote_file in votes
        ],
        "per_seed": [],
        "deployment_gate": {
            "human_preference_is_supplemental": True,
            "automated_evaluation_required": True,
        },
    }

    for pair in session["pairs"]:
        counts = per_pair[pair["pair_id"]]
        pair_result = {
            "pair_id": pair["pair_id"],
            "seed_track_id": pair["seed"]["track_id"],
            "anonymous_choices": {
                "list_a": counts["left"],
                "list_b": counts["right"],
                "ties": counts["tie"],
            },
        }
        result["per_seed"].append(pair_result)

    if key is not None:
        model_wins = Counter()
        for vote_file in votes:
            for vote in vote_file["votes"]:
                if vote["choice"] != "tie":
                    model_wins[key["placements"][vote["pair_id"]][vote["choice"]]] += 1
        non_ties = model_wins["model_1"] + model_wins["model_2"]
        result["models"] = []
        for model_key in ("model_1", "model_2"):
            details = key["models"][model_key]
            result["models"].append(
                {
                    **details,
                    "wins": model_wins[model_key],
                    "win_share_excluding_ties": (
                        round(model_wins[model_key] / non_ties, 6) if non_ties else 0.0
                    ),
                }
            )
        result["ties"] = totals["tie"]
        for pair_result in result["per_seed"]:
            placement = key["placements"][pair_result["pair_id"]]
            counts = per_pair[pair_result["pair_id"]]
            pair_result["model_wins"] = {
                model_key: sum(
                    counts[side]
                    for side in ("left", "right")
                    if placement[side] == model_key
                )
                for model_key in ("model_1", "model_2")
            }
        result["deployment_gate"]["automated_evaluations"] = {
            model_key: details["automated_evaluation"]
            for model_key, details in key["models"].items()
        }
    return result


def _seed_entries(value: dict) -> list[dict]:
    raw_seeds = value.get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise HumanPreferenceError("seed fixture must contain a non-empty seeds list")
    result = []
    seen = set()
    for index, raw_seed in enumerate(raw_seeds):
        context = f"seeds[{index}]"
        if not isinstance(raw_seed, dict):
            raise HumanPreferenceError(f"{context} must be an object")
        seed = {
            "track_id": _required_string(raw_seed, "seed_track_id", context),
            "name": _required_string(raw_seed, "name", context),
            "artist": _required_string(raw_seed, "artist", context),
        }
        if seed["track_id"] in seen:
            raise HumanPreferenceError(f"seed fixture repeats {seed['track_id']}")
        seen.add(seed["track_id"])
        result.append(seed)
    return result


def capture_recommendations(
    seeds: list[dict],
    *,
    model: dict,
    automated_evaluation: dict,
    k: int,
    read_only: bool = True,
) -> dict:
    """Snapshot the currently configured recommendation pipeline for later blinding."""
    if k < 1:
        raise HumanPreferenceError("k must be at least 1")
    # Lazy imports let prepare/vote/aggregate run without initializing app config
    # or requiring a database connection.
    from app.config import MMR_LAMBDA
    from app.routers.recommendations import build_recommendations

    recommendations = []

    def capture_loop() -> None:
        for seed in seeds:
            response = build_recommendations(
                seed["track_id"],
                k=k,
                lambda_param=MMR_LAMBDA,
                exclude=[],
                include_tags=False,
            )
            raw_tracks = (
                response.recommendations
                if hasattr(response, "recommendations")
                else response["recommendations"]
            )
            tracks = []
            for track in raw_tracks:
                if hasattr(track, "model_dump"):
                    track = track.model_dump()
                elif not isinstance(track, dict):
                    track = vars(track)
                tracks.append(_public_track(track, f"recommendations for {seed['track_id']}"))
            if not tracks:
                raise HumanPreferenceError(f"no recommendations returned for seed {seed['track_id']}")
            recommendations.append({"seed": seed, "tracks": tracks})

    if read_only:
        def reject_cold_seed(*_args, **_kwargs):
            raise HumanPreferenceError(
                "a seed lacks its configured embedding; read-only capture will not embed it. "
                "Warm the seed deliberately first or use --with-topup against local/staging."
            )

        with (
            patch("app.routers.recommendations.topup_from_lastfm", return_value=[]),
            patch(
                "app.routers.recommendations.ingest.embed_and_store_track",
                side_effect=reject_cold_seed,
            ),
        ):
            capture_loop()
    else:
        capture_loop()

    return _validate_model_export(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "human_preference_model_export",
            "model": model,
            "automated_evaluation": automated_evaluation,
            "capture": {
                "k": k,
                "lambda": float(MMR_LAMBDA),
                "read_only": read_only,
                "seed_digest": _digest(seeds),
            },
            "recommendations": recommendations,
        },
        "capture",
    )


def _add_metadata_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="repeatable metadata field",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blind A/B preference evaluation for recommendations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="snapshot the currently configured model")
    capture.add_argument("--seeds", required=True, help="JSON fixture containing seeds")
    capture.add_argument("--model-id", required=True)
    capture.add_argument("--model-version", required=True)
    capture.add_argument("--run-id", required=True)
    capture.add_argument("--automated-fixture", required=True)
    capture.add_argument("--automated-result", required=True)
    capture.add_argument("--automated-status", choices=sorted(AUTOMATED_STATUSES), required=True)
    capture.add_argument("--k", type=int, default=10)
    capture.add_argument("--env-file", help="dotenv file loaded before app imports")
    capture.add_argument("--with-topup", action="store_true", help="allow Last.fm top-up and DB writes")
    capture.add_argument("--out", required=True)
    capture.add_argument("--force", action="store_true")
    _add_metadata_argument(capture)

    prepare = subparsers.add_parser("prepare", help="create a blind session and private key")
    prepare.add_argument("--model-a", required=True, help="first captured model JSON")
    prepare.add_argument("--model-b", required=True, help="second captured model JSON")
    prepare.add_argument("--study-id", required=True)
    prepare.add_argument("--randomization-seed", required=True)
    prepare.add_argument("--session-out", required=True)
    prepare.add_argument("--key-out", required=True)
    prepare.add_argument("--force", action="store_true")

    vote = subparsers.add_parser("vote", help="run or resume a blind voting session")
    vote.add_argument("--session", required=True)
    vote.add_argument("--evaluator-id", required=True)
    vote.add_argument("--out", required=True)
    _add_metadata_argument(vote)

    aggregate = subparsers.add_parser("aggregate", help="aggregate blind or deblinded ballots")
    aggregate.add_argument("--session", required=True)
    aggregate.add_argument("--votes", nargs="+", required=True)
    aggregate.add_argument("--key", help="private key; omit to keep output blind")
    aggregate.add_argument("--out", help="write JSON here; defaults to stdout")
    aggregate.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            if args.env_file:
                from dotenv import load_dotenv

                load_dotenv(args.env_file, override=True)
            seed_fixture = _load_json(args.seeds)
            result = capture_recommendations(
                _seed_entries(seed_fixture),
                model={
                    "model_id": args.model_id,
                    "model_version": args.model_version,
                    "run_id": args.run_id,
                    "metadata": _parse_metadata(args.metadata),
                },
                automated_evaluation={
                    "fixture": args.automated_fixture,
                    "result": args.automated_result,
                    "status": args.automated_status,
                },
                k=args.k,
                read_only=not args.with_topup,
            )
            _write_json(args.out, result, overwrite=args.force)
            print(f"Captured {len(result['recommendations'])} seeds to {args.out}")
            return 0

        if args.command == "prepare":
            if Path(args.session_out).resolve() == Path(args.key_out).resolve():
                raise HumanPreferenceError("session and key outputs must be different files")
            if not args.force:
                existing = [
                    path
                    for path in (Path(args.session_out), Path(args.key_out))
                    if path.exists()
                ]
                if existing:
                    raise HumanPreferenceError(
                        f"refusing to overwrite {existing[0]}; pass --force to replace it"
                    )
            session, key = prepare_session(
                _load_json(args.model_a),
                _load_json(args.model_b),
                study_id=args.study_id,
                randomization_seed=args.randomization_seed,
            )
            _write_json(args.session_out, session, overwrite=args.force)
            _write_json(args.key_out, key, overwrite=args.force)
            print(f"Prepared {len(session['pairs'])} blind pairs in {args.session_out}")
            print(f"Keep the model placement key private during voting: {args.key_out}")
            return 0

        if args.command == "vote":
            vote_on_session(
                _load_json(args.session),
                output_path=args.out,
                evaluator_id=args.evaluator_id,
                session_metadata=_parse_metadata(args.metadata),
            )
            return 0

        if args.command == "aggregate":
            session = _load_json(args.session)
            key = _load_json(args.key) if args.key else None
            result = aggregate_votes(
                session,
                [_load_json(path) for path in args.votes],
                key=key,
            )
            if args.out:
                _write_json(args.out, result, overwrite=args.force)
                print(f"Wrote {'deblinded' if key else 'blind'} aggregate to {args.out}")
            else:
                print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
    except (HumanPreferenceError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
