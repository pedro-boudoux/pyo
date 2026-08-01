from contextlib import contextmanager
from pathlib import Path

import pytest

from jobs import train_colisten_embeddings as trainer


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_pipeline_holds_one_lock_through_publication(monkeypatch):
    events = []

    @contextmanager
    def lock():
        events.append("lock-enter")
        yield
        events.append("lock-exit")

    def train_candidate(**kwargs):
        events.append("train")
        return {"run_id": 17, "status": "candidate"}

    def validate(run_id, **kwargs):
        events.append(f"validate-{run_id}")
        return {"run_id": run_id, "passed": True}

    def publish(run_id, *, acquire_lock=True):
        events.append(f"publish-{run_id}-{acquire_lock}")
        return {"run_id": run_id, "status": "active"}

    monkeypatch.setattr(trainer, "model_lock", lock)
    monkeypatch.setattr(trainer, "_train_candidate", train_candidate)
    monkeypatch.setattr(trainer, "validate", validate)
    monkeypatch.setattr(trainer, "publish", publish)

    result = trainer.run_pipeline(training={}, validation={})

    assert result["publication"]["status"] == "active"
    assert events == [
        "lock-enter",
        "train",
        "validate-17",
        "publish-17-False",
        "lock-exit",
    ]


def test_run_pipeline_stops_before_publish_on_failed_gate(monkeypatch):
    @contextmanager
    def lock():
        yield

    monkeypatch.setattr(trainer, "model_lock", lock)
    monkeypatch.setattr(
        trainer,
        "_train_candidate",
        lambda **kwargs: {"run_id": 17, "status": "candidate"},
    )
    monkeypatch.setattr(
        trainer,
        "validate",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failed gate")),
    )
    monkeypatch.setattr(
        trainer,
        "publish",
        lambda *args, **kwargs: pytest.fail("publish must not run after failed validation"),
    )

    with pytest.raises(RuntimeError, match="failed gate"):
        trainer.run_pipeline(training={}, validation={})


def test_training_image_stays_alive_for_scheduled_execs():
    dockerfile = (REPO_ROOT / "Dockerfile.training").read_text()
    assert 'CMD ["sleep", "infinity"]' in dockerfile
    assert "ENTRYPOINT" not in dockerfile
