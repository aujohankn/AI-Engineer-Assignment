from pathlib import Path

import pytest

import app.worker as worker
from app.models import JobStatus, VariantResult
from tests.helpers import passing_evaluation, sample_job


class FakeStore:
    def __init__(self, job=None) -> None:
        self.job = job
        self.saved = []

    def get(self, job_id: str):
        return self.job

    def save(self, job) -> None:
        self.saved.append(job.model_copy(deep=True))


class Workflow:
    def __init__(self, variants=None, error: Exception | None = None) -> None:
        self.variants = variants
        self.error = error

    def run(self, job):
        if self.error:
            raise self.error
        return self.variants


def configure(monkeypatch: pytest.MonkeyPatch, store: FakeStore, workflow: Workflow) -> None:
    monkeypatch.setattr(worker.Redis, "from_url", lambda *args, **kwargs: object())
    monkeypatch.setattr(worker, "SyncRedisJobStore", lambda *args, **kwargs: store)
    monkeypatch.setattr(worker, "build_workflow", lambda settings: workflow)


def variant(job_id: str, *, passed: bool) -> VariantResult:
    return VariantResult(
        id="variant-1",
        image_url=f"/v1/jobs/{job_id}/variants/variant-1/image",
        selected_attempt=1,
        attempts=1,
        evaluation=passing_evaluation(recommendation=passed, brand=True),
    )


def test_worker_rejects_missing_job(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeStore()
    configure(monkeypatch, store, Workflow([]))
    with pytest.raises(ValueError, match="does not exist"):
        worker.process_job.run("missing")


@pytest.mark.parametrize(
    ("passed", "expected"),
    [(True, JobStatus.succeeded), (False, JobStatus.partially_succeeded)],
)
def test_worker_saves_success_or_partial_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, passed: bool, expected: JobStatus
) -> None:
    job = sample_job(tmp_path / "input.png")
    store = FakeStore(job)
    configure(monkeypatch, store, Workflow([variant(job.id, passed=passed)]))

    result = worker.process_job.run(job.id)

    assert result["status"] == expected.value
    assert store.saved[0].status is JobStatus.running
    assert store.saved[-1].status is expected
    assert result["variants"] == 1


def test_worker_persists_unexpected_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = sample_job(tmp_path / "input.png")
    store = FakeStore(job)
    configure(monkeypatch, store, Workflow(error=ValueError("bad workflow")))

    with pytest.raises(ValueError, match="bad workflow"):
        worker.process_job.run(job.id)

    assert store.saved[-1].status is JobStatus.failed
    assert "ValueError: bad workflow" in store.saved[-1].error


def test_worker_does_not_retry_exhausted_quota(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class QuotaError(Exception):
        body = {"type": "insufficient_quota", "code": "credit_balance_exhausted"}

    monkeypatch.setattr(worker, "RateLimitError", QuotaError)
    job = sample_job(tmp_path / "input.png")
    store = FakeStore(job)
    configure(monkeypatch, store, Workflow(error=QuotaError("no credits")))

    with pytest.raises(QuotaError):
        worker.process_job.run(job.id)

    assert store.saved[-1].status is JobStatus.failed
    assert "quota exhausted" in store.saved[-1].error


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"error": {"code": "credit_balance_exhausted"}}, True),
        ({"type": "insufficient_quota"}, True),
        ({"code": "rate_limit_exceeded"}, False),
        ("not-a-dict", False),
        ({"error": "not-a-dict"}, False),
    ],
)
def test_quota_error_classification(body, expected: bool) -> None:
    error = type("Error", (), {"body": body})()
    assert worker._is_quota_exhausted(error) is expected
