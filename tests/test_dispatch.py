from app.dispatch import CeleryDispatcher


def test_celery_dispatcher_enqueues_job(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("app.worker.process_job.delay", calls.append)
    CeleryDispatcher().dispatch("job-1")
    assert calls == ["job-1"]
