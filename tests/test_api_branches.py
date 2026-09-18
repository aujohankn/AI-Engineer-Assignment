import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from redis.exceptions import RedisError

import app.api as api_module
from app.api import create_app
from app.config import Settings
from app.models import JobStatus, VariantResult
from app.store import InMemoryJobStore
from tests.helpers import passing_evaluation, sample_job


class Dispatcher:
    def dispatch(self, job_id: str) -> None:
        pass


class RedisClientFake:
    def __init__(self, *, fail_ping: bool = False) -> None:
        self.fail_ping = fail_ping
        self.closed = False

    async def ping(self) -> bool:
        if self.fail_ping:
            raise RedisError("offline")
        return True

    async def aclose(self) -> None:
        self.closed = True

    async def get(self, key: str) -> None:
        return None

    async def set(self, *args: object, **kwargs: object) -> bool:
        return True


def valid_data() -> dict[str, str]:
    return {
        "recommendations": json.dumps(
            [{"id": "r1", "title": "Contrast", "description": "Increase it", "type": "contrast"}]
        ),
        "brand_guidelines": "{}",
        "variants": "1",
    }


def test_health_with_real_store_lifecycle_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = RedisClientFake()
    monkeypatch.setattr(api_module.redis, "from_url", lambda *args, **kwargs: client)
    app = create_app(settings=Settings(artifact_root=tmp_path), dispatcher=Dispatcher())
    with TestClient(app) as http:
        assert http.get("/health/live").json() == {"status": "ok"}
        assert http.get("/health/ready").json() == {"status": "ready"}
    assert client.closed is True


def test_ready_returns_503_when_redis_is_offline(tmp_path: Path) -> None:
    app = create_app(
        settings=Settings(artifact_root=tmp_path),
        store=InMemoryJobStore(),
        dispatcher=Dispatcher(),
    )
    with TestClient(app) as http:
        app.state.redis_client = RedisClientFake(fail_ping=True)
        response = http.get("/health/ready")
        app.state.redis_client = None
    assert response.status_code == 503


def test_job_validation_limits(tmp_path: Path) -> None:
    app = create_app(
        settings=Settings(artifact_root=tmp_path, max_variants=1, max_upload_bytes=1024),
        store=InMemoryJobStore(),
        dispatcher=Dispatcher(),
    )
    with TestClient(app) as http:
        too_many = http.post(
            "/v1/jobs",
            files={"image": ("x.png", b"anything", "image/png")},
            data={**valid_data(), "variants": "2"},
        )
        empty = http.post(
            "/v1/jobs",
            files={"image": ("x.png", b"anything", "image/png")},
            data={**valid_data(), "recommendations": "[]"},
        )
        oversized = http.post(
            "/v1/jobs",
            files={"image": ("x.png", b"x" * 1025, "image/png")},
            data=valid_data(),
        )
    assert too_many.status_code == 422
    assert empty.status_code == 422
    assert oversized.status_code == 413


def test_job_and_variant_not_found(tmp_path: Path) -> None:
    store = InMemoryJobStore()
    app = create_app(
        settings=Settings(artifact_root=tmp_path), store=store, dispatcher=Dispatcher()
    )
    job = sample_job(tmp_path / "input.png")
    store.jobs[job.id] = job
    with TestClient(app) as http:
        missing_job = http.get("/v1/jobs/00000000-0000-0000-0000-000000000099")
        missing_variant = http.get(f"/v1/jobs/{job.id}/variants/missing/image")
    assert missing_job.status_code == 404
    assert missing_variant.status_code == 404


def test_variant_download_success_and_missing_artifact(tmp_path: Path) -> None:
    store = InMemoryJobStore()
    app = create_app(
        settings=Settings(artifact_root=tmp_path), store=store, dispatcher=Dispatcher()
    )
    job = sample_job(tmp_path / "input.png")
    job.status = JobStatus.succeeded
    job.variants = [
        VariantResult(
            id="variant-1",
            image_url=f"/v1/jobs/{job.id}/variants/variant-1/image",
            selected_attempt=1,
            attempts=1,
            evaluation=passing_evaluation(),
        )
    ]
    store.jobs[job.id] = job
    directory = tmp_path / job.id
    directory.mkdir()
    artifact = directory / "variant-1-attempt-1.png"

    with TestClient(app) as http:
        missing = http.get(f"/v1/jobs/{job.id}/variants/variant-1/image")
        artifact.write_bytes(b"png-content")
        found = http.get(f"/v1/jobs/{job.id}/variants/variant-1/image")

    assert missing.status_code == 404
    assert found.status_code == 200
    assert found.content == b"png-content"
