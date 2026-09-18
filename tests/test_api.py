import json
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.api import create_app
from app.config import Settings
from app.store import InMemoryJobStore


class RecordingDispatcher:
    def __init__(self) -> None:
        self.job_ids: list[str] = []

    def dispatch(self, job_id: str) -> None:
        self.job_ids.append(job_id)


def image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 48), "blue").save(buffer, "PNG")
    return buffer.getvalue()


def test_create_and_get_job(tmp_path: Path) -> None:
    store = InMemoryJobStore()
    dispatcher = RecordingDispatcher()
    settings = Settings(artifact_root=tmp_path, max_variants=2)
    app = create_app(settings=settings, store=store, dispatcher=dispatcher)
    recommendation = [
        {"id": "r1", "title": "More contrast", "description": "Increase it", "type": "contrast"}
    ]
    guidelines = {"protected_regions": ["Keep logo"]}

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs",
            files={"image": ("creative.png", image_bytes(), "image/png")},
            data={
                "recommendations": json.dumps(recommendation),
                "brand_guidelines": json.dumps(guidelines),
                "variants": "2",
            },
        )
        assert response.status_code == 202
        payload = response.json()
        assert payload["status"] == "queued"
        assert dispatcher.job_ids == [payload["job_id"]]

        status_response = client.get(f"/v1/jobs/{payload['job_id']}")
        assert status_response.status_code == 200
        assert "input_path" not in status_response.json()
        assert status_response.json()["requested_variants"] == 2


def test_rejects_invalid_json_before_dispatch(tmp_path: Path) -> None:
    dispatcher = RecordingDispatcher()
    app = create_app(
        settings=Settings(artifact_root=tmp_path),
        store=InMemoryJobStore(),
        dispatcher=dispatcher,
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs",
            files={"image": ("creative.png", image_bytes(), "image/png")},
            data={"recommendations": "not-json", "brand_guidelines": "{}", "variants": "1"},
        )
    assert response.status_code == 422
    assert dispatcher.job_ids == []


def test_rejects_non_image_upload(tmp_path: Path) -> None:
    dispatcher = RecordingDispatcher()
    app = create_app(
        settings=Settings(artifact_root=tmp_path),
        store=InMemoryJobStore(),
        dispatcher=dispatcher,
    )
    recommendation = [
        {"id": "r1", "title": "Contrast", "description": "Increase it", "type": "contrast"}
    ]
    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs",
            files={"image": ("creative.txt", b"not an image", "text/plain")},
            data={
                "recommendations": json.dumps(recommendation),
                "brand_guidelines": "{}",
                "variants": "1",
            },
        )
    assert response.status_code == 422
    assert dispatcher.job_ids == []
