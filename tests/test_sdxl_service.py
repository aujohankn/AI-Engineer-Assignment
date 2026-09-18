import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import app.sdxl_service as sdxl_service


def test_readiness_reports_cuda(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    with TestClient(sdxl_service.app) as client:
        response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "cuda": True}


def test_readiness_fails_without_cuda(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    with TestClient(sdxl_service.app) as client:
        response = client.get("/health/ready")
    assert response.status_code == 503


def test_generate_returns_png_and_removes_temporary_file(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Generator:
        def __init__(self, settings) -> None:
            captured["settings"] = settings

        def generate(self, source: Path, prompt: str) -> bytes:
            captured["path"] = source
            captured["prompt"] = prompt
            assert source.exists()
            assert source.read_bytes() == b"source-image"
            return b"png-result"

    monkeypatch.setattr(sdxl_service, "SDXLImageGenerator", Generator)
    with TestClient(sdxl_service.app) as client:
        response = client.post(
            "/v1/generate",
            files={"image": ("creative.png", b"source-image", "image/png")},
            data={"prompt": "make it vivid"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == b"png-result"
    assert captured["prompt"] == "make it vivid"
    assert not captured["path"].exists()


def test_generate_reports_inference_failure(monkeypatch) -> None:
    class BrokenGenerator:
        def __init__(self, settings) -> None:
            pass

        def generate(self, source: Path, prompt: str) -> bytes:
            raise RuntimeError("out of memory")

    monkeypatch.setattr(sdxl_service, "SDXLImageGenerator", BrokenGenerator)
    with TestClient(sdxl_service.app) as client:
        response = client.post(
            "/v1/generate",
            files={"image": ("creative.png", b"source", "image/png")},
            data={"prompt": "edit"},
        )

    assert response.status_code == 503
    assert "out of memory" in response.json()["detail"]
