from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.sdxl_client import SDXLAPIImageGenerator


def _generator(handler) -> SDXLAPIImageGenerator:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return SDXLAPIImageGenerator(
        Settings(sdxl_api_url="http://inference:9000", sdxl_api_timeout_seconds=12),
        client=client,
    )


def test_sdxl_api_generator_posts_multipart_and_returns_image(tmp_path: Path) -> None:
    source = tmp_path / "creative.png"
    source.write_bytes(b"source-image")

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://inference:9000/v1/generate"
        body = request.read()
        assert b"source-image" in body
        assert b"make it vivid" in body
        return httpx.Response(200, content=b"png-result", headers={"content-type": "image/png"})

    assert _generator(handler).generate(source, "make it vivid") == b"png-result"


def test_sdxl_api_generator_surfaces_provider_error(tmp_path: Path) -> None:
    source = tmp_path / "creative.png"
    source.write_bytes(b"source")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="GPU unavailable")

    with pytest.raises(RuntimeError, match="HTTP 503.*GPU unavailable"):
        _generator(handler).generate(source, "edit")


def test_sdxl_api_generator_rejects_non_image_response(tmp_path: Path) -> None:
    source = tmp_path / "creative.png"
    source.write_bytes(b"source")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{}", headers={"content-type": "application/json"})

    with pytest.raises(RuntimeError, match="no image data"):
        _generator(handler).generate(source, "edit")
