from pathlib import Path

import httpx

from app.config import Settings


class SDXLAPIImageGenerator:
    """Image generator adapter for the self-hosted SDXL HTTP service."""

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.url = f"{settings.sdxl_api_url.rstrip('/')}/v1/generate"
        self.client = client or httpx.Client(timeout=settings.sdxl_api_timeout_seconds)

    def generate(self, source: Path, prompt: str) -> bytes:
        with source.open("rb") as image:
            response = self.client.post(
                self.url,
                files={"image": (source.name, image, "image/png")},
                data={"prompt": prompt},
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text[:500]
            raise RuntimeError(
                f"SDXL API returned HTTP {response.status_code}: {detail}"
            ) from exc
        content_type = response.headers.get("content-type", "")
        if not response.content or not content_type.startswith("image/"):
            raise RuntimeError("SDXL API returned no image data")
        return response.content
