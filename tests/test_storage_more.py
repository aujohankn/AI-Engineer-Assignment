import asyncio
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile
from PIL import Image

from app.storage import ArtifactStore, InvalidImageError, UploadTooLargeError, image_data_url


def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (10, 12), "green").save(output, "PNG")
    return output.getvalue()


def test_save_upload_and_artifact_paths(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = ArtifactStore(tmp_path, 10_000)
        upload = UploadFile(filename="input.png", file=BytesIO(png_bytes()))
        path, dimensions = await store.save_upload("job-1", upload)
        assert dimensions == (10, 12)
        assert path.is_file()
        assert store.dimensions(path) == (10, 12)
        assert store.attempt_path("job-1", 1, 2).name == "variant-2-attempt-2.png"
        assert image_data_url(path).startswith("data:image/png;base64,")

    asyncio.run(scenario())


def test_storage_rejects_bad_paths_large_and_invalid_images(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, 10)
    with pytest.raises(ValueError, match="invalid job id"):
        store.job_dir("../escape")

    async def large() -> None:
        upload = UploadFile(filename="large.png", file=BytesIO(b"x" * 11))
        with pytest.raises(UploadTooLargeError):
            await store.save_upload("large", upload)

    async def invalid() -> None:
        upload = UploadFile(filename="bad.png", file=BytesIO(b"bad"))
        with pytest.raises(InvalidImageError):
            await ArtifactStore(tmp_path, 100).save_upload("bad", upload)

    asyncio.run(large())
    asyncio.run(invalid())


def test_normalize_rejects_invalid_model_output() -> None:
    with pytest.raises(InvalidImageError, match="model returned"):
        ArtifactStore.normalize_to_canvas(b"not an image", (10, 10))
