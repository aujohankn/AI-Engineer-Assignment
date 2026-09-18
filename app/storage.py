import asyncio
from io import BytesIO
from pathlib import Path

from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError


class InvalidImageError(ValueError):
    pass


class UploadTooLargeError(ValueError):
    pass


class ArtifactStore:
    def __init__(self, root: Path, max_upload_bytes: int, archive_root: Path | None = None) -> None:
        self.root = root.resolve()
        self.max_upload_bytes = max_upload_bytes
        self.archive_root = archive_root.resolve() if archive_root else None

    def job_dir(self, job_id: str) -> Path:
        path = (self.root / job_id).resolve()
        if self.root not in path.parents:
            raise ValueError("invalid job id")
        return path

    async def save_upload(self, job_id: str, upload: UploadFile) -> tuple[Path, tuple[int, int]]:
        data = bytearray()
        while chunk := await upload.read(1024 * 1024):
            data.extend(chunk)
            if len(data) > self.max_upload_bytes:
                raise UploadTooLargeError(
                    f"image exceeds {self.max_upload_bytes // (1024 * 1024)} MiB limit"
                )

        try:
            with Image.open(BytesIO(data)) as image:
                image.verify()
            with Image.open(BytesIO(data)) as image:
                dimensions = image.size
                converted = ImageOps.exif_transpose(image).convert("RGBA")
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("uploaded file is not a valid image") from exc

        directory = self.job_dir(job_id)
        await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=False)
        destination = directory / "input.png"
        await asyncio.to_thread(converted.save, destination, "PNG")
        return destination, dimensions

    def attempt_path(self, job_id: str, variant_index: int, attempt: int) -> Path:
        directory = self.job_dir(job_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"variant-{variant_index + 1}-attempt-{attempt}.png"

    def save_attempt(
        self, job_id: str, variant_index: int, attempt: int, image_bytes: bytes
    ) -> Path:
        path = self.attempt_path(job_id, variant_index, attempt)
        path.write_bytes(image_bytes)
        if self.archive_root is not None:
            archive_directory = (self.archive_root / job_id).resolve()
            if self.archive_root not in archive_directory.parents:
                raise ValueError("invalid job id")
            archive_directory.mkdir(parents=True, exist_ok=True)
            (archive_directory / path.name).write_bytes(image_bytes)
        return path

    @staticmethod
    def normalize_to_canvas(image_bytes: bytes, target_size: tuple[int, int]) -> bytes:
        """Return a PNG with exact target dimensions while avoiding geometric distortion."""
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                normalized = ImageOps.fit(
                    image,
                    target_size,
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5),
                )
                output = BytesIO()
                normalized.save(output, "PNG", optimize=True)
                return output.getvalue()
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("model returned an invalid image") from exc

    @staticmethod
    def dimensions(path: Path) -> tuple[int, int]:
        with Image.open(path) as image:
            return image.size


def image_data_url(path: Path) -> str:
    import base64

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
