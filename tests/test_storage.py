from io import BytesIO

from PIL import Image

from app.storage import ArtifactStore


def png(size: tuple[int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, "purple").save(buffer, "PNG")
    return buffer.getvalue()


def test_normalize_to_canvas_has_exact_dimensions_without_stretching() -> None:
    result = ArtifactStore.normalize_to_canvas(png((1024, 1536)), (636, 1063))
    with Image.open(BytesIO(result)) as image:
        assert image.size == (636, 1063)
        assert image.format == "PNG"
