from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.config import get_settings
from app.local_models import SDXLImageGenerator

app = FastAPI(
    title="Local SDXL Inference API",
    version="1.0.0",
    description="GPU-backed image-to-image API used by the creative workflow backend.",
)
_generation_lock = Lock()


@app.get("/health/ready")
def readiness() -> dict[str, object]:
    try:
        import torch

        available = torch.cuda.is_available()
    except ImportError:
        available = False
    if not available:
        raise HTTPException(503, "CUDA is unavailable")
    return {"status": "ready", "cuda": True}


@app.post("/v1/generate", response_class=Response)
def generate(
    image: Annotated[UploadFile, File(description="Source creative")],
    prompt: Annotated[str, Form(min_length=1)],
) -> Response:
    suffix = Path(image.filename or "source.png").suffix or ".png"
    source_path: Path | None = None
    try:
        with NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(image.file.read())
            source_path = Path(temporary.name)
        with _generation_lock:
            output = SDXLImageGenerator(get_settings()).generate(source_path, prompt)
        return Response(content=output, media_type="image/png")
    except Exception as exc:
        raise HTTPException(503, f"SDXL generation failed: {type(exc).__name__}: {exc}") from exc
    finally:
        if source_path is not None:
            source_path.unlink(missing_ok=True)
