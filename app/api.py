import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID, uuid4

import redis.asyncio as redis
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import TypeAdapter, ValidationError
from redis.exceptions import RedisError
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.dispatch import CeleryDispatcher, JobDispatcher
from app.models import BrandGuidelines, JobAccepted, JobRecord, JobView, Recommendation
from app.storage import ArtifactStore, InvalidImageError, UploadTooLargeError
from app.store import AsyncJobStore, RedisJobStore


def create_app(
    *,
    settings: Settings | None = None,
    store: AsyncJobStore | None = None,
    dispatcher: JobDispatcher | None = None,
) -> FastAPI:
    config = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if store is None:
            client = redis.from_url(config.redis_url, decode_responses=True)
            app.state.redis_client = client
            app.state.job_store = RedisJobStore(client, config.job_ttl_seconds)
        else:
            app.state.redis_client = None
            app.state.job_store = store
        app.state.dispatcher = dispatcher or CeleryDispatcher()
        app.state.artifacts = ArtifactStore(config.artifact_root, config.max_upload_bytes)
        yield
        if app.state.redis_client is not None:
            await app.state.redis_client.aclose()

    app = FastAPI(
        title=config.app_name,
        version="0.1.0",
        description="Generate and evaluate brand-compliant marketing creative variants.",
        lifespan=lifespan,
    )

    @app.exception_handler(UploadTooLargeError)
    async def upload_too_large(_: Request, exc: UploadTooLargeError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"detail": str(exc)},
        )

    @app.exception_handler(InvalidImageError)
    async def invalid_image(_: Request, exc: InvalidImageError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc)},
        )

    @app.get("/health/live", tags=["health"])
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    async def ready(request: Request) -> dict[str, str]:
        client = request.app.state.redis_client
        if client is not None:
            try:
                await client.ping()
            except RedisError as exc:
                raise HTTPException(status_code=503, detail="Redis unavailable") from exc
        return {"status": "ready"}

    @app.post(
        "/v1/jobs",
        response_model=JobAccepted,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["jobs"],
    )
    async def create_job(
        request: Request,
        image: Annotated[UploadFile, File(description="Marketing creative")],
        recommendations: Annotated[str, Form(description="JSON array of recommendations")],
        brand_guidelines: Annotated[str, Form(description="JSON object of brand guidelines")],
        variants: Annotated[int, Form(ge=1)] = 1,
    ) -> JobAccepted:
        if variants > config.max_variants:
            raise HTTPException(422, f"variants must be at most {config.max_variants}")
        try:
            parsed_recommendations = TypeAdapter(list[Recommendation]).validate_python(
                json.loads(recommendations)
            )
            parsed_guidelines = BrandGuidelines.model_validate(json.loads(brand_guidelines))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise HTTPException(422, f"invalid workflow input: {exc}") from exc
        if not parsed_recommendations:
            raise HTTPException(422, "at least one recommendation is required")

        job_id = str(uuid4())
        artifacts: ArtifactStore = request.app.state.artifacts
        input_path, _ = await artifacts.save_upload(job_id, image)
        job = JobRecord(
            id=job_id,
            input_filename=image.filename or "creative.png",
            input_path=str(input_path),
            recommendations=parsed_recommendations,
            brand_guidelines=parsed_guidelines,
            requested_variants=variants,
        )
        job_store: AsyncJobStore = request.app.state.job_store
        await job_store.create(job)
        await run_in_threadpool(request.app.state.dispatcher.dispatch, job_id)
        return JobAccepted(
            job_id=job_id,
            status=job.status,
            status_url=str(request.url_for("get_job", job_id=job_id)),
        )

    @app.get(
        "/v1/jobs/{job_id}",
        response_model=JobView,
        tags=["jobs"],
    )
    async def get_job(job_id: UUID, request: Request) -> JobView:
        job_store: AsyncJobStore = request.app.state.job_store
        job = await job_store.get(str(job_id))
        if job is None:
            raise HTTPException(404, "job not found")
        return JobView.model_validate(job)

    @app.get(
        "/v1/jobs/{job_id}/variants/{variant_id}/image",
        response_class=FileResponse,
        tags=["jobs"],
    )
    async def get_variant_image(job_id: UUID, variant_id: str, request: Request) -> FileResponse:
        job_store: AsyncJobStore = request.app.state.job_store
        job = await job_store.get(str(job_id))
        if job is None:
            raise HTTPException(404, "job not found")
        variant = next((item for item in job.variants if item.id == variant_id), None)
        if variant is None:
            raise HTTPException(404, "variant not found")
        filename = f"{variant.id}-attempt-{variant.selected_attempt}.png"
        path = request.app.state.artifacts.job_dir(str(job_id)) / filename
        if not path.is_file():
            raise HTTPException(404, "variant artifact not found")
        return FileResponse(path, media_type="image/png", filename=f"{variant_id}.png")

    return app


app = create_app()
