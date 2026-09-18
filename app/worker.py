from celery import Celery
from openai import APIConnectionError, APITimeoutError, RateLimitError
from redis import Redis

from app.agents import build_workflow
from app.config import get_settings
from app.gemini import GeminiTransientError
from app.models import JobStatus
from app.store import SyncRedisJobStore

settings = get_settings()
celery_app = Celery(
    "creative_variant_service", broker=settings.redis_url, backend=settings.redis_url
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    broker_connection_retry_on_startup=True,
)


def _is_quota_exhausted(exc: RateLimitError) -> bool:
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return False
    error = body.get("error", body)
    if not isinstance(error, dict):
        return False
    return error.get("type") == "insufficient_quota" or error.get("code") in {
        "credit_balance_exhausted",
        "insufficient_quota",
    }


@celery_app.task(bind=True, max_retries=2, name="creative.process_job")
def process_job(self, job_id: str) -> dict[str, str | int]:
    store = SyncRedisJobStore(
        Redis.from_url(settings.redis_url, decode_responses=True), settings.job_ttl_seconds
    )
    job = store.get(job_id)
    if job is None:
        raise ValueError(f"job {job_id} does not exist")

    job.status = JobStatus.running
    job.error = None
    store.save(job)
    try:
        job.variants = build_workflow(settings).run(job)
        all_passed = all(
            item.evaluation.recommendation_applied and item.evaluation.brand_compliant
            for item in job.variants
        )
        job.status = JobStatus.succeeded if all_passed else JobStatus.partially_succeeded
        store.save(job)
        return {"job_id": job_id, "variants": len(job.variants), "status": job.status.value}
    except RateLimitError as exc:
        if _is_quota_exhausted(exc):
            job.status = JobStatus.failed
            job.error = (
                "OpenAI API quota exhausted; add project credits at "
                "https://platform.openai.com/settings/organization/billing/"
            )
            store.save(job)
            raise
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=2 ** (self.request.retries + 1)) from exc
        job.status = JobStatus.failed
        job.error = "OpenAI API rate limit persisted after retries"
        store.save(job)
        raise
    except (APIConnectionError, APITimeoutError) as exc:
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=2 ** (self.request.retries + 1)) from exc
        job.status = JobStatus.failed
        job.error = f"model provider temporarily unavailable: {type(exc).__name__}"
        store.save(job)
        raise
    except GeminiTransientError as exc:
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=2 ** (self.request.retries + 1)) from exc
        job.status = JobStatus.failed
        job.error = "Gemini API remained unavailable after retries"
        store.save(job)
        raise
    except Exception as exc:
        job.status = JobStatus.failed
        job.error = f"workflow failed: {type(exc).__name__}: {exc}"
        store.save(job)
        raise
