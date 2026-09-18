from typing import Protocol

import redis.asyncio as async_redis
from redis import Redis

from app.models import JobRecord, utc_now


def _key(job_id: str) -> str:
    return f"creative-job:{job_id}"


class AsyncJobStore(Protocol):
    async def create(self, job: JobRecord) -> None: ...
    async def get(self, job_id: str) -> JobRecord | None: ...


class RedisJobStore:
    def __init__(self, client: async_redis.Redis, ttl_seconds: int) -> None:
        self.client = client
        self.ttl_seconds = ttl_seconds

    async def create(self, job: JobRecord) -> None:
        await self.client.set(_key(job.id), job.model_dump_json(), ex=self.ttl_seconds, nx=True)

    async def get(self, job_id: str) -> JobRecord | None:
        value = await self.client.get(_key(job_id))
        return JobRecord.model_validate_json(value) if value else None


class SyncRedisJobStore:
    def __init__(self, client: Redis, ttl_seconds: int) -> None:
        self.client = client
        self.ttl_seconds = ttl_seconds

    def get(self, job_id: str) -> JobRecord | None:
        value = self.client.get(_key(job_id))
        return JobRecord.model_validate_json(value) if value else None

    def save(self, job: JobRecord) -> None:
        job.updated_at = utc_now()
        self.client.set(_key(job.id), job.model_dump_json(), ex=self.ttl_seconds)


class InMemoryJobStore:
    """Small test adapter; production always uses Redis."""

    def __init__(self) -> None:
        self.jobs: dict[str, JobRecord] = {}

    async def create(self, job: JobRecord) -> None:
        self.jobs[job.id] = job.model_copy(deep=True)

    async def get(self, job_id: str) -> JobRecord | None:
        job = self.jobs.get(job_id)
        return job.model_copy(deep=True) if job else None
