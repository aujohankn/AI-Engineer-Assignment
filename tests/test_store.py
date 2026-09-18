import asyncio
from pathlib import Path

from app.models import JobStatus
from app.store import InMemoryJobStore, RedisJobStore, SyncRedisJobStore
from tests.helpers import sample_job


class AsyncRedisFake:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def set(self, key: str, value: str, **_: object) -> bool:
        self.values[key] = value
        return True

    async def get(self, key: str) -> str | None:
        return self.values.get(key)


class SyncRedisFake:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def set(self, key: str, value: str, **_: object) -> bool:
        self.values[key] = value
        return True

    def get(self, key: str) -> str | None:
        return self.values.get(key)


def test_async_redis_store_round_trip_and_missing(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = AsyncRedisFake()
        store = RedisJobStore(client, 60)
        assert await store.get("missing") is None
        job = sample_job(tmp_path / "input.png")
        await store.create(job)
        loaded = await store.get(job.id)
        assert loaded is not None
        assert loaded.id == job.id

    asyncio.run(scenario())


def test_sync_redis_store_round_trip_and_update(tmp_path: Path) -> None:
    client = SyncRedisFake()
    store = SyncRedisJobStore(client, 60)
    assert store.get("missing") is None
    job = sample_job(tmp_path / "input.png")
    job.status = JobStatus.running
    store.save(job)
    loaded = store.get(job.id)
    assert loaded is not None
    assert loaded.status is JobStatus.running


def test_memory_store_returns_defensive_copies(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = InMemoryJobStore()
        job = sample_job(tmp_path / "input.png")
        await store.create(job)
        loaded = await store.get(job.id)
        assert loaded is not None
        loaded.status = JobStatus.failed
        loaded_again = await store.get(job.id)
        assert loaded_again is not None
        assert loaded_again.status is JobStatus.queued
        assert await store.get("missing") is None

    asyncio.run(scenario())
