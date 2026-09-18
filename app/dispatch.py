from typing import Protocol


class JobDispatcher(Protocol):
    def dispatch(self, job_id: str) -> None: ...


class CeleryDispatcher:
    def dispatch(self, job_id: str) -> None:
        from app.worker import process_job

        process_job.delay(job_id)
