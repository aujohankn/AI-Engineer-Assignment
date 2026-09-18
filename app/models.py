from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class Recommendation(BaseModel):
    id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1, max_length=4000)
    type: str = Field(min_length=1, max_length=100)


class BrandGuidelines(BaseModel):
    protected_regions: list[str] = Field(default_factory=list, max_length=20)
    typography: str | None = Field(default=None, max_length=2000)
    aspect_ratio: str | None = Field(default=None, max_length=500)
    brand_elements: str | None = Field(default=None, max_length=2000)


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    partially_succeeded = "partially_succeeded"
    failed = "failed"


class CheckResult(BaseModel):
    name: str
    passed: bool
    score: float = Field(ge=0, le=1)
    evidence: str


class EvaluationResult(BaseModel):
    recommendation_checks: list[CheckResult]
    guideline_checks: list[CheckResult]
    recommendation_score: float = Field(ge=0, le=1)
    brand_compliance_score: float = Field(ge=0, le=1)
    recommendation_applied: bool
    brand_compliant: bool
    summary: str
    improvement_feedback: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def verdicts_match_checks(self) -> "EvaluationResult":
        if not self.recommendation_checks:
            raise ValueError("at least one recommendation check is required")
        if not self.guideline_checks:
            raise ValueError("at least one guideline check is required")
        return self


class VariantResult(BaseModel):
    id: str
    image_url: str
    selected_attempt: int = Field(ge=1)
    attempts: int = Field(ge=1)
    evaluation: EvaluationResult


class JobRecord(BaseModel):
    id: str
    status: JobStatus = JobStatus.queued
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    input_filename: str
    input_path: str
    recommendations: list[Recommendation] = Field(min_length=1)
    brand_guidelines: BrandGuidelines
    requested_variants: int = Field(ge=1, le=8)
    variants: list[VariantResult] = Field(default_factory=list)
    error: str | None = None


class JobView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    input_filename: str
    recommendations: list[Recommendation]
    brand_guidelines: BrandGuidelines
    requested_variants: int
    variants: list[VariantResult]
    error: str | None


class JobAccepted(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str
