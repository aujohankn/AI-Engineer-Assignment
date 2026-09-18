from pathlib import Path

from PIL import Image

from app.models import (
    BrandGuidelines,
    CheckResult,
    EvaluationResult,
    JobRecord,
    Recommendation,
)


def passing_evaluation(*, recommendation: bool = True, brand: bool = True) -> EvaluationResult:
    return EvaluationResult(
        recommendation_checks=[
            CheckResult(
                name="recommendation",
                passed=recommendation,
                score=1.0 if recommendation else 0.2,
                evidence="checked",
            )
        ],
        guideline_checks=[
            CheckResult(name="brand", passed=brand, score=1.0 if brand else 0.2, evidence="checked")
        ],
        recommendation_score=1.0 if recommendation else 0.2,
        brand_compliance_score=1.0 if brand else 0.2,
        recommendation_applied=recommendation,
        brand_compliant=brand,
        summary="reviewed",
        improvement_feedback=[] if recommendation and brand else ["Fix the candidate"],
    )


def sample_job(
    input_path: Path, *, job_id: str = "00000000-0000-0000-0000-000000000001"
) -> JobRecord:
    if not input_path.exists():
        Image.new("RGB", (32, 48), "navy").save(input_path)
    return JobRecord(
        id=job_id,
        input_filename=input_path.name,
        input_path=str(input_path),
        recommendations=[
            Recommendation(
                id="r1", title="Contrast", description="Increase contrast", type="contrast"
            )
        ],
        brand_guidelines=BrandGuidelines(protected_regions=["Keep logo"]),
        requested_variants=1,
    )
