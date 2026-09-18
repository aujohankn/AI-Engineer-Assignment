from pathlib import Path

from PIL import Image

from app.agents import CreativeWorkflow
from app.models import (
    BrandGuidelines,
    CheckResult,
    EvaluationResult,
    JobRecord,
    Recommendation,
)
from app.storage import ArtifactStore


class CopyGenerator:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, source: Path, prompt: str) -> bytes:
        self.prompts.append(prompt)
        return source.read_bytes()


class RetryEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, original: Path, candidate: Path, job: JobRecord) -> EvaluationResult:
        self.calls += 1
        passed = self.calls > 1
        check = CheckResult(
            name="recommendation", passed=passed, score=1.0 if passed else 0.3, evidence="reviewed"
        )
        guideline = CheckResult(name="logo", passed=True, score=1, evidence="unchanged")
        return EvaluationResult(
            recommendation_checks=[check],
            guideline_checks=[guideline],
            recommendation_score=check.score,
            brand_compliance_score=1,
            recommendation_applied=passed,
            brand_compliant=True,
            summary="pass" if passed else "retry",
            improvement_feedback=[] if passed else ["Increase headline contrast."],
        )


def test_workflow_retries_with_feedback_and_selects_passing_attempt(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 80), "orange").save(source)
    job = JobRecord(
        id="00000000-0000-0000-0000-000000000001",
        input_filename="source.png",
        input_path=str(source),
        recommendations=[
            Recommendation(
                id="r1", title="Headline", description="Increase contrast", type="contrast"
            )
        ],
        brand_guidelines=BrandGuidelines(protected_regions=["Keep logo"]),
        requested_variants=1,
    )
    generator = CopyGenerator()
    workflow = CreativeWorkflow(
        generator=generator,
        evaluator=RetryEvaluator(),
        artifacts=ArtifactStore(tmp_path / "jobs", 1_000_000, tmp_path / "archive"),
        max_attempts=2,
    )

    variants = workflow.run(job)

    assert variants[0].attempts == 2
    assert variants[0].selected_attempt == 2
    assert variants[0].evaluation.recommendation_applied is True
    assert "EVALUATOR FEEDBACK TO CORRECT" in generator.prompts[1]
    assert (tmp_path / "jobs" / job.id / "variant-1-attempt-2.png").is_file()
    assert (tmp_path / "archive" / job.id / "variant-1-attempt-1.png").is_file()
    assert (tmp_path / "archive" / job.id / "variant-1-attempt-2.png").is_file()
