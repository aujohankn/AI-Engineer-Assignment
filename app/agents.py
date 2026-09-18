import base64
from pathlib import Path
from typing import Protocol

from openai import OpenAI

from app.config import Settings
from app.models import CheckResult, EvaluationResult, JobRecord, VariantResult
from app.prompts import build_evaluation_prompt, build_generation_prompt
from app.storage import ArtifactStore, image_data_url


class ImageGenerator(Protocol):
    def generate(self, source: Path, prompt: str) -> bytes: ...


class VariantEvaluator(Protocol):
    def evaluate(self, original: Path, candidate: Path, job: JobRecord) -> EvaluationResult: ...


class OpenAIImageGenerator:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def generate(self, source: Path, prompt: str) -> bytes:
        with source.open("rb") as image:
            result = self.client.images.edit(
                model=self.model,
                image=image,
                prompt=prompt,
                size="auto",
                quality="high",
                output_format="png",
            )
        if not result.data or not result.data[0].b64_json:
            raise RuntimeError("image model returned no image data")
        return base64.b64decode(result.data[0].b64_json)


class OpenAIVariantEvaluator:
    SYSTEM_PROMPT = (
        "You are a meticulous senior creative QA reviewer. Compare pixels and semantics, "
        "separating recommendation effectiveness from brand compliance."
    )

    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def evaluate(self, original: Path, candidate: Path, job: JobRecord) -> EvaluationResult:
        response = self.client.responses.parse(
            model=self.model,
            store=False,
            instructions=self.SYSTEM_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": build_evaluation_prompt(
                                job.recommendations, job.brand_guidelines
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": image_data_url(original),
                            "detail": "high",
                        },
                        {
                            "type": "input_image",
                            "image_url": image_data_url(candidate),
                            "detail": "high",
                        },
                    ],
                }
            ],
            text_format=EvaluationResult,
        )
        if response.output_parsed is None:
            raise RuntimeError("evaluation model returned no structured result")
        return response.output_parsed


class CreativeWorkflow:
    """Bounded plan -> generate -> evaluate -> revise workflow."""

    def __init__(
        self,
        *,
        generator: ImageGenerator,
        evaluator: VariantEvaluator,
        artifacts: ArtifactStore,
        max_attempts: int,
    ) -> None:
        self.generator = generator
        self.evaluator = evaluator
        self.artifacts = artifacts
        self.max_attempts = max_attempts

    def run(self, job: JobRecord) -> list[VariantResult]:
        source = Path(job.input_path)
        target_size = self.artifacts.dimensions(source)
        variants: list[VariantResult] = []

        for variant_index in range(job.requested_variants):
            feedback: list[str] = []
            candidates: list[tuple[Path, EvaluationResult, int]] = []

            for attempt in range(1, self.max_attempts + 1):
                prompt = build_generation_prompt(
                    job.recommendations,
                    job.brand_guidelines,
                    variant_index=variant_index,
                    feedback=feedback,
                )
                generated = self.generator.generate(source, prompt)
                normalized = self.artifacts.normalize_to_canvas(generated, target_size)
                path = self.artifacts.save_attempt(job.id, variant_index, attempt, normalized)

                evaluation = self.evaluator.evaluate(source, path, job)
                self._add_deterministic_checks(evaluation, path, target_size)
                candidates.append((path, evaluation, attempt))

                if evaluation.recommendation_applied and evaluation.brand_compliant:
                    break
                feedback = evaluation.improvement_feedback

            _, best_evaluation, selected_attempt = max(candidates, key=self._rank_candidate)
            variant_id = f"variant-{variant_index + 1}"
            variants.append(
                VariantResult(
                    id=variant_id,
                    image_url=f"/v1/jobs/{job.id}/variants/{variant_id}/image",
                    selected_attempt=selected_attempt,
                    attempts=len(candidates),
                    evaluation=best_evaluation,
                )
            )

        return variants

    @staticmethod
    def _rank_candidate(candidate: tuple[Path, EvaluationResult, int]) -> tuple[int, float]:
        _, evaluation, _ = candidate
        complete = int(evaluation.recommendation_applied and evaluation.brand_compliant)
        weighted_score = (
            0.45 * evaluation.recommendation_score + 0.55 * evaluation.brand_compliance_score
        )
        return complete, weighted_score

    @staticmethod
    def _add_deterministic_checks(
        evaluation: EvaluationResult, candidate: Path, target_size: tuple[int, int]
    ) -> None:
        with candidate.open("rb") as file:
            from PIL import Image

            with Image.open(file) as image:
                actual_size = image.size
        passed = actual_size == target_size
        evaluation.guideline_checks.append(
            CheckResult(
                name="Exact canvas dimensions",
                passed=passed,
                score=1.0 if passed else 0.0,
                evidence=(
                    f"Expected {target_size[0]}x{target_size[1]}, "
                    f"got {actual_size[0]}x{actual_size[1]}"
                ),
            )
        )
        if not passed:
            evaluation.brand_compliant = False
            evaluation.brand_compliance_score = min(evaluation.brand_compliance_score, 0.4)
            evaluation.improvement_feedback.append("Preserve the exact original canvas dimensions.")


def build_workflow(settings: Settings) -> CreativeWorkflow:
    client = None
    if settings.image_provider == "openai" or settings.evaluation_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for an OpenAI provider")
        # Celery owns retry policy. Disabling SDK retries avoids multiplying attempts.
        client = OpenAI(api_key=settings.openai_api_key, max_retries=0)

    if settings.image_provider == "sdxl":
        from app.sdxl_client import SDXLAPIImageGenerator

        generator: ImageGenerator = SDXLAPIImageGenerator(settings)
    else:
        assert client is not None
        generator = OpenAIImageGenerator(client, settings.image_model)

    if settings.evaluation_provider == "local":
        from app.local_models import LocalImageEvaluator

        evaluator: VariantEvaluator = LocalImageEvaluator(settings)
    elif settings.evaluation_provider == "gemini":
        from app.gemini import GeminiVariantEvaluator

        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini evaluation")
        evaluator = GeminiVariantEvaluator(settings.gemini_api_key, settings.gemini_model)
    else:
        assert client is not None
        evaluator = OpenAIVariantEvaluator(client, settings.evaluation_model)

    artifacts = ArtifactStore(
        settings.artifact_root,
        settings.max_upload_bytes,
        archive_root=settings.archive_root,
    )
    return CreativeWorkflow(
        generator=generator,
        evaluator=evaluator,
        artifacts=artifacts,
        max_attempts=settings.max_generation_attempts,
    )
