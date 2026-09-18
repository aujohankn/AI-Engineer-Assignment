from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageChops, ImageOps, ImageStat

from app.config import Settings
from app.models import CheckResult, EvaluationResult, JobRecord


def _multiple_of_eight_size(size: tuple[int, int], max_edge: int) -> tuple[int, int]:
    width, height = size
    scale = min(1.0, max_edge / max(width, height))
    scaled_width = max(64, round(width * scale / 8) * 8)
    scaled_height = max(64, round(height * scale / 8) * 8)
    return scaled_width, scaled_height


@lru_cache(maxsize=1)
def _load_sdxl_pipeline(model_id: str, cpu_offload: bool):
    try:
        import torch
        from diffusers import AutoPipelineForImage2Image
    except ImportError as exc:  # pragma: no cover - exercised inside the SDXL image
        raise RuntimeError("Install the 'sdxl' dependency extra to use local generation") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("SDXL requires a CUDA-capable GPU in this deployment")
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    enable_vae_tiling = getattr(pipeline, "enable_vae_tiling", None)
    if callable(enable_vae_tiling):
        enable_vae_tiling()
    else:
        pipeline.vae.enable_tiling()
    if cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


class SDXLImageGenerator:
    NEGATIVE_PROMPT = (
        "changed logo, misspelled text, distorted typography, altered face, altered product, "
        "extra objects, watermark, frame, low quality, blurry, deformed"
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(self, source: Path, prompt: str) -> bytes:
        import torch

        pipeline = _load_sdxl_pipeline(self.settings.sdxl_model_id, self.settings.sdxl_cpu_offload)
        with Image.open(source) as opened:
            source_image = ImageOps.exif_transpose(opened).convert("RGB")
            generation_size = _multiple_of_eight_size(
                source_image.size, self.settings.sdxl_max_edge
            )
            init_image = source_image.resize(generation_size, Image.Resampling.LANCZOS)

        generator = torch.Generator(device="cpu").manual_seed(torch.seed())
        result = pipeline(
            prompt=prompt,
            negative_prompt=self.NEGATIVE_PROMPT,
            image=init_image,
            strength=self.settings.sdxl_strength,
            guidance_scale=self.settings.sdxl_guidance_scale,
            num_inference_steps=self.settings.sdxl_steps,
            generator=generator,
            width=generation_size[0],
            height=generation_size[1],
        )
        if not result.images:
            raise RuntimeError("SDXL returned no image")
        output = BytesIO()
        result.images[0].save(output, "PNG")
        return output.getvalue()


class LocalImageEvaluator:
    """Offline proxy evaluator based on change magnitude and visual preservation.

    It is deterministic and free, but it cannot prove localized logo/face integrity or
    understand rewritten copy as reliably as a vision-language model.
    """

    def __init__(self, settings: Settings) -> None:
        self.min_change = settings.local_min_change
        self.min_preservation = settings.local_min_preservation

    def evaluate(self, original: Path, candidate: Path, job: JobRecord) -> EvaluationResult:
        with Image.open(original) as original_image, Image.open(candidate) as candidate_image:
            original_rgb = ImageOps.exif_transpose(original_image).convert("RGB")
            candidate_rgb = ImageOps.exif_transpose(candidate_image).convert("RGB")
            if candidate_rgb.size != original_rgb.size:
                candidate_rgb = candidate_rgb.resize(original_rgb.size, Image.Resampling.LANCZOS)
            difference = ImageChops.difference(original_rgb, candidate_rgb)
            channel_means = ImageStat.Stat(difference).mean

        normalized_difference = sum(channel_means) / (len(channel_means) * 255)
        preservation = max(0.0, 1.0 - normalized_difference)
        changed_enough = normalized_difference >= self.min_change
        preserved_enough = preservation >= self.min_preservation

        recommendation_checks = [
            CheckResult(
                name=f"{item.id}: {item.title}",
                passed=changed_enough,
                score=min(1.0, normalized_difference / max(self.min_change * 3, 0.001)),
                evidence=(
                    f"Offline proxy measured {normalized_difference:.3f} normalized pixel change; "
                    f"minimum is {self.min_change:.3f}. Semantic intent requires human review."
                ),
            )
            for item in job.recommendations
        ]
        guideline_names = [
            *job.brand_guidelines.protected_regions,
            *(
                [f"Typography: {job.brand_guidelines.typography}"]
                if job.brand_guidelines.typography
                else []
            ),
            *(
                [f"Brand elements: {job.brand_guidelines.brand_elements}"]
                if job.brand_guidelines.brand_elements
                else []
            ),
        ] or ["Global visual preservation"]
        guideline_checks = [
            CheckResult(
                name=name,
                passed=preserved_enough,
                score=preservation,
                evidence=(
                    f"Offline global preservation proxy scored {preservation:.3f}; minimum is "
                    f"{self.min_preservation:.3f}. Localized protected regions require human "
                    "review."
                ),
            )
            for name in guideline_names
        ]
        return EvaluationResult(
            recommendation_checks=recommendation_checks,
            guideline_checks=guideline_checks,
            recommendation_score=sum(check.score for check in recommendation_checks)
            / len(recommendation_checks),
            brand_compliance_score=preservation,
            recommendation_applied=all(check.passed for check in recommendation_checks),
            brand_compliant=all(check.passed for check in guideline_checks),
            summary=(
                "Offline heuristic evaluation completed. Human review is required for semantic "
                "recommendation fidelity, exact copy, and localized protected assets."
            ),
            improvement_feedback=(
                []
                if changed_enough and preserved_enough
                else [
                    "Increase the edit strength slightly."
                    if not changed_enough
                    else "Reduce edit strength to preserve more of the original creative."
                ]
            ),
        )
