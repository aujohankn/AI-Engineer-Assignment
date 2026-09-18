from pathlib import Path
from typing import Any

from app.models import EvaluationResult, JobRecord
from app.prompts import build_evaluation_prompt


class GeminiEvaluationError(RuntimeError):
    pass


class GeminiTransientError(GeminiEvaluationError):
    pass


class GeminiVariantEvaluator:
    SYSTEM_PROMPT = (
        "You are a meticulous senior creative QA reviewer. Compare the ORIGINAL and CANDIDATE "
        "images directly. Judge every recommendation and brand rule independently, quote visible "
        "evidence, and never infer compliance when text or a protected asset is illegible."
    )

    def __init__(self, api_key: str, model: str, client: Any | None = None) -> None:
        if client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - dependency is in the runtime image
                raise RuntimeError("Install google-genai to use Gemini evaluation") from exc
            client = genai.Client(api_key=api_key)
        self.client = client
        self.model = model

    def evaluate(self, original: Path, candidate: Path, job: JobRecord) -> EvaluationResult:
        from google.genai import types

        contents = [
            build_evaluation_prompt(job.recommendations, job.brand_guidelines),
            types.Part.from_text(text="ORIGINAL image:"),
            types.Part.from_bytes(data=original.read_bytes(), mime_type="image/png"),
            types.Part.from_text(text="CANDIDATE image:"),
            types.Part.from_bytes(data=candidate.read_bytes(), mime_type="image/png"),
        ]
        config = types.GenerateContentConfig(
            system_instruction=self.SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=EvaluationResult,
            temperature=0.1,
        )
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            if status in {429, 500, 502, 503, 504}:
                message = f"Gemini temporarily unavailable (HTTP {status})"
                raise GeminiTransientError(message) from exc
            raise GeminiEvaluationError(f"Gemini evaluation failed: {type(exc).__name__}") from exc

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, EvaluationResult):
            return parsed
        if parsed is not None:
            return EvaluationResult.model_validate(parsed)
        text = getattr(response, "text", None)
        if not text:
            raise GeminiEvaluationError("Gemini returned no structured evaluation")
        return EvaluationResult.model_validate_json(text)
