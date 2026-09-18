import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from app.gemini import GeminiEvaluationError, GeminiTransientError, GeminiVariantEvaluator
from tests.helpers import passing_evaluation, sample_job


class FakePart:
    @staticmethod
    def from_text(*, text: str) -> dict[str, object]:
        return {"text": text}

    @staticmethod
    def from_bytes(*, data: bytes, mime_type: str) -> dict[str, object]:
        return {"data": data, "mime_type": mime_type}


class FakeConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class FakeModels:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.kwargs: dict[str, object] = {}

    def generate_content(self, **kwargs: object):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def install_fake_types(monkeypatch: pytest.MonkeyPatch) -> None:
    google = ModuleType("google")
    google.__path__ = []
    genai = ModuleType("google.genai")
    genai.types = SimpleNamespace(Part=FakePart, GenerateContentConfig=FakeConfig)
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)


def image_paths(tmp_path: Path) -> tuple[Path, Path]:
    original = tmp_path / "original.png"
    candidate = tmp_path / "candidate.png"
    original.write_bytes(b"original")
    candidate.write_bytes(b"candidate")
    return original, candidate


def test_gemini_evaluator_uses_two_images_and_pydantic_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_fake_types(monkeypatch)
    expected = passing_evaluation()
    models = FakeModels(SimpleNamespace(parsed=expected, text=None))
    evaluator = GeminiVariantEvaluator(
        "not-a-real-key", "gemini-test", client=SimpleNamespace(models=models)
    )
    original, candidate = image_paths(tmp_path)

    result = evaluator.evaluate(original, candidate, sample_job(original))

    assert result is expected
    assert models.kwargs["model"] == "gemini-test"
    assert [part["data"] for part in models.kwargs["contents"] if "data" in part] == [
        b"original",
        b"candidate",
    ]
    assert models.kwargs["config"].kwargs["response_schema"].__name__ == "EvaluationResult"


def test_gemini_evaluator_accepts_json_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_fake_types(monkeypatch)
    payload = passing_evaluation().model_dump(mode="json")
    models = FakeModels(SimpleNamespace(parsed=None, text=json.dumps(payload)))
    evaluator = GeminiVariantEvaluator("key", "model", client=SimpleNamespace(models=models))
    original, candidate = image_paths(tmp_path)

    result = evaluator.evaluate(original, candidate, sample_job(original))

    assert result.brand_compliant is True


def test_gemini_evaluator_rejects_empty_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_fake_types(monkeypatch)
    models = FakeModels(SimpleNamespace(parsed=None, text=None))
    evaluator = GeminiVariantEvaluator("key", "model", client=SimpleNamespace(models=models))
    original, candidate = image_paths(tmp_path)

    with pytest.raises(GeminiEvaluationError, match="no structured evaluation"):
        evaluator.evaluate(original, candidate, sample_job(original))


@pytest.mark.parametrize(
    ("status", "exception_type"),
    [(429, GeminiTransientError), (503, GeminiTransientError), (401, GeminiEvaluationError)],
)
def test_gemini_evaluator_classifies_api_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: int,
    exception_type: type[Exception],
) -> None:
    install_fake_types(monkeypatch)
    error = type("ProviderError", (Exception,), {"code": status})("provider failure")
    evaluator = GeminiVariantEvaluator(
        "key", "model", client=SimpleNamespace(models=FakeModels(error=error))
    )
    original, candidate = image_paths(tmp_path)

    with pytest.raises(exception_type):
        evaluator.evaluate(original, candidate, sample_job(original))
