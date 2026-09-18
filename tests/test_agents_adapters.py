import base64
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import ValidationError

import app.agents as agents
from app.agents import OpenAIImageGenerator, OpenAIVariantEvaluator
from app.config import Settings
from app.models import CheckResult, EvaluationResult
from tests.helpers import passing_evaluation, sample_job


class FakeImages:
    def __init__(self, payload: bytes | None) -> None:
        encoded = base64.b64encode(payload).decode() if payload is not None else None
        self.result = SimpleNamespace(data=[SimpleNamespace(b64_json=encoded)] if encoded else [])
        self.kwargs: dict[str, object] = {}

    def edit(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return self.result


class FakeResponses:
    def __init__(self, parsed: EvaluationResult | None) -> None:
        self.parsed = parsed
        self.kwargs: dict[str, object] = {}

    def parse(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return SimpleNamespace(output_parsed=self.parsed)


def test_openai_image_generator_decodes_result(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    images = FakeImages(b"generated")
    client = SimpleNamespace(images=images)

    result = OpenAIImageGenerator(client, "image-model").generate(source, "edit it")

    assert result == b"generated"
    assert images.kwargs["model"] == "image-model"
    assert images.kwargs["quality"] == "high"
    assert images.kwargs["output_format"] == "png"


def test_openai_image_generator_rejects_empty_response(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    client = SimpleNamespace(images=FakeImages(None))
    with pytest.raises(RuntimeError, match="no image data"):
        OpenAIImageGenerator(client, "image-model").generate(source, "edit it")


def test_openai_evaluator_uses_two_images_and_structured_output(tmp_path: Path) -> None:
    original = tmp_path / "original.png"
    candidate = tmp_path / "candidate.png"
    Image.new("RGB", (8, 8), "red").save(original)
    Image.new("RGB", (8, 8), "blue").save(candidate)
    expected = passing_evaluation()
    responses = FakeResponses(expected)
    client = SimpleNamespace(responses=responses)

    result = OpenAIVariantEvaluator(client, "vision-model").evaluate(
        original, candidate, sample_job(original)
    )

    assert result is expected
    assert responses.kwargs["model"] == "vision-model"
    assert responses.kwargs["store"] is False
    content = responses.kwargs["input"][0]["content"]
    assert [item["type"] for item in content] == ["input_text", "input_image", "input_image"]
    assert all(item["image_url"].startswith("data:image/png;base64,") for item in content[1:])


def test_openai_evaluator_rejects_missing_structured_result(tmp_path: Path) -> None:
    original = tmp_path / "original.png"
    Image.new("RGB", (8, 8), "red").save(original)
    client = SimpleNamespace(responses=FakeResponses(None))
    with pytest.raises(RuntimeError, match="no structured result"):
        OpenAIVariantEvaluator(client, "vision-model").evaluate(
            original, original, sample_job(original)
        )


def test_deterministic_dimension_failure_updates_verdict(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.png"
    Image.new("RGB", (9, 10), "red").save(candidate)
    evaluation = passing_evaluation()

    agents.CreativeWorkflow._add_deterministic_checks(evaluation, candidate, (10, 10))

    assert evaluation.brand_compliant is False
    assert evaluation.brand_compliance_score == 0.4
    assert evaluation.guideline_checks[-1].passed is False
    assert evaluation.improvement_feedback


def test_build_workflow_requires_key(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        agents.build_workflow(
            Settings(
                openai_api_key=None,
                artifact_root=tmp_path,
                image_provider="openai",
                evaluation_provider="openai",
            )
        )


def test_build_workflow_wires_adapters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_client = object()
    captured: dict[str, object] = {}

    def fake_openai(**kwargs: object) -> object:
        captured.update(kwargs)
        return fake_client

    monkeypatch.setattr(agents, "OpenAI", fake_openai)
    settings = Settings(
        openai_api_key="not-a-real-key",
        artifact_root=tmp_path,
        image_model="image-model",
        evaluation_model="vision-model",
        max_generation_attempts=3,
        image_provider="openai",
        evaluation_provider="openai",
    )

    workflow = agents.build_workflow(settings)

    assert workflow.generator.client is fake_client
    assert workflow.generator.model == "image-model"
    assert workflow.evaluator.model == "vision-model"
    assert workflow.max_attempts == 3
    assert captured == {"api_key": "not-a-real-key", "max_retries": 0}


def test_build_workflow_defaults_to_local_providers(tmp_path: Path) -> None:
    workflow = agents.build_workflow(Settings(artifact_root=tmp_path))
    assert type(workflow.generator).__name__ == "SDXLAPIImageGenerator"
    assert type(workflow.evaluator).__name__ == "LocalImageEvaluator"


def test_evaluation_result_requires_both_check_groups() -> None:
    check = CheckResult(name="one", passed=True, score=1, evidence="yes")
    with pytest.raises(ValidationError):
        EvaluationResult(
            recommendation_checks=[],
            guideline_checks=[check],
            recommendation_score=1,
            brand_compliance_score=1,
            recommendation_applied=True,
            brand_compliant=True,
            summary="invalid",
        )
    with pytest.raises(ValidationError):
        EvaluationResult(
            recommendation_checks=[check],
            guideline_checks=[],
            recommendation_score=1,
            brand_compliance_score=1,
            recommendation_applied=True,
            brand_compliant=True,
            summary="invalid",
        )
