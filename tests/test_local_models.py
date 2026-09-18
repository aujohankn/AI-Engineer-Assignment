import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import app.local_models as local_models
from app.config import Settings
from tests.helpers import sample_job


class FakePipeline:
    def __init__(self, *, with_image: bool = True) -> None:
        self.images = [Image.new("RGB", (64, 80), "teal")] if with_image else []
        self.calls: list[dict[str, object]] = []
        self.actions: list[object] = []

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(images=self.images)

    def enable_attention_slicing(self, value: str) -> None:
        self.actions.append(("attention", value))

    def enable_vae_tiling(self) -> None:
        self.actions.append("tiling")

    def enable_model_cpu_offload(self) -> None:
        self.actions.append("offload")

    def to(self, device: str) -> None:
        self.actions.append(("to", device))

    def set_progress_bar_config(self, *, disable: bool) -> None:
        self.actions.append(("progress", disable))


class FakeTorch:
    float16 = "float16"

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class Generator:
        def __init__(self, device: str) -> None:
            self.device = device

        def manual_seed(self, seed: int):
            self.seed = seed
            return self

    @staticmethod
    def seed() -> int:
        return 123


def test_sdxl_working_size_is_bounded_and_divisible_by_eight() -> None:
    assert local_models._multiple_of_eight_size((636, 1063), 1024) == (616, 1024)
    assert local_models._multiple_of_eight_size((32, 48), 1024) == (64, 64)


def test_pipeline_loader_configures_offload_or_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[FakePipeline] = []

    class Factory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakePipeline:
            pipeline = FakePipeline()
            pipeline.actions.append(("load", model_id, kwargs))
            created.append(pipeline)
            return pipeline

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(AutoPipelineForImage2Image=Factory),
    )
    local_models._load_sdxl_pipeline.cache_clear()

    offloaded = local_models._load_sdxl_pipeline("model-a", True)
    on_gpu = local_models._load_sdxl_pipeline("model-b", False)

    assert "offload" in offloaded.actions
    assert ("to", "cuda") in on_gpu.actions
    assert ("progress", True) in on_gpu.actions


def test_pipeline_loader_supports_component_vae_tiling(monkeypatch: pytest.MonkeyPatch) -> None:
    class ModernPipeline(FakePipeline):
        enable_vae_tiling = None

        def __init__(self) -> None:
            super().__init__()
            self.vae = SimpleNamespace(enable_tiling=lambda: self.actions.append("vae-tiling"))

    class Factory:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> ModernPipeline:
            return ModernPipeline()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(AutoPipelineForImage2Image=Factory),
    )
    local_models._load_sdxl_pipeline.cache_clear()

    pipeline = local_models._load_sdxl_pipeline("model", True)

    assert "vae-tiling" in pipeline.actions


def test_pipeline_loader_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoCudaTorch(FakeTorch):
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

    monkeypatch.setitem(sys.modules, "torch", NoCudaTorch)
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(AutoPipelineForImage2Image=object()),
    )
    local_models._load_sdxl_pipeline.cache_clear()
    with pytest.raises(RuntimeError, match="CUDA-capable GPU"):
        local_models._load_sdxl_pipeline("model", True)


def test_sdxl_generator_returns_png(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pipeline = FakePipeline()
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(local_models, "_load_sdxl_pipeline", lambda *args: pipeline)
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 80), "orange").save(source)
    generator = local_models.SDXLImageGenerator(
        Settings(sdxl_max_edge=512, sdxl_steps=4, sdxl_strength=0.2)
    )

    output = generator.generate(source, "make it brighter")

    with Image.open(BytesIO(output)) as image:
        assert image.format == "PNG"
    assert pipeline.calls[0]["prompt"] == "make it brighter"
    assert pipeline.calls[0]["num_inference_steps"] == 4
    assert pipeline.calls[0]["width"] % 8 == 0


def test_sdxl_generator_rejects_empty_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(
        local_models, "_load_sdxl_pipeline", lambda *args: FakePipeline(with_image=False)
    )
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 80), "orange").save(source)
    with pytest.raises(RuntimeError, match="no image"):
        local_models.SDXLImageGenerator(Settings()).generate(source, "edit")


def test_local_evaluator_passes_meaningful_change_and_preservation(tmp_path: Path) -> None:
    original = tmp_path / "original.png"
    candidate = tmp_path / "candidate.png"
    Image.new("RGB", (32, 48), (100, 100, 100)).save(original)
    Image.new("RGB", (16, 24), (120, 120, 120)).save(candidate)
    job = sample_job(original)
    job.brand_guidelines.typography = "Keep type"
    job.brand_guidelines.brand_elements = "Keep product"
    evaluator = local_models.LocalImageEvaluator(
        Settings(local_min_change=0.02, local_min_preservation=0.5)
    )

    result = evaluator.evaluate(original, candidate, job)

    assert result.recommendation_applied is True
    assert result.brand_compliant is True
    assert len(result.guideline_checks) == 3
    assert result.improvement_feedback == []


def test_local_evaluator_requests_stronger_or_weaker_edit(tmp_path: Path) -> None:
    original = tmp_path / "original.png"
    same = tmp_path / "same.png"
    destroyed = tmp_path / "destroyed.png"
    Image.new("RGB", (16, 16), "black").save(original)
    Image.new("RGB", (16, 16), "black").save(same)
    Image.new("RGB", (16, 16), "white").save(destroyed)
    job = sample_job(original)
    job.brand_guidelines.protected_regions = []
    evaluator = local_models.LocalImageEvaluator(
        Settings(local_min_change=0.1, local_min_preservation=0.8)
    )

    subtle = evaluator.evaluate(original, same, job)
    excessive = evaluator.evaluate(original, destroyed, job)

    assert subtle.recommendation_applied is False
    assert "Increase" in subtle.improvement_feedback[0]
    assert excessive.brand_compliant is False
    assert "Reduce" in excessive.improvement_feedback[0]
    assert excessive.guideline_checks[0].name == "Global visual preservation"
