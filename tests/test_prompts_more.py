from app.models import BrandGuidelines, Recommendation
from app.prompts import build_evaluation_prompt, build_generation_prompt


def test_generation_prompt_handles_minimal_guidelines() -> None:
    prompt = build_generation_prompt(
        [Recommendation(id="r", title="Edit", description="Make change", type="mood")],
        BrandGuidelines(),
        variant_index=2,
    )
    assert "variant 3" in prompt
    assert "- None" in prompt
    assert "EVALUATOR FEEDBACK" not in prompt


def test_evaluation_prompt_contains_rules_and_verdict_logic() -> None:
    prompt = build_evaluation_prompt(
        [Recommendation(id="r", title="Edit", description="Make change", type="mood")],
        BrandGuidelines(
            protected_regions=["Keep logo"],
            typography="Keep font",
            aspect_ratio="Keep ratio",
            brand_elements="Keep product central",
        ),
    )
    assert "Keep logo" in prompt
    assert "Typography: Keep font" in prompt
    assert "recommendation_applied is true only" in prompt
