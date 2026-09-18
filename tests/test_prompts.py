from app.models import BrandGuidelines, Recommendation
from app.prompts import build_generation_prompt


def test_generation_prompt_makes_brand_rules_hard_constraints() -> None:
    prompt = build_generation_prompt(
        [
            Recommendation(
                id="rec-1",
                title="Strengthen headline",
                description="Increase contrast without increasing physical size.",
                type="contrast_salience",
            )
        ],
        BrandGuidelines(
            protected_regions=["Do not alter the model's face", "Do not modify the logo"],
            aspect_ratio="Maintain 1572x1720",
        ),
        variant_index=0,
        feedback=["The headline still lacks contrast."],
    )

    assert "HARD BRAND CONSTRAINTS" in prompt
    assert "Do not alter the model's face" in prompt
    assert "Increase contrast without increasing physical size" in prompt
    assert "EVALUATOR FEEDBACK TO CORRECT" in prompt
    assert prompt.index("Strengthen headline") < prompt.index("APPLY ALL REQUESTED")
