from app.models import BrandGuidelines, EvaluationResult, Recommendation


def build_generation_prompt(
    recommendations: list[Recommendation],
    guidelines: BrandGuidelines,
    *,
    variant_index: int,
    feedback: list[str] | None = None,
) -> str:
    core_titles = "; ".join(item.title for item in recommendations)
    core_protected = "; ".join(guidelines.protected_regions) or "preserve all brand assets"
    requested = "\n".join(
        f"- [{item.id}] {item.title} ({item.type}): {item.description}" for item in recommendations
    )
    protected = "\n".join(f"- {rule}" for rule in guidelines.protected_regions) or "- None"
    optional = "\n".join(
        f"- {label}: {value}"
        for label, value in (
            ("Typography", guidelines.typography),
            ("Aspect ratio", guidelines.aspect_ratio),
            ("Brand elements", guidelines.brand_elements),
        )
        if value
    )
    correction = ""
    if feedback:
        correction = "\nEVALUATOR FEEDBACK TO CORRECT:\n" + "\n".join(
            f"- {item}" for item in feedback
        )

    return f"""Edit the supplied marketing creative in place. Produce variant {variant_index + 1}.
CORE BRIEF: Apply all of these: {core_titles}. Protect: {core_protected}. Preserve readable text,
the original layout, and brand identity.

APPLY ALL REQUESTED RECOMMENDATIONS:
{requested}

HARD BRAND CONSTRAINTS — these override every requested edit:
{protected}
{optional}

Keep all protected people, products, logos, and brand marks pixel-faithful and clearly legible.
Preserve the original canvas composition and aspect ratio. Do not add a border, mockup frame,
watermark, commentary, or any text not explicitly required by a recommendation. Maintain the
existing visual identity and typography. Make a polished, production-ready marketing creative.
Vary only stylistic choices that are compatible with the recommendations and constraints.
{correction}""".strip()


def build_evaluation_prompt(
    recommendations: list[Recommendation], guidelines: BrandGuidelines
) -> str:
    requested = "\n".join(
        f"- [{item.id}] {item.title}: {item.description}" for item in recommendations
    )
    rules = "\n".join(f"- {rule}" for rule in guidelines.protected_regions)
    extra_rules = "\n".join(
        f"- {name}: {value}"
        for name, value in (
            ("Typography", guidelines.typography),
            ("Aspect ratio", guidelines.aspect_ratio),
            ("Brand elements", guidelines.brand_elements),
        )
        if value
    )
    schema_note = ", ".join(EvaluationResult.model_json_schema()["properties"])
    return f"""Compare ORIGINAL (first image) with CANDIDATE (second image).

Recommendations to verify independently:
{requested}

Brand constraints to verify independently:
{rules}
{extra_rules}

Be strict and evidence-based. Treat a protected logo, face, or product as compliant only when it
remains recognizably the same and has not been materially redrawn. Do not reward aesthetic quality
unless it proves a requested change. recommendation_applied is true only when every recommendation
has a passing check. brand_compliant is true only when every guideline check passes. Scores are 0-1.
Make improvement_feedback actionable and empty only when both verdicts pass.
Return the structured fields: {schema_note}.""".strip()
