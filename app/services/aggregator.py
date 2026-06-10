import logging

logger = logging.getLogger(__name__)

# ── Blend weights ──────────────────────────────────────────
RULE_WEIGHT = 0.40   # deterministic, factual
LLM_WEIGHT  = 0.60   # contextual, depth-aware


def aggregate_scores(
    rule_score: int,
    llm_score: int,
) -> dict:
    """
    Blend rule-based and LLM scores into a final score.

    Rule score → factual coverage (skills, seniority)
    LLM score  → contextual depth (project evidence, quality)
    """

    blended = round(
        (rule_score * RULE_WEIGHT) +
        (llm_score  * LLM_WEIGHT)
    )

    # clamp to 0-100
    final_score = min(max(blended, 0), 100)

    logger.info(
        f"Aggregator: rule={rule_score} llm={llm_score} "
        f"→ blended={final_score} "
        f"(weights: rule={RULE_WEIGHT} llm={LLM_WEIGHT})"
    )

    return {
        "final_score": final_score,
        "rule_score":  rule_score,
        "llm_score":   llm_score,
        "rule_weight": RULE_WEIGHT,
        "llm_weight":  LLM_WEIGHT,
    }


def derive_recommendation(score: int) -> str:
    """
    Programmatic recommendation — never delegated to LLM.
    Move derive_recommendation here from main.py.
    """
    if score >= 70:
        return "advance"
    if score >= 45:
        return "hold"
    return "reject"