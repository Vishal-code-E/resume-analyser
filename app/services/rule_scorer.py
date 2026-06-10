from app.models.schemas import CandidateProfile, JobProfile
from app.services.normalizer import candidate_has_skill

# ── Weights ────────────────────────────────────────────────
REQUIRED_SKILL_WEIGHT  = 0.55   # required skills coverage
PREFERRED_SKILL_WEIGHT = 0.20   # preferred skills coverage
SENIORITY_WEIGHT       = 0.25   # seniority alignment

# ── Seniority penalty ──────────────────────────────────────
SENIORITY_SCORES = {
    ("junior", "junior"):   1.0,
    ("junior", "mid"):      0.5,
    ("junior", "senior"):   0.1,
    ("mid",    "junior"):   0.9,   # overqualified — slight penalty
    ("mid",    "mid"):      1.0,
    ("mid",    "senior"):   0.5,
    ("senior", "junior"):   0.8,   # overqualified
    ("senior", "mid"):      0.9,
    ("senior", "senior"):   1.0,
}


def compute_rule_score(
    candidate: CandidateProfile,
    job: JobProfile
) -> dict:
    """
    Deterministic rule-based scoring.
    Returns score (0-100) + breakdown detail.
    """

    candidate_skills = candidate.skills

    # ── Required skill coverage ────────────────────────────
    required_hits   = []
    required_misses = []

    for skill in job.required_skills:
        if candidate_has_skill(candidate_skills, skill):
            required_hits.append(skill)
        else:
            required_misses.append(skill)

    required_total    = len(job.required_skills)
    required_coverage = len(required_hits) / required_total if required_total else 1.0

    # ── Preferred skill coverage ───────────────────────────
    preferred_hits   = []
    preferred_misses = []

    for skill in job.preferred_skills:
        if candidate_has_skill(candidate_skills, skill):
            preferred_hits.append(skill)
        else:
            preferred_misses.append(skill)

    preferred_total    = len(job.preferred_skills)
    preferred_coverage = len(preferred_hits) / preferred_total if preferred_total else 1.0

    # ── Seniority alignment ────────────────────────────────
    seniority_key   = (candidate.seniority_level, job.seniority_expected)
    seniority_score = SENIORITY_SCORES.get(seniority_key, 0.5)

    # ── Weighted final score ───────────────────────────────
    raw_score = (
        required_coverage  * REQUIRED_SKILL_WEIGHT  +
        preferred_coverage * PREFERRED_SKILL_WEIGHT +
        seniority_score    * SENIORITY_WEIGHT
    ) * 100

    final_score = round(min(max(raw_score, 0), 100))

    return {
        "rule_score":          final_score,
        "required_coverage":   round(required_coverage * 100),
        "preferred_coverage":  round(preferred_coverage * 100),
        "seniority_alignment": round(seniority_score * 100),
        "required_hits":       required_hits,
        "required_misses":     required_misses,
        "preferred_hits":      preferred_hits,
        "preferred_misses":    preferred_misses,
    }