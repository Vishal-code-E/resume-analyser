import re

# ── Synonym map ────────────────────────────────────────────
# canonical name → list of variants
SKILL_SYNONYMS: dict[str, list[str]] = {
    "rest api":           ["rest apis", "rest api design", "restful api", "restful apis", "rest"],
    "postgresql":         ["postgres", "psql", "pg"],
    "javascript":         ["js"],
    "typescript":         ["ts"],
    "python":             ["python3", "py"],
    "kubernetes":         ["k8s"],
    "elasticsearch":      ["elastic", "es"],
    "mongodb":            ["mongo"],
    "fastapi":            ["fast api"],
    "langchain":          ["lang chain"],
    "aws":                ["amazon web services", "aws cloud"],
    "gcp":                ["google cloud", "google cloud platform"],
    "azure":              ["microsoft azure"],
    "docker":             ["containerisation", "containerization"],
    "machine learning":   ["ml"],
    "natural language processing": ["nlp"],
    "large language model": ["llm", "llms"],
    "retrieval augmented generation": ["rag"],
    "vector database":    ["vector db", "vector store", "vectordb", "vector databases"],
    "multi-agent":        ["multi agent", "multiagent", "multi-agent system design"],
    "prompt engineering": ["prompt design", "prompting"],
}

def _normalize(skill: str) -> str:
    """Lowercase and strip a skill string."""
    return skill.lower().strip()

def _canonical(skill: str) -> str:
    """Map a skill to its canonical form using synonym map."""
    normalized = _normalize(skill)
    for canonical, variants in SKILL_SYNONYMS.items():
        if normalized == canonical or normalized in variants:
            return canonical
    return normalized

def _parse_or_conditions(skill: str) -> list[str]:
    """
    Split OR conditions into alternatives.
    'Django or FastAPI' → ['django', 'fastapi']
    'Django / FastAPI'  → ['django', 'fastapi']
    """
    parts = re.split(r'\s+or\s+|\s*/\s*', skill, flags=re.IGNORECASE)
    return [_canonical(p.strip()) for p in parts if p.strip()]

def normalize_skill_list(skills: list[str]) -> list[str]:
    """Normalize a flat skill list to canonical forms."""
    result = []
    for skill in skills:
        parts = _parse_or_conditions(skill)
        result.extend(parts)
    return list(dict.fromkeys(result))  # deduplicate, preserve order

def candidate_has_skill(
    candidate_skills: list[str],
    required_skill: str
) -> bool:
    """
    Check if candidate covers a required skill.
    Handles OR conditions — candidate needs only ONE side.
    """
    alternatives = _parse_or_conditions(required_skill)
    candidate_normalized = [_canonical(s) for s in candidate_skills]
    return any(alt in candidate_normalized for alt in alternatives)