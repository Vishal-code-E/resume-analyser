from pydantic import BaseModel, Field, field_validator
from typing import Literal


# ── Intermediate agent models ──────────────────────────────

class CandidateProfile(BaseModel):
    name: str
    skills: list[str]
    experience_years: int
    projects: list[str]
    seniority_level: Literal["junior", "mid", "senior"]

    @field_validator("experience_years", mode="before")
    @classmethod
    def coerce_experience_years(cls, v):
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            # handles "3+", "2-4", "5 years" etc
            digits = "".join(filter(str.isdigit, v.split("-")[0].split("+")[0]))
            if digits:
                return int(digits)
        raise ValueError(f"Cannot parse experience_years from: {v}")


class JobProfile(BaseModel):
    role_title: str
    required_skills: list[str]
    preferred_skills: list[str]
    domains: list[str]
    seniority_expected: Literal["junior", "mid", "senior"]


# ── Request ────────────────────────────────────────────────

class AnalyseRequest(BaseModel):
    resume_text: str = Field(..., min_length=50, max_length=15000)
    job_description: str = Field(..., min_length=50, max_length=8000)

    @field_validator("resume_text", "job_description", mode="before")
    @classmethod
    def strip_and_check(cls, v):
        if isinstance(v, str):
            v = v.strip()
        if not v:
            raise ValueError("Field cannot be empty or whitespace")
        return v


# ── Response ───────────────────────────────────────────────

class Gap(BaseModel):
    skill: str
    severity: Literal["must-have", "preferred"]


class InterviewQuestion(BaseModel):
    question: str
    targets_gap: str
    priority: Literal["must-ask", "optional"]


class AnalysisResponse(BaseModel):
    candidate_name: str
    overall_score: int = Field(..., ge=0, le=100)
    confidence: Literal["low", "medium", "high"]
    recommendation: Literal["advance", "hold", "reject"]
    strengths: list[str] 
    matching_skills: list[str]
    gaps: list[Gap]
    jd_match_breakdown: dict[str, int]
    interview_questions: list[InterviewQuestion]