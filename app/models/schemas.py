from pydantic import BaseModel, Field
from typing import Literal


# ── Intermediate agent models ──────────────────────────────

class CandidateProfile(BaseModel):
    name: str
    skills: list[str]
    experience_years: int
    projects: list[str]
    seniority_level: Literal["junior", "mid", "senior"]


class JobProfile(BaseModel):
    role_title: str
    required_skills: list[str]
    preferred_skills: list[str]
    domains: list[str]          # extracted dynamically from JD
    seniority_expected: Literal["junior", "mid", "senior"]


# ── Request ────────────────────────────────────────────────

class AnalyseRequest(BaseModel):
    resume_text: str = Field(..., min_length=50)
    job_description: str = Field(..., min_length=50)


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
    matching_skills: list[str]
    gaps: list[Gap]
    jd_match_breakdown: dict[str, int]   # dynamic domains from JD
    interview_questions: list[InterviewQuestion]