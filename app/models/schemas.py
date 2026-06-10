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


#--- Score Breakdown --------------------------------------

class ScoreBreakdown(BaseModel):
    final_score:          int
    rule_score:           int
    llm_score:            int
    rule_weight:          float
    llm_weight:           float
    required_coverage:    int    # % of required skills matched
    preferred_coverage:   int    # % of preferred skills matched
    seniority_alignment:  int    # % seniority match score


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
    score_breakdown:     ScoreBreakdown  


# ── Batch ──────────────────────────────────────────────────

class BatchAnalyseRequest(BaseModel):
    resumes: list[str] = Field(..., min_length=1, max_length=10)
    job_description: str = Field(..., min_length=50, max_length=8000)

    @field_validator("resumes")
    @classmethod
    def validate_resumes(cls, v):
        for i, resume in enumerate(v):
            if len(resume.strip()) < 50:
                raise ValueError(f"Resume at index {i} is too short")
        return v


class RankedCandidate(BaseModel):
    rank:             int
    candidate_name:   str
    overall_score:    int
    recommendation:   Literal["advance", "hold", "reject"]
    confidence:       Literal["low", "medium", "high"]
    strengths:        list[str]
    matching_skills:  list[str]
    gaps:             list[Gap]
    jd_match_breakdown: dict[str, int]
    interview_questions: list[InterviewQuestion]
    score_breakdown:  ScoreBreakdown


class BatchAnalysisResponse(BaseModel):
    total:      int
    advance:    int
    hold:       int
    reject:     int
    candidates: list[RankedCandidate]


# ── Results history ────────────────────────────────────────

class ScreeningResultResponse(BaseModel):
    id:                  int
    candidate_name:      str
    overall_score:       int
    recommendation:      Literal["advance", "hold", "reject"]
    confidence:          Literal["low", "medium", "high"]
    rule_score:          int
    llm_score:           int
    matching_skills:     list[str]
    gaps:                list[Gap]
    strengths:           list[str]
    jd_match_breakdown:  dict[str, int]
    interview_questions: list[InterviewQuestion]
    score_breakdown:     ScoreBreakdown
    resume_snippet:      str
    created_at:          str

    model_config = {"from_attributes": True}


class ResultsResponse(BaseModel):
    total:   int
    results: list[ScreeningResultResponse]