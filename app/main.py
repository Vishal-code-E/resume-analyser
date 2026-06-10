import os
import asyncio
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from app.models.schemas import AnalyseRequest, AnalysisResponse
from app.agents.resume_agent import extract_candidate_profile
from app.agents.jd_agent import extract_job_profile
from app.agents.match_agent import score_candidate
from app.agents.interview_agent import generate_interview_questions

load_dotenv()

app = FastAPI(
    title="Resume Analyser API",
    description="4-agent pipeline for candidate-JD fit analysis",
    version="1.0.0"
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyse", response_model=AnalysisResponse)
async def analyse(request: AnalyseRequest):

    # ── Input validation ───────────────────────────────────
    if not request.resume_text.strip():
        raise HTTPException(status_code=422, detail="resume_text cannot be empty")
    if not request.job_description.strip():
        raise HTTPException(status_code=422, detail="job_description cannot be empty")

    # ── Agent 1 + 2 in parallel ────────────────────────────
    try:
        candidate_profile, job_profile = await asyncio.gather(
            extract_candidate_profile(request.resume_text),
            extract_job_profile(request.job_description)
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # ── Agent 3 — scoring ──────────────────────────────────
    try:
        match_data = await score_candidate(candidate_profile, job_profile)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # ── Agent 4 — interview questions ──────────────────────
    try:
        questions = await generate_interview_questions(
            gaps=match_data["gaps"],
            candidate_name=match_data["candidate_name"]
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # ── Assemble final response ────────────────────────────
    try:
        result = AnalysisResponse(
            candidate_name=match_data["candidate_name"],
            overall_score=match_data["overall_score"],
            confidence=match_data["confidence"],
            recommendation=match_data["recommendation"],
            matching_skills=match_data["matching_skills"],
            gaps=match_data["gaps"],
            jd_match_breakdown=match_data["jd_match_breakdown"],
            interview_questions=questions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Response assembly failed: {e}")

    return result