import os
import asyncio
import logging
import logging.config
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from app.models.schemas import (
    AnalyseRequest, AnalysisResponse, ScoreBreakdown,
    BatchAnalyseRequest, BatchAnalysisResponse, RankedCandidate,
    ResultsResponse, ScreeningResultResponse,
    Gap, InterviewQuestion
)
from app.agents.resume_agent import extract_candidate_profile
from app.agents.jd_agent import extract_job_profile
from app.agents.match_agent import score_candidate
from app.agents.interview_agent import generate_interview_questions
from app.services.rule_scorer import compute_rule_score
from app.services.aggregator import aggregate_scores, derive_recommendation
from app.db.database import get_db, engine
from app.db.models import ScreeningResult, Base

load_dotenv()

# ── Create tables ──────────────────────────────────────────
Base.metadata.create_all(bind=engine)

# ── Logging ────────────────────────────────────────────────
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            "datefmt": "%H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
})

logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────
app = FastAPI(
    title="Resume Analyser API",
    description="4-agent pipeline for candidate-JD fit analysis",
    version="2.0.0"
)

# ── CORS ───────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Core pipeline ──────────────────────────────────────────
async def run_pipeline(resume_text: str, jd_text: str) -> dict:
    """Shared pipeline used by both /analyse and /batch."""

    # Agent 1 + 2 parallel
    candidate_profile, job_profile = await asyncio.wait_for(
        asyncio.gather(
            extract_candidate_profile(resume_text),
            extract_job_profile(jd_text)
        ),
        timeout=30.0
    )

    # Rule scorer
    rule_data = compute_rule_score(candidate_profile, job_profile)

    # Agent 3
    match_data = await asyncio.wait_for(
        score_candidate(candidate_profile, job_profile, rule_data),
        timeout=30.0
    )

    # Aggregate
    score_data = aggregate_scores(
        rule_score=rule_data["rule_score"],
        llm_score=match_data["overall_score"]
    )
    final_recommendation = derive_recommendation(score_data["final_score"])

    # Agent 4
    questions = await asyncio.wait_for(
        generate_interview_questions(
            gaps=match_data["gaps"],
            candidate_name=match_data["candidate_name"]
        ),
        timeout=30.0
    )

    return {
        "candidate_name":      match_data["candidate_name"],
        "overall_score":       score_data["final_score"],
        "confidence":          match_data["confidence"],
        "recommendation":      final_recommendation,
        "strengths":           match_data.get("strengths", []),
        "matching_skills":     match_data["matching_skills"],
        "gaps":                match_data["gaps"],
        "jd_match_breakdown":  match_data["jd_match_breakdown"],
        "interview_questions": questions,
        "score_breakdown":     ScoreBreakdown(
            final_score=score_data["final_score"],
            rule_score=score_data["rule_score"],
            llm_score=score_data["llm_score"],
            rule_weight=score_data["rule_weight"],
            llm_weight=score_data["llm_weight"],
            required_coverage=rule_data["required_coverage"],
            preferred_coverage=rule_data["preferred_coverage"],
            seniority_alignment=rule_data["seniority_alignment"],
        ),
        "resume_snippet": resume_text[:200],
    }


def persist_result(db: Session, result: dict):
    """Save a pipeline result to PostgreSQL."""
    record = ScreeningResult(
        candidate_name=result["candidate_name"],
        overall_score=result["overall_score"],
        recommendation=result["recommendation"],
        confidence=result["confidence"],
        rule_score=result["score_breakdown"].rule_score,
        llm_score=result["score_breakdown"].llm_score,
        matching_skills=result["matching_skills"],
        gaps=[g.model_dump() for g in result["gaps"]],
        strengths=result["strengths"],
        jd_match_breakdown=result["jd_match_breakdown"],
        interview_questions=[q.model_dump() for q in result["interview_questions"]],
        score_breakdown=result["score_breakdown"].model_dump(),
        resume_snippet=result["resume_snippet"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


# ── Routes ─────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyse", response_model=AnalysisResponse)
async def analyse(
    request: AnalyseRequest,
    db: Session = Depends(get_db)
):
    logger.info("POST /analyse — request received")
    try:
        result = await run_pipeline(request.resume_text, request.job_description)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Pipeline timed out")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    persist_result(db, result)
    logger.info(f"POST /analyse — complete | score={result['overall_score']} | rec={result['recommendation']}")

    return AnalysisResponse(**{k: v for k, v in result.items() if k != "resume_snippet"})


@app.post("/batch", response_model=BatchAnalysisResponse)
async def batch_analyse(
    request: BatchAnalyseRequest,
    db: Session = Depends(get_db)
):
    logger.info(f"POST /batch — {len(request.resumes)} resumes received")

    # Run all resumes in parallel against same JD
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *[run_pipeline(resume, request.job_description)
                  for resume in request.resumes],
                return_exceptions=True
            ),
            timeout=120.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Batch pipeline timed out")

    # Filter out failed pipelines
    successful = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Batch: resume {i} failed — {result}")
        else:
            successful.append(result)

    if not successful:
        raise HTTPException(status_code=502, detail="All resumes failed to process")

    # Sort by score descending
    successful.sort(key=lambda x: x["overall_score"], reverse=True)

    # Persist all
    for result in successful:
        persist_result(db, result)

    # Build ranked response
    candidates = []
    for rank, result in enumerate(successful, start=1):
        candidates.append(RankedCandidate(
            rank=rank,
            candidate_name=result["candidate_name"],
            overall_score=result["overall_score"],
            recommendation=result["recommendation"],
            confidence=result["confidence"],
            strengths=result["strengths"],
            matching_skills=result["matching_skills"],
            gaps=result["gaps"],
            jd_match_breakdown=result["jd_match_breakdown"],
            interview_questions=result["interview_questions"],
            score_breakdown=result["score_breakdown"],
        ))

    summary = {
        "advance": sum(1 for c in candidates if c.recommendation == "advance"),
        "hold":    sum(1 for c in candidates if c.recommendation == "hold"),
        "reject":  sum(1 for c in candidates if c.recommendation == "reject"),
    }

    logger.info(f"POST /batch — complete | {len(candidates)} ranked | advance={summary['advance']}")

    return BatchAnalysisResponse(
        total=len(candidates),
        candidates=candidates,
        **summary
    )


@app.get("/results", response_model=ResultsResponse)
async def get_results(db: Session = Depends(get_db)):
    logger.info("GET /results — fetching last 50")

    records = (
        db.query(ScreeningResult)
        .order_by(ScreeningResult.created_at.desc())
        .limit(50)
        .all()
    )

    results = []
    for r in records:
        results.append(ScreeningResultResponse(
            id=r.id,
            candidate_name=r.candidate_name,
            overall_score=r.overall_score,
            recommendation=r.recommendation,
            confidence=r.confidence,
            rule_score=r.rule_score,
            llm_score=r.llm_score,
            matching_skills=r.matching_skills,
            gaps=[Gap(**g) for g in r.gaps],
            strengths=r.strengths,
            jd_match_breakdown=r.jd_match_breakdown,
            interview_questions=[InterviewQuestion(**q) for q in r.interview_questions],
            score_breakdown=ScoreBreakdown(**r.score_breakdown),
            resume_snippet=r.resume_snippet,
            created_at=r.created_at.isoformat(),
        ))

    return ResultsResponse(total=len(results), results=results)