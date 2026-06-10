import os
import asyncio
import logging
import logging.config
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.models.schemas import AnalyseRequest, AnalysisResponse
from app.agents.resume_agent import extract_candidate_profile
from app.agents.jd_agent import extract_job_profile
from app.agents.match_agent import score_candidate
from app.agents.interview_agent import generate_interview_questions

load_dotenv()

# ── Logging config ─────────────────────────────────────────
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
    version="1.0.0"
)

# ── CORS ───────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ────────────────────────────────────────────────
def derive_recommendation(score: int) -> str:
    if score >= 70:
        return "advance"
    if score >= 45:
        return "hold"
    return "reject"


# ── Routes ─────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyse", response_model=AnalysisResponse)
async def analyse(request: AnalyseRequest):
    logger.info("POST /analyse — request received")

    # ── Agent 1 + 2 parallel with timeout ─────────────────
    try:
        candidate_profile, job_profile = await asyncio.wait_for(
            asyncio.gather(
                extract_candidate_profile(request.resume_text),
                extract_job_profile(request.job_description)
            ),
            timeout=30.0
        )
        logger.info(f"Extraction complete — candidate={candidate_profile.name}")
    except asyncio.TimeoutError:
        logger.error("Extraction agents timed out")
        raise HTTPException(status_code=504, detail="Extraction timed out — try again")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # ── Agent 3 — scoring with timeout ────────────────────
    try:
        match_data = await asyncio.wait_for(
            score_candidate(candidate_profile, job_profile),
            timeout=30.0
        )
        logger.info(f"Scoring complete — raw_score={match_data.get('overall_score')}")
    except asyncio.TimeoutError:
        logger.error("Match Agent timed out")
        raise HTTPException(status_code=504, detail="Scoring timed out — try again")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # ── Programmatic recommendation — overrides LLM ───────
    final_recommendation = derive_recommendation(match_data["overall_score"])
    logger.info(f"Recommendation derived — {final_recommendation}")

    # ── Agent 4 — interview questions with timeout ─────────
    try:
        questions = await asyncio.wait_for(
            generate_interview_questions(
                gaps=match_data["gaps"],
                candidate_name=match_data["candidate_name"]
            ),
            timeout=30.0
        )
        logger.info(f"Interview Agent complete — {len(questions)} questions")
    except asyncio.TimeoutError:
        logger.error("Interview Agent timed out")
        raise HTTPException(status_code=504, detail="Interview generation timed out — try again")
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
            recommendation=final_recommendation,
            matching_skills=match_data["matching_skills"],
            strengths=match_data.get("strengths", []), 
            gaps=match_data["gaps"],
            jd_match_breakdown=match_data["jd_match_breakdown"],
            interview_questions=questions
        )
    except Exception as e:
        logger.error(f"Response assembly failed — {e}")
        raise HTTPException(status_code=500, detail=f"Response assembly failed: {e}")

    logger.info(f"POST /analyse — complete | score={result.overall_score} | rec={result.recommendation}")
    return result