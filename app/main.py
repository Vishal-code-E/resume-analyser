import os
import asyncio
import logging
import logging.config
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.models.schemas import AnalyseRequest, AnalysisResponse, ScoreBreakdown
from app.agents.resume_agent import extract_candidate_profile
from app.agents.jd_agent import extract_job_profile
from app.agents.match_agent import score_candidate
from app.agents.interview_agent import generate_interview_questions
from app.services.rule_scorer import compute_rule_score
from app.services.aggregator import aggregate_scores, derive_recommendation

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyse", response_model=AnalysisResponse)
async def analyse(request: AnalyseRequest):
    logger.info("POST /analyse — request received")

    # ── Agent 1 + 2 parallel ──────────────────────────────
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

    # ── Rule scorer — pure Python, instant ────────────────
    rule_data = compute_rule_score(candidate_profile, job_profile)
    logger.info(
        f"Rule scorer complete — "
        f"score={rule_data['rule_score']} "
        f"required={rule_data['required_coverage']}% "
        f"seniority={rule_data['seniority_alignment']}%"
    )

    # ── Agent 3 — LLM depth scoring ───────────────────────
    try:
        match_data = await asyncio.wait_for(
            score_candidate(candidate_profile, job_profile, rule_data),
            timeout=30.0
        )
        logger.info(f"Match Agent complete — llm_score={match_data.get('overall_score')}")
    except asyncio.TimeoutError:
        logger.error("Match Agent timed out")
        raise HTTPException(status_code=504, detail="Scoring timed out — try again")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # ── Aggregate rule + LLM scores ───────────────────────
    score_data = aggregate_scores(
        rule_score=rule_data["rule_score"],
        llm_score=match_data["overall_score"]
    )
    final_recommendation = derive_recommendation(score_data["final_score"])
    logger.info(
        f"Aggregator complete — "
        f"final={score_data['final_score']} "
        f"rec={final_recommendation}"
    )

    # ── Agent 4 — interview questions ─────────────────────
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
            overall_score=score_data["final_score"],
            confidence=match_data["confidence"],
            recommendation=final_recommendation,
            strengths=match_data.get("strengths", []),
            matching_skills=match_data["matching_skills"],
            gaps=match_data["gaps"],
            jd_match_breakdown=match_data["jd_match_breakdown"],
            interview_questions=questions,
            score_breakdown=ScoreBreakdown(
                final_score=score_data["final_score"],
                rule_score=score_data["rule_score"],
                llm_score=score_data["llm_score"],
                rule_weight=score_data["rule_weight"],
                llm_weight=score_data["llm_weight"],
                required_coverage=rule_data["required_coverage"],
                preferred_coverage=rule_data["preferred_coverage"],
                seniority_alignment=rule_data["seniority_alignment"],
            )
        )
    except Exception as e:
        logger.error(f"Response assembly failed — {e}")
        raise HTTPException(status_code=500, detail=f"Response assembly failed: {e}")

    logger.info(
        f"POST /analyse — complete | "
        f"final={result.overall_score} | "
        f"rule={score_data['rule_score']} | "
        f"llm={score_data['llm_score']} | "
        f"rec={result.recommendation}"
    )
    return result