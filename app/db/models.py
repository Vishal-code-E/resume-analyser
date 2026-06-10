from datetime import datetime, timezone
from sqlalchemy import Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column
from app.db.database import Base


class ScreeningResult(Base):
    __tablename__ = "screening_results"

    id:               Mapped[int]      = mapped_column(Integer, primary_key=True, index=True)
    candidate_name:   Mapped[str]      = mapped_column(String, index=True)
    overall_score:    Mapped[int]      = mapped_column(Integer)
    recommendation:   Mapped[str]      = mapped_column(String)
    confidence:       Mapped[str]      = mapped_column(String)
    rule_score:       Mapped[int]      = mapped_column(Integer)
    llm_score:        Mapped[int]      = mapped_column(Integer)
    matching_skills:  Mapped[list]     = mapped_column(JSON)
    gaps:             Mapped[list]     = mapped_column(JSON)
    strengths:        Mapped[list]     = mapped_column(JSON)
    jd_match_breakdown: Mapped[dict]   = mapped_column(JSON)
    interview_questions: Mapped[list]  = mapped_column(JSON)
    score_breakdown:  Mapped[dict]     = mapped_column(JSON)
    resume_snippet:   Mapped[str]      = mapped_column(String)
    created_at:       Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )