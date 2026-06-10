import json
import logging
import openai
from app.core.client import openai_client
from app.models.schemas import CandidateProfile

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a resume parser. Extract structured information from resumes.
You must respond ONLY with a valid JSON object. No markdown, no explanation, no extra text.

Scoring rubric for seniority_level:
- junior: 0-2 years
- mid: 2-5 years
- senior: 5+ years
"""

def build_prompt(resume_text: str) -> str:
    return f"""Extract the following from this resume and return as JSON:

{{
  "name": "full name of candidate",
  "skills": ["list", "of", "technical", "skills"],
  "experience_years": <integer>,
  "projects": ["brief project descriptions"],
  "seniority_level": "junior | mid | senior"
}}

Resume:
{resume_text}
"""

async def extract_candidate_profile(resume_text: str) -> CandidateProfile:
    logger.info("Resume Agent: starting extraction")
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(resume_text)}
            ]
        )
        raw = response.choices[0].message.content.strip()
        logger.info("Resume Agent: extraction complete")
        data = json.loads(raw)
        return CandidateProfile(**data)

    except json.JSONDecodeError as e:
        logger.error(f"Resume Agent: JSON parse failed — {e}")
        raise ValueError(f"Resume Agent failed to parse JSON: {e}")
    except openai.AuthenticationError:
        logger.error("Resume Agent: invalid OpenAI API key")
        raise RuntimeError("Invalid OpenAI API key")
    except openai.RateLimitError:
        logger.error("Resume Agent: rate limit hit")
        raise RuntimeError("OpenAI rate limit reached — try again shortly")
    except openai.APITimeoutError:
        logger.error("Resume Agent: request timed out")
        raise RuntimeError("OpenAI request timed out")
    except openai.APIError as e:
        logger.error(f"Resume Agent: OpenAI API error — {e}")
        raise RuntimeError(f"OpenAI API error: {e}")
    except Exception as e:
        logger.error(f"Resume Agent: unexpected error — {e}")
        raise RuntimeError(f"Resume Agent error: {e}")