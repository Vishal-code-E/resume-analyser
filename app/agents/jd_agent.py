import json
import logging
import openai
from app.core.client import openai_client
from app.models.schemas import JobProfile

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a job description parser. Extract structured hiring requirements from JDs.
You must respond ONLY with a valid JSON object. No markdown, no explanation, no extra text.

Scoring rubric for seniority_expected:
- junior: 0-2 years required
- mid: 2-5 years required
- senior: 5+ years required

For domains: extract the actual skill domains mentioned in the JD dynamically.
Examples: "Python/Backend", "GenAI/RAG", "Infrastructure", "Frontend", "Data Engineering"
Do NOT hardcode categories — derive them from what the JD actually emphasises.
"""

def build_prompt(jd_text: str) -> str:
    return f"""Extract the following from this job description and return as JSON:

{{
  "role_title": "exact job title",
  "required_skills": ["skills explicitly marked required or must-have"],
  "preferred_skills": ["skills marked preferred, nice-to-have, or bonus"],
  "domains": ["dynamically extracted domain categories from this JD"],
  "seniority_expected": "junior | mid | senior"
}}


Critical extraction rules:
- If a requirement lists alternatives (e.g. "Django or FastAPI", "Postgres / MySQL"),
  preserve it as a SINGLE entry exactly as written e.g. "Django or FastAPI"
  Do NOT split OR conditions into separate skills
- Only split into separate entries if they are clearly independent requirements

Job Description:
{jd_text}
"""

async def extract_job_profile(jd_text: str) -> JobProfile:
    logger.info("JD Agent: starting extraction")
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(jd_text)}
            ]
        )
        raw = response.choices[0].message.content.strip()
        logger.info("JD Agent: extraction complete")
        data = json.loads(raw)
        return JobProfile(**data)

    except json.JSONDecodeError as e:
        logger.error(f"JD Agent: JSON parse failed — {e}")
        raise ValueError(f"JD Agent failed to parse JSON: {e}")
    except openai.AuthenticationError:
        logger.error("JD Agent: invalid OpenAI API key")
        raise RuntimeError("Invalid OpenAI API key")
    except openai.RateLimitError:
        logger.error("JD Agent: rate limit hit")
        raise RuntimeError("OpenAI rate limit reached — try again shortly")
    except openai.APITimeoutError:
        logger.error("JD Agent: request timed out")
        raise RuntimeError("OpenAI request timed out")
    except openai.APIError as e:
        logger.error(f"JD Agent: OpenAI API error — {e}")
        raise RuntimeError(f"OpenAI API error: {e}")
    except Exception as e:
        logger.error(f"JD Agent: unexpected error — {e}")
        raise RuntimeError(f"JD Agent error: {e}")