import json
import logging
import openai
from app.core.client import openai_client
from app.models.schemas import CandidateProfile, JobProfile, Gap

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior technical recruiter scoring candidate fit against a job description.
You must respond ONLY with a valid JSON object. No markdown, no explanation, no extra text.

Scoring rubric for overall_score:
- 0-25:   Major gaps, missing must-have skills, wrong seniority
- 26-50:  Partial match, several must-haves missing
- 51-75:  Decent match, some gaps but buildable
- 76-100: Strong match, most must-haves present, seniority aligned

Confidence rubric:
- low:    Resume is sparse or vague, hard to judge
- medium: Enough signal but some ambiguity
- high:   Clear evidence across most requirement areas

Gap severity:
- must-have: skill is in required_skills of the JD and candidate lacks it
- preferred: skill is in preferred_skills of the JD and candidate lacks it

For jd_match_breakdown:
- Use the exact domain names from the job profile domains list
- Score each domain 0-100 using the same rubric as overall_score
- Only score domains that exist in the JD


Critical matching rules:
- If a requirement states "X or Y" and the candidate has either X or Y, it is NOT a gap
- Only flag a gap if the candidate has NEITHER option in an OR requirement
- Match skills semantically — "REST APIs" matches "REST API design"
- Seniority mismatch is a soft signal, not a hard gap

Scoring rubric for overall_score:
"""

def build_prompt(candidate: CandidateProfile, job: JobProfile) -> str:
    return f"""Score this candidate against the job profile and return as JSON:

{{
  "candidate_name": "from candidate profile",
  "overall_score": <0-100 integer>,
  "confidence": "low | medium | high",
  "strengths": ["key strengths to highlight in recommendation"],
  "matching_skills": ["skills candidate has that appear in required or preferred lists"],
  "gaps": [
    {{"skill": "missing skill name", "severity": "must-have | preferred"}}
  ],
  "jd_match_breakdown": {{
    "<domain from job profile>": <0-100 integer>
  }}
}}

Candidate Profile:
{json.dumps(candidate.model_dump(), indent=2)}

Job Profile:
{json.dumps(job.model_dump(), indent=2)}
"""

async def score_candidate(
    candidate: CandidateProfile,
    job: JobProfile
) -> dict:
    logger.info(f"Match Agent: scoring {candidate.name}")
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(candidate, job)}
            ]
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        data["gaps"] = [Gap(**g) for g in data.get("gaps", [])]
        logger.info(f"Match Agent: scoring complete — score={data.get('overall_score')}")
        return data

    except json.JSONDecodeError as e:
        logger.error(f"Match Agent: JSON parse failed — {e}")
        raise ValueError(f"Match Agent failed to parse JSON: {e}")
    except openai.AuthenticationError:
        logger.error("Match Agent: invalid OpenAI API key")
        raise RuntimeError("Invalid OpenAI API key")
    except openai.RateLimitError:
        logger.error("Match Agent: rate limit hit")
        raise RuntimeError("OpenAI rate limit reached — try again shortly")
    except openai.APITimeoutError:
        logger.error("Match Agent: request timed out")
        raise RuntimeError("OpenAI request timed out")
    except openai.APIError as e:
        logger.error(f"Match Agent: OpenAI API error — {e}")
        raise RuntimeError(f"OpenAI API error: {e}")
    except Exception as e:
        logger.error(f"Match Agent: unexpected error — {e}")
        raise RuntimeError(f"Match Agent error: {e}")