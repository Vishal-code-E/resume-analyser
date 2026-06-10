import os
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
from app.models.schemas import CandidateProfile, JobProfile, Gap, AnalysisResponse

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

Recommendation rubric:
- advance: overall_score >= 70
- hold:    overall_score 45-69
- reject:  overall_score < 45

Gap severity:
- must-have: skill is in required_skills of the JD and candidate lacks it
- preferred: skill is in preferred_skills of the JD and candidate lacks it

For jd_match_breakdown:
- Use the exact domain names from the job profile domains list
- Score each domain 0-100 using the same rubric as overall_score
- Only score domains that exist in the JD
"""

def build_prompt(candidate: CandidateProfile, job: JobProfile) -> str:
    return f"""Score this candidate against the job profile and return as JSON:

{{
  "candidate_name": "from candidate profile",
  "overall_score": <0-100 integer>,
  "confidence": "low | medium | high",
  "recommendation": "advance | hold | reject",
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
    try:
        response = await client.chat.completions.create(
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

        # Validate gaps into typed Gap objects
        data["gaps"] = [Gap(**g) for g in data.get("gaps", [])]

        return data

    except json.JSONDecodeError as e:
        raise ValueError(f"Match Agent failed to parse JSON: {e}")
    except Exception as e:
        raise RuntimeError(f"Match Agent error: {e}")