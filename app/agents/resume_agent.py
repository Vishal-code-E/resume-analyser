import os
import json
from openai import AsyncOpenAI
from app.models.schemas import CandidateProfile

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(resume_text)}
            ]
        )

        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        return CandidateProfile(**data)

    except json.JSONDecodeError as e:
        raise ValueError(f"Resume Agent failed to parse JSON: {e}")
    except Exception as e:
        raise RuntimeError(f"Resume Agent error: {e}")