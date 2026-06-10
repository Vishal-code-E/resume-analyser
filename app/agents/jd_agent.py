import os
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
from app.models.schemas import JobProfile

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

Job Description:
{jd_text}
"""

async def extract_job_profile(jd_text: str) -> JobProfile:
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(jd_text)}
            ]
        )

        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        return JobProfile(**data)

    except json.JSONDecodeError as e:
        raise ValueError(f"JD Agent failed to parse JSON: {e}")
    except Exception as e:
        raise RuntimeError(f"JD Agent error: {e}")