import os
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
from app.models.schemas import Gap, InterviewQuestion

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a technical interviewer generating targeted questions from candidate skill gaps.
You must respond ONLY with a valid JSON object. No markdown, no explanation, no extra text.

Rules:
- Generate exactly 5 questions total
- Prioritise must-have gaps first, then preferred gaps
- If fewer than 5 gaps exist, generate multiple questions per critical gap
- Each question must directly probe the identified gap — no generic questions
- must-ask: targets a must-have gap
- optional: targets a preferred gap

Question quality rules:
- Be specific, not generic ("Walk me through containerising a FastAPI app" not "Do you know Docker?")
- Questions should reveal depth of knowledge, not just yes/no answers
- Reference the candidate's existing experience where possible to make it conversational
"""

def build_prompt(gaps: list[Gap], candidate_name: str) -> str:
    gaps_data = [
        {
            "skill": g.skill,
            "severity": g.severity
        }
        for g in gaps
    ]

    return f"""Generate exactly 5 targeted interview questions for {candidate_name} based on these gaps:

Gaps:
{json.dumps(gaps_data, indent=2)}

Return as JSON:
{{
  "interview_questions": [
    {{
      "question": "specific targeted question text",
      "targets_gap": "exact skill name from gaps list",
      "priority": "must-ask | optional"
    }}
  ]
}}

Rules:
- must-ask questions target must-have gaps
- optional questions target preferred gaps
- Generate exactly 5 questions total
- If no gaps exist return 5 general senior-level technical questions
"""

async def generate_interview_questions(
    gaps: list[Gap],
    candidate_name: str
) -> list[InterviewQuestion]:
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(gaps, candidate_name)}
            ]
        )

        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)

        questions = [
            InterviewQuestion(**q)
            for q in data.get("interview_questions", [])
        ]

        if not questions:
            raise ValueError("Interview Agent returned no questions")

        return questions

    except json.JSONDecodeError as e:
        raise ValueError(f"Interview Agent failed to parse JSON: {e}")
    except Exception as e:
        raise RuntimeError(f"Interview Agent error: {e}")