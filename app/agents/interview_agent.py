import json
import logging
import openai
from app.core.client import openai_client
from app.models.schemas import Gap, InterviewQuestion

logger = logging.getLogger(__name__)

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
- Be specific, not generic
- Questions should reveal depth of knowledge, not just yes/no answers
- Reference the candidate's existing experience where possible
"""

def build_prompt(gaps: list[Gap], candidate_name: str) -> str:
    gaps_data = [{"skill": g.skill, "severity": g.severity} for g in gaps]
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

- must-ask questions target must-have gaps
- optional questions target preferred gaps
- Generate exactly 5 questions total
- If no gaps exist return 5 general senior-level technical questions
"""

async def generate_interview_questions(
    gaps: list[Gap],
    candidate_name: str
) -> list[InterviewQuestion]:
    logger.info(f"Interview Agent: generating questions for {candidate_name}")
    try:
        response = await openai_client.chat.completions.create(
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
        questions = [InterviewQuestion(**q) for q in data.get("interview_questions", [])]

        if not questions:
            raise ValueError("Interview Agent returned no questions")

        logger.info(f"Interview Agent: {len(questions)} questions generated")
        return questions

    except json.JSONDecodeError as e:
        logger.error(f"Interview Agent: JSON parse failed — {e}")
        raise ValueError(f"Interview Agent failed to parse JSON: {e}")
    except openai.AuthenticationError:
        logger.error("Interview Agent: invalid OpenAI API key")
        raise RuntimeError("Invalid OpenAI API key")
    except openai.RateLimitError:
        logger.error("Interview Agent: rate limit hit")
        raise RuntimeError("OpenAI rate limit reached — try again shortly")
    except openai.APITimeoutError:
        logger.error("Interview Agent: request timed out")
        raise RuntimeError("OpenAI request timed out")
    except openai.APIError as e:
        logger.error(f"Interview Agent: OpenAI API error — {e}")
        raise RuntimeError(f"OpenAI API error: {e}")
    except Exception as e:
        logger.error(f"Interview Agent: unexpected error — {e}")
        raise RuntimeError(f"Interview Agent error: {e}")