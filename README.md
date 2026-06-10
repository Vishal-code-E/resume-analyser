# Resume Analyser API

A production-grade AI pipeline that analyses candidate-JD fit using a 4-agent architecture built on FastAPI and OpenAI.



---

## Architecture

```
POST /analyse
       │
       ├──► Agent 1: Resume Extractor ──── parallel ────┐
       │    → CandidateProfile                          │
       │      (name, skills, experience, seniority)     ├──► Agent 3: Match Scorer
       │                                                │    → rubric-anchored scoring
       ├──► Agent 2: JD Extractor ──────── parallel ───┘     → dynamic domain breakdown
            → JobProfile                                │     → gap detection with severity
              (required, preferred,                     │
               domains, seniority)                      ▼
                                               Rule-Based Scorer (pure Python)
                                               → required coverage %
                                               → preferred coverage %
                                               → seniority alignment %
                                                        │
                                                        ▼
                                               Score Aggregator
                                               → 40% rule + 60% LLM
                                               → deterministic recommendation
                                                        │
                                                        ▼
                                               Agent 4: Interview Generator
                                               → gap-targeted questions
                                               → must-ask / optional priority
                                                        │
                                                        ▼
                                               PostgreSQL persistence
                                               → ScreeningResult ORM
```

---

## Tech Stack

| Layer | Tech |
|---|---|
| API | FastAPI |
| AI | OpenAI GPT-4o |
| ORM | SQLAlchemy |
| Database | PostgreSQL (Docker) |
| Validation | Pydantic v2 |
| Containerisation | Docker + docker-compose |

---

## Project Structure

```
resume-analyser/
├── app/
│   ├── main.py                    # FastAPI app — all endpoints
│   ├── agents/
│   │   ├── resume_agent.py        # Agent 1 — extracts CandidateProfile
│   │   ├── jd_agent.py            # Agent 2 — extracts JobProfile
│   │   ├── match_agent.py         # Agent 3 — rubric-anchored scorer
│   │   └── interview_agent.py     # Agent 4 — gap question generator
│   ├── services/
│   │   ├── normalizer.py          # Skill synonym + OR condition handling
│   │   ├── rule_scorer.py         # Deterministic scoring (pure Python)
│   │   └── aggregator.py          # 40/60 rule/LLM blend
│   ├── models/
│   │   └── schemas.py             # All Pydantic models
│   ├── db/
│   │   ├── database.py            # SQLAlchemy engine + session
│   │   └── models.py              # ScreeningResult ORM table
│   └── core/
│       └── client.py              # Shared OpenAI client singleton
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Setup

### Local Development

**1 — Clone and install:**
```bash
git clone https://github.com/Vishal-code-E/resume-analyser.git
cd resume-analyser
pip install -r requirements.txt
```

**2 — Environment:**
```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

**3 — Start PostgreSQL via Docker:**
```bash
docker-compose up db -d
```

**4 — Run the API:**
```bash
uvicorn app.main:app --reload
```

API available at `http://localhost:8000`
Swagger docs at `http://localhost:8000/docs`

---

### Full Docker Setup

Run everything (API + DB) in Docker:
```bash
docker-compose up --build
```

---

## Environment Variables

```env
OPENAI_API_KEY=sk-your-key-here
DATABASE_URL=postgresql://resume:resume@localhost:5432/resume_db
```

---

## API Endpoints

### `POST /analyse`

Analyse a single candidate against a job description.

**Request:**
```json
{
  "resume_text": "Candidate resume as plain text (50–15000 chars)",
  "job_description": "Job description as plain text (50–8000 chars)"
}
```

**Response:**
```json
{
  "candidate_name": "Sarah Mitchell",
  "overall_score": 80,
  "confidence": "high",
  "recommendation": "advance",
  "strengths": ["Strong FastAPI fluency", "Deep RAG experience"],
  "matching_skills": ["Python", "FastAPI", "LangChain"],
  "gaps": [
    { "skill": "vector databases", "severity": "must-have" },
    { "skill": "Kubernetes", "severity": "preferred" }
  ],
  "jd_match_breakdown": {
    "GenAI/Backend": 90,
    "RAG": 85,
    "Infrastructure": 70
  },
  "interview_questions": [
    {
      "question": "Walk me through a vector database you've worked with...",
      "targets_gap": "vector databases",
      "priority": "must-ask"
    }
  ],
  "score_breakdown": {
    "final_score": 80,
    "rule_score": 73,
    "llm_score": 85,
    "rule_weight": 0.4,
    "llm_weight": 0.6,
    "required_coverage": 75,
    "preferred_coverage": 40,
    "seniority_alignment": 100
  }
}
```

---

### `POST /batch`

Analyse up to 10 candidates in parallel against one JD. Returns candidates ranked by score.

**Request:**
```json
{
  "resumes": [
    "Candidate 1 resume text...",
    "Candidate 2 resume text...",
    "Candidate 3 resume text..."
  ],
  "job_description": "Job description text..."
}
```

**Response:**
```json
{
  "total": 3,
  "advance": 1,
  "hold": 1,
  "reject": 1,
  "candidates": [
    {
      "rank": 1,
      "candidate_name": "Sarah Mitchell",
      "overall_score": 81,
      "recommendation": "advance",
      ...
    }
  ]
}
```

---

### `GET /results`

Fetch the last 50 screening results from PostgreSQL.

**Response:**
```json
{
  "total": 50,
  "results": [
    {
      "id": 1,
      "candidate_name": "Sarah Mitchell",
      "overall_score": 81,
      "recommendation": "advance",
      "rule_score": 74,
      "llm_score": 85,
      "resume_snippet": "Sarah Mitchell — Full Stack Engineer...",
      "created_at": "2026-06-10T09:51:00.696948+00:00",
      ...
    }
  ]
}
```

---

### `GET /health`

```json
{ "status": "ok" }
```

---

## Scoring System

### How scores are calculated

```
final_score = (rule_score × 0.40) + (llm_score × 0.60)
```

**Rule score** (deterministic, same input = same output):
- Required skill coverage: 55% weight
- Preferred skill coverage: 20% weight
- Seniority alignment: 25% weight

**LLM score** (contextual, evaluates project depth and evidence):
- Rubric-anchored prompt prevents score drift
- Temperature = 0 for scoring agents

**Recommendation thresholds:**
- `advance`: score ≥ 70
- `hold`: score 45–69
- `reject`: score < 45

---

## Skill Normalisation

The normalizer handles:
- **OR conditions** — `"Django or FastAPI"` → candidate needs only one
- **Synonyms** — `"REST APIs"` matches `"REST API design"`
- **Abbreviations** — `"k8s"` matches `"Kubernetes"`
- **Case insensitive** — all comparisons lowercased

---

## Error Handling

| Code | Meaning |
|---|---|
| 422 | Invalid input or agent parse failure |
| 502 | OpenAI API error (bad key, rate limit) |
| 504 | Agent timeout (30s per stage) |
| 500 | Response assembly failure |

---

## License

MIT
