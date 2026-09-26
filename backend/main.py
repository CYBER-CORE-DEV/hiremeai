import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field
from pypdf import PdfReader

# ==========================================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

client = Groq(api_key=GROQ_API_KEY)

model = "openai/gpt-oss-20b"

# ==========================================================
# FASTAPI APP
# ==========================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# PYDANTIC MODELS
# ==========================================================

class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = Field(default_factory=list)


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None

    total_experience_years: float | None = None

    skills: list[str] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    question: str


resume_schema = Resume.model_json_schema()

# ==========================================================
# PDF READER
# ==========================================================

def read_pdf(file_path: Path) -> str:
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ==========================================================
# RESUME PARSER
# ==========================================================

def parse_resume(resume_text: str) -> Resume:

    system_prompt = f"""
You are an expert resume parser.

Extract information from the resume based on meaning,
not only section headings.

Return ONLY valid JSON matching this schema:

{resume_schema}

Rules:

1. Do not invent information.
2. If unavailable return null.
3. Empty lists if no data exists.
4. Include internships inside experiences.
5. Extract skills from the entire resume.
"""

    user_prompt = f"""
Parse this resume:

{resume_text}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        response_format={"type": "json_object"}
    )

    raw_output = response.choices[0].message.content

    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid JSON returned by model: {raw_output}"
        )

    return Resume(**data)


# ==========================================================
# INTERVIEW CHATBOT
# ==========================================================

def ask_candidate(question: str, resume: Resume):

    system_prompt = f"""
You are an AI assistant representing a job candidate.

Candidate Information:

{resume.model_dump_json(indent=2)}

Rules:

1. Answer ONLY from the information provided.
2. Never hallucinate.

3. If information is unavailable, reply:

"I don't have enough information to answer that."

4. Be professional.
5. Answer as if HR is interviewing the candidate.
6.If asked 'what is your name?'
reply--my name is Subhajit Dhar.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.choices[0].message.content


# ==========================================================
# ROUTES
# ==========================================================

@app.get("/")
def home():
    return {
        "message": "Resume Interview Bot API Running"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    pdf_path = Path("Subhajit Dhar cv (1).pdf")

    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Resume PDF not found."
        )

    try:
        resume_text = read_pdf(pdf_path)

        resume = parse_resume(resume_text)

        answer = ask_candidate(
            request.question,
            resume
        )

        return {
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================================
# LOCAL RUN
# ==========================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )