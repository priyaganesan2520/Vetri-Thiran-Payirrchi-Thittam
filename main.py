"""
main.py
EduGenie: Google Gemini Powered Learning Assistant
FastAPI Application Entry Point
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, Header, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Ensure local module imports work regardless of working directory
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from explanation_module import explain_concept, MODEL_NAME as LAMINI_MODEL_NAME, get_pipeline
from qna import answer_question
from quiz_module import generate_quiz
from summary_module import summarize_passage
from learning_path import recommend_learning_path
from gemini_service import get_api_key, DEFAULT_MODEL as GEMINI_DEFAULT_MODEL

app = FastAPI(
    title="EduGenie: Google Gemini Powered Learning Assistant",
    description="A lightweight AI-powered educational assistant for students and self-learners.",
    version="1.0.0"
)

# Static files & template setup with resilient path resolution
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------------------------------------------------------
# Request Models
# ---------------------------------------------------------
class BaseRequest(BaseModel):
    input: Optional[str] = None
    api_key: Optional[str] = None


class QnARequest(BaseModel):
    question: Optional[str] = None
    input: Optional[str] = None
    api_key: Optional[str] = None


class ExplainRequest(BaseModel):
    topic: Optional[str] = None
    input: Optional[str] = None


class QuizRequest(BaseModel):
    topic: Optional[str] = None
    passage: Optional[str] = None
    input: Optional[str] = None
    api_key: Optional[str] = None


class SummarizeRequest(BaseModel):
    text: Optional[str] = None
    passage: Optional[str] = None
    input: Optional[str] = None
    api_key: Optional[str] = None


class LearningPathRequest(BaseModel):
    topic: Optional[str] = None
    subject: Optional[str] = None
    input: Optional[str] = None
    level: Optional[str] = "Beginner"
    api_key: Optional[str] = None


# Helper to extract API key from header or body
def resolve_api_key(body_key: Optional[str], header_key: Optional[str] = None) -> Optional[str]:
    return get_api_key(body_key or header_key)


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the EduGenie interactive dashboard."""
    has_api_key = bool(get_api_key())
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "has_api_key": has_api_key,
            "gemini_model": GEMINI_DEFAULT_MODEL,
            "local_model": LAMINI_MODEL_NAME
        }
    )


@app.get("/status")
async def system_status():
    """Returns runtime diagnostic information about configured models."""
    return {
        "status": "online",
        "gemini_api_configured": bool(get_api_key()),
        "gemini_model": GEMINI_DEFAULT_MODEL,
        "local_model": LAMINI_MODEL_NAME,
        "architecture": {
            "qa": "Google Gemini",
            "explain": f"Local {LAMINI_MODEL_NAME} (CPU compatible)",
            "quiz": "Google Gemini (Structured 3-MCQ)",
            "summarize": "Google Gemini",
            "learning_path": "Google Gemini"
        }
    }


@app.post("/qa")
async def endpoint_qa(
    payload: QnARequest,
    x_gemini_api_key: Optional[str] = Header(None)
):
    """
    Question and Answer:
    Answers academic or general knowledge questions concisely using Google Gemini.
    """
    question = (payload.question or payload.input or "").strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question is required. Please ask an educational question."
        )

    api_key = resolve_api_key(payload.api_key, x_gemini_api_key)
    result = answer_question(question=question, api_key=api_key)

    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "An error occurred while answering your question.")
        )
    return result


@app.post("/explain")
async def endpoint_explain(payload: ExplainRequest):
    """
    Concept Explanation:
    Explains any topic or concept using the lightweight local LaMini-Flan-T5-783M model.
    """
    topic = (payload.topic or payload.input or "").strip()
    if not topic:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Topic is required. Please enter a concept to explain."
        )

    result = explain_concept(topic=topic)
    return result


@app.post("/quiz")
async def endpoint_quiz(
    payload: QuizRequest,
    x_gemini_api_key: Optional[str] = Header(None)
):
    """
    Quiz Generation:
    Generates exactly 3 multiple-choice questions (4 options each) from a topic or passage.
    Returns structured JSON with markdown blocks cleaned.
    """
    topic_or_passage = (payload.passage or payload.topic or payload.input or "").strip()
    if not topic_or_passage:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Topic or passage is required to generate a quiz."
        )

    api_key = resolve_api_key(payload.api_key, x_gemini_api_key)
    result = generate_quiz(topic_or_passage=topic_or_passage, api_key=api_key)

    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "An error occurred while generating the quiz.")
        )
    return result


@app.post("/summarize")
async def endpoint_summarize(
    payload: SummarizeRequest,
    x_gemini_api_key: Optional[str] = Header(None)
):
    """
    Summarization:
    Condenses long educational texts into structured, quick-revision study notes using Gemini.
    """
    text = (payload.text or payload.passage or payload.input or "").strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text or passage is required to generate a summary."
        )

    api_key = resolve_api_key(payload.api_key, x_gemini_api_key)
    result = summarize_passage(text=text, api_key=api_key)

    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "An error occurred while summarizing the passage.")
        )
    return result


@app.post("/learn/recommendations")
async def endpoint_learning_recommendations(
    payload: LearningPathRequest,
    x_gemini_api_key: Optional[str] = Header(None)
):
    """
    Personalized Learning Path:
    Generates a structured beginner-to-advanced curriculum roadmap with recommended resources.
    """
    topic = (payload.topic or payload.subject or payload.input or "").strip()
    if not topic:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Subject or topic is required to generate a learning path."
        )

    api_key = resolve_api_key(payload.api_key, x_gemini_api_key)
    level = payload.level or "Beginner"
    result = recommend_learning_path(topic=topic, current_level=level, api_key=api_key)

    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "An error occurred while creating the learning path.")
        )
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
