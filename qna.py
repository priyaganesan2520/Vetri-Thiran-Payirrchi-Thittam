"""
qna.py
Question and Answer logic powered by Google Gemini.
Handles academic and general knowledge queries with structured, clear answers.
"""

from typing import Dict, Any, Optional
from gemini_service import call_gemini

SYSTEM_INSTRUCTION = (
    "You are EduGenie, an intelligent, empathetic, and encouraging educational tutor. "
    "Your mission is to help students and self-learners understand academic and general knowledge topics. "
    "Provide accurate, direct, and well-structured answers. "
    "Use bullet points, bold key terms, short paragraphs, or formulas where applicable. "
    "Keep answers concise, clear, and easy to grasp for students without skipping essential facts."
)


def answer_question(question: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Answers an academic or general knowledge question using Google Gemini.
    """
    if not question or not question.strip():
        return {
            "success": False,
            "error": "Question cannot be empty. Please ask an academic or general knowledge question.",
            "answer": ""
        }

    clean_question = question.strip()
    prompt = (
        f"Student Question: {clean_question}\n\n"
        "Please provide a comprehensive yet concise educational explanation and direct answer."
    )

    result = call_gemini(
        prompt=prompt,
        system_instruction=SYSTEM_INSTRUCTION,
        api_key=api_key,
        temperature=0.3
    )

    if not result.get("success"):
        return {
            "success": False,
            "error": result.get("error", "Failed to retrieve answer from Gemini."),
            "answer": ""
        }

    return {
        "success": True,
        "question": clean_question,
        "answer": result.get("output_text", ""),
        "model": result.get("model", "Gemini")
    }
