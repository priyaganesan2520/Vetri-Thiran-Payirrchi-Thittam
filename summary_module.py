"""
summary_module.py
Educational summarization logic powered by Google Gemini.
Condenses long educational texts into structured, revision-friendly summaries.
"""

from typing import Dict, Any, Optional
from gemini_service import call_gemini

SUMMARY_SYSTEM_INSTRUCTION = (
    "You are EduGenie's Study Revision Expert. "
    "Your objective is to read lengthy educational passages, extract the core ideas, "
    "and create a concise, high-impact study summary. "
    "Eliminate fluff and repetitive phrasing. "
    "Structure your response with:\n"
    "1. 📌 **Quick Overview** (1-2 sentences capturing the central thesis)\n"
    "2. 🔑 **Core Key Points & Concepts** (clean bullet points explaining the critical mechanisms or facts)\n"
    "3. ⚡ **Quick Revision Takeaway** (a brief memory hook or essential formula/rule)\n"
    "Ensure the summary is beginner-friendly and optimized for rapid exam or topic revision."
)


def summarize_passage(text: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Summarizes an educational passage into structured revision notes.
    """
    if not text or not text.strip():
        return {
            "success": False,
            "error": "Passage text cannot be empty. Please provide text or an article to summarize.",
            "summary": ""
        }

    clean_text = text.strip()
    prompt = (
        "Please summarize the following educational text for quick revision:\n\n"
        f"\"\"\"\n{clean_text}\n\"\"\"\n\n"
        "Provide a clean, bulleted, and structured summary preserving all critical facts."
    )

    result = call_gemini(
        prompt=prompt,
        system_instruction=SUMMARY_SYSTEM_INSTRUCTION,
        api_key=api_key,
        temperature=0.3
    )

    if not result.get("success"):
        return {
            "success": False,
            "error": result.get("error", "Failed to summarize text with Gemini."),
            "summary": ""
        }

    return {
        "success": True,
        "original_length": len(clean_text),
        "summary": result.get("output_text", ""),
        "model": result.get("model", "Gemini")
    }
