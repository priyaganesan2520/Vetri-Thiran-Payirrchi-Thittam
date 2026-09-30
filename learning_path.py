"""
learning_path.py
Personalized learning path generator powered by Google Gemini.
Creates structured, step-by-step educational roadmaps from beginner to advanced levels,
including recommended resources (books, videos, articles) and milestones.
"""

from typing import Dict, Any, Optional
from gemini_service import call_gemini

ROADMAP_SYSTEM_INSTRUCTION = (
    "You are EduGenie's Academic Curriculum and Career Mentor. "
    "Your goal is to build personalized, step-by-step learning paths for students and self-learners. "
    "Structure every learning path clearly into three progressive stages:\n"
    "1. 🌱 **Level 1: Beginner Fundamentals** (Core concepts, terminology, getting started, estimated timeline)\n"
    "2. 🌿 **Level 2: Intermediate Deep Dive** (Practical application, problem-solving, common patterns, estimated timeline)\n"
    "3. 🌳 **Level 3: Advanced Mastery & Projects** (Complex topics, optimization, real-world portfolio projects, estimated timeline)\n\n"
    "For each stage, specify:\n"
    "- Key topics to master (in increasing difficulty)\n"
    "- Actionable milestone project\n"
    "- Recommended learning resources (Specific high-rated books, video channels/tutorials, documentation, and practice platforms).\n\n"
    "Tone: Motivating, structured, highly practical, and clear."
)


def recommend_learning_path(
    topic: str,
    current_level: str = "Beginner",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a personalized, step-by-step learning roadmap from beginner to advanced.
    """
    if not topic or not topic.strip():
        return {
            "success": False,
            "error": "Subject or topic cannot be empty. Please enter what you want to learn (e.g., 'SQL', 'Data Science', 'Calculus').",
            "learning_path": ""
        }

    clean_topic = topic.strip()
    level_hint = (current_level or "Beginner").strip()

    prompt = (
        f"Create a comprehensive, step-by-step educational learning path for: **{clean_topic}**.\n"
        f"Learner's current level: **{level_hint}**.\n\n"
        "Please provide:\n"
        "1. Overview of the learning journey and target outcome\n"
        "2. Step-by-step curriculum organized from Beginner to Intermediate to Advanced with timelines\n"
        "3. Curated resources (Books, Free Videos/Courses, Documentation, Practice Sites)\n"
        "4. Hands-on milestones/projects to validate mastery\n"
        "5. Golden tips for retaining this knowledge."
    )

    result = call_gemini(
        prompt=prompt,
        system_instruction=ROADMAP_SYSTEM_INSTRUCTION,
        api_key=api_key,
        temperature=0.4
    )

    if not result.get("success"):
        return {
            "success": False,
            "error": result.get("error", "Failed to generate learning path with Gemini."),
            "learning_path": ""
        }

    return {
        "success": True,
        "topic": clean_topic,
        "current_level": level_hint,
        "learning_path": result.get("output_text", ""),
        "model": result.get("model", "Gemini")
    }
