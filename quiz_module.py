"""
quiz_module.py
Quiz-generation logic powered by Google Gemini.
Produces exactly 3 multiple-choice questions with 4 options each,
cleans Markdown blocks, validates JSON structure, and returns educational feedback.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from gemini_service import call_gemini, clean_markdown_json

logger = logging.getLogger("edugenie.quiz")

QUIZ_SYSTEM_INSTRUCTION = (
    "You are an expert educational exam and quiz creator. "
    "Your task is to generate high-quality multiple choice quizzes based on the user's topic or passage. "
    "Rules:\n"
    "1. You MUST generate exactly 3 multiple-choice questions.\n"
    "2. Each question MUST have exactly 4 options labeled or listed clearly.\n"
    "3. Exactly one option is correct; the remaining 3 options MUST be plausible distractors.\n"
    "4. Return ONLY valid JSON matching the specified schema with NO extra commentary.\n"
    "5. Do NOT include markdown code blocks if possible, or wrap strictly in ```json."
)

JSON_SCHEMA_EXAMPLE = """
[
  {
    "id": 1,
    "question": "What is the primary function of chlorophyll in plant cells?",
    "options": [
      "Absorbing light energy for photosynthesis",
      "Storing water in the central vacuole",
      "Synthesizing proteins in the cytoplasm",
      "Transporting glucose through the phloem"
    ],
    "correct_option_index": 0,
    "correct_answer": "Absorbing light energy for photosynthesis",
    "explanation": "Chlorophyll is a green pigment that absorbs light wavelengths necessary to drive photosynthesis."
  },
  {
    "id": 2,
    "question": "Which organelle contains chlorophyll?",
    "options": [
      "Mitochondria",
      "Chloroplast",
      "Ribosome",
      "Endoplasmic Reticulum"
    ],
    "correct_option_index": 1,
    "correct_answer": "Chloroplast",
    "explanation": "Chloroplasts are specialized plant organelles that house thylakoid membranes containing chlorophyll."
  },
  {
    "id": 3,
    "question": "What are the primary products of oxygenic photosynthesis?",
    "options": [
      "Glucose and Oxygen",
      "Carbon Dioxide and Water",
      "Lactic Acid and ATP",
      "Nitrogen and Starch"
    ],
    "correct_option_index": 0,
    "correct_answer": "Glucose and Oxygen",
    "explanation": "Plants convert carbon dioxide and water into glucose and release oxygen as a byproduct."
  }
]
"""


def normalize_quiz_json(raw_json: Any) -> List[Dict[str, Any]]:
    """
    Validates and normalizes the parsed JSON into exactly 3 questions with 4 options.
    """
    questions_list = []
    if isinstance(raw_json, dict):
        # Could be {"quiz": [...]} or {"questions": [...]}
        for key in ["quiz", "questions", "items", "data"]:
            if key in raw_json and isinstance(raw_json[key], list):
                raw_json = raw_json[key]
                break

    if not isinstance(raw_json, list):
        raise ValueError("Quiz root must be a JSON array of questions.")

    for i, item in enumerate(raw_json[:3], start=1):
        if not isinstance(item, dict):
            continue

        q_text = item.get("question") or f"Question {i}"
        opts = item.get("options") or []

        # Convert dict options {"A": "...", "B": "..."} to list
        if isinstance(opts, dict):
            opts = list(opts.values())

        if not isinstance(opts, list) or len(opts) < 2:
            continue

        # Ensure exactly 4 options
        while len(opts) < 4:
            opts.append(f"Additional option {len(opts) + 1}")
        opts = opts[:4]

        correct_idx = item.get("correct_option_index")
        correct_ans = item.get("correct_answer")

        # Determine correct index and text
        if correct_idx is None or not (0 <= int(correct_idx) < len(opts)):
            # Try to match correct_answer text with options
            if correct_ans and any(str(correct_ans).strip().lower() in str(o).lower() for o in opts):
                for idx, opt in enumerate(opts):
                    if str(correct_ans).strip().lower() in str(opt).lower():
                        correct_idx = idx
                        break
            else:
                correct_idx = 0

        correct_idx = int(correct_idx)
        final_correct_text = str(opts[correct_idx])

        explanation = item.get("explanation") or f"The correct answer is: {final_correct_text}."

        questions_list.append({
            "id": i,
            "question": str(q_text).strip(),
            "options": [str(o).strip() for o in opts],
            "correct_option_index": correct_idx,
            "correct_answer": final_correct_text,
            "explanation": str(explanation).strip()
        })

    if len(questions_list) < 3:
        raise ValueError(f"Expected 3 questions, parsed only {len(questions_list)}")

    return questions_list[:3]


def generate_fallback_quiz(topic_or_passage: str) -> List[Dict[str, Any]]:
    """Generates an emergency structured quiz if parsing completely fails."""
    t = topic_or_passage.strip()[:40]
    return [
        {
            "id": 1,
            "question": f"What is the central focus of {t}?",
            "options": [
                f"Understanding the core principles of {t}",
                "Unrelated mathematical theorem",
                "Historical trivia from ancient times",
                "Atmospheric weather observation"
            ],
            "correct_option_index": 0,
            "correct_answer": f"Understanding the core principles of {t}",
            "explanation": f"The primary focus is understanding the foundational principles of {t}."
        },
        {
            "id": 2,
            "question": f"Why is {t} important in learning and practice?",
            "options": [
                "It serves no practical purpose",
                "It enables structured problem solving and critical analysis",
                "It was completely replaced by random guessing",
                "It applies exclusively to artistic painting"
            ],
            "correct_option_index": 1,
            "correct_answer": "It enables structured problem solving and critical analysis",
            "explanation": f"Mastery of {t} provides analytical frameworks for deeper learning."
        },
        {
            "id": 3,
            "question": f"Which strategy is most effective when studying {t}?",
            "options": [
                "Memorizing without understanding concepts",
                "Breaking topics into fundamentals, practice, and review",
                "Ignoring all core definitions",
                "Skipping foundational principles entirely"
            ],
            "correct_option_index": 1,
            "correct_answer": "Breaking topics into fundamentals, practice, and review",
            "explanation": "Active learning with structured conceptual review delivers the best educational retention."
        }
    ]


def generate_quiz(topic_or_passage: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates exactly 3 MCQs (4 options each) from a topic or passage.
    Cleans markdown code fences and returns verified JSON.
    """
    if not topic_or_passage or not topic_or_passage.strip():
        return {
            "success": False,
            "error": "Please provide a topic or study passage to generate a quiz.",
            "quiz": []
        }

    clean_input = topic_or_passage.strip()
    prompt = (
        f"Generate a 3-question multiple-choice quiz based on the following input:\n"
        f"Input: \"\"\"\n{clean_input}\n\"\"\"\n\n"
        "Requirements:\n"
        "- Exactly 3 questions.\n"
        "- Exactly 4 options per question.\n"
        "- Exactly 1 correct answer with plausible distractors.\n"
        "- Include 'correct_option_index' (0, 1, 2, or 3), 'correct_answer' (text), and a brief 'explanation'.\n"
        "- Return JSON strictly adhering to this schema format:\n"
        f"{JSON_SCHEMA_EXAMPLE}\n"
    )

    result = call_gemini(
        prompt=prompt,
        system_instruction=QUIZ_SYSTEM_INSTRUCTION,
        api_key=api_key,
        json_mode=True,
        temperature=0.4
    )

    if not result.get("success"):
        return {
            "success": False,
            "error": result.get("error", "Failed to generate quiz with Gemini."),
            "quiz": []
        }

    raw_output = result.get("output_text", "")

    # Clean markdown fences (e.g. ```json ... ```)
    cleaned_json_str = clean_markdown_json(raw_output)

    try:
        parsed_data = json.loads(cleaned_json_str)
        validated_quiz = normalize_quiz_json(parsed_data)
        return {
            "success": True,
            "input": clean_input,
            "quiz": validated_quiz,
            "model": result.get("model", "Gemini")
        }
    except Exception as parse_err:
        logger.warning(f"Quiz JSON parsing error: {parse_err}. Attempting recovery.")
        # If strict JSON parsing failed, try extracting JSON array with regex
        import re
        array_match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", raw_output)
        if array_match:
            try:
                recovered = json.loads(array_match.group(0))
                validated_quiz = normalize_quiz_json(recovered)
                return {
                    "success": True,
                    "input": clean_input,
                    "quiz": validated_quiz,
                    "model": result.get("model", "Gemini")
                }
            except Exception:
                pass

        # Fallback if recovery failed
        logger.error("Failed to parse valid quiz JSON. Using resilient educational fallback.")
        return {
            "success": True,
            "input": clean_input,
            "quiz": generate_fallback_quiz(clean_input),
            "model": "EduGenie Resilient Engine",
            "warning": "Quiz was generated using fallback structure due to AI JSON formatting variance."
        }
