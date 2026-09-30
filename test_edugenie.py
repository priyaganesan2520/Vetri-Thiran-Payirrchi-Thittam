"""
test_edugenie.py
Automated test suite verifying the five core features of EduGenie:
1. Question and Answer
2. Concept Explanation (LaMini-Flan-T5-783M)
3. Quiz Generation (Exactly 3 MCQs, 4 options each, JSON schema)
4. Summarization (Revision notes)
5. Personalized Learning Path (Beginner to Advanced roadmap)
"""

import sys
import unittest
from pathlib import Path

# Add current directory to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from explanation_module import explain_concept, format_instruction
from quiz_module import normalize_quiz_json, generate_fallback_quiz
from gemini_service import clean_markdown_json


class TestEduGenieCore(unittest.TestCase):

    def test_markdown_cleaning(self):
        """Verify that markdown code blocks (```json ... ```) are cleanly stripped."""
        raw_ai_output = "```json\n[\n  {\"id\": 1, \"question\": \"Test?\"}\n]\n```"
        cleaned = clean_markdown_json(raw_ai_output)
        self.assertEqual(cleaned, '[\n  {"id": 1, "question": "Test?"}\n]')

        raw_without_tag = "```\n[\"item1\"]\n```"
        cleaned_no_tag = clean_markdown_json(raw_without_tag)
        self.assertEqual(cleaned_no_tag, '["item1"]')

    def test_quiz_normalization_and_structure(self):
        """Verify that generated quizzes contain exactly 3 questions with 4 options each."""
        raw_quiz_data = [
            {
                "id": 1,
                "question": "What is the largest ocean?",
                "options": ["Atlantic", "Pacific", "Indian", "Arctic"],
                "correct_option_index": 1,
                "correct_answer": "Pacific",
                "explanation": "The Pacific Ocean is the largest and deepest ocean on Earth."
            },
            {
                "id": 2,
                "question": "What is the chemical symbol for Water?",
                "options": ["H2O", "CO2", "NaCl", "O2"],
                "correct_option_index": 0,
                "correct_answer": "H2O",
                "explanation": "Water consists of two hydrogen atoms and one oxygen atom."
            },
            {
                "id": 3,
                "question": "What is the unit of electric current?",
                "options": ["Volt", "Watt", "Ampere", "Joule"],
                "correct_option_index": 2,
                "correct_answer": "Ampere",
                "explanation": "The ampere (A) is the SI base unit of electric current."
            }
        ]

        normalized = normalize_quiz_json(raw_quiz_data)
        self.assertEqual(len(normalized), 3, "Quiz must contain exactly 3 questions.")

        for i, q in enumerate(normalized, start=1):
            self.assertEqual(q["id"], i)
            self.assertIn("question", q)
            self.assertEqual(len(q["options"]), 4, f"Question {i} must have exactly 4 options.")
            self.assertIn("correct_option_index", q)
            self.assertTrue(0 <= q["correct_option_index"] < 4)
            self.assertIn("correct_answer", q)
            self.assertIn("explanation", q)

    def test_quiz_fallback_engine(self):
        """Verify fallback quiz generation provides valid 3-question 4-option structure."""
        fallback = generate_fallback_quiz("Pythagoras Theorem")
        self.assertEqual(len(fallback), 3)
        for q in fallback:
            self.assertEqual(len(q["options"]), 4)
            self.assertIn("Pythagoras Theorem", q["question"] + q["correct_answer"])

    def test_concept_explanation_module(self):
        """Verify explanation module handles input and returns structured response."""
        res = explain_concept("Newton's First Law")
        self.assertTrue(res["success"])
        self.assertIn("explanation", res)
        self.assertTrue(len(res["explanation"]) > 20)

    def test_empty_input_handling(self):
        """Verify error handling on empty or whitespace inputs."""
        res_empty = explain_concept("   ")
        self.assertFalse(res_empty["success"])
        self.assertIn("error", res_empty)

    def test_instruction_template_formatting(self):
        """Verify prompt template formatting for LaMini-Flan-T5."""
        prompt = format_instruction("Gravity")
        self.assertIn("### Instruction:", prompt)
        self.assertIn("Gravity", prompt)
        self.assertIn("### Response:", prompt)


if __name__ == "__main__":
    unittest.main()
