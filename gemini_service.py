"""
gemini_service.py
Unified Google Gemini API service layer for EduGenie.
Supports google-genai and google-generativeai with robust error handling,
model fallback, and Markdown block cleaning.
"""

import os
import re
import json
import logging
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("edugenie.gemini")

# Priority list of models to try
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FALLBACK_MODELS = [
    DEFAULT_MODEL,
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-3.8-flash"
]
# Remove duplicates while preserving order
MODELS_TO_TRY = list(dict.fromkeys([m for m in FALLBACK_MODELS if m]))


def get_api_key(override_key: Optional[str] = None) -> Optional[str]:
    """Retrieve Gemini API key from explicit parameter, environment, or .env."""
    key = (override_key or os.getenv("GEMINI_API_KEY") or "").strip()
    return key if key else None


def clean_markdown_json(text: str) -> str:
    """
    Cleans Markdown code fence blocks (e.g. ```json ... ``` or ``` ... ```)
    from AI-generated responses before JSON parsing.
    """
    if not text:
        return ""
    cleaned = text.strip()

    # Match code blocks with or without json/JSON tag
    pattern = r"```(?:json|JSON)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, cleaned)
    if match:
        cleaned = match.group(1).strip()
    else:
        # Also handle text that starts with ``` and doesn't close properly
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json|JSON)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

    return cleaned.strip()


def call_gemini(
    prompt: str,
    system_instruction: Optional[str] = None,
    api_key: Optional[str] = None,
    json_mode: bool = False,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Invokes Google Gemini with fallback model handling.
    Works with both google-genai and google-generativeai packages.
    """
    active_key = get_api_key(api_key)
    if not active_key:
        return {
            "success": False,
            "error": (
                "Google Gemini API Key is missing. "
                "Please configure GEMINI_API_KEY in your .env file or provide it in the request."
            ),
            "output_text": ""
        }

    # Attempt 1: Try modern google-genai SDK
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=active_key)

        for model_name in MODELS_TO_TRY:
            try:
                # Use interactions or models.generate_content
                config = {}
                if system_instruction:
                    config["system_instruction"] = system_instruction
                if json_mode:
                    config["response_mime_type"] = "application/json"
                if temperature is not None:
                    config["temperature"] = temperature

                # Try client.models.generate_content
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config) if config else None
                )

                response_text = ""
                if hasattr(response, "text") and response.text:
                    response_text = response.text
                elif hasattr(response, "output_text"):
                    response_text = response.output_text

                if response_text:
                    return {
                        "success": True,
                        "output_text": response_text.strip(),
                        "model": model_name,
                        "sdk": "google-genai"
                    }
            except Exception as model_err:
                logger.debug(f"Model {model_name} failed with google-genai: {model_err}")
                continue

    except ImportError:
        logger.debug("google-genai SDK not installed, checking google-generativeai")
    except Exception as e:
        logger.debug(f"google-genai execution error: {e}")

    # Attempt 2: Try legacy google-generativeai package if available
    try:
        import google.generativeai as legacy_genai

        legacy_genai.configure(api_key=active_key)

        for model_name in MODELS_TO_TRY:
            try:
                generation_config = {"temperature": temperature}
                if json_mode:
                    generation_config["response_mime_type"] = "application/json"

                model = legacy_genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instruction,
                    generation_config=generation_config
                )
                response = model.generate_content(prompt)
                if response and response.text:
                    return {
                        "success": True,
                        "output_text": response.text.strip(),
                        "model": model_name,
                        "sdk": "google-generativeai"
                    }
            except Exception as model_err:
                logger.debug(f"Model {model_name} failed with google-generativeai: {model_err}")
                continue

    except ImportError:
        logger.debug("google-generativeai not installed")
    except Exception as e:
        logger.debug(f"google-generativeai error: {e}")

    # If neither SDK succeeded or all models failed
    return {
        "success": False,
        "error": (
            "Unable to generate response from Google Gemini API. "
            "Please check that your GEMINI_API_KEY is valid, has quota enabled in Google AI Studio, "
            "and that either `google-genai` or `google-generativeai` is installed."
        ),
        "output_text": ""
    }
