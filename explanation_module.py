"""
explanation_module.py
Concept explanation logic using the local lightweight LaMini-Flan-T5-783M model.
"""

import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("edugenie.explanation")

# Global pipeline instance (lazy-loaded singleton)
_explainer_pipeline = None
_model_load_attempted = False
_model_load_error = None

MODEL_NAME = os.getenv("LAMINI_MODEL_NAME", "MBZUAI/LaMini-Flan-T5-783M")


def get_pipeline():
    """
    Lazily loads the LaMini-Flan-T5-783M model pipeline.
    Runs on CPU with memory-efficient configuration.
    """
    global _explainer_pipeline, _model_load_attempted, _model_load_error

    if _explainer_pipeline is not None:
        return _explainer_pipeline

    if _model_load_attempted and _model_load_error:
        return None

    _model_load_attempted = True
    try:
        logger.info(f"Loading local model '{MODEL_NAME}' on CPU...")
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

        # Check torch device (CPU preferred for lightweight compatibility)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device} for local explanation model")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        _explainer_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1
        )
        logger.info(f"Successfully loaded local model '{MODEL_NAME}'.")
        return _explainer_pipeline

    except ImportError as e:
        _model_load_error = f"Missing dependencies for local model: {e}. Please install transformers and torch."
        logger.warning(_model_load_error)
        return None
    except Exception as e:
        _model_load_error = f"Failed to load local model '{MODEL_NAME}': {str(e)}"
        logger.warning(_model_load_error)
        return None


def format_instruction(topic: str) -> str:
    """
    Formats the prompt according to LaMini instruction-tuning template.
    """
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n"
        f"Explain the concept of '{topic.strip()}' in simple, clear, and beginner-friendly terms for a student. "
        f"Break down difficult terms into everyday analogies without overwhelming the learner.\n\n"
        "### Response:"
    )


def explain_concept(topic: str, max_length: int = 512) -> Dict[str, Any]:
    """
    Explains an educational concept or topic in simple terms using LaMini-Flan-T5-783M.
    If the local model is not yet loaded or unavailable, provides a graceful explanation.
    """
    if not topic or not topic.strip():
        return {
            "success": False,
            "error": "Topic cannot be empty. Please provide a concept or topic to explain.",
            "explanation": ""
        }

    clean_topic = topic.strip()
    generator = get_pipeline()

    if generator is not None:
        try:
            prompt = format_instruction(clean_topic)
            outputs = generator(
                prompt,
                max_length=max_length,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.2
            )
            raw_text = outputs[0]["generated_text"].strip()
            # Clean response if template prefixes were echoed
            if "### Response:" in raw_text:
                raw_text = raw_text.split("### Response:")[-1].strip()

            return {
                "success": True,
                "topic": clean_topic,
                "explanation": raw_text,
                "model": MODEL_NAME,
                "source": "local_lamini"
            }
        except Exception as e:
            logger.error(f"Inference error with local model: {e}")
            # Fallback below

    # Fallback explanation mechanism if local model weights are downloading, uninstalled, or out of RAM
    note = (
        _model_load_error
        or f"Local model '{MODEL_NAME}' is currently initializing or downloading."
    )

    fallback_text = (
        f"### 💡 Understanding {clean_topic}\n\n"
        f"**Core Concept:**\n"
        f"{clean_topic} is a fundamental concept in its domain. At a high level, it represents "
        f"a foundational principle designed to solve specific problems or explain natural phenomena.\n\n"
        f"**Simple Analogy:**\n"
        f"Think of {clean_topic} like the foundation of a house: before building complex structures, "
        f"understanding the core rules makes everything else straightforward.\n\n"
        f"**Key Takeaways for Beginners:**\n"
        f"1. **Start with the basics:** Focus on why {clean_topic} exists and what problem it addresses.\n"
        f"2. **Real-world connection:** Relate {clean_topic} to everyday observations or practical examples.\n"
        f"3. **Practice:** Try summarizing what you learned in your own words.\n\n"
        f"> *Note from EduGenie: {note} To use local offline inference, verify `pip install transformers torch sentencepiece` and ensure model weights are cached.*"
    )

    return {
        "success": True,
        "topic": clean_topic,
        "explanation": fallback_text,
        "model": f"{MODEL_NAME} (Fallback Mode)",
        "source": "fallback",
        "notice": note
    }
