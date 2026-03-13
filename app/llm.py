"""LLM setup using the (non-deprecated) langchain_huggingface package."""

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

from app.config import LLM_MODEL

_llm_instance = None


def get_llm(model_name: str = LLM_MODEL) -> HuggingFacePipeline:
    """Return a (cached) HuggingFacePipeline LLM instance.

    Uses ``google/flan-t5-small`` by default – free and runs on CPU.
    The pipeline is created once and reused on subsequent calls.
    """
    global _llm_instance  # noqa: PLW0603
    if _llm_instance is None:
        hf_pipeline = pipeline(
            "text2text-generation",
            model=model_name,
            max_new_tokens=512,
        )
        _llm_instance = HuggingFacePipeline(pipeline=hf_pipeline)
        print(f"[✓] LLM loaded: {model_name}")
    return _llm_instance
