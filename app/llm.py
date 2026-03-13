"""LLM setup using AutoModelForSeq2SeqLM for T5-family models."""

from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.config import LLM_MODEL
from app.logger import get_logger

logger = get_logger(__name__)

_llm_instance = None


class Seq2SeqLLM(LLM):
    """LangChain LLM wrapper around a HuggingFace seq2seq model."""

    model: Any = None
    tokenizer: Any = None
    max_new_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "seq2seq"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        )
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def get_llm(model_name: str = LLM_MODEL) -> Seq2SeqLLM:
    """Return a (cached) Seq2SeqLLM instance.

    Uses ``google/flan-t5-small`` by default – free and runs on CPU.
    The instance is created once and reused on subsequent calls.
    """
    global _llm_instance  # noqa: PLW0603
    if _llm_instance is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _llm_instance = Seq2SeqLLM(model=model, tokenizer=tokenizer)
        logger.info("LLM loaded: %s", model_name)
    return _llm_instance
