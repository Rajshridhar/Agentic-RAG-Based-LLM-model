"""LLM setup using Airawat Llama API for response generation."""

import json
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.config import AIRAWAT_API_URL, AIRAWAT_MODEL, AIRAWAT_TOKEN
from app.logger import get_logger

logger = get_logger(__name__)

_llm_instance = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate response from LLM"""
        pass

    @abstractmethod
    def generate_response_stream(self, prompt: str, context: Optional[str] = None):
        """Generate response with streaming"""
        pass

class AirawatLlamaLLM(LLM):
    """Custom LangChain LLM wrapper for the Airawat Llama API"""

    api_url: str = ""
    token: str = ""
    model: str = "meta/llama-3.2-11b-vision-instruct"
    max_tokens: int = 512
    temperature: float = 0.7
    timeout: int = 120

    @property
    def _llm_type(self) -> str:
        return "airawat_llama"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Make API call to Airawat Llama service"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Cookie": "SERVERID=api-manager",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                verify=False,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0].get("message", {}).get("content", "")
            else:
                return str(result)

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Airawat API: {e}")
            return f"Error calling API: {str(e)}"


class AirawatProvider(LLMProvider):
    """Airawat LLM provider that wraps AirawatLlamaLLM for use in the RAG pipeline"""

    def __init__(self, api_url: str, token: str, model: str = "meta/llama-3.2-11b-vision-instruct",
                 temperature: float = 0.7, max_tokens: int = 512, timeout: int = 120):
        self.llm = AirawatLlamaLLM(
            api_url=api_url,
            token=token,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout
        )
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=CRICKET_PROMPT_TEMPLATE
        )
        logger.info(f"Initialized Airawat provider with model: {model}")

    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate response from Airawat Llama"""
        try:
            full_prompt = self._build_prompt(prompt, context)
            return self.llm.invoke(full_prompt)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"

    def generate_response_stream(self, prompt: str, context: Optional[str] = None):
        """Generate response (streaming not supported, falls back to normal)"""
        return self.generate_response(prompt, context)

    def _build_prompt(self, prompt: str, context: Optional[str] = None) -> str:
        """Build prompt with optional context using the regulatory template"""
        if context:
            return self.prompt_template.format(context=context, question=prompt)
        return prompt

    def build_rag_chain(self, vector_store):
        """Build a LangChain RAG chain with the given vector store"""
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        rag_chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        return rag_chain
    

# Prompt Template & Helpers

CRICKET_PROMPT_TEMPLATE = """You are an expert assistant specializing in Indian Cricket history, players, matches, tournaments, and records. Your expertise covers the Indian cricket team, IPL (Indian Premier League), domestic cricket, international tournaments, World Cups, and all aspects of cricket in India.

Your core tasks:
1. **Answer Based on Provided Documents**: Respond to queries using the provided document content. If the answer is not found in the documents, clearly state: "I don't have this information in the provided documents." Do not invent or assume facts beyond the context.
2. **Precision and Citations**: Be precise, cite relevant sections, pages, or document titles where applicable (e.g., "Page 3 of the Indian Cricket Report"). If statistics or numerical data are present, interpret and summarize them accurately without altering values.
3. **Out-of-Scope Queries**: If the query is unrelated to Indian Cricket or the provided documents, state: "I don't have this information in the provided cricket documents."
4. **Ambiguity Handling**: If the question is ambiguous or lacks specifics (e.g., no specific player, match, or tournament mentioned), ask for clarification politely (e.g., "Could you specify the player, match, or tournament you're referring to?").
5. **Comparisons**: For queries comparing players, teams, eras, or tournaments, provide a clear, structured comparison highlighting differences, similarities, and key statistics, based solely on the documents.
6. **Historical Context**: For queries about cricket history, milestones, or records, provide detailed context including dates, venues, scores, and notable performances as available in the documents.
7. **Structured Answers**: Organize responses clearly with headings, bullet points, or numbered lists where appropriate.
8. **Comprehensive Responses**: Try to give a full response that covers all aspects of the query.

Cricket Document Content:
{context}

User Question: {question}

Provide a clear, structured answer:"""


def _format_docs(docs):
    """Format retrieved documents for the RAG chain"""
    formatted = []
    for doc in docs:
        page_num = doc.metadata.get('page', 'N/A')

        table_info = ""
        table_ids_json = doc.metadata.get('table_ids', '')
        if table_ids_json:
            try:
                table_ids_list = json.loads(table_ids_json)
                if table_ids_list:
                    table_info = f" [Tables: {', '.join(str(t)[:8] for t in table_ids_list[:2])}]"
            except (json.JSONDecodeError, TypeError):
                pass

        formatted_doc = f"[Page {page_num}]{table_info}\n{doc.page_content}"
        formatted.append(formatted_doc)

    return "\n\n---\n\n".join(formatted)


def get_llm() -> AirawatLlamaLLM:
    """Return a (cached) AirawatLlamaLLM instance.

    Uses the Airawat Llama API (``meta/llama-3.2-11b-vision-instruct``
    by default).  Configure via ``AIRAWAT_API_URL``, ``AIRAWAT_TOKEN``,
    and ``AIRAWAT_MODEL`` environment variables.
    """
    global _llm_instance  # noqa: PLW0603
    if _llm_instance is None:
        _llm_instance = AirawatLlamaLLM(
            api_url=AIRAWAT_API_URL,
            token=AIRAWAT_TOKEN,
            model=AIRAWAT_MODEL,
        )
        logger.info("LLM loaded: %s (Airawat API)", AIRAWAT_MODEL)
    return _llm_instance
