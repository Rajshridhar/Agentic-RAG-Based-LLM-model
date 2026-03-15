"""Agentic RAG Orchestrator using LangGraph StateGraph.

Flow
----
question → route_query
  ├── not_relevant  → END (out-of-scope message)
  └── vectorstore   → retrieve → grade_docs → generate
                                    ↓
                              check_hallucination
                                ├── no  → regenerate (up to MAX_RETRIES)
                                └── yes → grade_answer
                                            ├── no  → fallback message
                                            └── yes → END (return answer)
"""

from typing import TypedDict, List, Optional

from langgraph.graph import END, StateGraph

from app.agents.answer_grader import grade_answer
from app.agents.hallucination_grader import grade_hallucination
from app.agents.query_router import route_query
from app.agents.retrieval_grader import grade_documents
from app.config import MAX_RETRIES
from app.formatter import format_response
from app.llm import get_llm
from app.logger import get_logger
from app.vector_store import get_retriever, load_vector_store
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = get_logger(__name__)


class AgentState(TypedDict):
    question: str
    documents: List
    answer: Optional[str]
    agent_trace: List[str]
    retries: int
    route: Optional[str]


# RAG generation helper
_RAG_TEMPLATE = """You are an expert assistant specializing in Indian Cricket history, players, matches, tournaments, and records. Your expertise covers the Indian cricket team, IPL (Indian Premier League), domestic cricket, international tournaments, World Cups, and all aspects of cricket in India.

Your core tasks:
1. **Answer Based on Provided Documents**: Respond to queries using the provided document content. If the answer is not found in the documents, clearly state: "I don't have this information in the provided documents.(I am Not like GPT🤣)" Do not invent or assume facts beyond the context.
2. **Precision and Citations**: Be precise, cite relevant sections, pages, or document titles where applicable (e.g., "Page 3 of the Indian Cricket Report"). If statistics or numerical data are present, interpret and summarize them accurately without altering values.
3. **Out-of-Scope Queries**: If the query is unrelated to Indian Cricket or the provided documents, state: "I don't have this information in the provided cricket documents."
4. **Ambiguity Handling**: If the question is ambiguous or lacks specifics (e.g., no specific player, match, or tournament mentioned), ask for clarification politely (e.g., "Could you specify the player, match, or tournament you're referring to?").
5. **Comparisons**: For queries comparing players, teams, eras, or tournaments, provide a clear, structured comparison highlighting differences, similarities, and key statistics, based solely on the documents.
6. **Historical Context**: For queries about cricket history, milestones, or records, provide detailed context including dates, venues, scores, and notable performances as available in the documents.
7. **Structured Answers**: Organize responses clearly with headings, bullet points, or numbered lists where appropriate.
8. **Comprehensive Responses**: Try to give a full response that covers all aspects of the query.

Context:
{context}

Question: {question}

Answer:"""


def _generate_answer(question: str, documents: list) -> str:
    logger.info("Generating answer from %d document(s) for: '%s'", len(documents), question[:100])
    llm = get_llm()
    prompt = PromptTemplate.from_template(_RAG_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    context = "\n\n".join(doc.page_content for doc in documents)
    answer = chain.invoke({"context": context, "question": question})
    logger.info("Generated answer (%d chars): '%s'", len(answer), answer[:150])
    return answer


# Graph nodes
def node_route_query(state: AgentState) -> AgentState:
    """Route the query: vectorstore or not_relevant."""
    decision = route_query(state["question"])
    trace = state.get("agent_trace", [])
    trace.append(f"route_query → {decision}")
    return {**state, "route": decision, "agent_trace": trace}


def node_retrieve(state: AgentState, vector_store=None) -> AgentState:
    """Retrieve documents from the vector store."""
    if vector_store is None:
        vector_store = load_vector_store()
    retriever = get_retriever(vector_store)
    docs = retriever.invoke(state["question"])
    trace = state.get("agent_trace", [])
    trace.append(f"retrieve → {len(docs)} document(s) retrieved")
    return {**state, "documents": docs, "agent_trace": trace}


def node_grade_docs(state: AgentState) -> AgentState:
    """Filter retrieved documents by relevance."""
    relevant = grade_documents(state["question"], state["documents"])
    trace = state.get("agent_trace", [])
    trace.append(
        f"grade_docs → {len(relevant)}/{len(state['documents'])} document(s) relevant"
    )
    return {**state, "documents": relevant, "agent_trace": trace}


def node_generate(state: AgentState) -> AgentState:
    """Generate an answer from the (filtered) documents."""
    answer = _generate_answer(state["question"], state["documents"])
    trace = state.get("agent_trace", [])
    trace.append("generate → answer produced")
    return {**state, "answer": answer, "agent_trace": trace}


def node_check_hallucination(state: AgentState) -> AgentState:
    """Grade whether the generated answer is grounded."""
    grounded = grade_hallucination(state["answer"], state["documents"])
    retries = state.get("retries", 0)
    trace = state.get("agent_trace", [])
    trace.append(f"check_hallucination → grounded={grounded} (retry {retries})")
    return {**state, "route": grounded, "retries": retries, "agent_trace": trace}


def node_grade_answer(state: AgentState) -> AgentState:
    """Grade whether the answer addresses the original question."""
    useful = grade_answer(state["question"], state["answer"])
    trace = state.get("agent_trace", [])
    trace.append(f"grade_answer → useful={useful}")
    return {**state, "route": useful, "agent_trace": trace}


def node_not_relevant(state: AgentState) -> AgentState:
    trace = state.get("agent_trace", [])
    trace.append("not_relevant → query out of scope")
    return {
        **state,
        "answer": "This system only answers questions about Indian Cricket history.",
        "agent_trace": trace,
    }


def node_fallback(state: AgentState) -> AgentState:
    trace = state.get("agent_trace", [])
    trace.append("fallback → insufficient information")
    return {
        **state,
        "answer": "I don't have enough information in the knowledge base to answer this question.(BHAG BHOSDIKE)",
        "agent_trace": trace,
    }


# Conditional edge functions
def edge_after_route(state: AgentState) -> str:
    return "retrieve" if state["route"] == "vectorstore" else "not_relevant"


def edge_after_hallucination(state: AgentState) -> str:
    if state["route"] == "yes":
        return "grade_answer"
    if state.get("retries", 0) < MAX_RETRIES:
        return "regenerate"
    return "grade_answer"  # Give up regenerating; move on


def edge_after_answer_grade(state: AgentState) -> str:
    return "end" if state["route"] == "yes" else "fallback"


# Build the graph (factory function – accepts optional vector_store for DI)
def build_agentic_rag(vector_store=None):
    def _retrieve(state):
        return node_retrieve(state, vector_store=vector_store)

    def _regenerate(state: AgentState) -> AgentState:
        """Re-generate and increment retry counter."""
        state = dict(state)
        state["retries"] = state.get("retries", 0) + 1
        return node_generate(state)

    graph = StateGraph(AgentState)

    graph.add_node("route_query", node_route_query)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("grade_docs", node_grade_docs)
    graph.add_node("generate", node_generate)
    graph.add_node("check_hallucination", node_check_hallucination)
    graph.add_node("grade_answer", node_grade_answer)
    graph.add_node("not_relevant", node_not_relevant)
    graph.add_node("fallback", node_fallback)
    graph.add_node("regenerate", _regenerate)

    graph.set_entry_point("route_query")

    graph.add_conditional_edges("route_query", edge_after_route, {
        "retrieve": "retrieve",
        "not_relevant": "not_relevant",
    })
    graph.add_edge("retrieve", "grade_docs")
    graph.add_edge("grade_docs", "generate")
    graph.add_edge("generate", "check_hallucination")
    graph.add_conditional_edges("check_hallucination", edge_after_hallucination, {
        "grade_answer": "grade_answer",
        "regenerate": "regenerate",
    })
    graph.add_edge("regenerate", "check_hallucination")
    graph.add_conditional_edges("grade_answer", edge_after_answer_grade, {
        "end": END,
        "fallback": "fallback",
    })
    graph.add_edge("not_relevant", END)
    graph.add_edge("fallback", END)

    return graph.compile()


def query_agentic(question: str, vector_store=None) -> dict:
    logger.info("=== Agentic RAG query started: '%s' ===", question[:150])
    app = build_agentic_rag(vector_store=vector_store)
    initial_state: AgentState = {
        "question": question,
        "documents": [],
        "answer": None,
        "agent_trace": [],
        "retries": 0,
        "route": None,
    }
    final_state = app.invoke(initial_state)

    sources = [
        {
            "page": doc.metadata.get("page", "N/A"),
            "chunk_id": doc.metadata.get("chunk_id", "N/A"),
            "content_preview": doc.page_content[:300],
        }
        for doc in final_state.get("documents", [])
    ]

    result = {
        "answer": format_response(final_state.get("answer", "No answer generated.")),
        "sources": sources,
        "agent_trace": final_state.get("agent_trace", []),
    }
    logger.info("=== Agentic RAG query completed — trace: %s ===", result["agent_trace"])
    return result
