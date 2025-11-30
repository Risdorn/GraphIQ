# backend/reasoning_agent.py

import os
import json
from typing import Any, Dict, List, Optional

import dotenv
from openai import OpenAI

from retrieval_agent import hybrid_retrieve

dotenv.load_dotenv()

# ---------------- Global LLM client (OpenRouter) ----------------

llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# You can change this to any model exposed by OpenRouter
DEFAULT_MODEL = "openai/gpt-oss-20b:free"


# ---------------- Prompts ----------------

SYSTEM_PROMPT = """
You are a careful, truthful AI research assistant.

You are given:
  1. A user question.
  2. Retrieved context from a vector store (text chunks and summaries).
  3. Retrieved context from a graph database (nodes/relations).

Your job:
  - Read all context carefully.
  - Use it to answer the question as accurately and concretely as possible.
  - If some detail is not present in the context, you may use your general
    knowledge, but clearly mark it as such (e.g., "More generally, ...").
  - If the question cannot be answered reliably from the context and general
    knowledge, say you are uncertain and explain why.
  - Be concise but clear. Use bullet points or short sections when helpful.
  - Do NOT fabricate citations or IDs. Only reference chunks by their given
    IDs or indices if explicitly provided.
"""


# ---------------- Helpers ----------------

def _format_vector_context(vector_results: List[Dict[str, Any]],
                           max_chars: int = 4000) -> str:
    """
    Turn vector store search results into a compact text context.
    Expects each result to look like:
      {
        "chunk_id": ...,
        "text": ...,
        "summary": ...,
        "score": ...,
        ...
      }
    but is robust to missing fields.
    """
    lines: List[str] = []
    total = 0

    for i, r in enumerate(vector_results):
        chunk_id = r.get("chunk_id", f"chunk_{i}")
        score = r.get("score", None)
        summary = r.get("summary", None)
        text = r.get("text", "")

        header = f"[VECTOR CHUNK {i} | id={chunk_id}"
        if score is not None:
            header += f" | score={score:.4f}"
        header += "]"

        parts = [header]
        if summary:
            parts.append(f"Summary: {summary}")
        if text:
            parts.append(f"Text: {text}")

        block = "\n".join(parts) + "\n\n"
        if total + len(block) > max_chars:
            break

        lines.append(block)
        total += len(block)

    if not lines:
        return "No relevant vector-store chunks were retrieved.\n"

    return "".join(lines)


def _format_graph_context(graph_results: List[Dict[str, Any]],
                          max_chars: int = 3000) -> str:
    """
    Turn graph DB results into a JSON-like snippet.
    We keep it compact to avoid blowing up the prompt.
    """
    if not graph_results:
        return "No relevant graph relations were retrieved.\n"

    # Try to compress: keep at most first N records
    max_records = 15
    subset = graph_results[:max_records]

    try:
        serialized = json.dumps(subset, indent=2, default=str)
    except TypeError:
        # In case there are non-serializable objects
        serialized = "\n".join(str(r) for r in subset)

    if len(serialized) > max_chars:
        serialized = serialized[:max_chars] + "\n... (truncated)"

    return serialized


def _build_context_block(retrieval_output: Dict[str, Any]) -> str:
    """
    Build the combined context string given the output from hybrid_retrieve.
    """
    entities = retrieval_output.get("entities", [])
    vector_results = retrieval_output.get("vector_results", [])
    graph_results = retrieval_output.get("graph_results", [])

    ent_str = ", ".join(entities) if entities else "None"

    vector_ctx = _format_vector_context(vector_results)
    graph_ctx = _format_graph_context(graph_results)

    context = f"""
[ENTITIES]
{ent_str}

[VECTOR STORE RESULTS]
{vector_ctx}

[GRAPH DB RESULTS]
{graph_ctx}
"""
    return context.strip()


def _call_llm_with_context(
    question: str,
    context: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> str:
    """
    Make a single LLM call to answer the question given the context.
    """
    used_model = model or DEFAULT_MODEL

    completion = llm_client.chat.completions.create(
        model=used_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {
                "role": "user",
                "content": (
                    "You are given the following retrieved context and a user question.\n\n"
                    "=== RETRIEVED CONTEXT START ===\n"
                    f"{context}\n"
                    "=== RETRIEVED CONTEXT END ===\n\n"
                    f"User question: {question}"
                ),
            },
        ],
        temperature=temperature,
    )

    return completion.choices[0].message.content.strip()


# ---------------- Public API ----------------

def answer_with_retrieval(
    question: str,
    top_k: int = 5,
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    High-level entry point for the app / API.

    - Runs hybrid retrieval (vector + graph).
    - Builds a prompt with the retrieved context.
    - Calls the LLM to generate an answer.
    - Returns both the answer and raw retrieval artifacts.

    Returns:
    {
      "question": str,
      "answer": str,
      "entities": [...],
      "vector_results": [...],
      "graph_results": [...],
      "model": str
    }
    """
    retrieval_output = hybrid_retrieve(question, top_k=top_k)
    context_block = _build_context_block(retrieval_output)

    answer = _call_llm_with_context(
        question=question,
        context=context_block,
        model=model,
        temperature=temperature,
    )

    return {
        "question": question,
        "answer": answer,
        "entities": retrieval_output.get("entities", []),
        "vector_results": retrieval_output.get("vector_results", []),
        "graph_results": retrieval_output.get("graph_results", []),
        "model": model or DEFAULT_MODEL,
    }


# # Optional: simple CLI for quick testing
# if __name__ == "__main__":
#     import sys

#     q = " ".join(sys.argv[1:]) or "Explain the Jaynes–Cummings model and its relation to the Rabi model."
#     out = answer_with_retrieval(q, top_k=5)
#     print("\n=== ANSWER ===\n")
#     print(out["answer"])
#     print("\n=== ENTITIES ===\n", out["entities"])
