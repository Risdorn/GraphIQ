import re
import json

def approx_tokens(text):
    return max(1, len(text) // 4)

def split_paragraphs(text):
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # If PDF extractor dumps everything on one line,
    # force-break on periods + space + capital letters.
    text = re.sub(r"(?<=[.!?])\s+(?=[A-Z])", "\n\n", text)

    # Now split on actual blank lines (if any)
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]

    return paras

def paragraph_chunking(text, max_tokens = 512):
    paras = split_paragraphs(text)
    chunks = []
    buffer = ""
    for p in paras:
        if not buffer:
            buffer = p
        else:
            candidate = buffer + "\n\n" + p
            if approx_tokens(candidate) <= max_tokens:
                buffer = candidate
            else:
                chunks.append(buffer)
                buffer = p
    if buffer:
        chunks.append(buffer)
    # further split very large chunks by sentence if available
    final_chunks = []
    for c in chunks:
        if approx_tokens(c) > max_tokens:
            new_chunks = sentence_chunking(c, max_tokens)
            final_chunks.extend(new_chunks)
        else:
            final_chunks.append(c)
    return final_chunks

def sentence_chunking(c, max_tokens):
    sents = re.split(r'(?<=[.!?])\s+', c)
    new_chunks = []
    buffer = ""
    for s in sents:
        candidate = (buffer + ' ' + s).strip() if buffer else s
        if approx_tokens(candidate) <= max_tokens:
            buffer = candidate
        elif buffer:
            new_chunks.append(buffer)
        buffer = s
    if buffer:
        new_chunks.append(buffer)
    return new_chunks

def format_vector_context(vector_results, max_chars = 4000):
    """
    Turn vector store search results into a compact text context.
    Expects each result to look like:
      {
        "chunk_id": ...,
        "text": ...,
        "summary": ...,
        "combined_score": ...,
        ...
      }
    but is robust to missing fields.
    """
    lines = []
    total = 0

    for i, r in enumerate(vector_results):
        chunk_id = r.get("chunk_id", f"chunk_{i}")
        #score = r.get("combined_score", None)
        summary = r.get("summary", None)
        text = r.get("text", "")

        header = f"[VECTOR CHUNK {i} | id={chunk_id}"
        # if score is not None:
        #     header += f" | score={score:.4f}"
        # header += "]"

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

def format_graph_context(graph_results, max_chars = 3000):
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