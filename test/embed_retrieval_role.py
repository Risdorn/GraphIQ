# embed_retrieval_gemini_fallback.py
import os
import json
import pickle
import math
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
import faiss  # faiss-cpu required
import pdfplumber  # optional, used only if you call pdf_to_chunks()

# Optional: google-genai (only used if GEMINI_API_KEY present and client works)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_GENAI_API_KEY", "")
_use_gemini = False
_gemini_client = None
if GEMINI_API_KEY:
    try:
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        # test minimal call (do not send user data) — postpone actual run until embedding call
        _use_gemini = True
    except Exception:
        _use_gemini = False

# Neo4j environment
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# Configuration
EMBED_DIM = 512               # fallback embedding dim
GEMINI_MODEL = "gemini-text-embedding-1"  # change if needed (will attempt)
BATCH_SIZE = 64               # embedding batch size for API
INDEX_PREFIX_DEFAULT = "emb_index"

# -------------------------
# PDF -> chunks helper
# -------------------------
def pdf_to_chunks(pdf_path: str, chunk_chars: int = 1200, overlap: int = 256):
    path = Path(pdf_path)
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for pnum, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            # naive split on paragraphs
            paras = [p.strip() for p in text.split("\n\n") if p.strip()]
            if not paras:
                paras = [text]
            for i, para in enumerate(paras):
                start = 0
                while start < len(para):
                    end = start + chunk_chars
                    chunk = para[start:end].strip()
                    if chunk:
                        chunks.append({
                            "id": f"{path.name}_p{pnum}_para{i}_c{start}",
                            "text": chunk,
                            "meta": {"source": path.name, "page": pnum}
                        })
                    start = end - overlap
                    if start < 0:
                        start = 0
    return chunks

# -------------------------
# Gemini embedding (remote) — batched
# -------------------------
def _gemini_embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Uses google-genai client to generate embeddings. Returns list of embeddings (list of floats).
    Falls back to raising an exception if the call fails.
    """
    global _gemini_client
    if not _gemini_client:
        raise RuntimeError("Gemini client not initialized.")
    embeddings = []
    # batch
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        # google-genai embeddings API (library versions differ slightly). We try a typical call pattern.
        # If your installed google-genai uses a different call signature, adjust accordingly.
        resp = _gemini_client.embeddings.create(model=GEMINI_MODEL, input=batch)
        # resp.data is expected to be a sequence with embedding vectors
        # Each item might be dict with 'embedding' key; adapt if your client returns slightly different shape.
        for item in resp.data:
            emb = item.embedding if hasattr(item, "embedding") else item["embedding"]
            embeddings.append(list(emb))
    return embeddings

# -------------------------
# Fallback local hashing embedding
# -------------------------
def _hashing_char_ngrams(texts: List[str], dim: int = EMBED_DIM, ngram: int = 3) -> np.ndarray:
    """
    Fast deterministic fallback embedding: counts character n-grams (n=3 default),
    hashes them into `dim` buckets and L2-normalizes.
    Returns numpy array shape (len(texts), dim).
    """
    mats = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        s = t.lower()
        counts = {}
        for j in range(len(s) - ngram + 1):
            g = s[j:j+ngram]
            h = hash(g) % dim
            mats[i, h] += 1.0
        # also add simple token-level counts as weak signal
        for w in s.split():
            h = hash(w) % dim
            mats[i, h] += 0.5
        # normalize
        norm = np.linalg.norm(mats[i])
        if norm > 0:
            mats[i] /= norm
    return mats

# -------------------------
# Embedding manager (tries Gemini, falls back)
# -------------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Returns numpy array of shape (len(texts), D) with dtype float32 and L2-normalized rows.
    Attempts to use Gemini (google-genai) if configured; otherwise uses local hashing.
    """
    if len(texts) == 0:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    if _use_gemini:
        try:
            embs = _gemini_embed_texts(texts)
            arr = np.array(embs, dtype=np.float32)
            # normalize
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
            return arr
        except Exception as e:
            # log but fall back
            print("Gemini embedding failed, falling back to local hashing embedding. Error:", str(e))

    # fallback
    return _hashing_char_ngrams(texts, dim=EMBED_DIM, ngram=3)

# -------------------------
# EmbedIndex using FAISS (CPU)
# -------------------------
class EmbedIndex:
    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim
        # use inner product on normalized vectors => cosine similarity
        self.index = faiss.IndexFlatIP(self.dim)
        self.id_to_meta: Dict[int, Dict[str, Any]] = {}

    def build(self, chunks: List[Dict[str, Any]]):
        texts = [c["text"] for c in chunks]
        embs = embed_texts(texts)
        if embs.shape[1] != self.dim:
            # if gemini model returns different dim, adapt
            if embs.shape[1] > self.dim:
                # truncate
                embs = embs[:, : self.dim]
            else:
                # pad with zeros
                pad = np.zeros((embs.shape[0], self.dim - embs.shape[1]), dtype=np.float32)
                embs = np.concatenate([embs, pad], axis=1)
        self.index.add(embs.astype(np.float32))
        start = len(self.id_to_meta)
        for i, c in enumerate(chunks):
            self.id_to_meta[start + i] = {"chunk_id": c.get("id"), "text": c.get("text"), "meta": c.get("meta", {})}

    def save(self, prefix: str = INDEX_PREFIX_DEFAULT):
        faiss.write_index(self.index, f"{prefix}.index")
        with open(f"{prefix}.meta.pkl", "wb") as f:
            pickle.dump(self.id_to_meta, f)

    def load(self, prefix: str = INDEX_PREFIX_DEFAULT):
        self.index = faiss.read_index(f"{prefix}.index")
        with open(f"{prefix}.meta.pkl", "rb") as f:
            self.id_to_meta = pickle.load(f)

    def query(self, q_text: str, top_k: int = 5):
        q_emb = embed_texts([q_text])
        if q_emb.shape[1] != self.dim:
            # pad/truncate similarly to build
            if q_emb.shape[1] > self.dim:
                q_emb = q_emb[:, : self.dim]
            else:
                pad = np.zeros((1, self.dim - q_emb.shape[1]), dtype=np.float32)
                q_emb = np.concatenate([q_emb, pad], axis=1)
        D, I = self.index.search(q_emb.astype(np.float32), top_k)
        res = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            meta = self.id_to_meta.get(int(idx), {})
            res.append({
                "index": int(idx),
                "chunk_id": meta.get("chunk_id"),
                "score": float(score),
                "text": meta.get("text"),
                "meta": meta.get("meta", {})
            })
        return res

# -------------------------
# Neo4j evidence helper
# -------------------------
class Neo4jEvidence:
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USERNAME, password: str = NEO4J_PASSWORD):
        if not uri or not user or not password:
            self.driver = None
        else:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        if self.driver:
            self.driver.close()

    def neighbors_of_entities(self, names: List[str], limit: int = 5):
        if not self.driver or not names:
            return []
        q = """
        UNWIND $names AS name
        MATCH (e {name: name})-[r]-(n)
        RETURN e.name AS entity, type(r) AS rel, n.name AS neighbor, labels(n) AS neighbor_labels
        LIMIT $limit
        """
        out = []
        with self.driver.session() as ses:
            res = ses.run(q, names=names, limit=limit)
            for r in res:
                out.append({
                    "entity": r["entity"],
                    "rel": r["rel"],
                    "neighbor": r["neighbor"],
                    "neighbor_labels": r["neighbor_labels"]
                })
        return out

# -------------------------
# Top-level functions for your role
# -------------------------
def build_index_from_json(chunks: List[Dict[str, Any]], save_prefix: str = INDEX_PREFIX_DEFAULT) -> Dict[str, Any]:
    idx = EmbedIndex(dim=EMBED_DIM)
    idx.build(chunks)
    idx.save(save_prefix)
    return {"status": "ok", "saved_prefix": save_prefix, "count": len(chunks), "embed_dim": EMBED_DIM, "used_gemini": _use_gemini}

def hybrid_retrieve(query: str, top_k: int = 5, index_prefix: str = INDEX_PREFIX_DEFAULT, use_graph: bool = True) -> Dict[str, Any]:
    idx = EmbedIndex(dim=EMBED_DIM)
    idx.load(index_prefix)
    hits = idx.query(query, top_k=top_k)
    neo = Neo4jEvidence() if use_graph else None
    for h in hits:
        entities = h.get("meta", {}).get("entities", [])
        ge = neo.neighbors_of_entities(entities, limit=5) if (neo and entities) else []
        boost = 0.05 * len(ge)
        h["graph_evidence"] = ge
        h["combined_score"] = h["score"] + boost
    if neo:
        neo.close()
    hits_sorted = sorted(hits, key=lambda x: x.get("combined_score", x["score"]), reverse=True)
    return {"query": query, "results": hits_sorted}

# -------------------------
# if run as script: small demo
# -------------------------
if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", help="chunks.json", default=None)
    ap.add_argument("--query", help="text query", default=None)
    ap.add_argument("--index_prefix", default=INDEX_PREFIX_DEFAULT)
    args = ap.parse_args()

    if args.build:
        with open(args.build, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(build_index_from_json(chunks, save_prefix=args.index_prefix))
        sys.exit(0)

    if args.query:
        print(json.dumps(hybrid_retrieve(args.query, top_k=5, index_prefix=args.index_prefix), indent=2))
        sys.exit(0)

    print("Usage: --build chunks.json  OR  --query 'text'")
