"""
Ingestion & Enrichment pipeline (Python)

Features implemented:
- Parsing: PDF, DOCX, HTML, TXT
- Cleaning & normalization
- Chunking into semantically meaningful chunks (by heading/paragraph + token limit)
- Summarization (pluggable): default uses OpenAI if API key provided, otherwise a simple gensim/summarizer placeholder
- Embedding (pluggable): default uses sentence-transformers if installed, otherwise a stub
- Storage: saves raw files, processed chunks and metadata to disk (JSONL) + optional FAISS index
- Event publishing: simple Redis pub/sub or webhook call

This is a template / working reference implementation. Swap-in your own summarizer/embedding/storage backends
for production (OpenAI, HuggingFace, Pinecone, Chroma, Redis Streams, Kafka, etc.).

Usage:
    python ingest_enrich_pipeline.py --input /path/to/file.pdf --outdir /tmp/ingest_output

Requirements (suggested):
    pip install pdfplumber python-docx beautifulsoup4 nltk sentence-transformers faiss-cpu redis requests

Note: The code is intentionally defensive and pluggable. See the classes:
- ParserRegistry: select parser by mimetype or extension
- Cleaner: normalize text
- Chunker: chunk by paragraphs + token limit
- Summarizer: pluggable summarizer
- Embedder: pluggable embedder
- Storage: simple JSONL + optional FAISS
- EventPublisher: Redis or webhook

"""

import os
import re
import json
import uuid
import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Parsers
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx
except Exception:
    docx = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

# NLP helpers
try:
    import nltk
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk = None

# Embedding / FAISS
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None

# Optional event publisher
try:
    import redis
except Exception:
    redis = None

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------- Parsing -------------------------
class Parser:
    def parse(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        """Return (text, metadata)"""
        raise NotImplementedError


class PDFParser(Parser):
    def parse(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        if pdfplumber is None:
            raise RuntimeError("pdfplumber not installed")
        text_chunks = []
        metadata = {"source": str(path), "type": "pdf"}
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                text_chunks.append(page_text)
        text = "\n\n".join(text_chunks)
        return text, metadata


class DOCXParser(Parser):
    def parse(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        if docx is None:
            raise RuntimeError("python-docx not installed")
        doc = docx.Document(str(path))
        paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        text = "\n\n".join(paras)
        metadata = {"source": str(path), "type": "docx"}
        return text, metadata


class HTMLParser(Parser):
    def parse(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        if BeautifulSoup is None:
            raise RuntimeError("beautifulsoup4 not installed")
        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        for s in soup(['script', 'style', 'noscript']):
            s.decompose()
        text = soup.get_text(separator='\n')
        metadata = {"source": str(path), "type": "html"}
        return text, metadata


class TextParser(Parser):
    def parse(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        metadata = {"source": str(path), "type": "text"}
        return text, metadata


class ParserRegistry:
    def __init__(self):
        self.map = {
            '.pdf': PDFParser(),
            '.docx': DOCXParser(),
            '.html': HTMLParser(),
            '.htm': HTMLParser(),
            '.txt': TextParser(),
        }

    def get(self, path: Path) -> Parser:
        ext = path.suffix.lower()
        return self.map.get(ext, TextParser())


# ------------------------- Cleaning & Normalization -------------------------
class Cleaner:
    def __init__(self, remove_long_whitespace: bool = True):
        self.remove_long_whitespace = remove_long_whitespace

    def clean(self, text: str) -> str:
        # Normalize unicode, remove non-printables, fix spacing
        text = text.replace('\u00A0', ' ')
        text = re.sub(r'\r\n?', '\n', text)
        text = re.sub(r'\t+', ' ', text)
        # Remove repeated blank lines
        if self.remove_long_whitespace:
            text = re.sub(r"\n{3,}", '\n\n', text)
        # Trim
        text = text.strip()
        return text


# ------------------------- Chunking -------------------------
class Chunker:
    def __init__(self, max_tokens: int = 512, overlap: int = 32):
        self.max_tokens = max_tokens
        self.overlap = overlap
        # If nltk is available, use sent_tokenize
        if nltk:
            try:
                from nltk import sent_tokenize
                self.sent_tokenize = sent_tokenize
            except Exception:
                self.sent_tokenize = None
        else:
            self.sent_tokenize = None

    def approx_tokens(self, text: str) -> int:
        # conservative token estimate: 1 token per 4 chars
        return max(1, len(text) // 4)

    def chunk_by_paragraph(self, text: str) -> List[str]:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks = []
        buffer = ""
        for p in paras:
            if not buffer:
                buffer = p
            else:
                candidate = buffer + "\n\n" + p
                if self.approx_tokens(candidate) <= self.max_tokens:
                    buffer = candidate
                else:
                    chunks.append(buffer)
                    buffer = p
        if buffer:
            chunks.append(buffer)
        # further split very large chunks by sentence if available
        final_chunks = []
        for c in chunks:
            if self.approx_tokens(c) > self.max_tokens:
                final_chunks.extend(self.split_by_sentences(c))
            else:
                final_chunks.append(c)
        return final_chunks

    def split_by_sentences(self, text: str) -> List[str]:
        if self.sent_tokenize:
            sents = self.sent_tokenize(text)
        else:
            # fallback simple split on punctuation
            sents = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        buffer = ""
        for s in sents:
            candidate = (buffer + ' ' + s).strip() if buffer else s
            if self.approx_tokens(candidate) <= self.max_tokens:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                buffer = s
        if buffer:
            chunks.append(buffer)
        return chunks


# ------------------------- Summarization -------------------------
class Summarizer:
    def __init__(self, provider: str = 'openai', openai_api_key: Optional[str] = None):
        self.provider = provider
        self.openai_api_key = openai_api_key
        # we keep provider options minimal here; users can extend

    def summarize(self, text: str, max_tokens: int = 128) -> str:
        if self.provider == 'openai' and self.openai_api_key:
            try:
                import openai
                openai.api_key = self.openai_api_key
                prompt = f"Summarize the following content in one concise paragraph:\n\n{text}\n\nSummary:" 
                resp = openai.Completion.create(
                    model='text-davinci-003', prompt=prompt, max_tokens=max_tokens, temperature=0.2
                )
                return resp.choices[0].text.strip()
            except Exception as e:
                logger.warning("OpenAI summarization failed: %s", e)
        # fallback: simple extractive summary (first 2 sentences)
        sents = re.split(r'(?<=[.!?])\s+', text)
        summary = ' '.join(sents[:2]).strip()
        if not summary:
            summary = text[:max_tokens*4]
        return summary


# ------------------------- Embedding -------------------------
class Embedder:
    def __init__(self, provider: str = 'sentence-transformers', model_name: str = 'all-MiniLM-L6-v2'):
        self.provider = provider
        self.model_name = model_name
        self.model = None
        if provider == 'sentence-transformers' and SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                logger.warning("Failed to load SentenceTransformer: %s", e)
                self.model = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.model is not None:
            return self.model.encode(texts, show_progress_bar=False).tolist()
        # fallback: simple hashing-based embedding (not useful for real retrieval)
        embeddings = []
        for t in texts:
            h = abs(hash(t)) % (10**8)
            vec = [((h >> i) & 0xFF) / 255.0 for i in range(32)]
            embeddings.append(vec)
        return embeddings


# ------------------------- Storage -------------------------
class Storage:
    def __init__(self, outdir: Path, use_faiss: bool = False):
        self.outdir = outdir
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.chunks_file = self.outdir / 'chunks.jsonl'
        self.raw_dir = self.outdir / 'raw'
        self.raw_dir.mkdir(exist_ok=True)
        self.use_faiss = use_faiss and (faiss is not None)
        self._faiss_index = None
        self._dimension = None

    def save_raw(self, path: Path) -> str:
        target = self.raw_dir / f"{uuid.uuid4().hex}_{path.name}"
        target.write_bytes(path.read_bytes())
        return str(target)

    def save_chunk(self, chunk_doc: Dict[str, Any]):
        # append to JSONL
        with open(self.chunks_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(chunk_doc, ensure_ascii=False) + '\n')

    def init_faiss(self, dimension: int):
        if not self.use_faiss:
            return
        self._dimension = dimension
        self._faiss_index = faiss.IndexFlatL2(dimension)

    def add_to_faiss(self, vectors: List[List[float]]):
        if self._faiss_index is None:
            raise RuntimeError('FAISS not initialized')
        import numpy as np
        arr = np.array(vectors).astype('float32')
        self._faiss_index.add(arr)

    def dump_faiss(self, fname: str = 'index.faiss'):
        if self._faiss_index is None:
            return
        faiss.write_index(self._faiss_index, str(self.outdir / fname))


# ------------------------- Event Publisher -------------------------
class EventPublisher:
    def __init__(self, redis_url: Optional[str] = None, webhook_url: Optional[str] = None):
        self.redis_url = redis_url
        self.webhook_url = webhook_url
        if self.redis_url and redis is not None:
            self._redis = redis.from_url(redis_url)
        else:
            self._redis = None

    def publish(self, channel: str, payload: Dict[str, Any]):
        payload_json = json.dumps(payload, default=str)
        if self._redis:
            self._redis.publish(channel, payload_json)
        if self.webhook_url:
            try:
                requests.post(self.webhook_url, json=payload, timeout=3)
            except Exception as e:
                logger.warning("Webhook publish failed: %s", e)


# ------------------------- Pipeline Orchestration -------------------------
class IngestEnrichPipeline:
    def __init__(
        self,
        outdir: Path,
        summarizer: Optional[Summarizer] = None,
        embedder: Optional[Embedder] = None,
        storage: Optional[Storage] = None,
        publisher: Optional[EventPublisher] = None,
        chunker: Optional[Chunker] = None,
        cleaner: Optional[Cleaner] = None,
    ):
        self.registry = ParserRegistry()
        self.cleaner = cleaner or Cleaner()
        self.chunker = chunker or Chunker()
        self.summarizer = summarizer or Summarizer(provider='openai')
        self.embedder = embedder or Embedder()
        self.storage = storage or Storage(outdir)
        self.publisher = publisher or EventPublisher()

    def ingest_file(self, path: Path, extra_meta: Dict[str, Any] = None) -> Dict[str, Any]:
        # 1. parse
        parser = self.registry.get(path)
        raw_text, md = parser.parse(path)
        md.update(extra_meta or {})
        logger.info('Parsed file: %s (chars=%d)', path.name, len(raw_text))

        # save raw
        saved_raw = self.storage.save_raw(path)
        md['raw_saved_path'] = saved_raw

        # 2. clean
        cleaned = self.cleaner.clean(raw_text)

        # 3. chunk
        chunks = self.chunker.chunk_by_paragraph(cleaned)
        logger.info('Produced %d chunks', len(chunks))

        # 4. summarize and embed
        summaries = []
        for i, c in enumerate(chunks):
            # use the chunk index as the chunk_id (as requested)
            chunk_id = str(i)
            summary = self.summarizer.summarize(c)
            summaries.append(summary)
            doc = {
                'chunk_id': chunk_id,
                'text': c,
                'summary': summary,
            }
            self.storage.save_chunk(doc)

        # embeddings
        vectors = self.embedder.embed(chunks)

        # attach embeddings to chunks and save again with embeddings
        for i, vec in enumerate(vectors):
            # use the same index-based chunk_id for embeddings
            chunk_doc = {
                'chunk_id': str(i),
                'text': chunks[i],
                'summary': summaries[i],
            }
            self.storage.save_chunk(chunk_doc)

        # optional FAISS
        if self.storage.use_faiss and vectors:
            dim = len(vectors[0])
            if self.storage._faiss_index is None:
                self.storage.init_faiss(dim)
            self.storage.add_to_faiss(vectors)

        # 5. publish ingestion event
        event = {
            'event': 'ingestion_completed',
            'source': str(path),
            'num_chunks': len(chunks),
            'sample_summary': summaries[0] if summaries else None,
        }
        self.publisher.publish('ingestion.events', event)
        return event


# ------------------------- CLI -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Input file path')
    parser.add_argument('--outdir', '-o', required=True, help='Output directory')
    parser.add_argument('--openai-key', help='OpenAI API key (optional)')
    parser.add_argument('--redis-url', help='Redis URL for events (optional)')
    parser.add_argument('--webhook-url', help='Webhook URL for events (optional)')
    parser.add_argument('--use-faiss', action='store_true', help='Enable FAISS index')
    args = parser.parse_args()

    p = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summarizer = Summarizer(provider='openai', openai_api_key=args.openai_key)
    embedder = Embedder()
    storage = Storage(outdir, use_faiss=args.use_faiss)
    publisher = EventPublisher(redis_url=args.redis_url, webhook_url=args.webhook_url)

    pipeline = IngestEnrichPipeline(
        outdir=outdir,
        summarizer=summarizer,
        embedder=embedder,
        storage=storage,
        publisher=publisher,
    )

    # If input is a directory, process supported files inside it (skip dirs).
    if p.is_dir():
        supported = {'.pdf', '.docx', '.html', '.htm', '.txt'}
        for f in sorted(p.iterdir()):
            if not f.is_file():
                logger.debug('Skipping non-file: %s', f)
                continue
            if f.suffix.lower() not in supported:
                logger.info('Skipping unsupported file type: %s', f.name)
                continue
            # If PDF support isn't available, skip PDF files gracefully
            if f.suffix.lower() == '.pdf' and pdfplumber is None:
                logger.warning('Skipping PDF %s: pdfplumber not installed', f.name)
                continue
            try:
                event = pipeline.ingest_file(f)
                print('Ingestion event for', f.name, json.dumps(event, indent=2))
            except Exception:
                logger.exception('Failed to ingest file: %s', f)
        return

    # Single file input
    # If PDF support isn't available, skip single-file PDF inputs gracefully
    if p.suffix.lower() == '.pdf' and pdfplumber is None:
        logger.warning('Skipping PDF %s: pdfplumber not installed', p.name)
        return
    try:
        event = pipeline.ingest_file(p)
        print('Ingestion event:', json.dumps(event, indent=2))
    except Exception:
        logger.exception('Failed to ingest file: %s', p)


if __name__ == '__main__':
    main()
