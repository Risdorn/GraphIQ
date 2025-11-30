#!/usr/bin/env python3
"""
Ingestion agent: watches a data directory and runs the ingestion pipeline
on new or changed files. It reuses the classes in `ingest_enrich_pipeline.py`.

Usage:
  python GraphIQ/scripts/ingest_agent.py --data-dir GraphIQ/data --outdir GraphIQ/data/output

The agent keeps a small `processed.json` state file in the output directory to
avoid reprocessing the same files unless modified.
"""
import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Dict

logger = logging.getLogger("ingest_agent")
logging.basicConfig(level=logging.INFO)


def import_pipeline_module(module_path: Path):
    # Load the pipeline module directly from the file path to avoid
    # executing any CLI top-level code accidentally via regular import.
    # This uses importlib to load the module from the scripts folder.
    from importlib import util

    pipeline_path = module_path / 'ingest_enrich_pipeline.py'
    if not pipeline_path.exists():
        raise ImportError(f"Pipeline module not found at {pipeline_path}")

    spec = util.spec_from_file_location('graphiq.ingest_enrich_pipeline', str(pipeline_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {pipeline_path}")
    mod = util.module_from_spec(spec)
    # Execute the module in its own namespace (it will not be __main__)
    spec.loader.exec_module(mod)
    return mod


class IngestAgent:
    def __init__(self, data_dir: Path, outdir: Path, poll_interval: float = 5.0, openai_key: str = None, use_faiss: bool = False):
        self.data_dir = data_dir
        self.outdir = outdir
        self.poll_interval = poll_interval
        self.openai_key = openai_key
        self.use_faiss = use_faiss

        # import pipeline module from same folder
        pkg_dir = Path(__file__).resolve().parent
        self.pipeline_module = import_pipeline_module(pkg_dir)

        # build pipeline components
        self.summarizer = self.pipeline_module.Summarizer(provider='openai', openai_api_key=self.openai_key)
        self.embedder = self.pipeline_module.Embedder()
        self.storage = self.pipeline_module.Storage(self.outdir, use_faiss=self.use_faiss)
        self.publisher = self.pipeline_module.EventPublisher()
        self.pipeline = self.pipeline_module.IngestEnrichPipeline(
            outdir=self.outdir,
            summarizer=self.summarizer,
            embedder=self.embedder,
            storage=self.storage,
            publisher=self.publisher,
        )

        self.state_file = self.outdir / 'processed.json'
        self.processed = self._load_state()
        self._stop = False

    def _load_state(self) -> Dict[str, float]:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding='utf-8'))
            except Exception:
                logger.exception('Failed to read state file; starting fresh')
        return {}

    def _save_state(self):
        try:
            self.outdir.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(self.processed, indent=2), encoding='utf-8')
        except Exception:
            logger.exception('Failed to write state file')

    def _should_process(self, p: Path) -> bool:
        if not p.is_file():
            return False
        # only process supported extensions
        supported = {'.pdf', '.docx', '.html', '.htm', '.txt'}
        if p.suffix.lower() not in supported:
            return False
        mtime = p.stat().st_mtime
        key = str(p.resolve())
        prev = self.processed.get(key)
        if prev is None or prev != mtime:
            return True
        return False

    def mark_processed(self, p: Path):
        key = str(p.resolve())
        self.processed[key] = p.stat().st_mtime
        self._save_state()

    def run_once(self):
        for p in sorted(self.data_dir.iterdir()):
            try:
                if not self._should_process(p):
                    continue
                logger.info('Processing: %s', p)
                try:
                    event = self.pipeline.ingest_file(p)
                    logger.info('Ingested %s -> event=%s', p.name, event)
                    self.mark_processed(p)
                except Exception:
                    logger.exception('Failed to ingest %s', p)
            except Exception:
                logger.exception('Error handling %s', p)

    def run(self, process_existing: bool = True):
        # optional initial pass
        if process_existing:
            logger.info('Initial scan of %s', self.data_dir)
            self.run_once()

        # setup graceful shutdown
        def _signal_handler(sig, frame):
            logger.info('Received signal %s, stopping agent...', sig)
            self._stop = True

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        logger.info('Starting ingestion agent; polling every %.1fs', self.poll_interval)
        while not self._stop:
            try:
                self.run_once()
            except Exception:
                logger.exception('Error during run loop')
            time.sleep(self.poll_interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='GraphIQ/data', help='Directory to watch for input files')
    ap.add_argument('--outdir', default='GraphIQ/data/output', help='Output directory where pipeline writes results')
    ap.add_argument('--poll', type=float, default=5.0, help='Polling interval in seconds')
    ap.add_argument('--openai-key', default=None, help='OpenAI API key (optional)')
    ap.add_argument('--no-initial', dest='initial', action='store_false', help='Do not process existing files on startup')
    ap.add_argument('--use-faiss', action='store_true', help='Enable FAISS in storage if available')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    if not data_dir.exists():
        logger.error('Data dir does not exist: %s', data_dir)
        return

    agent = IngestAgent(data_dir, outdir, poll_interval=args.poll, openai_key=args.openai_key, use_faiss=args.use_faiss)
    agent.run(process_existing=args.initial)


if __name__ == '__main__':
    main()
