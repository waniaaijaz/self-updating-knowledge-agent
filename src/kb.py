"""Top-level facade. One import gets you a wired system."""

from __future__ import annotations

from pathlib import Path

from . import config
from .agent import AgentRunner
from .chunking import chunk_file, chunk_markdown
from .embeddings import get_embedder
from .graph_store import get_graph_store
from .hitl import ReviewQueue
from .nli import get_detector
from .pipeline import Services
from .query import QueryEngine
from .vector_store import get_vector_store


class KnowledgeBase:
    def __init__(self, offline: bool | None = None):
        self.embedder = get_embedder(force_offline=offline)
        self.detector = get_detector(force_offline=offline)
        self.vectors = get_vector_store(dim=self.embedder.dim)
        self.graph = get_graph_store()
        self.queue = ReviewQueue()
        self.services = Services(
            self.embedder, self.detector, self.vectors, self.graph, self.queue
        )
        self.agent = AgentRunner(self.services)
        self.query_engine = QueryEngine(self.services)

    # ---------------------------------------------------------------- info
    def backends(self) -> dict:
        return {
            "embedder": self.embedder.name,
            "nli": self.detector.backend,
            "vectors": self.vectors.backend,
            "graph": self.graph.backend,
            "orchestrator": self.agent.mode,
        }

    # -------------------------------------------------------------- ingest
    def ingest_markdown(self, text, doc_id, version, timestamp, verbose=True, force_reingest=False):
        chunks = chunk_markdown(text, doc_id, version, timestamp)
        return self._ingest_chunks(chunks, verbose, force_reingest)

    def ingest_file(self, path, doc_id, version, timestamp, verbose=True, force_reingest=False):
        chunks = chunk_file(Path(path), doc_id, version, timestamp)
        return self._ingest_chunks(chunks, verbose, force_reingest)

    def _ingest_chunks(self, chunks, verbose, force_reingest=False):
        summary = {"INSERT": 0, "SUPERSEDE": 0, "HITL": 0, "DUPLICATE": 0}
        logs = []
        for chunk in chunks:
            state = self.agent.ingest_chunk(chunk.to_dict(), force=force_reingest)
            summary[state.get("action", "INSERT")] += 1
            logs.extend(state.get("log", []))
        if verbose:
            for line in logs:
                print("  " + line)
        return {"chunks": len(chunks), "summary": summary, "log": logs}

    # --------------------------------------------------------------- query
    def ask(self, question, top_k=5):
        return self.query_engine.ask(question, top_k=top_k)

    # --------------------------------------------------------------- admin
    def reset(self):
        self.vectors.reset()
        self.graph.reset()
        self.queue.reset()
        self.services._hashes = None

    def stats(self):
        nodes = self.graph.nodes()
        return {
            "nodes": len(nodes),
            "active": sum(1 for n in nodes if n.get("status") == "ACTIVE"),
            "superseded": sum(1 for n in nodes if n.get("status") == "SUPERSEDED"),
            "pending_review": sum(1 for n in nodes if n.get("status") == "PENDING_REVIEW"),
            "supersedes_edges": len(self.graph.edges()),
            "open_reviews": len(self.queue.open_items()),
        }

    def close(self):
        if hasattr(self, "vectors") and hasattr(self.vectors, "close"):
            self.vectors.close()
