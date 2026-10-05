"""The ingest-and-reconcile pipeline, as plain functions.

Each function is one node of the LangGraph state machine in agent.py. They
live here, framework-free, so they can be unit-tested and so the pipeline
still runs if langgraph is not installed.

Flow per incoming chunk:

    dedup_gate ──(duplicate)──────────────────────────────► END
        │
        ▼
    retrieve_candidates
        │
        ▼
    detect_contradiction
        │
        ├─ score >= 0.88 ──► supersede_node      (auto)
        ├─ 0.50..0.87    ──► human_review_node   (HITL queue)
        └─ otherwise     ──► insert_node         (clean insert)
"""

from __future__ import annotations

from typing import Any, TypedDict

from . import config
from .access import DEFAULT_TENANT
from .chunking import Chunk
from .freshness import STATUS_ACTIVE, STATUS_PENDING_REVIEW
from .nli import CONTRADICTION  # noqa: F401


class IngestState(TypedDict, total=False):
    chunk: dict
    vector: Any
    candidates: list[dict]
    conflict: dict | None
    action: str  # DUPLICATE | INSERT | SUPERSEDE | HITL
    log: list[str]
    force: bool  # skip the dedup hash check, for re-testing without a reset


class Services:
    """Bundle of the four collaborators, so nodes stay pure-ish and testable."""

    def __init__(self, embedder, detector, vector_store, graph_store, review_queue):
        self.embedder = embedder
        self.detector = detector
        self.vectors = vector_store
        self.graph = graph_store
        self.queue = review_queue
        self._hashes: set[str] | None = None

    def known_hashes(self) -> set[str]:
        if self._hashes is None:
            self._hashes = {
                p.get("content_hash")
                for p in self.vectors.all_payloads()
                if p.get("content_hash")
            }
        return self._hashes


# ------------------------------------------------------------------ nodes

def dedup_gate(state: IngestState, svc: Services) -> IngestState:
    """SHA-256 check before spending any compute on embeddings or NLI.

    Reformatting a document should not re-run the whole pipeline over every
    unchanged clause. In a 40-page handbook where one rule changed, this
    skips ~95% of the work.

    state["force"] bypasses the check. Mainly for re-testing the same file
    during dev without wiping storage first."""
    chunk = state["chunk"]
    log = state.get("log", [])
    if not state.get("force") and chunk["content_hash"] in svc.known_hashes():
        log.append(f"DUPLICATE  {chunk['id']} (hash match, skipped)")
        return {**state, "action": "DUPLICATE", "log": log}
    return {**state, "action": "", "log": log}


def retrieve_candidates(state: IngestState, svc: Services) -> IngestState:
    """Find prior chunks this one might contradict or supersede.

    Scoped to the same doc_id: contradiction detection compares different
    versions of the *same* document (v1's 4.1 vs v2's 4.1), never unrelated
    documents that happen to use similar wording or the same section number.
    Two different policies both having a clause "4.1" is a coincidence, not
    evidence of a conflict — without this scoping, an unrelated document
    could silently mark another document's chunk as superseded.

    Candidates are excluded by their own chunk id, not by version string.
    Excluding by version would silently hide contradictions any time two
    genuinely different revisions share a version label (a mislabel, or a
    workflow that dates documents instead of versioning them) — which
    previously caused ingestion to report "0 candidate(s)" and never
    compare them at all."""
    chunk = state["chunk"]
    vector = svc.embedder.encode_one(chunk["text"])
    # Pull a wider pool than we'll actually use: the similarity floor below
    # is a lexical proxy and cheap embedders (offline hashing mode) can put
    # a genuine same-section rewrite (all new wording, e.g. "hybrid allowed"
    # -> "remote prohibited") right at or under that floor, while unrelated
    # sections in the same doc sit only slightly lower. A wider pool lets
    # the section_path check below rescue those before they're lost.
    hits = svc.vectors.search(
        vector,
        top_k=max(config.TOP_K_CANDIDATES * 4, 12),
        doc_id=chunk["doc_id"],
        exclude_id=chunk["id"],
        # same doc_id in two tenants is two different documents, one tenant's
        # upload must never supersede the other's. No-op for default access.
        tenant_id=chunk.get("tenant_id") or DEFAULT_TENANT,
    )
    min_sim = getattr(svc.embedder, "min_candidate_sim", config.MIN_CANDIDATE_SIM)
    candidates = [
        {"payload": h.payload, "similarity": h.score}
        for h in hits
        if h.payload.get("status", STATUS_ACTIVE) == STATUS_ACTIVE
        # Sibling sections of the same revision are different rules, not
        # competing ones. Revisions are told apart by (version, timestamp)
        # rather than version alone, so the mislabel case above still works.
        and (h.payload.get("version"), h.payload.get("timestamp"))
        != (chunk.get("version"), chunk.get("timestamp"))
        and (
            h.score >= min_sim
            # Same section_path across versions is the same rule by
            # construction (see the boost in detect_contradiction below) —
            # let it in even if wording changed enough to tank raw lexical
            # similarity, instead of silently dropping the clearest
            # contradictions before they're ever scored.
            or (
                chunk.get("section_path")
                and h.payload.get("section_path") == chunk.get("section_path")
            )
        )
    ][: config.TOP_K_CANDIDATES]
    log = state.get("log", [])
    log.append(f"RETRIEVE   {chunk['id']} -> {len(candidates)} candidate(s)")
    return {**state, "vector": vector, "candidates": candidates, "log": log}


def detect_contradiction(state: IngestState, svc: Services) -> IngestState:
    """Score every candidate, keep the strongest contradiction, then route."""
    chunk = state["chunk"]
    log = state.get("log", [])
    best = None

    for cand in state.get("candidates", []):
        old_text = cand["payload"].get("body") or cand["payload"]["text"]
        result = svc.detector.evaluate_bidirectional(old_text, chunk["body"])

        # A section-path match is corroborating evidence: "4.2" in v1 and "4.2"
        # in v2 are the same rule by construction, so we trust the classifier
        # a little more there. Capped so it can never manufacture a conflict
        # out of a low-confidence score on its own.
        boost = 0.05 if cand["payload"].get("section_path") == chunk.get("section_path") else 0.0
        score = min(result.contradiction_score + boost, 0.999)

        if best is None or score > best["score"]:
            best = {
                "score": score,
                "raw": result.to_dict(),
                "old": cand["payload"],
                "similarity": cand["similarity"],
                "section_boost": boost,
            }

    if best is None:
        log.append(f"INSERT     {chunk['id']} (no comparable prior chunk)")
        return {**state, "conflict": None, "action": "INSERT", "log": log}

    auto_thr = config.AUTO_THRESHOLD_OVERRIDE
    hitl_thr = config.HITL_THRESHOLD_OVERRIDE
    if auto_thr is None:
        auto_thr = svc.detector.auto_threshold
    if hitl_thr is None:
        hitl_thr = svc.detector.hitl_threshold

    score = best["score"]
    if score >= auto_thr:
        action = "SUPERSEDE"
    elif score >= hitl_thr:
        action = "HITL"
    else:
        action = "INSERT"

    log.append(
        f"{action:<10} {chunk['id']}  contradiction={score:.3f} "
        f"vs {best['old']['id']} (sim={best['similarity']:.3f})"
    )
    return {**state, "conflict": best, "action": action, "log": log}


def route_action(state: IngestState) -> str:
    return state.get("action", "INSERT")


def insert_node(state: IngestState, svc: Services) -> IngestState:
    chunk = state["chunk"]
    svc.vectors.upsert([Chunk(**{k: v for k, v in chunk.items()})], state["vector"][None, :])
    svc.graph.add_chunk(chunk, status=STATUS_ACTIVE)
    svc.known_hashes().add(chunk["content_hash"])
    return state


def supersede_node(state: IngestState, svc: Services) -> IngestState:
    chunk, conflict = state["chunk"], state["conflict"]
    insert_node(state, svc)
    svc.graph.supersede(
        new_id=chunk["id"],
        old_id=conflict["old"]["id"],
        reason=f"NLI_AUTOMATED:{conflict['raw']['backend']}",
        nli_score=conflict["score"],
    )
    svc.vectors.set_payload(conflict["old"]["id"], {"status": "SUPERSEDED"})
    return state


def human_review_node(state: IngestState, svc: Services) -> IngestState:
    """Ambiguous. Store the new chunk as PENDING_REVIEW and queue the pair.

    Critically, the old chunk is left ACTIVE. Until a human confirms, the
    system keeps serving the rule it already trusted rather than acting on
    a coin-flip."""
    chunk, conflict = state["chunk"], state["conflict"]
    svc.vectors.upsert([Chunk(**chunk)], state["vector"][None, :])
    svc.vectors.set_payload(chunk["id"], {"status": STATUS_PENDING_REVIEW})
    svc.graph.add_chunk(chunk, status=STATUS_PENDING_REVIEW)
    svc.known_hashes().add(chunk["content_hash"])
    svc.queue.add(chunk, conflict["old"], conflict["raw"], conflict["similarity"])
    return state


# ------------------------------------------------------- fallback executor

def run_sequential(chunk_dict: dict, svc: Services, force: bool = False) -> IngestState:
    """Same graph, executed by hand. Used when langgraph is unavailable."""
    state: IngestState = {"chunk": chunk_dict, "log": [], "force": force}

    state = dedup_gate(state, svc)
    if state["action"] == "DUPLICATE":
        return state

    state = retrieve_candidates(state, svc)
    state = detect_contradiction(state, svc)

    {"INSERT": insert_node, "SUPERSEDE": supersede_node, "HITL": human_review_node}[
        state["action"]
    ](state, svc)
    return state
