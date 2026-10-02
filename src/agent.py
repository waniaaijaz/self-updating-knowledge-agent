"""LangGraph state machine.

Why LangGraph and not a plain LangChain chain: a chain is a directed acyclic
pass-through. What we need is conditional branching on a classifier score,
with one branch that mutates persistent state (the graph edge) and another
that parks the item in a human queue and stops. That is a state machine, and
LangGraph is the piece of the stack that models it explicitly.

              ┌──────────────┐
              │  dedup_gate  │──── DUPLICATE ──────────────► END
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │   retrieve   │
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │ detect (NLI) │
              └──────┬───────┘
          ┌──────────┼──────────┐
          ▼          ▼          ▼
     supersede    review     insert          ───► END
"""

from __future__ import annotations

from . import pipeline
from .pipeline import IngestState, Services


def build_agent(svc: Services):
    from langgraph.graph import END, StateGraph

    def node(fn):
        return lambda state: fn(state, svc)

    wf = StateGraph(IngestState)
    wf.add_node("dedup", node(pipeline.dedup_gate))
    wf.add_node("retrieve", node(pipeline.retrieve_candidates))
    wf.add_node("detect", node(pipeline.detect_contradiction))
    wf.add_node("insert", node(pipeline.insert_node))
    wf.add_node("supersede", node(pipeline.supersede_node))
    wf.add_node("review", node(pipeline.human_review_node))

    wf.set_entry_point("dedup")
    wf.add_conditional_edges(
        "dedup",
        lambda s: "DUPLICATE" if s.get("action") == "DUPLICATE" else "CONTINUE",
        {"DUPLICATE": END, "CONTINUE": "retrieve"},
    )
    wf.add_edge("retrieve", "detect")
    wf.add_conditional_edges(
        "detect",
        pipeline.route_action,
        {"INSERT": "insert", "SUPERSEDE": "supersede", "HITL": "review"},
    )
    for terminal in ("insert", "supersede", "review"):
        wf.add_edge(terminal, END)

    return wf.compile()


class AgentRunner:
    """Uses LangGraph when available, falls back to the hand-rolled executor
    so the project is never blocked on an install."""

    def __init__(self, svc: Services):
        self.svc = svc
        try:
            self.app = build_agent(svc)
            self.mode = "langgraph"
        except Exception as exc:  # noqa: BLE001
            print(f"[agent] langgraph unavailable ({exc}); using sequential runner.")
            self.app = None
            self.mode = "sequential"

    def ingest_chunk(self, chunk_dict: dict, force: bool = False) -> IngestState:
        if self.app is not None:
            return self.app.invoke({"chunk": chunk_dict, "log": [], "force": force})
        return pipeline.run_sequential(chunk_dict, self.svc, force=force)
