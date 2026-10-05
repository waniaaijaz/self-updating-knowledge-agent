"""Knowledge graph: nodes are chunks, edges record what overwrote what.

    (:Chunk {id:"hr::v2::4.2::0"})
      -[:SUPERSEDES {detected_at, reason, nli_score}]->
    (:Chunk {id:"hr::v1::4.2::0", status:"SUPERSEDED"})

The edge is the audit trail. Being able to answer "why is this rule no longer
being served, and what replaced it, and when" is the thing that separates this
from a vector database with a date filter on it.

NetworkX + JSON is the default (zero setup). Neo4j AuraDB free tier is a drop-in
swap via env vars.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from . import config
from .freshness import STATUS_ACTIVE, STATUS_SUPERSEDED


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BaseGraphStore:
    backend = "base"

    def add_chunk(self, chunk_dict: dict, status: str = STATUS_ACTIVE) -> None: ...
    def supersede(self, new_id: str, old_id: str, reason: str, nli_score: float) -> None: ...
    def set_status(self, chunk_id: str, status: str) -> None: ...
    def get(self, chunk_id: str) -> dict | None: ...
    def nodes(self) -> list[dict]: ...
    def edges(self) -> list[dict]: ...
    def lineage(self, chunk_id: str) -> list[dict]: ...
    def reset(self) -> None: ...


class NetworkXGraphStore(BaseGraphStore):
    backend = "networkx"

    def __init__(self, path=None):
        import networkx as nx

        self.path = path or config.GRAPH_JSON_PATH
        self.g = nx.DiGraph()
        self._load()

    def _load(self):
        if self.path.exists():
            blob = json.loads(self.path.read_text())
            for n in blob.get("nodes", []):
                self.g.add_node(n["id"], **n)
            for e in blob.get("edges", []):
                self.g.add_edge(e["source"], e["target"], **e)

    def _save(self):
        self.path.write_text(json.dumps({
            "nodes": [dict(d) for _, d in self.g.nodes(data=True)],
            "edges": [dict(d) for *_, d in self.g.edges(data=True)],
        }, indent=2))

    def add_chunk(self, chunk_dict, status=STATUS_ACTIVE):
        data = dict(chunk_dict)
        data["status"] = status
        data.setdefault("ingested_at", _now())
        self.g.add_node(data["id"], **data)
        self._save()

    def supersede(self, new_id, old_id, reason, nli_score):
        self.g.add_edge(new_id, old_id, **{
            "source": new_id, "target": old_id, "type": "SUPERSEDES",
            "detected_at": _now(), "reason": reason,
            "nli_score": round(float(nli_score), 4),
        })
        if old_id in self.g.nodes:
            self.g.nodes[old_id]["status"] = STATUS_SUPERSEDED
            self.g.nodes[old_id]["freshness_score"] = 0.0
            self.g.nodes[old_id]["superseded_by"] = new_id
        self._save()

    def set_status(self, chunk_id, status):
        if chunk_id in self.g.nodes:
            self.g.nodes[chunk_id]["status"] = status
            self._save()

    def get(self, chunk_id):
        return dict(self.g.nodes[chunk_id]) if chunk_id in self.g.nodes else None

    def nodes(self):
        return [dict(d) for _, d in self.g.nodes(data=True)]

    def edges(self):
        return [dict(d) for *_, d in self.g.edges(data=True)]

    def lineage(self, chunk_id):
        """Walk the SUPERSEDES chain backwards: what did this rule replace?"""
        trail, cursor, seen = [], chunk_id, set()
        while cursor and cursor not in seen:
            seen.add(cursor)
            nxt = None
            for _, target, data in self.g.out_edges(cursor, data=True):
                if data.get("type") == "SUPERSEDES":
                    trail.append({"from": cursor, "to": target, **data})
                    nxt = target
                    break
            cursor = nxt
        return trail

    def reset(self):
        self.g.clear()
        if self.path.exists():
            self.path.unlink()


class Neo4jGraphStore(BaseGraphStore):
    backend = "neo4j"

    def __init__(self):
        from neo4j import GraphDatabase

        if not config.NEO4J_URI:
            raise RuntimeError("NEO4J_URI is not set")
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS "
                  "FOR (c:Chunk) REQUIRE c.id IS UNIQUE")

    def add_chunk(self, chunk_dict, status=STATUS_ACTIVE):
        # string lists are fine in neo4j and allowed_roles must survive, or
        # the UI would read a missing field as "visible to all"
        props = {k: v for k, v in chunk_dict.items()
                 if not isinstance(v, dict)
                 and not (isinstance(v, list) and not all(isinstance(x, str) for x in v))}
        props["status"] = status
        props["ingested_at"] = _now()
        with self.driver.session() as s:
            s.run("MERGE (c:Chunk {id: $id}) SET c += $props",
                  id=chunk_dict["id"], props=props)

    def supersede(self, new_id, old_id, reason, nli_score):
        with self.driver.session() as s:
            s.run(
                """
                MATCH (new:Chunk {id: $new_id})
                MATCH (old:Chunk {id: $old_id})
                SET old.status = 'SUPERSEDED',
                    old.freshness_score = 0.0,
                    old.superseded_by = $new_id
                MERGE (new)-[r:SUPERSEDES]->(old)
                SET r.detected_at = $ts, r.reason = $reason, r.nli_score = $score
                """,
                new_id=new_id, old_id=old_id, ts=_now(),
                reason=reason, score=float(nli_score),
            )

    def set_status(self, chunk_id, status):
        with self.driver.session() as s:
            s.run("MATCH (c:Chunk {id: $id}) SET c.status = $status",
                  id=chunk_id, status=status)

    def get(self, chunk_id):
        with self.driver.session() as s:
            rec = s.run("MATCH (c:Chunk {id: $id}) RETURN c", id=chunk_id).single()
            return dict(rec["c"]) if rec else None

    def nodes(self):
        with self.driver.session() as s:
            return [dict(r["c"]) for r in s.run("MATCH (c:Chunk) RETURN c")]

    def edges(self):
        with self.driver.session() as s:
            return [
                {"source": r["a"], "target": r["b"], "type": "SUPERSEDES", **dict(r["r"])}
                for r in s.run(
                    "MATCH (a:Chunk)-[r:SUPERSEDES]->(b:Chunk) "
                    "RETURN a.id AS a, b.id AS b, r"
                )
            ]

    def lineage(self, chunk_id):
        with self.driver.session() as s:
            return [
                {"from": r["a"], "to": r["b"], **dict(r["r"])}
                for r in s.run(
                    "MATCH p=(start:Chunk {id:$id})-[:SUPERSEDES*]->(old:Chunk) "
                    "UNWIND relationships(p) AS r "
                    "RETURN startNode(r).id AS a, endNode(r).id AS b, r",
                    id=chunk_id,
                )
            ]

    def reset(self):
        with self.driver.session() as s:
            s.run("MATCH (c:Chunk) DETACH DELETE c")


def get_graph_store() -> BaseGraphStore:
    backend = config.GRAPH_BACKEND
    if backend in ("auto", "neo4j") and config.NEO4J_URI:
        try:
            return Neo4jGraphStore()
        except Exception as exc:  # noqa: BLE001
            if backend == "neo4j":
                raise
            print(f"[graph] neo4j unavailable ({exc}); using networkx store.")
    return NetworkXGraphStore()
