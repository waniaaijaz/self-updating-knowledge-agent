"""Human-in-the-loop review queue.

Production systems do not delete knowledge on the strength of a probabilistic
classifier. Anything in the ambiguous confidence band lands here and the
chunk stays PENDING_REVIEW — retrievable, but marked, and never silently
used to overwrite the existing rule.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from . import config
from .freshness import STATUS_ACTIVE, STATUS_SUPERSEDED, STATUS_PENDING_REVIEW


class ReviewQueue:
    def __init__(self, path=None):
        self.path = path or config.HITL_QUEUE_PATH
        self.items: list[dict] = []
        if self.path.exists():
            self.items = json.loads(self.path.read_text())

    def _save(self):
        self.path.write_text(json.dumps(self.items, indent=2))

    def add(self, new_chunk: dict, old_payload: dict, nli: dict, similarity: float):
        self.items.append({
            "review_id": f"rev-{len(self.items) + 1:04d}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "OPEN",
            "new_chunk_id": new_chunk["id"],
            "new_text": new_chunk["body"],
            "old_chunk_id": old_payload["id"],
            "old_text": old_payload.get("body", old_payload.get("text", "")),
            "section_path": new_chunk.get("section_path"),
            "nli": nli,
            "similarity": round(similarity, 4),
        })
        self._save()
        return self.items[-1]

    def open_items(self):
        return [i for i in self.items if i["status"] == "OPEN"]

    def resolve(self, review_id: str, approved: bool, graph_store, vector_store, reviewer="admin"):
        """Approving means: yes, this really is a contradiction — apply the
        supersede edge that the automated gate declined to apply on its own."""
        for item in self.items:
            if item["review_id"] != review_id:
                continue
            item["status"] = "APPROVED" if approved else "REJECTED"
            item["resolved_at"] = datetime.now(timezone.utc).isoformat()
            item["reviewer"] = reviewer

            if approved:
                graph_store.supersede(
                    item["new_chunk_id"], item["old_chunk_id"],
                    reason=f"HITL_APPROVED:{reviewer}",
                    nli_score=item["nli"]["contradiction_score"],
                )
                vector_store.set_payload(item["old_chunk_id"],
                                         {"status": STATUS_SUPERSEDED})
            graph_store.set_status(item["new_chunk_id"], STATUS_ACTIVE)
            vector_store.set_payload(item["new_chunk_id"], {"status": STATUS_ACTIVE})
            self._save()
            return item
        raise KeyError(review_id)

    def reset(self):
        self.items = []
        if self.path.exists():
            self.path.unlink()
