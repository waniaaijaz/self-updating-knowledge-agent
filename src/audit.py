"""Append-only audit log of queries, one JSON object per line.

Only ids go in here, never chunk text, so the log itself doesn't become a
second copy of restricted content.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from . import config


class AuditLog:
    def __init__(self, path=None):
        self.path = path or config.AUDIT_LOG_PATH

    def record(self, user, query, retrieved, blocked, superseded, decayed=()):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": user.user if user else None,
            "role": user.role if user else None,
            "tenant": user.tenant_id if user else None,
            "query": query,
            "retrieved": list(retrieved),
            "filtered_permissions": list(blocked),
            "filtered_superseded": list(superseded),
            # dropped by the freshness floor, not superseded. rare, but
            # leaving them out would make the log look incomplete
            "filtered_decayed": list(decayed),
        }
        # "a" mode only, we never rewrite earlier lines
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        return entry

    def entries(self, tenant_id: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if tenant_id is None or entry.get("tenant") == tenant_id:
                out.append(entry)
        return out

    def reset(self):
        if self.path.exists():
            self.path.unlink()
