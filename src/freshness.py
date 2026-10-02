"""Temporal freshness decay.

    S_freshness = S_base * exp(-lambda * delta_t_days)

and the final ranking score fuses it with cosine similarity:

    S_final = S_vector * S_freshness

Two rules make this behave sensibly:
  * a node explicitly marked SUPERSEDED is forced to 0.0 — no amount of
    semantic similarity should resurrect a rule that was overwritten;
  * decay is computed from the document's own timestamp, not from ingest
    time, so back-filling an old document does not make it look fresh.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from . import config

STATUS_ACTIVE = "ACTIVE"
STATUS_SUPERSEDED = "SUPERSEDED"
STATUS_PENDING_REVIEW = "PENDING_REVIEW"


def parse_timestamp(value) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            dt = datetime.strptime(text[:10], "%Y-%m-%d")
    return dt.replace(tzinfo=dt.tzinfo or timezone.utc)


def age_in_days(created_at, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    delta = now - parse_timestamp(created_at)
    return max(delta.total_seconds() / 86400.0, 0.0)


def calculate_freshness(
    base_score: float,
    created_at,
    decay_lambda: float | None = None,
    status: str = STATUS_ACTIVE,
    now: datetime | None = None,
) -> float:
    if status == STATUS_SUPERSEDED:
        return 0.0
    lam = config.DECAY_LAMBDA if decay_lambda is None else decay_lambda
    score = base_score * math.exp(-lam * age_in_days(created_at, now))
    return round(min(max(score, 0.0), 1.0), 4)


def half_life_days(decay_lambda: float | None = None) -> float:
    lam = config.DECAY_LAMBDA if decay_lambda is None else decay_lambda
    return math.log(2) / lam if lam else float("inf")


def final_score(vector_similarity: float, freshness: float) -> float:
    return round(vector_similarity * freshness, 4)


def passes_gate(freshness: float, floor: float | None = None) -> bool:
    return freshness >= (config.FRESHNESS_FLOOR if floor is None else floor)
