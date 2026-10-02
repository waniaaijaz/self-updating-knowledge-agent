"""Test suite. Runs fully offline: no model downloads, no network, no servers.

    KB_OFFLINE=1 python -m pytest tests/ -v
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ["KB_OFFLINE"] = "1"
os.environ["KB_VECTOR_BACKEND"] = "numpy"
os.environ["KB_GRAPH_BACKEND"] = "networkx"
os.environ["KB_STORAGE_DIR"] = tempfile.mkdtemp(prefix="kbtest-")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from src import config  # noqa: E402
from src.chunking import chunk_markdown, estimate_tokens, sha256  # noqa: E402
from src.freshness import (  # noqa: E402
    STATUS_SUPERSEDED,
    calculate_freshness,
    final_score,
    passes_gate,
)
from src.kb import KnowledgeBase  # noqa: E402
from src.nli import CONTRADICTION, HeuristicDetector  # noqa: E402

SAMPLE = """# Handbook

## 3 Allowances

### 3.1 Stipend
Remote employees receive an annual home office stipend of $500.

### 3.2 Internet
Reimbursement is capped at $40 per month.
"""


# ---------------------------------------------------------------- chunking

def test_chunker_builds_breadcrumbs_and_section_paths():
    chunks = chunk_markdown(SAMPLE, "hb", "v1", "2024-01-01")
    assert len(chunks) == 2
    stipend = next(c for c in chunks if c.section_path == "3.1")
    assert stipend.breadcrumb == "Handbook > 3 Allowances > 3.1 Stipend"
    assert stipend.text.startswith(stipend.breadcrumb)
    assert "$500" in stipend.body
    # parent_id is built from the FULL header path, not just the leaf's own
    # section number/slug — see test_sibling_sections_with_the_same_heading_
    # dont_collide below for why that distinction matters.
    assert stipend.parent_id == "hb::v1::handbook/3-allowances/3-1-stipend"


def test_sibling_sections_with_the_same_heading_dont_collide():
    """Regression test: two different chapters each having their own
    "## Overview" subsection used to be assigned the same chunk/parent ID
    (derived from the leaf heading only), so the second one silently
    overwrote the first in the vector and graph stores. Full ancestor path
    must be part of the ID."""
    doc = (
        "# Part A: Hiring\n## Overview\nHiring is done through referrals.\n\n"
        "# Part B: Onboarding\n## Overview\nOnboarding takes two weeks.\n"
    )
    chunks = chunk_markdown(doc, "handbook", "v1", "2024-01-01")
    assert len(chunks) == 2
    ids = {c.id for c in chunks}
    assert len(ids) == 2, "sibling sections collided onto the same chunk id"
    bodies = {c.parent_id: c.body for c in chunks}
    assert any("referrals" in b for b in bodies.values())
    assert any("two weeks" in b for b in bodies.values())


def test_chunks_stay_under_cross_encoder_token_limit():
    chunks = chunk_markdown(SAMPLE * 12, "hb", "v1", "2024-01-01")
    assert all(c.token_estimate < 512 for c in chunks)


def test_hash_ignores_whitespace_and_case():
    assert sha256("Stipend is $500") == sha256("  stipend   IS $500  ")
    assert sha256("Stipend is $500") != sha256("Stipend is $600")


def test_long_section_splits_on_sentence_boundaries():
    body = "# D\n\n## 1 S\n\n" + " ".join(f"Rule number {i} applies." for i in range(200))
    chunks = chunk_markdown(body, "d", "v1", "2024-01-01", max_tokens=60)
    assert len(chunks) > 1
    assert all(c.body.strip().endswith(".") for c in chunks)


# --------------------------------------------------------------- freshness

def test_freshness_decays_with_age():
    now = datetime.now(timezone.utc)
    fresh = calculate_freshness(1.0, now - timedelta(days=1))
    old = calculate_freshness(1.0, now - timedelta(days=900))
    assert fresh > old
    assert 0.0 <= old <= 1.0


def test_superseded_is_forced_to_zero_regardless_of_age():
    now = datetime.now(timezone.utc)
    assert calculate_freshness(1.0, now, status=STATUS_SUPERSEDED) == 0.0


def test_gate_and_final_score():
    assert passes_gate(0.9)
    assert not passes_gate(0.0)
    assert final_score(0.8, 0.5) == 0.4


def test_old_but_uncontradicted_content_survives_the_gate():
    """Decay re-ranks; it must not silently delete a rule that is still in
    force. Only an explicit SUPERSEDED status does that."""
    ancient = datetime.now(timezone.utc) - timedelta(days=365 * 5)
    score = calculate_freshness(1.0, ancient)
    assert score < 0.5, "five-year-old content should be down-weighted"
    assert passes_gate(score), "but it must still be retrievable"


# --------------------------------------------------------------------- nli

@pytest.fixture
def detector():
    return HeuristicDetector()


def test_detects_polarity_flip(detector):
    r = detector.evaluate_bidirectional(
        "Remote employees receive an annual home office stipend of $500.",
        "The annual home office stipend is suspended effective Q1 2026.",
    )
    assert r.label == CONTRADICTION
    assert r.contradiction_score > 0.5


def test_detects_numeric_conflict(detector):
    r = detector.evaluate_bidirectional(
        "The minimum required unit test coverage for merging into main is 70%.",
        "The minimum required unit test coverage for merging into main is 85%.",
    )
    assert r.label == CONTRADICTION


def test_unrelated_topics_are_not_contradictions(detector):
    r = detector.evaluate_bidirectional(
        "Monitors and keyboards are ordered through the IT portal.",
        "Restore drills are conducted once per quarter.",
    )
    assert r.label != CONTRADICTION


def test_bidirectional_takes_the_stronger_signal(detector):
    a = "Deployments on Friday are permitted."
    b = "Deployments on Friday are prohibited."
    both = detector.evaluate_bidirectional(a, b)
    assert both.contradiction_score >= detector.evaluate(a, b).contradiction_score


# ------------------------------------------------------------------- e2e

@pytest.fixture
def kb():
    instance = KnowledgeBase(offline=True)
    instance.reset()
    return instance


def test_end_to_end_v2_supersedes_v1(kb):
    v1 = """# HR

## 3 Allowances

### 3.1 Stipend
Remote employees receive an annual home office stipend of $500 each January.
"""
    v2 = """# HR

## 3 Allowances

### 3.1 Stipend
The annual home office stipend is suspended and no payments will be issued.
"""
    kb.ingest_markdown(v1, "hr", "v1", "2024-01-01", verbose=False)
    kb.ingest_markdown(v2, "hr", "v2", "2026-01-01", verbose=False)

    edges = kb.graph.edges()
    assert len(edges) == 1, "expected exactly one SUPERSEDES edge"
    edge = edges[0]
    assert "v2" in edge["source"] and "v1" in edge["target"]
    assert edge["type"] == "SUPERSEDES"
    assert "nli_score" in edge and "detected_at" in edge

    old = kb.graph.get(edge["target"])
    assert old["status"] == STATUS_SUPERSEDED


def test_superseded_chunk_never_reaches_the_answer(kb):
    kb.ingest_markdown(
        "# HR\n\n## 3 A\n\n### 3.1 Stipend\nRemote employees receive an annual "
        "home office stipend of $500 each January.\n",
        "hr", "v1", "2024-01-01", verbose=False)
    kb.ingest_markdown(
        "# HR\n\n## 3 A\n\n### 3.1 Stipend\nThe annual home office stipend is "
        "suspended and no payments will be issued.\n",
        "hr", "v2", "2026-01-01", verbose=False)

    used, dropped = kb.query_engine.retrieve("what is the home office stipend?")
    used_versions = {c.version for c in used}
    assert "v1" not in used_versions
    assert any(c.version == "v1" for c in dropped)
    assert all(c.freshness == 0.0 for c in dropped if c.status == STATUS_SUPERSEDED)


def test_identical_content_is_deduplicated_not_reprocessed(kb):
    doc = ("# HR\n\n## 1 Scope\n\n### 1.1 Coverage\nThis handbook applies to all "
           "full-time and part-time employees of the company.\n")
    kb.ingest_markdown(doc, "hr", "v1", "2024-01-01", verbose=False)
    result = kb.ingest_markdown(doc, "hr", "v2", "2026-01-01", verbose=False)

    assert result["summary"]["DUPLICATE"] == 1
    assert result["summary"]["SUPERSEDE"] == 0
    assert kb.stats()["supersedes_edges"] == 0


def test_force_reingest_bypasses_the_dedup_gate(kb):
    doc = ("# HR\n\n## 1 Scope\n\n### 1.1 Coverage\nThis handbook applies to all "
           "full-time and part-time employees of the company.\n")
    kb.ingest_markdown(doc, "hr", "v1", "2024-01-01", verbose=False)

    # same text, same doc_id/version re-ingested without force -> skipped
    result = kb.ingest_markdown(doc, "hr", "v1", "2024-01-01", verbose=False)
    assert result["summary"]["DUPLICATE"] == 1

    # with force_reingest the hash check is bypassed and it goes through
    # the pipeline again (re-inserted, not re-superseded, since the text
    # didn't actually change)
    result = kb.ingest_markdown(
        doc, "hr", "v1", "2024-01-01", verbose=False, force_reingest=True
    )
    assert result["summary"]["DUPLICATE"] == 0
    assert result["summary"]["INSERT"] == 1


def test_v2_edit_still_supersedes_after_a_forced_v1_reingest(kb):
    """The scenario from the bug report: someone re-runs v1 (e.g. while
    testing) with force_reingest, then ingests an edited v2 on top. The
    changed clause must still produce exactly one SUPERSEDES edge."""
    v1 = ("# HR\n\n## 3 Allowances\n\n### 3.1 Stipend\nRemote employees receive "
          "an annual home office stipend of $500.\n")
    v2 = ("# HR\n\n## 3 Allowances\n\n### 3.1 Stipend\nThe annual home office "
          "stipend is suspended effective Q1 2026.\n")

    kb.ingest_markdown(v1, "hr", "v1", "2024-01-01", verbose=False)
    kb.ingest_markdown(v1, "hr", "v1", "2024-01-01", verbose=False, force_reingest=True)
    result = kb.ingest_markdown(v2, "hr", "v2", "2026-01-01", verbose=False)

    assert result["summary"]["SUPERSEDE"] == 1
    assert kb.stats()["supersedes_edges"] == 1


def test_additive_section_inserts_without_superseding_anything(kb):
    kb.ingest_markdown(
        "# HR\n\n## 2 Work\n\n### 2.1 Hours\nCore hours are 11:00 to 16:00 local time.\n",
        "hr", "v1", "2024-01-01", verbose=False)
    result = kb.ingest_markdown(
        "# HR\n\n## 6 Data\n\n### 6.1 Devices\nStoring customer data on personal "
        "devices is prohibited under all circumstances.\n",
        "hr", "v2", "2026-01-01", verbose=False)

    assert result["summary"]["SUPERSEDE"] == 0
    assert kb.stats()["supersedes_edges"] == 0
    assert kb.stats()["active"] == 2


def test_hitl_leaves_the_old_rule_active_until_a_human_approves(kb, monkeypatch):
    # Force everything into the ambiguous band so the review path is exercised.
    monkeypatch.setattr(config, "AUTO_THRESHOLD_OVERRIDE", 0.99)
    monkeypatch.setattr(config, "HITL_THRESHOLD_OVERRIDE", 0.30)

    kb.ingest_markdown(
        "# Ops\n\n## 1 Deploy\n\n### 1.1 Window\nDeployments to production are "
        "permitted on any weekday including Fridays.\n",
        "ops", "v1", "2024-01-01", verbose=False)
    kb.ingest_markdown(
        "# Ops\n\n## 1 Deploy\n\n### 1.1 Window\nDeployments to production are "
        "prohibited on Fridays and weekends.\n",
        "ops", "v2", "2026-01-01", verbose=False)

    open_items = kb.queue.open_items()
    assert len(open_items) == 1
    assert kb.stats()["supersedes_edges"] == 0, "must not act before human approval"

    old_id = open_items[0]["old_chunk_id"]
    assert kb.graph.get(old_id)["status"] == "ACTIVE"

    kb.queue.resolve(open_items[0]["review_id"], approved=True,
                     graph_store=kb.graph, vector_store=kb.vectors)

    assert kb.stats()["supersedes_edges"] == 1
    assert kb.graph.get(old_id)["status"] == STATUS_SUPERSEDED
    assert kb.queue.open_items() == []


def test_lineage_trail_is_walkable(kb):
    for version, ts, amount in [("v1", "2022-01-01", "$300"),
                                ("v2", "2024-01-01", "$500"),
                                ("v3", "2026-01-01", "$900")]:
        kb.ingest_markdown(
            f"# HR\n\n## 3 A\n\n### 3.1 Stipend\nThe annual home office stipend "
            f"is {amount} paid each January.\n",
            "hr", version, ts, verbose=False)

    newest = next(n["id"] for n in kb.graph.nodes()
                  if n["version"] == "v3" and n["status"] == "ACTIVE")
    trail = kb.graph.lineage(newest)
    assert len(trail) >= 1
    assert all("detected_at" in step for step in trail)
