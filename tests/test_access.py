"""Unit tests for access metadata, the store-level filter and the audit log.

    KB_OFFLINE=1 python -m pytest tests/test_access.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["KB_OFFLINE"] = "1"
os.environ["KB_VECTOR_BACKEND"] = "numpy"
os.environ["KB_GRAPH_BACKEND"] = "networkx"
os.environ.setdefault("KB_STORAGE_DIR", tempfile.mkdtemp(prefix="kbtest-"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src import config  # noqa: E402
from src.access import UserContext, can_see, normalize_roles  # noqa: E402
from src.audit import AuditLog  # noqa: E402
from src.chunking import chunk_markdown, sha256  # noqa: E402
from src.embeddings import HashingEmbedder  # noqa: E402
from src.kb import KnowledgeBase  # noqa: E402
from src.vector_store import NumpyVectorStore, QdrantVectorStore  # noqa: E402

# .env may hold a real key; tests must never call an actual LLM
config.GEMINI_API_KEY = config.OPENAI_API_KEY = config.GROQ_API_KEY = ""

DOC = """# Handbook

## 1 General

### 1.1 Hours
Core hours are 10:00 to 16:00 on weekdays.

### 1.2 Salary Bands
<!-- roles: hr -->
Engineer salary bands range from 60,000 to 80,000 per year.

### 1.3 Hiring
<!-- roles: Manager , HR -->
Managers may hire up to 2 new staff per quarter.
"""


@pytest.fixture
def kb():
    instance = KnowledgeBase(offline=True)
    instance.reset()
    return instance


# ------------------------------------------------------------- metadata

def test_defaults_keep_the_old_chunk_format():
    for c in chunk_markdown("# H\n\n## 1 A\n\n### 1.1 B\nSome rule text that is long enough.\n",
                            "hr", "v1", "2024-01-01"):
        assert c.tenant_id == "default"
        assert c.allowed_roles == ["all"]
        assert c.content_hash == sha256(c.body)  # same hash as before
        assert c.id.startswith("hr::v1::")       # same id as before


def test_section_marker_sets_roles_and_is_stripped_from_text():
    chunks = {c.section_path: c for c in chunk_markdown(DOC, "hb", "v1", "2024-01-01",
                                                        tenant_id="acme")}
    assert chunks["1.1"].allowed_roles == ["all"]
    assert chunks["1.2"].allowed_roles == ["hr"]
    assert chunks["1.3"].allowed_roles == ["hr", "manager"]  # trimmed, lowercased
    for c in chunks.values():
        assert "<!--" not in c.text and "<!--" not in c.parent_text
        assert c.tenant_id == "acme"
        assert c.id.startswith("acme::hb::v1::")


def test_document_roles_apply_unless_a_section_overrides_them():
    chunks = {c.section_path: c for c in chunk_markdown(
        DOC, "hb", "v1", "2024-01-01", tenant_id="acme", allowed_roles=["manager", "hr"])}
    assert chunks["1.1"].allowed_roles == ["hr", "manager"]
    assert chunks["1.2"].allowed_roles == ["hr"]


def test_empty_roles_are_refused():
    with pytest.raises(ValueError):
        normalize_roles([])
    with pytest.raises(ValueError):
        normalize_roles(" , ")
    with pytest.raises(ValueError):
        chunk_markdown("# H\n## 1 A\n<!-- roles: -->\nSome rule text that is long enough.\n",
                       "hb", "v1", "2024-01-01")


def test_can_see_rules():
    hr_only = {"tenant_id": "acme", "allowed_roles": ["hr"]}
    legacy = {}  # stored before this feature existed
    assert can_see(hr_only, UserContext("acme", "hr"))
    assert can_see(hr_only, UserContext("acme", "HR"))
    assert not can_see(hr_only, UserContext("acme", "employee"))
    assert not can_see(hr_only, UserContext("globex", "hr"))
    assert can_see(legacy, UserContext("default", "employee"))
    assert not can_see(legacy, UserContext("acme", "employee"))
    assert can_see(hr_only, None)  # no user = old unfiltered behaviour


# ------------------------------------------------------ store-level filter

def _store(kind, tmp_path, monkeypatch, dim):
    if kind == "numpy":
        return NumpyVectorStore(path=tmp_path / "vectors.json")
    monkeypatch.setattr(config, "QDRANT_URL", "")
    monkeypatch.setattr(config, "QDRANT_LOCAL_PATH", str(tmp_path / "qdrant"))
    return QdrantVectorStore(dim)


def _add_legacy_point(store, emb, cid, text):
    """A payload with no tenant_id / allowed_roles, like old storage."""
    payload = {"id": cid, "text": text, "doc_id": "old", "version": "v1"}
    vec = emb.encode_one(text)
    if isinstance(store, NumpyVectorStore):
        store.ids.append(cid)
        store.payloads[cid] = payload
        store.matrix = vec[None, :] if store.matrix is None else np.vstack([store.matrix, vec[None, :]])
    else:
        from qdrant_client.models import PointStruct
        from src.vector_store import _point_uuid
        store.client.upsert(collection_name=store.collection,
                            points=[PointStruct(id=_point_uuid(cid), vector=vec.tolist(), payload=payload)])


@pytest.mark.parametrize("kind", ["numpy", "qdrant"])
def test_store_filter_on_its_own(kind, tmp_path, monkeypatch):
    """No query engine here, so no second check behind it: the store alone
    must return only what the user may see."""
    emb = HashingEmbedder()
    store = _store(kind, tmp_path, monkeypatch, emb.dim)
    chunks = (chunk_markdown(DOC, "hb", "v1", "2024-01-01", tenant_id="acme")
              + chunk_markdown(DOC, "hb", "v1", "2024-01-01", tenant_id="globex"))
    store.upsert(chunks, emb.encode([c.text for c in chunks]))
    _add_legacy_point(store, emb, "legacy::1", "Engineer salary bands from an old upload.")

    vec = emb.encode_one("engineer salary bands")
    everything = {h.id for h in store.search(vec, top_k=50)}
    assert len(everything) == len(chunks) + 1

    for tenant in ["acme", "globex", "default"]:
        for role in ["employee", "manager", "hr"]:
            got = {h.id for h in store.search(vec, top_k=50, tenant_id=tenant, role=role)}
            want = {c.id for c in chunks if can_see(c.to_dict(), UserContext(tenant, role))}
            if tenant == "default":
                want.add("legacy::1")
            assert got == want, (kind, tenant, role)

    # tenant only (what ingest uses for candidate lookup)
    got = {h.id for h in store.search(vec, top_k=50, tenant_id="acme")}
    assert got == {c.id for c in chunks if c.tenant_id == "acme"}

    # hidden chunks can't take top_k slots
    top1 = store.search(vec, top_k=1, tenant_id="acme", role="employee")
    assert len(top1) == 1 and top1[0].payload["allowed_roles"] == ["all"]

    # search_ids only carries ids
    assert set(store.search_ids(vec, 50)) == everything
    store.close()


# ------------------------------------------------------------- ingest

def test_same_clause_in_two_tenants_is_not_deduplicated(kb):
    doc = "# P\n\n## 1 A\n\n### 1.1 Hours\nCore hours are 10:00 to 16:00 on weekdays.\n"
    a = kb.ingest_markdown(doc, "policy", "v1", "2024-01-01", verbose=False, tenant_id="acme")
    b = kb.ingest_markdown(doc, "policy", "v1", "2024-01-01", verbose=False, tenant_id="globex")
    assert a["summary"]["INSERT"] == 1 and b["summary"]["INSERT"] == 1
    tenants = sorted(p["tenant_id"] for p in kb.vectors.all_payloads())
    assert tenants == ["acme", "globex"]  # two points, neither overwrote the other


def test_one_tenants_v2_never_supersedes_another_tenants_v1(kb):
    v1 = "# P\n\n## 1 A\n\n### 1.1 Remote\nStaff may work remotely up to 2 days per week.\n"
    v2 = "# P\n\n## 1 A\n\n### 1.1 Remote\nStaff may work remotely up to 4 days per week.\n"
    kb.ingest_markdown(v1, "policy", "v1", "2024-01-01", verbose=False, tenant_id="acme")
    res = kb.ingest_markdown(v2, "policy", "v2", "2026-01-01", verbose=False, tenant_id="globex")
    assert res["summary"]["SUPERSEDE"] == 0
    assert kb.stats()["supersedes_edges"] == 0
    # same tenant still supersedes as before
    res = kb.ingest_markdown(v2, "policy", "v2", "2026-01-01", verbose=False, tenant_id="acme")
    assert res["summary"]["SUPERSEDE"] == 1


def test_ingest_without_access_args_retrieves_like_before(kb):
    kb.ingest_markdown(DOC, "hb", "v1", "2024-01-01", verbose=False)
    used, _ = kb.query_engine.retrieve("engineer salary bands")
    # the markers still apply, but with no user nothing is filtered
    assert any(rc.breadcrumb.endswith("Salary Bands") for rc in used)
    res = kb.ask("engineer salary bands")
    assert res.blocked_ids == []


def test_review_queue_items_are_checked_per_chunk(kb, monkeypatch):
    monkeypatch.setattr(config, "AUTO_THRESHOLD_OVERRIDE", 0.99)
    monkeypatch.setattr(config, "HITL_THRESHOLD_OVERRIDE", 0.30)
    kb.ingest_markdown("# O\n\n## 1 D\n\n### 1.1 Pay\n<!-- roles: hr -->\nBonus is paid "
                       "on any weekday including Fridays.\n", "ops", "v1", "2024-01-01",
                       verbose=False, tenant_id="acme")
    kb.ingest_markdown("# O\n\n## 1 D\n\n### 1.1 Pay\n<!-- roles: hr -->\nBonus is "
                       "prohibited on Fridays and weekends.\n", "ops", "v2", "2026-01-01",
                       verbose=False, tenant_id="acme")
    item = kb.queue.open_items()[0]
    for role, ok in [("hr", True), ("manager", False), ("employee", False)]:
        user = UserContext("acme", role)
        assert kb.can_see_id(item["old_chunk_id"], user) is ok
        assert kb.can_see_id(item["new_chunk_id"], user) is ok
    assert not kb.can_see_id(item["old_chunk_id"], UserContext("globex", "hr"))


# --------------------------------------------------------------- audit

def test_audit_log_appends_ids_only_and_reset_clears_it(kb):
    kb.ingest_markdown(DOC, "hb", "v1", "2024-01-01", verbose=False, tenant_id="acme")
    kb.ask("engineer salary bands", user=UserContext("acme", "employee", "alice"))
    kb.ask("engineer salary bands", user=UserContext("acme", "hr", "bob"))
    kb.ask("core hours")  # no user, still logged

    lines = config.AUDIT_LOG_PATH.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    first, second, third = (json.loads(x) for x in lines)
    keys = {"timestamp", "user", "role", "tenant", "query", "retrieved",
            "filtered_permissions", "filtered_superseded", "filtered_decayed"}
    assert set(first) == keys
    assert (first["user"], first["role"], first["tenant"]) == ("alice", "employee", "acme")
    salary = next(p["id"] for p in kb.vectors.all_payloads() if p["section_path"] == "1.2")
    assert salary in first["filtered_permissions"] and salary not in first["retrieved"]
    assert salary in second["retrieved"]
    assert third["user"] is None and third["filtered_permissions"] == []

    # no chunk text in the log, just ids
    raw = config.AUDIT_LOG_PATH.read_text(encoding="utf-8")
    assert "60,000" not in raw

    assert [e["user"] for e in kb.audit.entries(tenant_id="acme")] == ["alice", "bob"]
    kb.reset()
    assert kb.audit.entries() == []
    assert not config.AUDIT_LOG_PATH.exists()


def test_audit_log_is_append_only_across_instances(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditLog(path).record(None, "q1", [], [], [])
    AuditLog(path).record(None, "q2", [], [], [])
    assert [e["query"] for e in AuditLog(path).entries()] == ["q1", "q2"]
