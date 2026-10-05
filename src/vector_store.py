"""Vector storage.

Qdrant is the default. Note that `QdrantClient(path=...)` runs Qdrant fully
embedded on local disk — same client, same API, no server and no cloud
signup. Setting QDRANT_URL switches the identical code to a free cloud
cluster, which is what makes this deployable without a rewrite.

The numpy backend is a dependency-free fallback so the pipeline still runs
if qdrant-client is not installed.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

import numpy as np

from . import config
from .access import ALL_ROLES, DEFAULT_TENANT, chunk_roles, chunk_tenant
from .chunking import Chunk


@dataclass
class SearchHit:
    id: str
    score: float
    payload: dict


def _point_uuid(chunk_id: str) -> str:
    """Qdrant point IDs must be UUIDs or ints, but our chunk IDs are readable
    strings. Derive a stable UUID5 so ingesting twice updates in place."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class BaseVectorStore:
    backend = "base"

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    # tenant_id / role are the permission filter. They are applied inside the
    # search, so hidden chunks never come back from the store at all.
    def search(self, vector: np.ndarray, top_k: int, doc_id: str | None = None, exclude_id: str | None = None,
               tenant_id: str | None = None, role: str | None = None) -> list[SearchHit]: ...
    # ids only (no text), for the audit log's "blocked by permissions" list
    def search_ids(self, vector: np.ndarray, top_k: int) -> list[str]: ...
    def set_payload(self, chunk_id: str, updates: dict) -> None: ...
    def all_payloads(self) -> list[dict]: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...


class NumpyVectorStore(BaseVectorStore):
    backend = "numpy"

    def __init__(self, path=None):
        self.path = path or (config.STORAGE_DIR / "vectors.json")
        self.ids: list[str] = []
        self.payloads: dict[str, dict] = {}
        self.matrix: np.ndarray | None = None
        self._load()

    def _load(self):
        if self.path and self.path.exists():
            blob = json.loads(self.path.read_text())
            self.ids = blob["ids"]
            self.payloads = blob["payloads"]
            self.matrix = np.asarray(blob["vectors"], dtype=np.float32) if blob["vectors"] else None

    def _save(self):
        self.path.write_text(json.dumps({
            "ids": self.ids,
            "payloads": self.payloads,
            "vectors": [] if self.matrix is None else self.matrix.tolist(),
        }))

    def upsert(self, chunks, vectors):
        vectors = np.asarray(vectors, dtype=np.float32)
        for chunk, vec in zip(chunks, vectors):
            payload = chunk.to_dict()
            if chunk.id in self.payloads:
                idx = self.ids.index(chunk.id)
                self.matrix[idx] = vec
                self.payloads[chunk.id].update(payload)
                continue
            self.ids.append(chunk.id)
            self.payloads[chunk.id] = payload
            self.matrix = vec[None, :] if self.matrix is None else np.vstack([self.matrix, vec[None, :]])
        self._save()

    def _ranked(self, vector):
        vector = np.asarray(vector, dtype=np.float32)
        norms = np.linalg.norm(self.matrix, axis=1) * np.linalg.norm(vector)
        norms[norms == 0] = 1e-9
        sims = (self.matrix @ vector) / norms
        return sims, np.argsort(-sims)

    def search(self, vector, top_k, doc_id=None, exclude_id=None, tenant_id=None, role=None):
        if self.matrix is None or not self.ids:
            return []
        sims, order = self._ranked(vector)

        hits = []
        for idx in order:
            cid = self.ids[idx]
            payload = self.payloads[cid]
            if doc_id is not None and payload["doc_id"] != doc_id:
                continue
            # checked before the hit is appended, so a hidden chunk can't
            # take a top_k slot or reach the caller
            if tenant_id is not None and chunk_tenant(payload) != tenant_id:
                continue
            if role is not None:
                roles = chunk_roles(payload)
                if ALL_ROLES not in roles and role.lower() not in roles:
                    continue
            # Exclude the literal chunk being compared against itself, not
            # every chunk that happens to share its version label. Two
            # different document revisions can legitimately carry the same
            # version string (a mislabel, a shared "v1" across sub-docs,
            # etc.) — excluding by version would hide real contradictions
            # between them, which is exactly the bug this used to have.
            if exclude_id is not None and cid == exclude_id:
                continue
            hits.append(SearchHit(cid, float(sims[idx]), payload))
            if len(hits) >= top_k:
                break
        return hits

    def search_ids(self, vector, top_k):
        if self.matrix is None or not self.ids:
            return []
        _, order = self._ranked(vector)
        return [self.ids[i] for i in order[:top_k]]

    def set_payload(self, chunk_id, updates):
        if chunk_id in self.payloads:
            self.payloads[chunk_id].update(updates)
            self._save()

    def all_payloads(self):
        return list(self.payloads.values())

    def reset(self):
        self.ids, self.payloads, self.matrix = [], {}, None
        if self.path.exists():
            self.path.unlink()


class QdrantVectorStore(BaseVectorStore):
    backend = "qdrant"

    def __init__(self, dim: int):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.dim = dim
        if config.QDRANT_URL:
            self.client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY or None)
            self.backend = "qdrant:cloud"
        else:
            self.client = QdrantClient(path=config.QDRANT_LOCAL_PATH)
            self.backend = "qdrant:local"

        self.collection = config.QDRANT_COLLECTION
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, chunks, vectors):
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=_point_uuid(c.id), vector=v.tolist(), payload=c.to_dict())
            for c, v in zip(chunks, np.asarray(vectors, dtype=np.float32))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector, top_k, doc_id=None, exclude_id=None, tenant_id=None, role=None):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must, must_not = [], []
        if doc_id is not None:
            must.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
        if exclude_id is not None:
            # Exclude the literal chunk itself, not everything sharing its
            # version label — see NumpyVectorStore.search for why.
            must_not.append(FieldCondition(key="id", match=MatchValue(value=exclude_id)))
        must.extend(_access_conditions(tenant_id, role))
        flt = Filter(must=must or None, must_not=must_not or None) if (must or must_not) else None

        res = self.client.query_points(
            collection_name=self.collection,
            query=np.asarray(vector, dtype=np.float32).tolist(),
            limit=top_k,
            query_filter=flt,
            with_payload=True,
        ).points
        return [SearchHit(p.payload["id"], float(p.score), p.payload) for p in res]

    def search_ids(self, vector, top_k):
        res = self.client.query_points(
            collection_name=self.collection,
            query=np.asarray(vector, dtype=np.float32).tolist(),
            limit=top_k,
            with_payload=["id"],
        ).points
        return [p.payload["id"] for p in res]

    def set_payload(self, chunk_id, updates):
        self.client.set_payload(
            collection_name=self.collection,
            payload=updates,
            points=[_point_uuid(chunk_id)],
        )

    def all_payloads(self):
        out, offset = [], None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection, limit=256,
                offset=offset, with_payload=True,
            )
            out.extend(p.payload for p in points)
            if offset is None:
                break
        return out

    def reset(self):
        from qdrant_client.models import PointIdsList
        # Delete the points rather than drop/recreate the collection: on
        # Windows the local backend can't remove the locked sqlite file, so
        # recreate silently reopens the old data and reset becomes a no-op.
        while True:
            points, _ = self.client.scroll(
                collection_name=self.collection, limit=256, with_payload=False,
            )
            if not points:
                break
            self.client.delete(
                collection_name=self.collection,
                points_selector=PointIdsList(points=[p.id for p in points]),
            )

    def close(self):
        if hasattr(self, "client") and self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass


def _access_conditions(tenant_id, role) -> list:
    """Qdrant version of the tenant/role check in NumpyVectorStore.search."""
    from qdrant_client.models import (
        FieldCondition, Filter, IsEmptyCondition, MatchAny, MatchValue, PayloadField,
    )

    out = []
    if tenant_id is not None:
        same_tenant = FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))
        if tenant_id == DEFAULT_TENANT:
            # chunks stored before tenants existed have no tenant_id at all
            out.append(Filter(should=[same_tenant,
                                      IsEmptyCondition(is_empty=PayloadField(key="tenant_id"))]))
        else:
            out.append(same_tenant)
    if role is not None:
        # MatchAny on a list field = "any element of allowed_roles is in here".
        # is_empty covers old chunks with no allowed_roles (= visible to all);
        # an explicit [] can't be stored, normalize_roles refuses it.
        out.append(Filter(should=[
            FieldCondition(key="allowed_roles", match=MatchAny(any=[role.lower(), ALL_ROLES])),
            IsEmptyCondition(is_empty=PayloadField(key="allowed_roles")),
        ]))
    return out


def get_vector_store(dim: int) -> BaseVectorStore:
    backend = config.VECTOR_BACKEND
    if backend in ("auto", "qdrant"):
        try:
            return QdrantVectorStore(dim)
        except Exception as exc:  # noqa: BLE001
            if backend == "qdrant":
                raise
            print(f"[vector] qdrant unavailable ({exc}); using numpy store.")
    return NumpyVectorStore()
