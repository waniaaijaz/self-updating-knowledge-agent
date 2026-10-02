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
    def search(self, vector: np.ndarray, top_k: int, doc_id: str | None = None, exclude_id: str | None = None) -> list[SearchHit]: ...
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

    def search(self, vector, top_k, doc_id=None, exclude_id=None):
        if self.matrix is None or not self.ids:
            return []
        vector = np.asarray(vector, dtype=np.float32)
        norms = np.linalg.norm(self.matrix, axis=1) * np.linalg.norm(vector)
        norms[norms == 0] = 1e-9
        sims = (self.matrix @ vector) / norms

        hits = []
        for idx in np.argsort(-sims):
            cid = self.ids[idx]
            payload = self.payloads[cid]
            if doc_id is not None and payload["doc_id"] != doc_id:
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

    def search(self, vector, top_k, doc_id=None, exclude_id=None):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must, must_not = [], []
        if doc_id is not None:
            must.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
        if exclude_id is not None:
            # Exclude the literal chunk itself, not everything sharing its
            # version label — see NumpyVectorStore.search for why.
            must_not.append(FieldCondition(key="id", match=MatchValue(value=exclude_id)))
        flt = Filter(must=must or None, must_not=must_not or None) if (must or must_not) else None

        res = self.client.query_points(
            collection_name=self.collection,
            query=np.asarray(vector, dtype=np.float32).tolist(),
            limit=top_k,
            query_filter=flt,
            with_payload=True,
        ).points
        return [SearchHit(p.payload["id"], float(p.score), p.payload) for p in res]

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
