"""Embedding backends.

Default is a local sentence-transformers model (all-MiniLM-L6-v2, ~80 MB,
runs fine on CPU, costs nothing). The hashing fallback exists so the test
suite and CI can run with no model download and no network.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from . import config


class BaseEmbedder:
    dim: int = 0
    name: str = "base"

    # Minimum cosine similarity for a prior chunk to even be considered a
    # candidate for contradiction scoring. Embedder-specific because
    # different backends produce very different similarity distributions
    # for the same pair of sentences — a single global threshold tuned for
    # one backend can silently exclude real matches on another (see
    # HashingEmbedder below).
    min_candidate_sim: float = config.MIN_CANDIDATE_SIM

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.name = model_name or config.EMBED_MODEL
        self.model = SentenceTransformer(self.name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


class HashingEmbedder(BaseEmbedder):
    """Deterministic bag-of-ngrams embedder. No download, no network.

    Not competitive with a real transformer, but it is a genuine lexical
    similarity signal, which is enough to exercise the whole pipeline offline.
    """

    name = "hashing-ngram"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        feats = list(words)
        feats += [f"{a}_{b}" for a, b in zip(words, words[1:])]
        return feats

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = int.from_bytes(
                    hashlib.md5(feat.encode()).digest()[:8], "little", signed=False
                )
                out[row, h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


_CACHE: dict[bool, BaseEmbedder] = {}


def get_embedder(force_offline: bool | None = None) -> BaseEmbedder:
    """Cached per offline/online mode, not globally.

    A single process can legitimately need both: the Streamlit sidebar lets
    the user flip between "Fast / Offline" and "Transformer" mode without
    restarting, and a single pytest session may exercise both too. A single
    module-level singleton silently served whichever backend loaded first
    for every request after that — flipping the UI toggle to "Transformer"
    did nothing if the app had already initialized in Fast mode. Keying the
    cache by the requested mode keeps each backend loaded at most once
    while letting both coexist.
    """
    offline = config.OFFLINE if force_offline is None else force_offline
    if offline in _CACHE:
        return _CACHE[offline]

    if not offline:
        try:
            embedder = SentenceTransformerEmbedder()
            _CACHE[offline] = embedder
            return embedder
        except Exception as exc:  # noqa: BLE001
            print(f"[embeddings] sentence-transformers unavailable ({exc}); "
                  f"falling back to hashing embedder.")

    embedder = HashingEmbedder()
    _CACHE[offline] = embedder
    return embedder


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)
