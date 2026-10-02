"""Central configuration. Everything is env-overridable so the project runs
at $0 with zero signups by default, and upgrades to cloud backends by
setting a few environment variables."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STORAGE_DIR = Path(os.getenv("KB_STORAGE_DIR", ROOT / "storage"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- models
EMBED_MODEL = os.getenv("KB_EMBED_MODEL", "all-MiniLM-L6-v2")
NLI_MODEL = os.getenv("KB_NLI_MODEL", "cross-encoder/nli-deberta-v3-base")

# Force the offline/deterministic backends (used by CI and by the tests).
OFFLINE = os.getenv("KB_OFFLINE", "0") == "1"

# ---------------------------------------------------------------- chunking
CHUNK_MAX_TOKENS = int(os.getenv("KB_CHUNK_MAX_TOKENS", "220"))
CHUNK_MIN_TOKENS = int(os.getenv("KB_CHUNK_MIN_TOKENS", "5"))

# ---------------------------------------------------------------- retrieval
TOP_K_CANDIDATES = int(os.getenv("KB_TOP_K", "4"))
# Candidates below this cosine similarity are never even sent to the NLI model.
MIN_CANDIDATE_SIM = float(os.getenv("KB_MIN_CANDIDATE_SIM", "0.35"))

# ---------------------------------------------------------------- routing gate
# >= AUTO  -> supersede automatically
# >= HITL  -> send to the human review queue
# <  HITL  -> treat as a normal, non-conflicting insert
#
# Thresholds are a property of the CLASSIFIER, not of the system. A probability
# of 0.88 from DeBERTa and a score of 0.88 from a keyword heuristic are not the
# same evidence. So each detector carries its own calibrated defaults (measured
# with scripts/evaluate.py) and these env vars only override them when set.
_auto = os.getenv("KB_AUTO_THRESHOLD")
_hitl = os.getenv("KB_HITL_THRESHOLD")
AUTO_THRESHOLD_OVERRIDE = float(_auto) if _auto else None
HITL_THRESHOLD_OVERRIDE = float(_hitl) if _hitl else None

# Kept for reference and for scripts that want a single default number.
AUTO_SUPERSEDE_THRESHOLD = AUTO_THRESHOLD_OVERRIDE or 0.88
HITL_THRESHOLD = HITL_THRESHOLD_OVERRIDE or 0.50

# ---------------------------------------------------------------- freshness
# Expressed as a half-life because "lambda = 0.0015" means nothing to a reader.
# 3 years: a policy nobody has touched in three years is worth half as much as
# a fresh one, but is still retrievable.
#
# Decay must NOT be the mechanism that hides an old rule. An unchanged clause
# from 2019 that was never contradicted is still in force, and gating it out on
# age alone silently deletes valid policy. Supersession is the hard gate
# (freshness forced to 0); decay only re-ranks. Hence the very low floor.
DECAY_HALF_LIFE_DAYS = float(os.getenv("KB_DECAY_HALF_LIFE_DAYS", "1095"))
DECAY_LAMBDA = float(os.getenv("KB_DECAY_LAMBDA", "0")) or (0.693147 / DECAY_HALF_LIFE_DAYS)
FRESHNESS_FLOOR = float(os.getenv("KB_FRESHNESS_FLOOR", "0.05"))

# ---------------------------------------------------------------- backends
VECTOR_BACKEND = os.getenv("KB_VECTOR_BACKEND", "auto")  # auto | qdrant | numpy
GRAPH_BACKEND = os.getenv("KB_GRAPH_BACKEND", "auto")  # auto | neo4j | networkx

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_LOCAL_PATH = str(STORAGE_DIR / "qdrant")
QDRANT_COLLECTION = os.getenv("KB_COLLECTION", "kb_chunks")

NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
GRAPH_JSON_PATH = STORAGE_DIR / "graph.json"
HITL_QUEUE_PATH = STORAGE_DIR / "hitl_queue.json"

# ---------------------------------------------------------------- LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("KB_GEMINI_MODEL", "gemini-3.6-flash")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("KB_GROQ_MODEL", "openai/gpt-oss-120b")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("KB_OPENAI_MODEL", "gpt-5-mini")

