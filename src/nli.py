"""Contradiction detection via Natural Language Inference.

Three gotchas I ran into while building this, noting them here so I don't
forget why the code looks the way it does:

- Label order isn't fixed across checkpoints. A lot of example code
  hardcodes ["contradiction", "entailment", "neutral"], which happens to
  match nli-deberta-v3-base but not every NLI model. Reading
  `model.config.id2label` instead means swapping checkpoints doesn't
  silently flip contradiction and entailment.
- CrossEncoder.predict returns raw logits, not probabilities. Taking
  argmax on the raw numbers and treating that as a confidence score gives
  values like 4.7, which blow past a 0.88 threshold and fire on everything.
  Needs an explicit softmax first (took me a while to catch this one).
- Entailment is directional (entails(a, b) != entails(b, a)), and
  contradiction is only roughly symmetric in practice, so both orderings
  get scored and the stronger one wins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from . import config

CONTRADICTION = "CONTRADICTION"
ENTAILMENT = "ENTAILMENT"
NEUTRAL = "NEUTRAL"


@dataclass
class NLIResult:
    label: str
    confidence: float  # probability of the winning label
    contradiction_score: float  # probability of CONTRADICTION specifically
    backend: str

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "contradiction_score": round(self.contradiction_score, 4),
            "backend": self.backend,
        }


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


class BaseDetector:
    backend = "base"
    # Calibrated per backend — see scripts/evaluate.py --sweep.
    auto_threshold = 0.88
    hitl_threshold = 0.50

    def evaluate(self, premise: str, hypothesis: str) -> NLIResult:
        raise NotImplementedError

    def evaluate_bidirectional(self, a: str, b: str) -> NLIResult:
        forward = self.evaluate(a, b)
        backward = self.evaluate(b, a)
        return max(forward, backward, key=lambda r: r.contradiction_score)


class CrossEncoderDetector(BaseDetector):
    backend = "cross-encoder"
    # DeBERTa softmax probabilities on a clear rule reversal sit above 0.95,
    # so a high bar here costs almost no recall and protects against the one
    # failure mode that actually hurts: deleting a rule that is still in force.
    auto_threshold = 0.88
    hitl_threshold = 0.50

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or config.NLI_MODEL
        self.model = CrossEncoder(self.model_name)
        self.backend = f"cross-encoder:{self.model_name}"

        id2label = getattr(self.model.model.config, "id2label", None) or {}
        self.labels = [
            str(id2label.get(i, f"LABEL_{i}")).upper()
            for i in range(len(id2label) or 3)
        ]
        if CONTRADICTION not in self.labels:
            # Some checkpoints use LABEL_0/1/2. Fall back to the MNLI default.
            self.labels = [CONTRADICTION, ENTAILMENT, NEUTRAL]
        self.contra_idx = self.labels.index(CONTRADICTION)

    def evaluate(self, premise: str, hypothesis: str) -> NLIResult:
        logits = np.asarray(self.model.predict([(premise, hypothesis)])[0], dtype=float)
        probs = _softmax(logits)
        best = int(probs.argmax())
        return NLIResult(
            label=self.labels[best],
            confidence=float(probs[best]),
            contradiction_score=float(probs[self.contra_idx]),
            backend=self.backend,
        )


# --------------------------------------------------------------------------
# Offline fallback
# --------------------------------------------------------------------------

NEGATORS = {
    "no", "not", "never", "cannot", "cant", "won't", "wont", "prohibited",
    "forbidden", "banned", "suspended", "discontinued", "revoked", "disallowed",
    "removed", "ineligible", "denied", "blocked", "restricted",
}
REVOCATION = {
    "suspended", "discontinued", "revoked", "prohibited", "forbidden",
    "banned", "withdrawn", "rescinded", "cancelled", "canceled",
}
AFFIRMERS = {
    "allowed", "permitted", "eligible", "granted", "approved", "available",
    "entitled", "may", "can", "will", "provided", "offered",
}
STOPWORDS = {
    "the", "a", "an", "of", "to", "is", "are", "be", "and", "or", "for", "in",
    "on", "at", "as", "by", "with", "from", "that", "this", "it", "per", "all",
    "any", "must", "shall", "should", "each", "policy", "employees", "employee",
}
NUM_RE = re.compile(r"\$?\d+(?:[.,]\d+)?%?")


def _tokens(text: str) -> list[str]:
    # The "." must only survive inside a decimal (2.5), never as sentence-final
    # punctuation — otherwise "forbidden." never matches the "forbidden"
    # keyword and every rule at the end of a sentence is silently invisible.
    return re.findall(r"[a-z0-9$%]+(?:\.[0-9]+)?", text.lower())


def _content_tokens(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in STOPWORDS and len(t) > 2}


def _numbers(text: str) -> set[str]:
    return {m.replace(",", "") for m in NUM_RE.findall(text)}


class HeuristicDetector(BaseDetector):
    """Rule-based stand-in used when transformers are unavailable.

    Topical overlap gates everything; then a polarity flip or a numeric
    disagreement on the same measure raises the contradiction score, combined
    with a noisy-OR so two weak signals reinforce instead of saturating.

    This is NOT a substitute for a real NLI model. It exists so the pipeline,
    the graph updates and the tests can run offline. Its thresholds are lower
    because its scores are not calibrated probabilities.
    """

    backend = "heuristic"
    auto_threshold = 0.60
    hitl_threshold = 0.35

    W_POLARITY_FLIP = 0.72
    W_AFFIRM_VS_NEGATE = 0.30
    W_NUMERIC_DISJOINT = 0.65
    W_NUMERIC_PARTIAL = 0.35
    W_REVOCATION_TERM = 0.45

    def evaluate(self, premise: str, hypothesis: str) -> NLIResult:
        a_tok, b_tok = _content_tokens(premise), _content_tokens(hypothesis)
        if not a_tok or not b_tok:
            return NLIResult(NEUTRAL, 0.5, 0.0, self.backend)

        overlap = len(a_tok & b_tok) / min(len(a_tok), len(b_tok))
        # This used to hard-gate at overlap < 0.25, returning a flat
        # NEUTRAL/0.02 before polarity-flip or revocation signals were even
        # checked. That was backwards for exactly the clearest
        # contradictions: a full policy reversal ("eligible for hybrid
        # work" -> "strictly prohibited without authorization") rewrites
        # nearly the whole sentence, so it shares only one or two words and
        # got silently zeroed out, while a trivial numeric edit that keeps
        # most of the sentence intact sailed through.
        #
        # But dropping the gate entirely is its own bug: two genuinely
        # unrelated sections that happen to each contain a number (e.g. a
        # headcount in one, a dollar figure in another) then trip
        # W_NUMERIC_DISJOINT on numbers that have nothing to do with each
        # other, with no topical connection to restrain it. So the gate
        # stays, just at the much lower bar of "share at least one content
        # word" rather than 25% overlap — zero shared vocabulary means
        # unrelated topics and is still a hard NEUTRAL; anything above that
        # only scales the score down (below, via the 0.85 + 0.15 * overlap
        # multiplier), as the rest of this method already documents.
        if overlap <= 0:
            return NLIResult(NEUTRAL, 0.6, 0.02, self.backend)

        a_words, b_words = set(_tokens(premise)), set(_tokens(hypothesis))
        a_neg, b_neg = bool(a_words & NEGATORS), bool(b_words & NEGATORS)
        a_aff, b_aff = bool(a_words & AFFIRMERS), bool(b_words & AFFIRMERS)

        signals: list[float] = []
        if a_neg != b_neg:
            signals.append(self.W_POLARITY_FLIP)
            if (a_aff and b_neg) or (b_aff and a_neg):
                signals.append(self.W_AFFIRM_VS_NEGATE)
        if bool(a_words & REVOCATION) != bool(b_words & REVOCATION):
            signals.append(self.W_REVOCATION_TERM)

        a_nums, b_nums = _numbers(premise), _numbers(hypothesis)
        if a_nums and b_nums and a_nums != b_nums:
            signals.append(
                self.W_NUMERIC_DISJOINT if not (a_nums & b_nums)
                else self.W_NUMERIC_PARTIAL
            )

        # Noisy-OR: independent weak signals accumulate without ever exceeding 1.
        combined = 1.0
        for sig in signals:
            combined *= (1 - sig)
        score = 1 - combined

        # Overlap mostly gates rather than scales, so a genuine contradiction
        # phrased very differently is not penalised into invisibility.
        score = min(score * (0.85 + 0.15 * overlap), 0.99)

        if score >= self.hitl_threshold:
            label = CONTRADICTION if score >= self.auto_threshold else NEUTRAL
            return NLIResult(label, max(score, 0.5), score, self.backend)
        if overlap > 0.7:
            return NLIResult(ENTAILMENT, 0.7 + 0.25 * overlap, score, self.backend)
        return NLIResult(NEUTRAL, max(0.4, 1 - score), score, self.backend)


_CACHE: dict[bool, BaseDetector] = {}


def get_detector(force_offline: bool | None = None) -> BaseDetector:
    """Cached per offline/online mode — see the matching note in embeddings.py.
    Without this, flipping the Streamlit sidebar to "Transformer" mode after
    the app had already loaded once in "Fast / Offline" mode would silently
    keep returning the heuristic detector."""
    offline = config.OFFLINE if force_offline is None else force_offline
    if offline in _CACHE:
        return _CACHE[offline]

    if not offline:
        try:
            detector = CrossEncoderDetector()
            _CACHE[offline] = detector
            return detector
        except Exception as exc:  # noqa: BLE001
            print(f"[nli] cross-encoder unavailable ({exc}); using heuristic detector.")

    detector = HeuristicDetector()
    _CACHE[offline] = detector
    return detector
