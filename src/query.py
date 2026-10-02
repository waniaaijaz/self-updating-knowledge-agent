"""Query-time retrieval.

The whole point of the ingest work shows up here: before ranking, we drop
anything the graph marked SUPERSEDED and anything whose freshness has decayed
below the floor. The LLM never sees the stale rule, so it cannot blend the two
versions together.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config
from .freshness import (
    STATUS_ACTIVE,
    STATUS_SUPERSEDED,
    calculate_freshness,
    final_score,
    passes_gate,
)


@dataclass
class RetrievedChunk:
    id: str
    text: str
    parent_text: str
    breadcrumb: str
    version: str
    doc_id: str
    similarity: float
    freshness: float
    score: float
    status: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class QueryResult:
    question: str
    answer: str
    used: list[RetrievedChunk]
    filtered_out: list[RetrievedChunk]
    generator: str


PROMPT = """You are an internal policy assistant. Answer using ONLY the context below.
Every passage in the context is the currently active version of the policy;
superseded versions have already been removed. If the context does not answer
the question, say so plainly. Cite the section breadcrumb you relied on.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


class QueryEngine:
    def __init__(self, svc):
        self.svc = svc
        self._llm = None
        self.generator_name = "extractive"

    # ------------------------------------------------------------ retrieval
    def retrieve(self, question: str, top_k: int = 8):
        vector = self.svc.embedder.encode_one(question)
        hits = self.svc.vectors.search(vector, top_k=top_k * 3)

        used, dropped = [], []
        for hit in hits:
            payload = hit.payload
            node = self.svc.graph.get(payload["id"]) or {}
            status = node.get("status", payload.get("status", STATUS_ACTIVE))

            freshness = calculate_freshness(
                base_score=1.0, created_at=payload["timestamp"], status=status
            )
            rc = RetrievedChunk(
                id=payload["id"],
                text=payload["text"],
                parent_text=payload.get("parent_text", payload["text"]),
                breadcrumb=payload.get("breadcrumb", ""),
                version=payload.get("version", ""),
                doc_id=payload.get("doc_id", ""),
                similarity=round(hit.score, 4),
                freshness=freshness,
                score=final_score(hit.score, freshness),
                status=status,
            )
            if status == STATUS_SUPERSEDED or not passes_gate(freshness):
                dropped.append(rc)
            else:
                used.append(rc)

        used.sort(key=lambda r: r.score, reverse=True)
        return used[:top_k], dropped

    # ------------------------------------------------------------ generation
    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        if config.GEMINI_API_KEY:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._llm = ChatGoogleGenerativeAI(
                    model=config.GEMINI_MODEL,
                    temperature=0,
                    google_api_key=config.GEMINI_API_KEY,
                )
                self.generator_name = f"gemini:{config.GEMINI_MODEL}"
                return self._llm
            except Exception as exc:  # noqa: BLE001
                print(f"[query] Gemini unavailable ({exc}); trying other LLMs or extractive.")
        if config.OPENAI_API_KEY:
            try:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    temperature=0,
                    api_key=config.OPENAI_API_KEY,
                    model_name=config.OPENAI_MODEL,
                )
                self.generator_name = f"openai:{config.OPENAI_MODEL}"
                return self._llm
            except Exception as exc:  # noqa: BLE001
                print(f"[query] OpenAI unavailable ({exc}); trying Groq or extractive.")
        if config.GROQ_API_KEY:
            try:
                from langchain_groq import ChatGroq

                self._llm = ChatGroq(
                    temperature=0,
                    groq_api_key=config.GROQ_API_KEY,
                    model_name=config.GROQ_MODEL,
                )
                self.generator_name = f"groq:{config.GROQ_MODEL}"
                return self._llm
            except Exception as exc:  # noqa: BLE001
                print(f"[query] Groq unavailable ({exc}); returning extractive answer.")
        self._llm = None
        self.generator_name = "extractive"
        return self._llm

    def ask(self, question: str, top_k: int = 5) -> QueryResult:
        used, dropped = self.retrieve(question, top_k=top_k)
        if not used:
            return QueryResult(question, "No active policy covers that question.",
                               [], dropped, "none")

        # Parent-child expansion: retrieve on the precise child chunk, then
        # hand the LLM the full section it came from.
        seen, blocks = set(), []
        for rc in used:
            if rc.parent_text in seen:
                continue
            seen.add(rc.parent_text)
            blocks.append(f"[{rc.breadcrumb} | {rc.doc_id} {rc.version}]\n{rc.parent_text}")
        context = "\n\n---\n\n".join(blocks)

        llm = self._get_llm()
        if llm is None:
            answer = (
                "(extractive — no LLM key configured)\n\n"
                + "\n\n".join(f"• {rc.breadcrumb}: {rc.text.split(': ', 1)[-1]}"
                              for rc in used[:3])
            )
            return QueryResult(question, answer, used, dropped, self.generator_name)

        try:
            response = llm.invoke(PROMPT.format(context=context, question=question))
            raw_content = getattr(response, "content", str(response))
            if isinstance(raw_content, list):
                text = "".join(item.get("text", "") for item in raw_content if isinstance(item, dict) and "text" in item)
            else:
                text = str(raw_content)
            return QueryResult(question, text, used, dropped, self.generator_name)
        except Exception as exc:
            print(f"[query] LLM invocation failed ({exc}); falling back to extractive answer.")
            answer = (
                "(extractive — LLM unavailable or key invalid)\n\n"
                + "\n\n".join(f"• {rc.breadcrumb}: {rc.text.split(': ', 1)[-1]}"
                              for rc in used[:3])
            )
            return QueryResult(question, answer, used, dropped, "extractive")

