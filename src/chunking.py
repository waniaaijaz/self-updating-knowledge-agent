"""Structural, document-aware chunking.

Fixed-character splitting destroys exactly the thing this project cares about:
a policy update usually rewrites one clause and leaves the rest identical. If a
chunk boundary lands in the middle of that clause, the contradiction becomes
invisible to the NLI model.

So we split on markdown headers instead, keep the header path as a breadcrumb,
and store a parent/child pair:

  * child  -> a single rule (fed to the cross-encoder, must stay under 512 tok)
  * parent -> the whole section (fed to the LLM at answer time, for context)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from typing import Iterator

from . import config

HEADER_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
# "4.2 Equipment Stipend" -> section number 4.2
SECTION_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+")


@dataclass
class Chunk:
    id: str
    text: str  # breadcrumb + body, this is what gets embedded
    body: str  # body only, no breadcrumb
    breadcrumb: str  # "HR Policy 2026 > Remote Work > Equipment Stipend"
    section_path: str  # "4.2" when the headings are numbered, else the slug
    doc_id: str
    version: str
    timestamp: str  # ISO date
    parent_id: str
    parent_text: str
    content_hash: str
    token_estimate: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_tokens(text: str) -> int:
    """Cheap token proxy. Words * 1.3 tracks real BPE counts closely enough
    for a 512-token guardrail, and costs nothing to compute."""
    return int(len(text.split()) * 1.3) + 1


def sha256(text: str) -> str:
    """Normalised hash. Whitespace and case changes are not content changes,
    so we strip them before hashing to avoid re-running NLI on reformatting."""
    normalised = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _split_long_body(body: str, max_tokens: int) -> list[str]:
    """Only used when a single section is too long for the cross-encoder.
    Splits on sentence/bullet boundaries, never mid-sentence."""
    if estimate_tokens(body) <= max_tokens:
        return [body]

    units = re.split(r"(?<=[.!?])\s+|\n(?=\s*[-*\d])", body)
    out, current = [], []
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        candidate = current + [unit]
        if estimate_tokens(" ".join(candidate)) > max_tokens and current:
            out.append(" ".join(current))
            current = [unit]
        else:
            current = candidate
    if current:
        out.append(" ".join(current))
    return out


def _sections(markdown: str) -> Iterator[tuple[list[str], str]]:
    """Yield (header_stack, body) for every section in the document.

    Text with no heading above it — either a whole document with no `#`
    lines, or a preamble before the first one — used to be dropped here,
    because nothing was ever appended to `stack`. That's a silent failure:
    the document "ingests" with zero chunks and is never retrievable, with
    no error to say why. It now falls back to a synthetic "Document"
    section instead, so any markdown or plain-text file produces at least
    one chunk.
    """
    stack: list[str] = []
    buffer: list[str] = []

    def flush() -> tuple[list[str], str] | None:
        body = "\n".join(buffer).strip()
        if not body:
            return None
        return (list(stack), body) if stack else (["Document"], body)

    for line in markdown.splitlines():
        match = HEADER_RE.match(line)
        if not match:
            buffer.append(line)
            continue

        section = flush()
        if section:
            yield section
        buffer = []

        level = len(match.group(1))
        title = match.group(2).strip()
        stack = stack[: level - 1]
        while len(stack) < level - 1:
            stack.append("")
        stack.append(title)

    section = flush()
    if section:
        yield section


def chunk_markdown(
    markdown: str,
    doc_id: str,
    version: str,
    timestamp: str,
    max_tokens: int | None = None,
    min_tokens: int | None = None,
) -> list[Chunk]:
    max_tokens = max_tokens or config.CHUNK_MAX_TOKENS
    min_tokens = min_tokens or config.CHUNK_MIN_TOKENS

    chunks: list[Chunk] = []
    for stack, body in _sections(markdown):
        titles = [t for t in stack if t]
        breadcrumb = " > ".join(titles)
        leaf = titles[-1] if titles else doc_id

        num_match = SECTION_NUM_RE.match(leaf)
        # section_path: used as a *hint* for cross-version matching (a 0.05
        # boost in detect_contradiction when v1 and v2 share one). A bare
        # section number or leaf slug is fine for that — it doesn't need to
        # be unique, and staying stable even when a parent chapter is
        # renamed between versions is what makes it useful there.
        section_path = num_match.group(1) if num_match else _slug(leaf)

        # full_path_slug: used for the chunk/parent ID, which MUST be unique
        # per location in the document. Two different chapters can each have
        # their own "## Overview" or their own "1.1" — using only the leaf
        # (the old behaviour) gave both the same ID, so the second one
        # silently overwrote the first in the vector and graph stores.
        full_path_slug = "/".join(_slug(t) for t in titles) if titles else "document"

        parent_id = f"{doc_id}::{version}::{full_path_slug}"
        pieces = _split_long_body(body, max_tokens)

        for i, piece in enumerate(pieces):
            if estimate_tokens(piece) < min_tokens:
                continue
            text = f"{breadcrumb}: {piece}" if breadcrumb else piece
            chunks.append(
                Chunk(
                    id=f"{parent_id}::{i}",
                    text=text,
                    body=piece,
                    breadcrumb=breadcrumb,
                    section_path=section_path,
                    doc_id=doc_id,
                    version=version,
                    timestamp=timestamp,
                    parent_id=parent_id,
                    parent_text=body,
                    content_hash=sha256(piece),
                    token_estimate=estimate_tokens(text),
                )
            )
    return chunks


def chunk_file(path, doc_id: str, version: str, timestamp: str) -> list[Chunk]:
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8")
    return chunk_markdown(text, doc_id, version, timestamp)
