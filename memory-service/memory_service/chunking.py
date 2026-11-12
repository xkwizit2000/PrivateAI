"""Chunking tuned for chat turns.

* Most chat messages are short (median ~180 chars): one message = one chunk,
  so the chunk is a coherent unit and the hit maps 1:1 to a turn.
* Long messages (assistant answers, pasted logs) are split on paragraph, then
  line, then sentence boundaries into <= `size` chars; fenced code blocks are
  kept together when they fit. A small tail-merge avoids tiny trailing chunks.
* No overlap: a hit always links back to the whole turn via get_conversation,
  so overlap would only add duplicate embeddings.
"""

from __future__ import annotations

import re

_SENT = re.compile(r"(?<=[.!?])\s+")


def _split_long(piece: str, size: int) -> list[str]:
    """Split one oversized paragraph on lines, sentences, then hard-cut."""
    out: list[str] = []
    for line in piece.split("\n"):
        if len(line) <= size:
            out.append(line)
            continue
        for sent in _SENT.split(line):
            while len(sent) > size:
                cut = sent.rfind(" ", 0, size)
                cut = cut if cut > size // 2 else size
                out.append(sent[:cut].strip())
                sent = sent[cut:].strip()
            if sent:
                out.append(sent)
    return out


def chunk_text(text: str, size: int = 700) -> list[str]:
    """Split text into chunks of at most `size` chars (never empty strings)."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    # Paragraph units (blank-line separated); keep fenced code blocks atomic.
    units: list[str] = []
    for block in re.split(r"(```.*?```)", text, flags=re.S):
        if block.startswith("```"):
            units.append(block.strip())
        else:
            units.extend(p.strip() for p in re.split(r"\n\s*\n", block) if p.strip())

    atoms: list[str] = []
    for unit in units:
        atoms.extend([unit] if len(unit) <= size else _split_long(unit, size))

    chunks: list[str] = []
    current = ""
    for atom in atoms:
        if not atom:
            continue
        joined = f"{current}\n{atom}" if current else atom
        if len(joined) <= size:
            current = joined
        else:
            if current:
                chunks.append(current)
            current = atom
    if current:
        chunks.append(current)

    # Merge a tiny tail into the previous chunk when it still fits (<= 1.25x size).
    if len(chunks) > 1 and len(chunks[-1]) < size // 5 and len(chunks[-2]) + len(chunks[-1]) + 1 <= int(size * 1.25):
        chunks[-2] = f"{chunks[-2]}\n{chunks[-1]}"
        chunks.pop()
    return [c for c in chunks if c.strip()]
