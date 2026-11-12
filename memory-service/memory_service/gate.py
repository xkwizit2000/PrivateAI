"""Cheap (no-LLM) gate deciding whether to auto-inject recalled memory.

Signals, all computed in a few milliseconds on CPU:
  1. a regex "refers to the past" cue on the user message,
  2. the top vector cosine of the message against the archive (one embedding of the
     short message + one matrix-vector product).
Fire when: (cue AND cos >= gate_cue_min_cos) OR cos >= gate_strong_cos.
"""

from __future__ import annotations

import re

_CUES = re.compile(
    r"""(?ix)\b(?:
        (?:do|did|can|could|would)\s+you\s+(?:still\s+)?(?:remember|recall) |
        remember\s+(?:when|what|how|which|where|who|the\s+\w+\s+(?:we|i|you)) |
        as\s+you\s+(?:remember|recall) | remind\s+me\s+(?:what|how|which|when|where|who|of|about) |
        you\s+(?:forgot|already\s+(?:said|told|know)) |
        earlier | previous(?:ly)? | (?:said|mentioned|told\s+you|discussed|talked\s+about)\s+(?:this\s+)?before |
        last\s+(?:time|week|month|night|session|conversation) |
        yesterday | the\s+other\s+day | back\s+when | (?:a\s+)?(?:few|couple\s+of)\s+(?:days|weeks|months)\s+ago |
        \d+\s+(?:days?|weeks?|months?)\s+ago | (?:last|this)\s+(?:mon|tues|wednes|thurs|fri|satur|sun)day |
        we\s+(?:talked|discussed|decided|agreed|chose|spoke) | (?:what|which|where|when|how)\s+(?:did|have)\s+(?:we|i)\s+\w+ |
        you\s+(?:said|told\s+me|mentioned|suggested|recommended|gave\s+me|showed\s+me|explained) |
        i\s+(?:told|mentioned|said)\s+(?:you|that) | (?:as|like)\s+(?:we|i)\s+(?:discussed|said|mentioned) |
        our\s+(?:earlier|previous|last|old|past)\s+\w+ | (?:history|record)\s+of\s+(?:our|my)
    )\b""",
)


def refers_to_past(text: str) -> bool:
    """True if the message contains an explicit reference to earlier conversation."""
    return bool(_CUES.search(text))


def should_inject(text: str, top_cos: float | None, cue_min_cos: float, strong_cos: float) -> tuple[bool, str]:
    """Return (inject?, reason). `top_cos` is the best vector cosine (None = empty archive)."""
    if top_cos is None:
        return False, "empty archive"
    cue = refers_to_past(text)
    if top_cos >= strong_cos:
        return True, f"strong match cos={top_cos:.2f}"
    if cue and top_cos >= cue_min_cos:
        return True, f"past-reference cue + cos={top_cos:.2f}"
    return False, f"no gate (cue={cue}, cos={top_cos:.2f})"
