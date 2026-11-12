"""Environment-backed settings for the memory service."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """All tunables. Every field has an env var of the same name, upper-cased."""

    db_path: str = "/data/memory.db"
    embed_model: str = "snowflake/snowflake-arctic-embed-xs"
    # "fastembed" (real model) or "hash" (dependency-free test embedder).
    embed_backend: str = "fastembed"
    embed_cache_dir: str = "/opt/models"
    embed_threads: int = 2
    embed_local_only: bool = True  # never touch the network at runtime
    host: str = "0.0.0.0"
    port: int = 8765
    auth_token: str = ""  # empty = no auth (internal Docker network only!)
    # Output budgets (characters; ~4 chars per token).
    search_max_chars: int = 1800
    search_snippet_chars: int = 420
    conversation_max_chars: int = 3000
    facts_max_chars: int = 1500
    recall_max_chars: int = 700
    # Retrieval.
    rrf_k: int = 60
    vec_weight: float = 1.0
    fts_weight: float = 1.0
    candidates: int = 50
    recency_weight: float = 0.15  # max boost for a brand-new chunk
    recency_tau_days: float = 45.0
    # Gate for server-side auto-injection (see gate.py); calibrated per model.
    gate_cue_min_cos: float = 0.50
    gate_strong_cos: float = 0.85
    # Min cosine for facts/summaries to appear in search_memory output.
    small_min_cos: float = 0.55
    # Facts.
    fact_dedupe_cos: float = 0.97
    fact_supersede_cos: float = 0.85
    # Ingest.
    redact: bool = True
    chunk_chars: int = 700
    # Consolidation.
    idle_seconds: float = 300.0
    consolidate_max_input_chars: int = 6000
    ollama_host: str = "http://host.docker.internal:11434"
    ollama_model: str = "gemma4:e4b-jt-8k"
    log_level: str = "INFO"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        for name, default in list(d.__dict__.items()):
            if name == "extra":
                continue
            raw = os.getenv(name.upper())
            if raw is None or raw.strip() == "":
                continue
            if isinstance(default, bool):
                setattr(d, name, _bool(name.upper(), default))
            elif isinstance(default, int):
                setattr(d, name, int(raw))
            elif isinstance(default, float):
                setattr(d, name, float(raw))
            else:
                setattr(d, name, raw.strip())
        return d
