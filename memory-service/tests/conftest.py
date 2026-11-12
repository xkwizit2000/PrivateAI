import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from memory_service.config import Settings  # noqa: E402
from memory_service.embedder import HashEmbedder  # noqa: E402
from memory_service.service import MemoryService  # noqa: E402
from memory_service.store import MemoryStore  # noqa: E402

BASE = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)  # a Monday

# Entirely synthetic conversation (no real user data).
SYNTH = [
    (0, "user", "I want to set up a nightly backup of the photo server to the NAS using restic."),
    (0, "assistant", "Use restic with a repository on the NAS and a systemd timer at 02:30; keep 7 daily and 4 weekly snapshots."),
    (0, "user", "My cat is named Biscuit and she hates the vacuum cleaner."),
    (0, "assistant", "Noted! Biscuit sounds like a character."),
    (1, "user", "What is the capital of Australia?"),
    (1, "assistant", "The capital of Australia is Canberra."),
    (1, "user", "Remind me how to rotate the TLS certificate for the grafana dashboard."),
    (1, "assistant", "Run certbot renew, then reload nginx; grafana sits behind the nginx proxy on port 3000."),
    (8, "user", "Let's decide: we will use PostgreSQL 16 for the inventory app, not MySQL."),
    (8, "assistant", "Agreed: PostgreSQL 16 for the inventory app. I will draft the schema with a products table."),
    (9, "user", "The inventory app needs an index on products.sku for fast lookups."),
    (9, "assistant", "CREATE UNIQUE INDEX products_sku_idx ON products (sku); that makes lookups fast."),
    (15, "user", "Tell me a joke about compilers."),
    (15, "assistant", "Why did the compiler break up with the interpreter? Too many unresolved references."),
]


def make_turns(session="s1"):
    out = []
    for i, (day, role, content) in enumerate(SYNTH):
        ts = BASE + timedelta(days=day, minutes=i)
        out.append({"session_id": session, "role": role, "content": content, "timestamp": ts.isoformat()})
    return out


@pytest.fixture()
def settings():
    s = Settings()
    s.embed_backend = "hash"
    s.recency_weight = 0.0
    s.small_min_cos = 0.1  # hash embedder cosines are low
    return s


@pytest.fixture()
def svc(settings):
    store = MemoryStore(":memory:", HashEmbedder(), settings)
    store.add_turns(make_turns())
    return MemoryService(store, settings)
