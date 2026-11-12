import sqlite3

import pytest

from conftest import make_turns
from memory_service.embedder import HashEmbedder
from memory_service.store import EmbedModelMismatch, MemoryStore


def test_ingest_idempotent(svc):
    before = svc.store.stats()
    res = svc.store.add_turns(make_turns())
    assert res.turns_new == 0
    assert svc.store.stats()["turns"] == before["turns"] == 14


def test_turns_immutable(svc):
    with pytest.raises(sqlite3.DatabaseError):
        svc.store.conn.execute("UPDATE turns SET content='x' WHERE id=1")
    with pytest.raises(sqlite3.DatabaseError):
        svc.store.conn.execute("DELETE FROM turns WHERE id=1")


def test_purge_is_audited_and_rebuilds_index(svc):
    n_chunks = svc.store.stats()["chunks"]
    assert svc.store.purge_turns([1]) == 1
    st = svc.store.stats()
    assert st["turns"] == 13 and st["chunks"] == n_chunks - 1 and svc.store.index.n == n_chunks - 1
    with pytest.raises(sqlite3.DatabaseError):  # trigger restored
        svc.store.conn.execute("DELETE FROM turns WHERE id=2")
    assert not svc.search_chunks("restic NAS", mode="keyword").hits or all(h.turn_id != 1 for h in svc.search_chunks("restic", mode="keyword").hits)


def test_keyword_finds_distinctive_term(svc):
    res = svc.search_chunks("Biscuit", mode="keyword", k=3)
    assert res.hits and "Biscuit" in res.hits[0].text


def test_hybrid_finds_topic(svc):
    res = svc.search_chunks("which database did we pick for the inventory application", k=3)
    ids = [h.turn_id for h in res.hits]
    assert any("PostgreSQL" in h.text for h in res.hits), ids


def test_date_filters(svc):
    res = svc.search_chunks("inventory", since="2026-03-09", until="2026-03-10", k=5)
    assert res.hits and all(h.day in ("2026-03-09", "2026-03-10") for h in res.hits)
    none = svc.search_chunks("inventory", until="2026-03-03", k=5, mode="keyword")
    assert none.hits == []


def test_session_filter(svc):
    svc.store.add_turns(make_turns("s2")[:2])
    assert all(h.session_id == "s2" for h in svc.search_chunks("restic", session="s2").hits)
    assert svc.search_chunks("restic", session="nope").hits == []


def test_exclude_turns(svc):
    top = svc.search_chunks("Biscuit", mode="keyword", k=1).hits[0].turn_id
    again = svc.search_chunks("Biscuit", mode="keyword", k=3, exclude_turns=[top])
    assert all(h.turn_id != top for h in again.hits)


def test_output_budget_capped(svc):
    text = svc.format_search(svc.search("backup restic nas", k=10), max_chars=200)
    assert len(text) <= 200


def test_no_match_message(svc):
    assert svc.format_search(svc.search_chunks("zzzzqqqq", mode="keyword")) == "No matching memory."


def test_long_turn_chunked_and_best_chunk_returned(svc):
    long = " ".join(f"filler sentence {i} about nothing in particular." for i in range(120))
    long += " The secret codename for the migration project is Heliotrope."
    svc.store.add_turns([{"session_id": "s1", "role": "assistant", "content": long, "timestamp": "2026-04-01T10:00:00+00:00"}])
    assert svc.store.stats()["chunks"] > 15
    h = svc.search_chunks("Heliotrope codename", k=1).hits[0]
    assert "Heliotrope" in h.text and len(h.text) <= svc.s.search_snippet_chars + 2


def test_reload_index_from_disk(tmp_path, settings):
    p = tmp_path / "m.db"
    s1 = MemoryStore(p, HashEmbedder(), settings)
    s1.add_turns(make_turns())
    s1.close()
    s2 = MemoryStore(p, HashEmbedder(), settings)
    assert s2.index.n == s2.stats()["chunks"] > 0


def test_model_mismatch_detected(tmp_path, settings):
    p = tmp_path / "m.db"
    MemoryStore(p, HashEmbedder(), settings).add_turns(make_turns())
    other = HashEmbedder(dim=128)
    other.name = "other-model"
    with pytest.raises(EmbedModelMismatch):
        MemoryStore(p, other, settings)


def test_redaction_at_ingest(svc):
    svc.store.add_turns([{"session_id": "s1", "role": "user", "content": "my api_key = abcdef123456789 ok",
                          "timestamp": "2026-04-02T10:00:00+00:00"}])
    row = svc.store.conn.execute("SELECT content, redactions FROM turns ORDER BY id DESC LIMIT 1").fetchone()
    assert "abcdef123456789" not in row["content"] and row["redactions"] == 1


def test_get_conversation_around_and_date(svc):
    h = svc.search_chunks("PostgreSQL", k=1).hits[0]
    text = svc.get_conversation(around_turn_id=h.turn_id, window=1)
    assert f"[t{h.turn_id} " in text and text.count("\n") >= 1
    day = svc.get_conversation(date="2026-03-02")
    assert "restic" in day and len(day) <= svc.s.conversation_max_chars
    assert svc.get_conversation(date="2020-01-01").startswith("No conversation")
    assert svc.get_conversation().startswith("Give")
