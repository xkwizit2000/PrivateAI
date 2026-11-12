from datetime import datetime, timezone

from memory_service import consolidate
from memory_service.gate import refers_to_past, should_inject


def test_fact_add_dedupe_supersede(svc):
    a = svc.remember_fact("The user prefers dark mode in every editor", "preference")
    assert a.startswith("saved fact#")
    assert svc.remember_fact("the user prefers dark mode in every editor", "preference").startswith("already known")
    svc.s.fact_supersede_cos = 0.5  # hash embedder: near-duplicates are only moderately similar
    b = svc.remember_fact("The user prefers dark mode in the terminal editor", "preference")
    assert "replaces" in b
    active = svc.store.list_facts("preference")
    assert len(active) == 1 and "terminal" in active[0]["statement"]
    old = svc.store.list_facts("preference", status="superseded")
    assert len(old) == 1 and old[0]["superseded_by"] == active[0]["id"]


def test_fact_category_normalised_and_listing(svc):
    svc.remember_fact("Deploy target is pai01", "bogus-category")
    assert "[note]" in svc.format_facts()
    assert svc.format_facts("task").startswith("No task")
    assert svc.remember_fact("x") .startswith("Error")


def test_facts_surface_in_search(svc):
    svc.remember_fact("The user's cat is named Biscuit", "profile")
    res = svc.search("what is my cat called", k=3)
    assert res.facts and "Biscuit" in res.facts[0]["statement"]
    assert svc.format_search(res).startswith("[fact#")


def test_gate_cues():
    assert refers_to_past("do you remember what we decided about the database?")
    assert refers_to_past("Last time you said to use restic")
    assert not refers_to_past("write a haiku about spring")


def test_gate_decisions():
    ok, _ = should_inject("what did we decide", 0.5, 0.45, 0.72)
    assert ok
    assert not should_inject("write a haiku", 0.5, 0.45, 0.72)[0]
    assert should_inject("write a haiku", 0.8, 0.45, 0.72)[0]
    assert not should_inject("anything", None, 0.45, 0.72)[0]


def test_recall_excludes_recent_and_gates(svc):
    r = svc.recall("do you remember which database we chose for the inventory app?", exclude_recent=2)
    assert r["hits"]
    quiet = svc.recall("zzz qqq")
    assert quiet["inject"] is False and quiet["text"] == ""


def test_core_profile(svc):
    svc.remember_fact("Prefers concise answers", "preference")
    core = svc.core("s1", recent_turns=2)
    assert "concise" in core["profile"] and len(core["turns"]) == 4


NOW = datetime(2026, 4, 1, tzinfo=timezone.utc)


def test_plan_and_run_hierarchy(svc):
    units = consolidate.plan(svc, NOW)
    assert [u.level for u in units] == ["day"] * 5  # 5 distinct days, all complete before NOW
    prompts = []

    def fake(p):
        prompts.append(p)
        return "summary-" + str(len(prompts))

    out = consolidate.run(svc, fake, now=NOW, force=True, max_units=0)
    assert len(out["done"]) == 5
    assert svc.store.stats()["summaries"] == 5
    # weeks W10 (Mar 2,3) and W11 (Mar 10,11) have 2 summarised days; Mar 17 is alone
    weeks = [u for u in consolidate.plan(svc, NOW) if u.level == "week"]
    assert [u.period for u in weeks] == ["2026-W10", "2026-W11"]
    consolidate.run(svc, fake, now=NOW, force=True, max_units=0)
    assert any(p.startswith("Merge") for p in prompts)
    # idempotent: nothing left
    assert consolidate.plan(svc, NOW) == []


def test_run_refuses_when_busy(svc):
    svc.touch()
    out = consolidate.run(svc, lambda p: "x", now=NOW, idle_seconds=300)
    assert out["done"] == [] and "not idle" in out["skipped"]


def test_oversize_day_is_map_reduced_and_capped():
    unit = consolidate.Unit("day", "2026-03-02", "s", "sha", 100, 1, 100, "\n".join("u: " + "word " * 100 for _ in range(100)))
    calls = []

    def fake(p):
        calls.append(len(p))
        return "ok"

    consolidate.summarize_unit(unit, fake, max_input_chars=2000)
    assert len(calls) > 3 and max(calls) < 3000  # each LLM call input is size-capped


def test_stale_summary_rebuilt(svc):
    consolidate.run(svc, lambda p: "v1", now=NOW, force=True, max_units=0)
    svc.store.add_turns([{"session_id": "s1", "role": "user", "content": "one more thing that day",
                          "timestamp": "2026-03-02T23:00:00+00:00"}])
    stale = [u for u in consolidate.plan(svc, NOW) if u.level == "day"]
    assert [u.period for u in stale] == ["2026-03-02"]


def test_summary_used_by_get_conversation(svc):
    consolidate.run(svc, lambda p: "Planned the restic backup", now=NOW, force=True, max_units=0)
    assert "[day summary 2026-03-02] Planned the restic backup" in svc.get_conversation(date="2026-03-02")
