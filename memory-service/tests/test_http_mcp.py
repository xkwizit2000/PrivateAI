import json

import pytest

pytest.importorskip("mcp")
httpx = pytest.importorskip("httpx")
from starlette.testclient import TestClient  # noqa: E402

from memory_service.http_api import build_app  # noqa: E402
from memory_service.mcp_server import build_mcp  # noqa: E402

SCHEMA_BUDGET = 2000


def _tools(svc):
    mcp = build_mcp(svc)
    return mcp, [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
        for t in mcp._tool_manager.list_tools()
    ]


def test_tool_names_and_schema_budget(svc):
    _, tools = _tools(svc)
    names = {t["function"]["name"] for t in tools}
    assert names == {"search_memory", "get_conversation", "remember_fact", "list_facts", "memory_stats"}
    size = len(json.dumps(tools))
    assert size < SCHEMA_BUDGET, size
    blob = json.dumps(tools)
    assert "anyOf" not in blob and '"title"' not in blob


def test_search_tool_params(svc):
    _, tools = _tools(svc)
    props = next(t for t in tools if t["function"]["name"] == "search_memory")["function"]["parameters"]
    assert props["required"] == ["query"]
    assert set(props["properties"]) == {"query", "k", "since", "until", "session"}


@pytest.fixture()
def client(svc):
    return TestClient(build_app(svc, build_mcp(svc), auth_token="s3cret"))


H = {"Authorization": "Bearer s3cret"}


def test_http_auth(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/v1/stats").status_code == 401
    assert client.get("/v1/stats", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/v1/stats", headers=H).json()["turns"] == 14


def test_http_search_recall_turns_facts_core(client):
    r = client.post("/v1/search", headers=H, json={"query": "Biscuit cat", "k": 2}).json()
    assert "Biscuit" in r["text"] and len(r["text"]) <= 1800
    rec = client.post("/v1/recall", headers=H, json={"query": "remember the postgres decision for the inventory app?", "session": "s1"}).json()
    assert "inject" in rec and len(rec["text"]) <= 700
    n = client.post("/v1/turns", headers=H, json={"session_id": "s9", "turns": [
        {"role": "user", "content": "hello from http"}, {"role": "assistant", "content": "hi"}]}).json()
    assert n["turns_new"] == 2
    assert "saved" in client.post("/v1/facts", headers=H, json={"statement": "Likes tea", "category": "profile"}).json()["result"]
    assert "Likes tea" in client.get("/v1/facts", headers=H).json()["text"]
    core = client.get("/v1/core?session=s9&turns=1", headers=H).json()
    assert "Likes tea" in core["profile"] and len(core["turns"]) == 2
    conv = client.post("/v1/conversation", headers=H, json={"date": "2026-03-02"}).json()
    assert "restic" in conv["text"]
