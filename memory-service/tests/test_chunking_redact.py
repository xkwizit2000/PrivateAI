from memory_service.chunking import chunk_text
from memory_service.redact import redact


def test_short_text_single_chunk():
    assert chunk_text("hello there") == ["hello there"]
    assert chunk_text("   ") == []


def test_long_text_split_within_limit_and_lossless():
    para = ". ".join(f"Sentence number {i} about the backup server" for i in range(60))
    chunks = chunk_text(para, 300)
    assert len(chunks) > 3
    assert all(len(c) <= 380 for c in chunks)
    joined = " ".join(chunks)
    for i in (0, 30, 59):
        assert f"Sentence number {i} " in joined


def test_code_block_kept_atomic_when_it_fits():
    text = "Intro paragraph. " * 20 + "\n\n```\nline1\nline2\nline3\n```\n\n" + "Outro. " * 40
    chunks = chunk_text(text, 400)
    assert any("```\nline1\nline2\nline3\n```" in c for c in chunks)


def test_redaction_kinds():
    tg = "123456789:" + "A" * 35
    text = f"token {tg} and password: hunter2hunter and key sk-abcdefghijklmnopqrstuv end"
    out, n = redact(text)
    assert n >= 3
    assert tg not in out and "hunter2hunter" not in out and "sk-abcdefghij" not in out
    assert "[REDACTED:" in out


def test_redaction_leaves_normal_text():
    s = "We talked about the password policy and token rotation schedule."
    assert redact(s) == (s, 0)


def test_mnemonic_redacted():
    out, n = redact("my mnemonic: " + " ".join(["abandon"] * 11 + ["about"]))
    assert n == 1 and "abandon" not in out
