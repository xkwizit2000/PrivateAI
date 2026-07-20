import tempfile
import unittest
from pathlib import Path

from src.rag import RagStore, _cosine


class FakeEmbedder:
    """Deterministic fixed-dimension bag-of-words embeddings for tests."""

    DIM = 32

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.DIM
            for token in text.lower().split():
                vector[hash(token) % self.DIM] += 1.0
            vectors.append(vector)
        return vectors


class RagStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.embedder = FakeEmbedder()
        self.store = RagStore(
            Path(self._tmpdir.name),
            top_k=3,
            max_context_chars=2000,
            min_score=0.1,
            embed_fn=self.embedder,
            recent_turns=1,
        )

    async def asyncTearDown(self) -> None:
        self._tmpdir.cleanup()

    async def test_empty_store_returns_nothing(self):
        snippets = await self.store.retrieve("123", "anything")
        self.assertEqual(snippets, [])

    async def test_indexes_and_ranks_relevant_chunks(self):
        await self.store.index_turn(
            "123",
            "The VPN subnet is 10.8.0.0/24",
            "Understood, VPN uses 10.8.0.0/24.",
        )
        await self.store.index_turn(
            "123",
            "My favorite tea is oolong",
            "Noted about oolong tea.",
        )

        snippets = await self.store.retrieve("123", "What is the VPN subnet?")
        self.assertTrue(snippets)
        joined = " ".join(item["content"] for item in snippets)
        self.assertIn("10.8.0.0/24", joined)
        self.assertGreaterEqual(snippets[0]["score"], self.store.min_score)

    async def test_score_floor_filters_weak_matches(self):
        strict = RagStore(
            Path(self._tmpdir.name) / "strict",
            top_k=3,
            min_score=0.99,
            embed_fn=self.embedder,
        )
        await strict.index_turn("123", "alpha beta", "gamma delta")
        snippets = await strict.retrieve("123", "completely unrelated zucchini")
        self.assertEqual(snippets, [])

    async def test_excludes_recent_turn_texts(self):
        await self.store.index_turn("123", "secret project codename Orion", "ok")
        snippets = await self.store.retrieve(
            "123",
            "codename Orion",
            exclude_texts=["secret project codename Orion", "ok"],
        )
        self.assertEqual(snippets, [])

    async def test_backfill_from_transcript(self):
        records = [
            {"role": "user", "content": "Backup disk is /mnt/backup"},
            {"role": "assistant", "content": "Noted /mnt/backup."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        inserted = await self.store.backfill_from_transcript("99", records)
        self.assertEqual(inserted, 4)
        # Second backfill should insert nothing (dedupe by content hash).
        inserted_again = await self.store.backfill_from_transcript("99", records)
        self.assertEqual(inserted_again, 0)

        snippets = await self.store.retrieve("99", "Where is the backup disk?")
        joined = " ".join(item["content"] for item in snippets)
        self.assertIn("/mnt/backup", joined)

    async def test_format_context(self):
        text = self.store.format_context(
            [{"role": "user", "content": "hello", "score": 0.9}]
        )
        self.assertIn("Relevant past context:", text)
        self.assertIn("user: hello", text)

    def test_cosine_identical_vectors(self):
        self.assertAlmostEqual(_cosine([1.0, 0.0], [1.0, 0.0]), 1.0)


if __name__ == "__main__":
    unittest.main()
