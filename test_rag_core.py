import os
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import rag_core


class FakeCollection:
    def __init__(self):
        self.ids = []
        self.upserts = []
        self.deleted = []

    def get(self, include):
        return {"ids": list(self.ids)}

    def delete(self, ids):
        self.deleted.extend(ids)
        self.ids = [item for item in self.ids if item not in ids]

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)
        self.ids.extend(kwargs["ids"])

    def count(self):
        return 2

    def query(self, **kwargs):
        return {
            "documents": [["Reset on the login page.", "Contact your admin."]],
            "metadatas": [[
                {"source": "faq_auth.md", "chunk_index": 0},
                {"source": "faq_sso.md", "chunk_index": 0},
            ]],
            "distances": [[0.1, 0.3]],
        }


class RagCoreTests(unittest.TestCase):
    def test_chunking_respects_size_and_preserves_words(self):
        text = "alpha beta gamma delta epsilon zeta eta theta"
        chunks = rag_core._chunk_text(text, size=18)
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= 18 for chunk in chunks))
        self.assertEqual(" ".join(chunks), text)

    def test_chunk_id_is_stable_and_source_scoped(self):
        first = rag_core._chunk_id("one.md", 0, "same content")
        self.assertEqual(first, rag_core._chunk_id("one.md", 0, "same content"))
        self.assertNotEqual(first, rag_core._chunk_id("two.md", 0, "same content"))

    def test_reindex_only_embeds_new_chunks_and_deletes_stale(self):
        collection = FakeCollection()
        collection.ids = ["stale"]
        records = [
            ("faq_auth.md", 0, "auth content"),
            ("faq_sso.md", 0, "sso content"),
        ]
        with (
            patch.object(rag_core, "_get_collection", return_value=collection),
            patch.object(rag_core, "_load_and_chunk_faqs", return_value=records),
            patch.object(
                rag_core,
                "_embed_texts",
                side_effect=lambda texts: [[0.1, 0.2] for _ in texts],
            ),
        ):
            rag_core.reindex()

        self.assertEqual(collection.deleted, ["stale"])
        self.assertEqual(len(collection.upserts), 1)
        self.assertEqual(len(collection.upserts[0]["ids"]), 2)

    def test_ask_faq_returns_exact_contract_and_retrieval_order(self):
        collection = FakeCollection()
        with (
            patch.object(rag_core, "_get_collection", return_value=collection),
            patch.object(rag_core, "_load_and_chunk_faqs") as scan_sources,
            patch.object(rag_core, "_embed_query", return_value=[0.1, 0.2]),
            patch.object(rag_core, "_generate_answer", return_value="Answer"),
        ):
            result = rag_core.ask_faq_core("How?", top_k=4)

        self.assertEqual(list(result), ["answer", "sources"])
        self.assertEqual(result["sources"], ["faq_auth.md", "faq_sso.md"])
        scan_sources.assert_not_called()

    def test_ask_faq_requires_an_existing_index(self):
        collection = FakeCollection()
        collection.count = lambda: 0
        with (
            patch.object(rag_core, "_get_collection", return_value=collection),
            patch.object(rag_core, "_embed_query") as embed_query,
        ):
            with self.assertRaisesRegex(RuntimeError, "--reindex"):
                rag_core.ask_faq_core("How?", top_k=4)

        embed_query.assert_not_called()

    def test_invalid_inputs_are_rejected(self):
        for question, top_k in [("", 4), ("question", 0), ("question", 11)]:
            with self.subTest(question=question, top_k=top_k):
                with self.assertRaises(ValueError):
                    rag_core.ask_faq_core(question, top_k=top_k)

    def test_reindex_cli_prints_statistics_and_does_not_prompt(self):
        stats = {
            "source_files": 3,
            "chunks_total": 7,
            "chunks_added": 2,
            "chunks_deleted": 1,
            "chunks_unchanged": 5,
        }
        output = io.StringIO()
        with (
            patch.object(rag_core, "reindex", return_value=stats) as reindex,
            patch("builtins.input", side_effect=AssertionError("unexpected prompt")),
            redirect_stdout(output),
        ):
            rag_core.main_cli(["--reindex"])

        reindex.assert_called_once_with()
        self.assertEqual(json.loads(output.getvalue()), stats)


if __name__ == "__main__":
    unittest.main()
