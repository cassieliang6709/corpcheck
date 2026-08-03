import unittest
from unittest.mock import patch

from corpcheck.ingestion.processors.chunker import Chunker


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False, truncation=False):
        return text.split()

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(token_ids)


class ChunkerBehaviorTests(unittest.TestCase):
    @patch("corpcheck.ingestion.processors.chunker.get_tokenizer", return_value=FakeTokenizer())
    def test_narrative_chunks_pack_full_sentences_without_cutting(self, _mock_tokenizer) -> None:
        chunker = Chunker(chunk_size=14, overlap=7)
        text = (
            "Alpha sentence includes substantial descriptive wording today. "
            "Beta sentence preserves complete narrative continuity here. "
            "Gamma sentence extends detailed financial explanation forward. "
            "Delta sentence closes the broader discussion coherently."
        )

        chunks = chunker.chunk_section("MD&A", text)
        chunk_texts = [chunk_text for _, chunk_text, _ in chunks]

        self.assertEqual(
            chunk_texts,
            [
                "Alpha sentence includes substantial descriptive wording today. Beta sentence preserves complete narrative continuity here.",
                "Beta sentence preserves complete narrative continuity here. Gamma sentence extends detailed financial explanation forward.",
                "Gamma sentence extends detailed financial explanation forward. Delta sentence closes the broader discussion coherently.",
            ],
        )
        self.assertTrue(all(chunk.endswith(".") for chunk in chunk_texts))

    @patch("corpcheck.ingestion.processors.chunker.get_tokenizer", return_value=FakeTokenizer())
    def test_table_chunks_keep_header_and_row_structure(self, _mock_tokenizer) -> None:
        chunker = Chunker(chunk_size=10, overlap=2)
        table_text = "\n".join(
            [
                "[TABLE] Revenue Table",
                "[HEADER] Year | Revenue",
                "[ROW] 2023 | 100",
                "[ROW] 2022 | 90",
                "[ROW] 2021 | 80",
                "[ROW] 2020 | 70",
                "[/TABLE]",
            ]
        )

        chunks = chunker.chunk_section("Financial Statements", table_text)
        self.assertGreaterEqual(len(chunks), 2)
        for _, chunk_text, _ in chunks:
            self.assertIn("[TABLE] Revenue Table", chunk_text)
            self.assertIn("[HEADER] Year | Revenue", chunk_text)
            self.assertIn("[ROW]", chunk_text)


if __name__ == "__main__":
    unittest.main()
