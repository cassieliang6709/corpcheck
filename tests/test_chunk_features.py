import unittest

from corpcheck.ingestion.chunk_features import compute_chunk_features


class ChunkFeatureTests(unittest.TestCase):
    def test_quantitative_financial_statement_chunk_scores_high(self) -> None:
        features = compute_chunk_features(
            "Financial Statements",
            "Revenue was $22.7 billion, gross margin was 46%, and operating income was $1.3 billion.",
        )
        self.assertGreaterEqual(features["numeric_token_count"], 3)
        self.assertTrue(features["is_quantitative"])

    def test_narrative_chunk_scores_low(self) -> None:
        features = compute_chunk_features(
            "Risk Factors",
            "We face intense competition, rapid technological change, and macroeconomic uncertainty.",
        )
        self.assertEqual(features["numeric_token_count"], 0)
        self.assertFalse(features["is_quantitative"])


if __name__ == "__main__":
    unittest.main()
