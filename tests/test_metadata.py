import unittest
from unittest.mock import patch

from corpcheck.ingestion.metadata import (
    UNKNOWN_SECTOR,
    fetch_upstream_company_metadata,
    is_unresolved_company_name,
    is_unresolved_sector,
    resolve_company_metadata,
)


class MetadataResolutionTests(unittest.TestCase):
    def tearDown(self) -> None:
        fetch_upstream_company_metadata.cache_clear()

    def test_in_universe_ticker_uses_upstream_name_and_normalized_sector(self) -> None:
        metadata = resolve_company_metadata(
            "AMD",
            {
                "longName": "Advanced Micro Devices, Inc.",
                "sector": "Technology",
                "industry": "Semiconductors",
            },
        )
        self.assertEqual(metadata["name"], "Advanced Micro Devices, Inc.")
        self.assertEqual(metadata["sector"], "tech")

    def test_out_of_universe_ticker_uses_upstream_metadata(self) -> None:
        metadata = resolve_company_metadata(
            "AXP",
            {
                "longName": "American Express Company",
                "sector": "Financial Services",
                "industry": "Credit Services",
            },
        )
        self.assertEqual(metadata["name"], "American Express Company")
        self.assertEqual(metadata["sector"], "banking")

    def test_missing_upstream_data_falls_back_to_config(self) -> None:
        metadata = resolve_company_metadata("MSFT", {})
        self.assertEqual(metadata["name"], "Microsoft Corporation")
        self.assertEqual(metadata["sector"], "tech")

    def test_unknown_ticker_without_metadata_is_deterministic(self) -> None:
        metadata = resolve_company_metadata("ZZZZ", {})
        self.assertEqual(metadata["name"], "ZZZZ")
        self.assertEqual(metadata["sector"], UNKNOWN_SECTOR)

    def test_fetch_upstream_metadata_handles_errors(self) -> None:
        with patch("corpcheck.ingestion.metadata.yf.Ticker", side_effect=RuntimeError("boom")):
            metadata = resolve_company_metadata("ERR")
        self.assertEqual(metadata["name"], "ERR")
        self.assertEqual(metadata["sector"], UNKNOWN_SECTOR)

    def test_unresolved_helpers(self) -> None:
        self.assertTrue(is_unresolved_company_name("AMD", "AMD"))
        self.assertFalse(is_unresolved_company_name("Advanced Micro Devices, Inc.", "AMD"))
        self.assertTrue(is_unresolved_sector("unknown"))
        self.assertTrue(is_unresolved_sector(""))
        self.assertFalse(is_unresolved_sector("tech"))


if __name__ == "__main__":
    unittest.main()
