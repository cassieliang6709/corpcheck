import re

import pytest
from evaluation.table_representation_profiles import (
    BASELINE_SOURCE_COMMIT,
    REPRESENTATION_PROFILES,
    cleaner_for_profile,
    profile_source_fingerprint,
)

from corpcheck.ingestion.processors.html_cleaner import HTMLCleaner


def _serialized(profile: str, html: str) -> str | None:
    cleaner = cleaner_for_profile(profile, "10-K")
    table = cleaner._load_soup(html).find("table")
    assert table is not None
    return cleaner._serialize_table(table, 1)


def test_semantic_title_profile_preserves_baseline_and_matches_candidate() -> None:
    html = """
    <table>
      <tr><td>Consolidated Statements of Income</td><td>2023</td><td>2022</td></tr>
      <tr><td>Revenue</td><td>100</td><td>90</td></tr>
    </table>
    """

    assert _serialized("baseline", html) == (
        "[TABLE] Table 1\n"
        "[ROW] Consolidated Statements of Income | 2023 | 2022\n"
        "[ROW] Revenue | 100 | 90\n"
        "[/TABLE]"
    )
    candidate = _serialized("candidate", html)
    current = HTMLCleaner(filing_type="10-K")._serialize_table(
        HTMLCleaner(filing_type="10-K")._load_soup(html).find("table"),
        1,
    )
    assert candidate == current
    assert candidate == (
        "[TABLE] Consolidated Statements of Income\n"
        "[HEADER] 2023 | 2022\n"
        "[ROW] Revenue | 100 | 90\n"
        "[/TABLE]"
    )


def test_td_header_profile_preserves_baseline_and_matches_candidate() -> None:
    html = """
    <table>
      <caption>Consolidated Balance Sheets</caption>
      <tr><td>2023</td><td>2022</td></tr>
      <tr><td>Total assets</td><td>250</td><td>225</td></tr>
    </table>
    """

    assert _serialized("baseline", html) == (
        "[TABLE] Consolidated Balance Sheets\n"
        "[ROW] 2023 | 2022\n"
        "[ROW] Total assets | 250 | 225\n"
        "[/TABLE]"
    )
    candidate = _serialized("candidate", html)
    current_cleaner = HTMLCleaner(filing_type="10-K")
    current_table = current_cleaner._load_soup(html).find("table")
    assert current_table is not None
    assert candidate == current_cleaner._serialize_table(current_table, 1)
    assert candidate == (
        "[TABLE] Consolidated Balance Sheets\n"
        "[HEADER] 2023 | 2022\n"
        "[ROW] Total assets | 250 | 225\n"
        "[/TABLE]"
    )


def test_profiles_and_invalid_profile() -> None:
    assert REPRESENTATION_PROFILES == ("baseline", "candidate")
    assert BASELINE_SOURCE_COMMIT == "3774e69ce74d7c010aecbb07b5ce5faebf2f1d95"
    assert type(cleaner_for_profile("baseline", "10-Q")) is not HTMLCleaner
    assert type(cleaner_for_profile("candidate", "10-Q")) is HTMLCleaner
    with pytest.raises(ValueError, match="unknown representation profile"):
        cleaner_for_profile("other", "10-K")
    with pytest.raises(ValueError, match="unknown representation profile"):
        profile_source_fingerprint("other")


def test_profile_fingerprints_are_stable_valid_and_different() -> None:
    fingerprints = {
        profile: profile_source_fingerprint(profile) for profile in REPRESENTATION_PROFILES
    }

    assert fingerprints == {
        profile: profile_source_fingerprint(profile) for profile in REPRESENTATION_PROFILES
    }
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in fingerprints.values())
    assert fingerprints["baseline"] != fingerprints["candidate"]
