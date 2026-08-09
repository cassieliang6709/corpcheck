import unittest

from corpcheck.ingestion.processors.html_cleaner import HTMLCleaner


class HTMLCleanerTableTests(unittest.TestCase):
    def test_incorporated_annual_report_supplies_financial_statement_tables(self) -> None:
        cleaner = HTMLCleaner(filing_type="10-K", min_section_length=1)
        submission = """
        <DOCUMENT>
        <TYPE>10-K
        <TEXT><html><body>
          <p>ITEM 8. Financial Statements and Supplementary Data</p>
          <p>The financial statements in the Annual Report are incorporated by reference.</p>
          <p>ITEM 9. Changes in and Disagreements with Accountants</p>
          <p>There were no disagreements.</p>
        </body></html>
        </DOCUMENT>
        <DOCUMENT>
        <TYPE>EX-13.1
        <TEXT><html><body>
          <h1>Financial Statements and Supplementary Data</h1>
          <p>Audited consolidated results follow.</p>
          <table>
            <caption>Consolidated Statements of Operations</caption>
            <tr><th>In millions</th><th>2018</th><th>2017</th></tr>
            <tr><td>Total revenues</td><td>194,579</td><td>184,786</td></tr>
          </table>
          <table>
            <caption>Consolidated Balance Sheets</caption>
            <tr><th>In millions</th><th>2018</th><th>2017</th></tr>
            <tr><td>Property and equipment, net</td><td>11,349</td><td>10,292</td></tr>
          </table>
        </body></html>
        </DOCUMENT>
        <DOCUMENT>
        <TYPE>EX-99.1
        <TEXT><html><body><p>Unrelated exhibit secret marker.</p></body></html>
        </DOCUMENT>
        """

        segments = cleaner.clean_text_segments(submission)
        financial_text = "\n".join(
            segment.text
            for segment in segments
            if segment.section_name == "Financial Statements"
        )

        self.assertIn("Total revenues | 194,579 | 184,786", financial_text)
        self.assertIn("Property and equipment, net | 11,349 | 10,292", financial_text)
        self.assertNotIn("secret marker", "\n".join(segment.text for segment in segments))

    def test_annual_report_exhibit_requires_explicit_incorporation(self) -> None:
        cleaner = HTMLCleaner(filing_type="10-K", min_section_length=1)
        submission = """
        <DOCUMENT>
        <TYPE>10-K
        <TEXT><html><body>
          <p>ITEM 8. Financial Statements and Supplementary Data</p>
          <p>The complete financial statements appear in this filing.</p>
        </body></html>
        </DOCUMENT>
        <DOCUMENT>
        <TYPE>ARS
        <TEXT><html><body>
          <h1>Financial Statements and Supplementary Data</h1>
          <p>Consolidated Statements of Operations</p>
          <p>Consolidated Balance Sheets</p>
          <p>Attachment-only marker.</p>
        </body></html>
        </DOCUMENT>
        """

        segments = cleaner.clean_text_segments(submission)

        self.assertNotIn("Attachment-only marker", "\n".join(s.text for s in segments))

    def test_clean_text_serializes_html_table_into_structured_lines(self) -> None:
        cleaner = HTMLCleaner(filing_type="10-K")
        html = """
        <html>
          <body>
            <h1>ITEM 8. Financial Statements</h1>
            <p>Summary of results.</p>
            <table>
              <caption>Consolidated Statements of Operations</caption>
              <tr><th>Year</th><th>Revenue</th></tr>
              <tr><td>2023</td><td>$22,680</td></tr>
              <tr><td>2022</td><td>$23,601</td></tr>
            </table>
          </body>
        </html>
        """

        sections = cleaner.clean_text(html, filing_type="10-K")
        self.assertTrue(sections)

        self.assertTrue(sections)
        fs_text = sections[0][1]
        self.assertIn("[TABLE] Consolidated Statements of Operations", fs_text)
        self.assertIn("[HEADER] Year | Revenue", fs_text)
        self.assertIn("[ROW] 2023 | $22,680", fs_text)
        self.assertIn("[ROW] 2022 | $23,601", fs_text)
        self.assertIn("[/TABLE]", fs_text)

    def test_clean_text_promotes_single_row_item_tables_into_sections(self) -> None:
        cleaner = HTMLCleaner(filing_type="10-K", min_section_length=1)
        html = """
        <html>
          <body>
            <p>INDEX</p>
            <table>
              <tr><td>Item 1.</td><td>Business</td><td>3</td></tr>
              <tr><td>Item 7.</td><td>Management's Discussion and Analysis</td><td>20</td></tr>
            </table>
            <table>
              <tr><td>Item 1.</td><td>Business</td></tr>
            </table>
            <p>Business overview paragraph.</p>
            <table>
              <tr><td>Item 7.</td><td>Management's Discussion and Analysis</td></tr>
            </table>
            <p>MD&A paragraph.</p>
          </body>
        </html>
        """

        sections = cleaner.clean_text(html, filing_type="10-K")
        section_names = [section_name for section_name, _ in sections]

        self.assertIn("Business Description", section_names)
        self.assertIn("MD&A", section_names)
        self.assertNotIn("Full Document", section_names)

    def test_clean_text_falls_back_to_generic_heading_lines(self) -> None:
        cleaner = HTMLCleaner(filing_type="10-K", min_section_length=1)
        html = """
        <html>
          <body>
            <p>Forward-looking preamble.</p>
            <div></div>
            <div></div>
            <div>Risk Factors</div>
            <div></div>
            <p>Risk discussion paragraph.</p>
            <div></div>
            <div></div>
            <div>Controls and Procedures</div>
            <div></div>
            <p>Controls paragraph.</p>
          </body>
        </html>
        """

        sections = cleaner.clean_text(html, filing_type="10-K")
        section_names = [section_name for section_name, _ in sections]

        self.assertIn("Risk Factors", section_names)
        self.assertIn("Controls and Procedures", section_names)
        self.assertNotIn("Full Document", section_names)

    def test_clean_text_maps_business_heading_to_business_description(self) -> None:
        cleaner = HTMLCleaner(filing_type="10-K", min_section_length=1)
        html = """
        <html>
          <body>
            <p>Introductory text.</p>
            <div></div>
            <div></div>
            <div>Business</div>
            <div></div>
            <p>Business overview paragraph.</p>
            <div></div>
            <div></div>
            <div>Risk Factors</div>
            <div></div>
            <p>Risk paragraph.</p>
          </body>
        </html>
        """

        sections = cleaner.clean_text(html, filing_type="10-K")
        section_names = [section_name for section_name, _ in sections]

        self.assertIn("Business Description", section_names)
        self.assertIn("Risk Factors", section_names)

    def test_remove_cover_page_does_not_trim_to_late_exhibit_item(self) -> None:
        cleaner = HTMLCleaner(filing_type="10-Q", min_section_length=1)
        text = (
            "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\n"
            "Washington, D.C. 20549\n"
            "FORM 10-Q\n"
            + ("Cover boilerplate\n" * 400)
            + "\nOverview\nQuarterly overview paragraph.\n"
            + ("Body text\n" * 500)
            + "\nItem 6. Exhibits\nExhibit list paragraph.\n"
        )

        trimmed = cleaner._remove_cover_page(text)

        self.assertEqual(trimmed, text)
        self.assertIn("Overview", trimmed)


class HTMLCleaner10QPartTests(unittest.TestCase):
    """Part I and Part II of a 10-Q both number items from 1."""

    MAYBE_HTML = """
    <html>
      <body>
        <p>PART I &#8212; FINANCIAL INFORMATION</p>
        <p>Item 1. Financial Statements</p>
        <p>Condensed consolidated balance sheet discussion.</p>
        <p>Item 2. Management's Discussion and Analysis</p>
        <p>Revenue increased during the quarter.</p>
        <p>Item 3. Quantitative and Qualitative Disclosures About Market Risk</p>
        <p>Our exposure to interest rate risk is unchanged.</p>
        <p>Item 4. Controls and Procedures</p>
        <p>Disclosure controls were effective as of the period end.</p>
        <p>PART II &#8212; OTHER INFORMATION</p>
        <p>Item 1. Legal Proceedings</p>
        <p>We are party to various claims arising in the ordinary course.</p>
        <p>Item 1A. Risk Factors</p>
        <p>There have been no material changes to our risk factors.</p>
        <p>Item 2. Unregistered Sales of Equity Securities</p>
        <p>No unregistered sales occurred during the quarter.</p>
        <p>Item 3. Defaults Upon Senior Securities</p>
        <p>None reported for the quarter.</p>
        <p>Item 4. Mine Safety Disclosures</p>
        <p>Not applicable to our operations.</p>
        <p>Item 5. Other Information</p>
        <p>No director or officer adopted a trading arrangement.</p>
      </body>
    </html>
    """

    def _section_names(self) -> list[str]:
        cleaner = HTMLCleaner(filing_type="10-Q", min_section_length=1)
        sections = cleaner.clean_text(self.MAYBE_HTML, filing_type="10-Q")
        return [section_name for section_name, _ in sections]

    def test_part_i_items_3_and_4_use_part_i_names(self) -> None:
        section_names = self._section_names()

        self.assertIn(
            "Quantitative and Qualitative Disclosures about Market Risk",
            section_names,
        )
        self.assertIn("Controls and Procedures", section_names)

    def test_part_ii_items_3_and_4_use_part_ii_names(self) -> None:
        section_names = self._section_names()

        self.assertIn("Defaults upon Senior Securities", section_names)
        self.assertIn("Mine Safety Disclosures", section_names)

    def test_part_i_and_part_ii_items_do_not_collide(self) -> None:
        section_names = self._section_names()

        # Every Part I / Part II pair with a shared item number resolves to two
        # distinct names, in document order.
        expected = [
            "Financial Statements",
            "MD&A",
            "Quantitative and Qualitative Disclosures about Market Risk",
            "Controls and Procedures",
            "Legal Proceedings",
            "Risk Factors",
            "Unregistered Sales of Equity Securities",
            "Defaults upon Senior Securities",
            "Mine Safety Disclosures",
            "Other Information",
        ]
        self.assertEqual(section_names, expected)

    def test_unknown_part_falls_back_to_part_i_for_shared_numbers(self) -> None:
        html = """
        <html>
          <body>
            <p>Item 3. Quantitative and Qualitative Disclosures About Market Risk</p>
            <p>Our exposure to interest rate risk is unchanged.</p>
            <p>Item 4. Controls and Procedures</p>
            <p>Disclosure controls were effective as of the period end.</p>
          </body>
        </html>
        """
        cleaner = HTMLCleaner(filing_type="10-Q", min_section_length=1)
        section_names = [
            section_name
            for section_name, _ in cleaner.clean_text(html, filing_type="10-Q")
        ]

        self.assertEqual(
            section_names,
            [
                "Quantitative and Qualitative Disclosures about Market Risk",
                "Controls and Procedures",
            ],
        )


if __name__ == "__main__":
    unittest.main()
