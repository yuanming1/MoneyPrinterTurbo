import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
SCRIPT_PATH = ROOT / "scripts" / "import_recap_catalog.py"
SPEC = importlib.util.spec_from_file_location("import_recap_catalog", SCRIPT_PATH)
import_recap_catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(import_recap_catalog)


class QingqueImporterTests(unittest.TestCase):
    def setUp(self):
        self.fixture_html = (ROOT / "test" / "fixtures" / "qingque_catalog_table.html").read_text(
            encoding="utf-8"
        )

    def test_parser_extracts_required_columns_and_baidu_password(self):
        rows = import_recap_catalog.parse_qingque_table(self.fixture_html)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["title"], "小小年岁平安顺遂")
        self.assertEqual(rows[0]["content_type"], "animated")
        self.assertEqual(rows[0]["download"]["password"], "pu2e")
        self.assertEqual(rows[1]["content_type"], "live_short_drama")
        self.assertEqual(rows[1]["access"], "paid")

    def test_diff_reports_added_changed_and_removed_items(self):
        existing = {
            "items": [
                {"id": "one", "title": "One"},
                {"id": "removed", "title": "Removed"},
            ]
        }
        imported = [
            {"id": "one", "title": "Changed"},
            {"id": "added", "title": "Added"},
        ]

        diff = import_recap_catalog.build_diff(existing, imported)

        self.assertEqual(diff["added"], ["added"])
        self.assertEqual(diff["changed"], ["one"])
        self.assertEqual(diff["removed"], ["removed"])

    def test_fetch_decodes_utf8_bytes_when_the_server_omits_a_charset(self):
        response = SimpleNamespace(
            content=self.fixture_html.encode("utf-8"),
            text="garbled response text",
            raise_for_status=lambda: None,
        )
        args = SimpleNamespace(html_file=None, source_url="https://docs.qingque.cn/example")

        with patch.object(import_recap_catalog.requests, "get", return_value=response):
            document = import_recap_catalog._load_document(args)

        self.assertIn("任务名称", document)

    def test_fetch_retries_when_qingque_returns_an_ssr_failure_page(self):
        failed_response = SimpleNamespace(
            content=b"<html><script>window.ssrStatInfo={isSuccess:false}</script></html>",
            raise_for_status=lambda: None,
        )
        successful_response = SimpleNamespace(
            content=self.fixture_html.encode("utf-8"),
            raise_for_status=lambda: None,
        )
        args = SimpleNamespace(html_file=None, source_url="https://docs.qingque.cn/example")

        with patch.object(
            import_recap_catalog.requests,
            "get",
            side_effect=[failed_response, successful_response],
        ) as get:
            document = import_recap_catalog._load_document(args)

        self.assertIn("素材链接", document)
        self.assertEqual(get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
