import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import RecapCatalogSelection
from app.services import recap_catalog


def _catalog_payload(*, item_id="qingque-kuaishou:alpha", url=None):
    return {
        "schema_version": 1,
        "catalog_id": "qingque-kuaishou",
        "title": "Qingque Kuaishou recap catalog",
        "source_url": "https://docs.qingque.cn/s/home/example",
        "refreshed_at": "2026-08-11",
        "items": [
            {
                "id": item_id,
                "title": "Alpha",
                "content_type": "animated",
                "access": "free",
                "theater": "Example theater",
                "download": {
                    "provider": "baidu_pan",
                    "url": url or "https://pan.baidu.com/s/example?pwd=abcd",
                    "password": "abcd",
                },
                "rights": {"status": "requires_confirmation"},
            }
        ],
    }


class RecapCatalogTests(unittest.TestCase):
    def _write_catalog(self, directory, payload):
        Path(directory, "qingque-kuaishou.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def test_resolves_a_catalog_item_and_returns_a_sanitized_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_catalog(directory, _catalog_payload())
            with patch.object(recap_catalog, "CATALOG_DIR", Path(directory)):
                item = recap_catalog.resolve_selection(
                    "qingque-kuaishou", "qingque-kuaishou:alpha"
                )

            self.assertEqual(item.title, "Alpha")
            self.assertEqual(item.content_type, "animated")
            self.assertEqual(item.download_password, "abcd")
            self.assertNotIn("download", item.snapshot())
            self.assertNotIn("password", item.snapshot())

    def test_requires_confirmation_before_selecting_an_unverified_item(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_catalog(directory, _catalog_payload())
            selection = RecapCatalogSelection(
                catalog_id="qingque-kuaishou",
                item_id="qingque-kuaishou:alpha",
            )
            with patch.object(recap_catalog, "CATALOG_DIR", Path(directory)):
                with self.assertRaisesRegex(ValueError, "confirm source reuse rights"):
                    recap_catalog.validate_selection(selection)

    def test_returns_a_snapshot_after_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_catalog(directory, _catalog_payload())
            selection = RecapCatalogSelection(
                catalog_id="qingque-kuaishou",
                item_id="qingque-kuaishou:alpha",
                rights_confirmed=True,
            )
            with patch.object(recap_catalog, "CATALOG_DIR", Path(directory)):
                snapshot = recap_catalog.validate_selection(selection)

            self.assertEqual(snapshot["catalog_id"], "qingque-kuaishou")
            self.assertEqual(snapshot["item_id"], "qingque-kuaishou:alpha")
            self.assertTrue(snapshot["rights_confirmed"])
            self.assertIn("rights_confirmed_at", snapshot)
            self.assertNotIn("download", snapshot)

    def test_rejects_non_baidu_pan_download_urls(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_catalog(
                directory,
                _catalog_payload(url="https://example.test/not-an-authorized-provider"),
            )
            with patch.object(recap_catalog, "CATALOG_DIR", Path(directory)):
                with self.assertRaisesRegex(ValueError, "baidu_pan URL"):
                    recap_catalog.load_catalogs()


class QingqueCatalogDataTests(unittest.TestCase):
    def test_migrated_qingque_catalog_keeps_the_existing_playlist_records(self):
        catalog = recap_catalog.load_catalogs()["qingque-kuaishou"]

        self.assertEqual(len(catalog.items), 50)
        self.assertTrue(
            any(
                item.title == "小小年岁平安顺遂"
                and item.content_type == "animated"
                and item.rights_status == "requires_confirmation"
                for item in catalog.items.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
