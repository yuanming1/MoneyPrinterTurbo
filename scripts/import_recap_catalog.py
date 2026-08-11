"""Review and import the Qingque recap catalog without downloading media."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests


ROOT_DIR = Path(__file__).resolve().parent.parent
CATALOG_ID = "qingque-kuaishou"
CATALOG_PATH = ROOT_DIR / "webui" / "data" / "recap_catalogs" / f"{CATALOG_ID}.json"
CATALOG_SOURCE_URL = "https://docs.qingque.cn/s/home/eZQBGcZW8Gc2gdfFbwROXAN_v"
_HEADERS = ("任务名称", "短剧/真人剧", "付费/免费", "剧场", "素材链接")
_SPACE_PATTERN = re.compile(r"\s+")
_PAN_URL_PATTERN = re.compile(r"https://(?:www\.)?pan\.baidu\.com/[^\s<]+")
_PASSWORD_PATTERN = re.compile(r"提取码\s*[:：]?\s*([A-Za-z0-9_-]+)")


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._table_depth = 0
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        if tag == "table":
            self._table_depth += 1
            return
        if self._table_depth != 1:
            return
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            self._current_cell = []

    def handle_data(self, data: str):
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str):
        if tag == "table":
            self._table_depth -= 1
            return
        if self._table_depth != 1:
            return
        if tag in {"td", "th"} and self._current_cell is not None:
            self._current_row.append(_normalize_text("".join(self._current_cell)))
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = None


def _normalize_text(value: str) -> str:
    return _SPACE_PATTERN.sub(" ", value).strip()


def _parse_download(value: str) -> dict[str, str]:
    match = _PAN_URL_PATTERN.search(value)
    if not match:
        raise ValueError("catalog row does not contain a Baidu Pan HTTPS URL")
    url = match.group().rstrip("，。；;)")
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {"pan.baidu.com", "www.pan.baidu.com"}:
        raise ValueError("catalog row has an invalid Baidu Pan URL")
    query_password = parse_qs(parsed.query).get("pwd", [""])[0]
    text_match = _PASSWORD_PATTERN.search(value)
    text_password = text_match.group(1) if text_match else ""
    return {
        "provider": "baidu_pan",
        "url": url,
        "password": text_password or query_password,
    }


def _content_type(value: str) -> str:
    if "漫" in value:
        return "animated"
    if "真人" in value or "短剧" in value:
        return "live_short_drama"
    raise ValueError(f"unsupported Qingque content type: {value}")


def _access(value: str) -> str:
    if "免费" in value:
        return "free"
    if "付费" in value:
        return "paid"
    raise ValueError(f"unsupported Qingque access value: {value}")


def parse_qingque_table(document: str) -> list[dict[str, Any]]:
    """Parse a server-rendered Qingque table into validated catalog rows."""
    parser = _TableParser()
    parser.feed(document)
    parser.close()

    for row_index, headers in enumerate(parser.rows):
        normalized_headers = [_normalize_text(value) for value in headers]
        if not all(header in normalized_headers for header in _HEADERS):
            continue
        column = {header: normalized_headers.index(header) for header in _HEADERS}
        rows: list[dict[str, Any]] = []
        for cells in parser.rows[row_index + 1 :]:
            if len(cells) <= max(column.values()):
                continue
            title = _normalize_text(cells[column["任务名称"]])
            if not title:
                continue
            rows.append(
                {
                    "title": title,
                    "content_type": _content_type(cells[column["短剧/真人剧"]]),
                    "access": _access(cells[column["付费/免费"]]),
                    "theater": _normalize_text(cells[column["剧场"]]),
                    "download": _parse_download(cells[column["素材链接"]]),
                    "rights": {"status": "requires_confirmation"},
                }
            )
        if rows:
            return rows
    raise ValueError("could not find the Qingque recap catalog table")


def _item_key(item: dict[str, Any]) -> tuple[str, str, str]:
    download = item.get("download") or {}
    return (
        str(item.get("title", "")),
        str(item.get("theater", "")),
        str(download.get("url", "")),
    )


def _new_item_id(item: dict[str, Any]) -> str:
    fingerprint = "\x00".join(_item_key(item)).encode("utf-8")
    return f"{CATALOG_ID}:{hashlib.sha256(fingerprint).hexdigest()[:16]}"


def build_catalog(
    rows: list[dict[str, Any]],
    existing_catalog: dict[str, Any] | None = None,
    *,
    source_url: str = CATALOG_SOURCE_URL,
) -> dict[str, Any]:
    existing_ids = {
        _item_key(item): item["id"]
        for item in (existing_catalog or {}).get("items", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    items = []
    seen_ids: set[str] = set()
    for row in rows:
        item = dict(row)
        item_id = existing_ids.get(_item_key(item), _new_item_id(item))
        if item_id in seen_ids:
            raise ValueError(f"duplicate Qingque catalog item: {item_id}")
        seen_ids.add(item_id)
        item["id"] = item_id
        items.append(item)

    return {
        "schema_version": 1,
        "catalog_id": CATALOG_ID,
        "title": "快手萤光计划短剧&漫剧达人二创片单",
        "source_url": source_url,
        "refreshed_at": datetime.now(timezone.utc).date().isoformat(),
        "items": items,
    }


def build_diff(existing_catalog: dict[str, Any], imported_items: list[dict[str, Any]]) -> dict[str, list[str]]:
    existing = {
        item["id"]: item
        for item in existing_catalog.get("items", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    imported = {
        item["id"]: item
        for item in imported_items
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    return {
        "added": sorted(imported.keys() - existing.keys()),
        "changed": sorted(
            item_id
            for item_id in imported.keys() & existing.keys()
            if imported[item_id] != existing[item_id]
        ),
        "removed": sorted(existing.keys() - imported.keys()),
    }


def _load_catalog(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"items": []}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_catalog(path: Path, catalog: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(catalog, output, ensure_ascii=False, indent=2)
            output.write("\n")
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_document(args: argparse.Namespace) -> str:
    if args.html_file:
        return Path(args.html_file).read_text(encoding="utf-8")
    for _ in range(3):
        response = requests.get(
            args.source_url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        response.raise_for_status()
        document = response.content.decode("utf-8")
        if all(header in document for header in _HEADERS):
            return document
    raise ValueError("Qingque did not return the expected SSR catalog table")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import the Qingque recap catalog without downloading media."
    )
    parser.add_argument("--source-url", default=CATALOG_SOURCE_URL)
    parser.add_argument("--html-file")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--write", action="store_true")
    args = parser.parse_args()

    existing_catalog = _load_catalog(CATALOG_PATH)
    imported_catalog = build_catalog(
        parse_qingque_table(_load_document(args)),
        existing_catalog,
        source_url=args.source_url,
    )
    diff = build_diff(existing_catalog, imported_catalog["items"])
    print(json.dumps(diff, ensure_ascii=False, indent=2))
    if args.write:
        _write_catalog(CATALOG_PATH, imported_catalog)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
