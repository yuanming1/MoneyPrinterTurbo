"""Validated local catalogs for source videos used by recap jobs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.models.schema import RecapCatalogSelection
from app.utils import utils


CATALOG_DIR = Path(utils.root_dir()) / "webui" / "data" / "recap_catalogs"
_ALLOWED_DOWNLOAD_PROVIDERS = {"baidu_pan"}
_ALLOWED_CONTENT_TYPES = {"animated", "live_short_drama"}
_ALLOWED_ACCESS_VALUES = {"free", "paid"}
_ALLOWED_RIGHTS_STATUSES = {
    "verified",
    "requires_confirmation",
    "blocked",
}


@dataclass(frozen=True)
class CatalogItem:
    catalog_id: str
    catalog_source_url: str
    catalog_refreshed_at: str
    id: str
    title: str
    content_type: str
    access: str
    theater: str
    download_provider: str
    download_url: str
    download_password: str
    rights_status: str
    rights_allowed_platforms: tuple[str, ...]

    def snapshot(self, *, rights_confirmed: bool = False) -> dict[str, Any]:
        """Return task provenance without remote-download secrets."""
        payload: dict[str, Any] = {
            "catalog_id": self.catalog_id,
            "item_id": self.id,
            "title": self.title,
            "content_type": self.content_type,
            "access": self.access,
            "theater": self.theater,
            "catalog_source_url": self.catalog_source_url,
            "catalog_refreshed_at": self.catalog_refreshed_at,
            "rights_status": self.rights_status,
            "rights_confirmed": rights_confirmed,
        }
        if self.rights_allowed_platforms:
            payload["rights_allowed_platforms"] = list(self.rights_allowed_platforms)
        return payload


@dataclass(frozen=True)
class Catalog:
    id: str
    title: str
    source_url: str
    refreshed_at: str
    items: dict[str, CatalogItem]


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"recap catalog {field} must be a non-empty string")
    return value.strip()


def _require_https_url(value: Any, field: str, *, baidu_pan_only: bool = False) -> str:
    url = _require_string(value, field)
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"recap catalog {field} must be an HTTPS URL")
    host = parsed.hostname or ""
    if baidu_pan_only and host not in {"pan.baidu.com", "www.pan.baidu.com"}:
        raise ValueError("recap catalog baidu_pan URL must use pan.baidu.com")
    return url


def _load_item(
    raw_item: Any,
    *,
    catalog_id: str,
    source_url: str,
    refreshed_at: str,
) -> CatalogItem:
    if not isinstance(raw_item, dict):
        raise ValueError("recap catalog item must be an object")

    item_id = _require_string(raw_item.get("id"), "item id")
    content_type = _require_string(raw_item.get("content_type"), "content_type")
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise ValueError(f"recap catalog item has unsupported content_type: {content_type}")
    access = _require_string(raw_item.get("access"), "access")
    if access not in _ALLOWED_ACCESS_VALUES:
        raise ValueError(f"recap catalog item has unsupported access: {access}")

    download = raw_item.get("download")
    if not isinstance(download, dict):
        raise ValueError("recap catalog item download must be an object")
    provider = _require_string(download.get("provider"), "download.provider")
    if provider not in _ALLOWED_DOWNLOAD_PROVIDERS:
        raise ValueError(f"recap catalog item has unsupported download provider: {provider}")
    download_url = _require_https_url(
        download.get("url"), "download.url", baidu_pan_only=provider == "baidu_pan"
    )
    password = download.get("password", "")
    if password is None:
        password = ""
    if not isinstance(password, str):
        raise ValueError("recap catalog download.password must be a string")

    rights = raw_item.get("rights", {})
    if not isinstance(rights, dict):
        raise ValueError("recap catalog item rights must be an object")
    rights_status = _require_string(
        rights.get("status", "requires_confirmation"), "rights.status"
    )
    if rights_status not in _ALLOWED_RIGHTS_STATUSES:
        raise ValueError(f"recap catalog item has unsupported rights.status: {rights_status}")
    platforms = rights.get("allowed_platforms", [])
    if not isinstance(platforms, list) or not all(
        isinstance(platform, str) and platform.strip() for platform in platforms
    ):
        raise ValueError("recap catalog rights.allowed_platforms must be a list of strings")

    return CatalogItem(
        catalog_id=catalog_id,
        catalog_source_url=source_url,
        catalog_refreshed_at=refreshed_at,
        id=item_id,
        title=_require_string(raw_item.get("title"), "item title"),
        content_type=content_type,
        access=access,
        theater=_require_string(raw_item.get("theater"), "item theater"),
        download_provider=provider,
        download_url=download_url,
        download_password=password.strip(),
        rights_status=rights_status,
        rights_allowed_platforms=tuple(platform.strip() for platform in platforms),
    )


def _load_catalog(path: Path) -> Catalog:
    try:
        raw_catalog = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid recap catalog JSON: {path.name}") from exc
    if not isinstance(raw_catalog, dict):
        raise ValueError(f"recap catalog must be an object: {path.name}")
    if raw_catalog.get("schema_version") != 1:
        raise ValueError(f"recap catalog has unsupported schema_version: {path.name}")

    catalog_id = _require_string(raw_catalog.get("catalog_id"), "catalog_id")
    source_url = _require_https_url(raw_catalog.get("source_url"), "source_url")
    refreshed_at = _require_string(raw_catalog.get("refreshed_at"), "refreshed_at")
    raw_items = raw_catalog.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("recap catalog items must be a list")

    items: dict[str, CatalogItem] = {}
    for raw_item in raw_items:
        item = _load_item(
            raw_item,
            catalog_id=catalog_id,
            source_url=source_url,
            refreshed_at=refreshed_at,
        )
        if item.id in items:
            raise ValueError(f"recap catalog has duplicate item id: {item.id}")
        items[item.id] = item

    return Catalog(
        id=catalog_id,
        title=_require_string(raw_catalog.get("title"), "catalog title"),
        source_url=source_url,
        refreshed_at=refreshed_at,
        items=items,
    )


def load_catalogs(directory: Path | None = None) -> dict[str, Catalog]:
    """Load all local catalog files, failing clearly on invalid source data."""
    catalog_directory = directory or CATALOG_DIR
    if not catalog_directory.exists():
        return {}

    catalogs: dict[str, Catalog] = {}
    for path in sorted(catalog_directory.glob("*.json")):
        catalog = _load_catalog(path)
        if catalog.id in catalogs:
            raise ValueError(f"recap catalog has duplicate catalog_id: {catalog.id}")
        catalogs[catalog.id] = catalog
    return catalogs


def list_items(catalog_id: str) -> list[CatalogItem]:
    catalog = _get_catalog(catalog_id)
    return list(catalog.items.values())


def _get_catalog(catalog_id: str) -> Catalog:
    try:
        return load_catalogs()[catalog_id]
    except KeyError as exc:
        raise ValueError(f"recap catalog not found: {catalog_id}") from exc


def resolve_selection(catalog_id: str, item_id: str) -> CatalogItem:
    catalog = _get_catalog(catalog_id)
    try:
        return catalog.items[item_id]
    except KeyError as exc:
        raise ValueError(f"recap catalog item not found: {item_id}") from exc


def validate_selection(selection: RecapCatalogSelection) -> dict[str, Any]:
    """Validate a client selection and return immutable task provenance."""
    item = resolve_selection(selection.catalog_id, selection.item_id)
    if item.rights_status == "blocked":
        raise ValueError("selected recap catalog item is blocked")
    if item.rights_status != "verified" and not selection.rights_confirmed:
        raise ValueError("confirm source reuse rights before generating a recap")

    snapshot = item.snapshot(rights_confirmed=selection.rights_confirmed)
    if selection.rights_confirmed:
        snapshot["rights_confirmed_at"] = datetime.now(timezone.utc).isoformat()
    return snapshot
