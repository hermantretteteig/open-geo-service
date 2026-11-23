"""Table dataset templates: data from external systems that arrives as rows (CLIM-1556).

A table template is a YAML entry in a `tables/` folder, read from core
(`open_geo_service/plugins/tables/`), then installed plugin packages, then the instance
`plugins_dir`; a later template with the same id replaces an earlier one. It names its schema
(`tables.schemas`), the default aggregation schema when it names none, and a plugin
(`tables.plugin.BaseTablePlugin`) that returns the rows of one period at a time. Its source,
licence and providers are the table's, as a raster or vector template's are its dataset's.
"""

from __future__ import annotations

import functools
import importlib.resources
import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from open_geo_service import config as api_config
from open_geo_service.shared.urls import is_segment_safe_id
from open_geo_service.tables.schemas import DEFAULT_SCHEMA, TableSchema, get_schema

logger = logging.getLogger(__name__)

PERIOD_TYPES = ("daily", "weekly", "dekadal", "monthly", "yearly")
"""The period types a table template can declare: those whose periods the store keeps as their first day."""


def list_table_templates() -> list[dict[str, Any]]:
    """Every table template, each with its `schema` filled in."""
    return list(_templates_cached(*_cache_key()).values())


def get_table_template(template_id: str) -> dict[str, Any] | None:
    """The table template with *template_id*, or None when no table template declares it."""
    return _templates_cached(*_cache_key()).get(template_id)


def is_table_template(template: dict[str, Any]) -> bool:
    """Whether *template* is a table template: every loaded one has a `schema`, which no raster template has."""
    return "schema" in template


def template_schema(template: dict[str, Any]) -> TableSchema:
    """The schema a table template's rows follow."""
    return get_schema(str(template["schema"]))


def reset_table_template_caches() -> None:
    """Forget the parsed templates, for a test that changes `plugins_dir` or installs a plugin."""
    _builtin.cache_clear()
    _templates_cached.cache_clear()


def _cache_key() -> tuple[str | None, str | None]:
    config = api_config.get_config() or {}
    path = api_config.get_config_path()
    plugins_dir = config.get("plugins_dir")
    return (str(path) if path else None, str(plugins_dir) if plugins_dir is not None else None)


@functools.lru_cache(maxsize=8)
def _templates_cached(config_path: str | None, plugins_dir: str | None) -> dict[str, dict[str, Any]]:
    from open_geo_service.plugin_discovery import iter_plugin_subdirs

    merged = {str(template["id"]): template for template in _builtin()}
    for plugin_name, _package, folder in iter_plugin_subdirs("tables"):
        for template in _parse_files(folder.iterdir(), source=f"plugin '{plugin_name}'"):
            merged[str(template["id"])] = template
    if plugins_dir:
        root = ((Path(config_path).parent if config_path else Path()) / plugins_dir).resolve()
        # As for raster templates: a plugin module in `plugins_dir` is imported by its path below it.
        if root.is_dir() and str(root) not in sys.path:
            sys.path.append(str(root))
        folder = root / "tables"
        if folder.is_dir():
            for template in _parse_files(sorted(folder.iterdir()), source=str(folder)):
                merged[str(template["id"])] = template
    return merged


@functools.lru_cache(maxsize=1)
def _builtin() -> tuple[dict[str, Any], ...]:
    folder = importlib.resources.files("open_geo_service") / "plugins" / "tables"
    return tuple(_parse_files(folder.iterdir(), source="core"))


def _parse_files(files: Iterable[Any], *, source: str) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    for file in files:
        if not file.name.endswith((".yaml", ".yml")):
            continue
        declared = yaml.safe_load(file.read_text(encoding="utf-8"))
        if not isinstance(declared, list):
            raise ValueError(f"{source} ({file.name}) must contain a list of table templates")
        templates += [_validated(item, source=f"{source} ({file.name})") for item in declared]
    return templates


def _validated(template: object, *, source: str) -> dict[str, Any]:
    """*template* with its schema filled in, or ValueError naming *source* and what is wrong."""
    if not isinstance(template, dict):
        raise ValueError(f"{source} contains a table template that is not a mapping")
    template_id = template.get("id")
    if not isinstance(template_id, str) or not is_segment_safe_id(template_id):
        raise ValueError(f"{source} contains a table template with a missing or invalid id {template_id!r}")
    where = f"Table template '{template_id}' in {source}"
    name = template.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{where} must define a non-empty 'name'")
    schema_id = template.get("schema", DEFAULT_SCHEMA)
    if not isinstance(schema_id, str):
        raise ValueError(f"{where} names its schema by id; declare the schema in a tables/schemas/ file")
    try:
        schema = get_schema(schema_id)
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from None
    # Ingest and sync go period by period, so the rows need a period column to be sorted into periods.
    if schema.represented_by("period_start") is None:
        raise ValueError(f"{where}: schema '{schema.id}' has no column representing period_start")
    if template.get("period_type") not in PERIOD_TYPES:
        raise ValueError(f"{where} must define a period_type, one of {list(PERIOD_TYPES)}")
    collection = template.get("collection")
    if schema.represented_by("collection_id") is not None:
        if not isinstance(collection, str) or not collection:
            raise ValueError(
                f"{where}: schema '{schema.id}' reports on features; name their feature collection in 'collection'"
            )
    elif collection is not None:
        raise ValueError(f"{where} names a collection, but schema '{schema.id}' has no feature column")
    ingestion = template.get("ingestion")
    plugin = ingestion.get("plugin") if isinstance(ingestion, dict) else None
    if not isinstance(plugin, str) or not plugin.strip():
        raise ValueError(f"{where} must define 'ingestion.plugin', the plugin that fetches its rows")
    params = ingestion.get("params") if isinstance(ingestion, dict) else None
    if params is not None and not isinstance(params, dict):
        raise ValueError(f"{where} has 'ingestion.params' that is not a mapping")
    return {**template, "schema": schema.id}
