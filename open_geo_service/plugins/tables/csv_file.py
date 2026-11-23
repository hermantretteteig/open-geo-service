"""Rows from a CSV file, read from a path or a URL (CLIM-1556)."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator, Mapping
from datetime import date
from pathlib import Path
from typing import Any

from open_geo_service import config as api_config
from open_geo_service.tables.plugin import FieldsTablePlugin


class CsvTablePlugin(FieldsTablePlugin):
    """A table from one CSV file, its columns read by `FieldsTablePlugin`'s rules.

    *path* is a file path, relative to the instance config file, or an http(s) URL. The file is
    read whole once per ingestion or sync.
    """

    def __init__(self, *, path: str, delimiter: str = ",", **context: Any) -> None:
        super().__init__(**context)
        self.path = path
        self.delimiter = delimiter

    def records(self, start: date | None, end: date | None) -> Iterator[tuple[str, Mapping[str, Any]]]:
        reader = csv.DictReader(io.StringIO(self._text()), delimiter=self.delimiter)
        for line, record in enumerate(reader, start=2):
            yield f"CSV line {line}", record

    def _text(self) -> str:
        if self.path.startswith(("http://", "https://")):
            import httpx

            response = httpx.get(self.path, follow_redirects=True, timeout=60)
            response.raise_for_status()
            return response.content.decode("utf-8-sig")
        config_path = api_config.get_config_path()
        path = (config_path.parent if config_path else Path()) / self.path
        return path.read_text(encoding="utf-8-sig")
