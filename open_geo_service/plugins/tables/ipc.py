"""IPC acute food insecurity per analysis area, from the dataset IPC publishes on HDX (CLIM-1556).

Rows report on the areas of `plugins/vectors/ipc.py` by name, from the country's `area_long`
file: one row per area, analysis period and phase. Only *validity* periods are read, `current` by
default: a projection is published again by the next analysis, so projections repeat their keys.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator, Mapping
from datetime import date
from typing import Any

from open_geo_service.tables.plugin import FieldsTablePlugin

COLUMNS = {"feature_id": "Area", "feature_name": "Area", "period_start": "From", "class": "Phase", "value": "Number"}
"""The aggregation schema's columns, and the field of IPC's area file each is read from."""


class IpcAreasPlugin(FieldsTablePlugin):
    """IPC's population per phase per area for *country*, the instance extent's country by default."""

    def __init__(self, *, country: str | None = None, validity: str = "current", **context: Any) -> None:
        super().__init__(columns=COLUMNS, **context)
        self.country = country
        self.validity = validity

    def records(self, start: date | None, end: date | None) -> Iterator[tuple[str, Mapping[str, Any]]]:
        from open_geo_service.plugins.vectors.ipc import ipc_file

        text = ipc_file("ipc_{iso3}_area_long.csv", self.country)
        for line, row in enumerate(csv.DictReader(io.StringIO(text)), start=2):
            if row.get("Validity period") == self.validity:
                yield f"IPC areas line {line}", row
