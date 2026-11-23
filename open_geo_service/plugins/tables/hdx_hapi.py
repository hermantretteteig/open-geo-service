"""Rows from HDX HAPI, the Humanitarian Data Exchange's API of subnational tables (CLIM-1556).

HAPI serves values per admin area and reference period, such as rainfall, baseline population
and conflict events, at https://hapi.humdata.org/docs. One ingestion or sync makes one query
for the requested periods, paged by 10,000 rows, and keeps the rows on the plugin.

HAPI asks every client for an app identifier, which encodes an application name and a contact
email (`/api/v2/encode_app_identifier`). Set it as `HDX_HAPI_APP_IDENTIFIER` in `.env`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import date
from typing import Any

from open_geo_service.tables.plugin import FieldsTablePlugin

BASE_URL = "https://hapi.humdata.org/api/v2"
PAGE_SIZE = 10_000
"""The most rows HAPI returns in one response."""


class HapiTablePlugin(FieldsTablePlugin):
    """A table from one HAPI endpoint, such as `climate/rainfall`, for one country.

    Rows are the *admin_level* (1 or 2) areas of *location_code*, the instance extent's country
    by default, and name their feature by its p-code (`admin2_code`) and their period by its
    start (`reference_period_start`). The template maps the value field, as `columns: {value:
    rainfall}`. *filters* are further query parameters, such as `{aggregation_period: dekad,
    version: final}`, which keep one row per area and period.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        admin_level: int = 2,
        location_code: str | None = None,
        filters: Mapping[str, Any] | None = None,
        columns: Mapping[str, str] | None = None,
        **context: Any,
    ) -> None:
        if admin_level not in (1, 2):
            raise ValueError(f"HAPI admin_level must be 1 or 2, got {admin_level!r}")
        fields = {
            "feature_id": f"admin{admin_level}_code",
            "feature_name": f"admin{admin_level}_name",
            "period_start": "reference_period_start",
        }
        # Every HAPI row has these fields, so the columns representing them read them by default.
        defaults = {
            column: field
            for meaning, field in fields.items()
            if (column := context["schema"].represented_by(meaning)) is not None
        }
        super().__init__(columns={**defaults, **(columns or {})}, **context)
        self.endpoint = endpoint.strip("/")
        self.admin_level = admin_level
        self.location_code = location_code
        self.filters = dict(filters or {})

    def records(self, start: date | None, end: date | None) -> Iterator[tuple[str, Mapping[str, Any]]]:
        import httpx

        from open_geo_service.extents.services import get_extent
        from open_geo_service.plugins.rasters.providers import require_env

        location = self.location_code or (get_extent() or {}).get("country_code")
        if not location:
            raise ValueError("HAPI needs a location_code; set it in the template or the extent's country_code")
        query: dict[str, Any] = {
            **self.filters,
            "app_identifier": require_env(
                "HDX_HAPI_APP_IDENTIFIER",
                hint="Encode an application name and a contact email at https://hapi.humdata.org/docs "
                "(encode_app_identifier).",
            ),
            "location_code": location,
            "admin_level": self.admin_level,
            "output_format": "json",
            "limit": PAGE_SIZE,
        }
        if start is not None:
            query["start_date"] = start.isoformat()
        if end is not None:
            query["end_date"] = end.isoformat()
        with httpx.Client(timeout=60) as client:
            offset = 0
            while True:
                response = client.get(f"{BASE_URL}/{self.endpoint}", params={**query, "offset": offset})
                response.raise_for_status()
                page = response.json()["data"]
                for index, record in enumerate(page, start=offset):
                    yield f"HAPI {self.endpoint} row {index}", record
                if len(page) < PAGE_SIZE:
                    return
                offset += PAGE_SIZE
