"""WFP market food prices, from the dataset WFP publishes for each country on HDX (CLIM-1556).

Rows report on the markets of `plugins/vectors/wfp.py` by WFP's `market_id`. The price file is
read whole once per ingestion or sync, a few MB per country.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import date
from typing import Any

from open_geo_service.tables.plugin import FieldsTablePlugin

COLUMNS = {
    "feature_id": "market_id",
    "feature_name": "market",
    "period_start": "date",
    "commodity": "commodity",
    "commodity_category": "category",
    "price_type": "pricetype",
    "measure": "unit",
    "price": "price",
    "currency": "currency",
    "usd_price": "usdprice",
}
"""The `market_prices` schema's columns, and the field of WFP's price file each is read from."""


class WfpPricesPlugin(FieldsTablePlugin):
    """WFP's market prices for *country*, the instance extent's country by default."""

    def __init__(self, *, country: str | None = None, **context: Any) -> None:
        super().__init__(columns=COLUMNS, **context)
        self.country = country

    def records(self, start: date | None, end: date | None) -> Iterator[tuple[str, Mapping[str, Any]]]:
        from open_geo_service.plugins.vectors.wfp import wfp_csv

        for line, row in enumerate(wfp_csv("wfp_food_prices_", self.country), start=2):
            yield f"WFP prices line {line}", row
