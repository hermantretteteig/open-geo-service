"""WFP market monitoring: the markets of this instance's country, as points (CLIM-1556).

The World Food Programme publishes each country's market prices on the Humanitarian Data
Exchange itself, as a dataset `wfp-food-prices-for-<country>` with two CSV files: the markets,
with WFP's own `market_id` and coordinates, and the prices, per market, commodity and month
(`plugins/tables/wfp.py`). Prices report on markets by `market_id`, so they need no admin codes;
a market's admin area, where needed, comes from its location. WFP's own admin names are not
used as ids: they can be wrong (Sierra Leone's Bo market is listed in Bonthe district).
"""

from __future__ import annotations

import csv
import io
from typing import Any

from open_geo_service.features.providers import feature_provider
from open_geo_service.shared.hdx import country_files, download


def wfp_csv(prefix: str, country: str | None) -> list[dict[str, str]]:
    """The rows of the file starting with *prefix*, such as `wfp_markets_`, of WFP's dataset for *country*."""
    files = country_files("wfp", "wfp-food-prices-for-*", country)
    url = next((url for name, url in files.items() if name.startswith(prefix)), None)
    if url is None:
        raise ValueError(f"WFP's food price dataset has no {prefix}* file")
    return list(csv.DictReader(io.StringIO(download(url))))


@feature_provider("wfp_markets")
def market_features(*, country: str | None = None) -> dict[str, Any]:
    """The markets WFP monitors in *country*, the instance extent's country by default."""
    features = [
        {
            "type": "Feature",
            "properties": {key: row[key] for key in ("market_id", "market", "admin1", "admin2")},
            "geometry": {"type": "Point", "coordinates": [float(row["longitude"]), float(row["latitude"])]},
        }
        for row in wfp_csv("wfp_markets_", country)
        if row.get("latitude") and row.get("longitude")
    ]
    return {"type": "FeatureCollection", "features": features}
