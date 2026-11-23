"""IPC analysis areas of this instance's country, as polygons (CLIM-1556).

The Integrated Food Security Phase Classification publishes each country's analyses on the
Humanitarian Data Exchange itself, as a dataset `<country>-acute-food-insecurity-country-data`:
a GeoJSON of the latest analysis's areas and CSV files of every analysis
(`plugins/tables/ipc.py`). The tables name an area only by its name, so the name is the id. The
areas are the latest analysis's; a row of an earlier analysis for an area since renamed or redrawn
has no feature and is refused.
"""

from __future__ import annotations

import json
from typing import Any

from open_geo_service.features.providers import feature_provider
from open_geo_service.shared.hdx import country_files, download

DATASET = "*-acute-food-insecurity-country-data"


def ipc_file(prefix: str, country: str | None) -> str:
    """The text of the file named *prefix* plus the country code, such as `ipc_sle.geojson`."""
    from open_geo_service.shared.hdx import iso3

    files = country_files("ipc", DATASET, country)
    name = prefix.format(iso3=iso3(country))
    if name not in files:
        raise ValueError(f"IPC's dataset has no file {name}")
    return download(files[name])


@feature_provider("ipc_areas")
def area_features(*, country: str | None = None) -> dict[str, Any]:
    """The areas of IPC's latest analysis of *country*, the instance extent's country by default."""
    collection = json.loads(ipc_file("ipc_{iso3}.geojson", country))
    features = [
        {**feature, "properties": {"area": feature["properties"]["title"]}}
        for feature in collection["features"]
        if feature.get("geometry")
    ]
    return {"type": "FeatureCollection", "features": features}
