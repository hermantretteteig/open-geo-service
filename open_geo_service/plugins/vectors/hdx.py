"""HDX COD-AB feature provider: a country's common operational administrative boundaries (CLIM-1556).

The Humanitarian Data Exchange publishes each country's COD-AB as a dataset `cod-ab-<iso3>`,
whose GeoJSON download is a zip of one file per admin level. The p-codes in it are the ids
humanitarian tables report on, HDX HAPI's among them. The edge-matched layer (`_em`), which HDX
recommends for most uses, is read: its boundaries are aligned to the international borders.

One refresh downloads the zip once, which ranges from a few MB to over 100 MB per country, and
keeps one level. Countries whose COD-AB HDX has not yet republished in this layout have no
GeoJSON download and are refused. The edition in the files (`version`, such as `v02`) is the
collection's version.
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from open_geo_service.extents.services import get_extent
from open_geo_service.features.providers import feature_provider

PACKAGE_URL = "https://data.humdata.org/api/3/action/package_show"
SEARCH_URL = "https://data.humdata.org/api/3/action/package_search"


def iso3(country: str | None) -> str:
    """*country* (ISO 3166-1 alpha-2 or alpha-3), the instance extent's country by default, as lower-case alpha-3."""
    from iso3166 import countries

    declared = country or (get_extent() or {}).get("country_code")
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError("this data source needs a country; set `country` or the extent's country_code")
    try:
        return str(countries.get(declared.strip()).alpha3).lower()
    except KeyError:
        raise ValueError(f"{declared!r} is not an ISO 3166-1 country code") from None


def country_files(organization: str, name: str, country: str | None) -> dict[str, str]:
    """The download URL of each file, by name, of *organization*'s dataset on HDX for *country*.

    The one dataset whose name matches *name*, a CKAN pattern such as `wfp-food-prices-for-*`, is
    found by the country's HDX group, its alpha-3 code, since dataset names spell out the country.
    """
    import httpx

    code = iso3(country)
    query: dict[str, str | int] = {"fq": f"organization:{organization} AND groups:{code} AND name:{name}", "rows": 2}
    response = httpx.get(SEARCH_URL, params=query, timeout=60, follow_redirects=True)
    response.raise_for_status()
    results = response.json()["result"]["results"]
    if len(results) != 1:
        raise ValueError(f"{organization} publishes no dataset {name} for {code.upper()} on HDX")
    return {str(resource["url"]).rsplit("/", 1)[-1]: str(resource["url"]) for resource in results[0]["resources"]}


def download(url: str) -> str:
    """The text of one file of an HDX dataset."""
    import httpx

    response = httpx.get(url, timeout=120, follow_redirects=True)
    response.raise_for_status()
    return response.content.decode("utf-8-sig")


@feature_provider("hdx_cod_ab")
def cod_ab_features(*, level: int = 2, country: str | None = None) -> tuple[dict[str, Any], str | None]:
    """Admin *level* boundaries of *country* (ISO 3166-1), the instance extent's country by default."""
    import httpx

    code = iso3(country)
    with httpx.Client(timeout=300, follow_redirects=True) as client:
        package = client.get(PACKAGE_URL, params={"id": f"cod-ab-{code}"})
        if package.status_code == 404:
            raise ValueError(f"HDX has no COD-AB dataset for {code.upper()}")
        package.raise_for_status()
        resources = package.json()["result"]["resources"]
        zip_url = next((r["url"] for r in resources if str(r.get("name", "")).endswith(".geojson.zip")), None)
        if zip_url is None:
            raise ValueError(
                f"HDX COD-AB for {code.upper()} has no GeoJSON download; its boundaries are not yet "
                "published in HDX's standard layout"
            )
        member = f"{code}_admin{level}_em.geojson"
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "cod-ab.zip"
            with client.stream("GET", zip_url) as response, archive.open("wb") as file:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    file.write(chunk)
            with zipfile.ZipFile(archive) as opened:
                if member not in opened.namelist():
                    raise ValueError(f"HDX COD-AB for {code.upper()} has no admin level {level}")
                collection = json.loads(opened.read(member))
    keep = [f"adm{each}_{part}" for each in range(level + 1) for part in ("pcode", "name")]
    features = [
        {**feature, "properties": {name: feature["properties"].get(name) for name in keep}}
        for feature in collection["features"]
    ]
    editions = {str(feature["properties"].get("version")) for feature in collection["features"]}
    return {"type": "FeatureCollection", "features": features}, editions.pop() if len(editions) == 1 else None
