"""Finding a publisher's dataset for a country on the Humanitarian Data Exchange (CLIM-1556).

WFP and IPC publish each country's data on HDX themselves, as one dataset per country whose name
spells out the country. HDX's package search finds it by the publisher and the country's HDX
group, its alpha-3 code. Shared by the WFP and IPC feature providers and table plugins.
"""

from __future__ import annotations

from open_geo_service.extents.services import get_extent

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
