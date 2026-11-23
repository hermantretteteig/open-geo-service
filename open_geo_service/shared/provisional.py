"""When a period fetched from a preliminary product is due to be fetched again (CLIM-931).

Some sources publish recent periods as a preliminary product and replace them with final values
later: ERA5-Land-T about two months after the month in question. A dataset template declares
how long after a period's end its final version is expected, as ``sync.refresh_after``, an ISO
8601 duration. Every period written to a store carries ``fetched_at``, the instant it was
fetched, so a period is due exactly once: when it was fetched before its final version was
expected, and that moment has since passed. See docs/zarr_and_geozarr.md.

This module is the one place that rule is decided.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from open_geo_service.shared.time import next_period_string, parse_period_string_to_datetime

FETCHED_AT = "fetched_at"
"""Name of the coordinate along a store's time axis holding when each period was fetched."""

FETCHED_AT_ENCODING: dict[str, Any] = {
    # float64 rather than int64: NaN decodes to NaT on its own, while xarray's append path
    # decodes the existing array without masking and fails on an integer NaT sentinel. Seconds
    # are exact in float64 for any instant this code will see.
    "units": "seconds since 1970-01-01",
    "calendar": "proleptic_gregorian",
    "dtype": "float64",
}
"""Pinned so that every append and region write encodes ``fetched_at`` the same way."""

_DURATION = re.compile(
    r"^P(?!$)(?:(?P<years>\d+)Y)?(?:(?P<months>\d+)M)?(?:(?P<weeks>\d+)W)?(?:(?P<days>\d+)D)?"
    r"(?:T(?=\d)(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)


@dataclass(frozen=True)
class RefreshAfter:
    """A parsed ``sync.refresh_after``: calendar months plus a fixed duration."""

    months: int
    delta: timedelta

    def add_to(self, instant: datetime) -> datetime:
        """Return ``instant`` plus this duration, clamping to the end of a shorter month."""
        month_index = instant.month - 1 + self.months
        year = instant.year + month_index // 12
        month = month_index % 12 + 1
        day = min(instant.day, calendar.monthrange(year, month)[1])
        return instant.replace(year=year, month=month, day=day) + self.delta


def parse_refresh_after(value: object) -> RefreshAfter | None:
    """Parse an ISO 8601 duration such as ``P4M``; None when the template declares none.

    Raises:
        ValueError: for anything other than a positive ISO 8601 duration.
    """
    if value is None:
        return None
    match = _DURATION.match(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError(f"sync.refresh_after must be an ISO 8601 duration such as 'P4M', got {value!r}")
    parts = {key: int(number) for key, number in match.groupdict().items() if number is not None}
    window = RefreshAfter(
        months=parts.get("years", 0) * 12 + parts.get("months", 0),
        delta=timedelta(
            weeks=parts.get("weeks", 0),
            days=parts.get("days", 0),
            hours=parts.get("hours", 0),
            minutes=parts.get("minutes", 0),
            seconds=parts.get("seconds", 0),
        ),
    )
    if window.months == 0 and window.delta == timedelta(0):
        raise ValueError(f"sync.refresh_after must be longer than zero, got {value!r}")
    return window


def refresh_after_from_template(dataset: Mapping[str, Any]) -> RefreshAfter | None:
    """The template's current ``sync.refresh_after``, read at sync time and never stored."""
    sync = dataset.get("sync")
    return parse_refresh_after(sync.get("refresh_after")) if isinstance(sync, Mapping) else None


def final_after(period_id: str, period_type: str, window: RefreshAfter) -> datetime:
    """The instant from which a period's final version is expected: its end in UTC plus ``window``.

    All built-in datasets run in UTC; ``utc_offset_hours`` is not applied.
    """
    period_end = parse_period_string_to_datetime(next_period_string(period_id, period_type))
    return window.add_to(period_end)


@dataclass(frozen=True)
class Provisional:
    """A stored period fetched before its final version was expected."""

    period: str
    fetched_at: datetime
    final_after: datetime


def provisional_periods(
    period_ids: Sequence[str],
    fetched_at: Sequence[datetime | None],
    period_type: str,
    window: RefreshAfter,
) -> list[Provisional]:
    """The stored periods whose values are preliminary: fetched before ``final_after``, in order.

    An unknown ``fetched_at`` (None, from a store written before it was recorded) counts as final.
    """
    found: list[Provisional] = []
    for period_id, fetched in zip(period_ids, fetched_at, strict=True):
        if fetched is None:
            continue
        matures = final_after(period_id, period_type, window)
        if _as_utc(fetched) < matures:
            found.append(Provisional(period_id, _as_utc(fetched), matures))
    return found


def refresh_due(
    period_ids: Sequence[str],
    fetched_at: Sequence[datetime | None],
    period_type: str,
    window: RefreshAfter,
    now: datetime,
) -> list[str]:
    """The preliminary periods whose final version is now expected, in stored order.

    A period is due when ``fetched_at < final_after <= now``. Refetched after maturity, its new
    ``fetched_at`` is past ``final_after``, so it is never due again, and a period that matured
    while no sync ran is still due at the next one. Periods inside the window are not due: their
    preliminary values do not change in place, so refetching them would cost one window of
    fetches per sync for nothing.
    """
    now = _as_utc(now)
    return [
        found.period
        for found in provisional_periods(period_ids, fetched_at, period_type, window)
        if found.final_after <= now
    ]


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
