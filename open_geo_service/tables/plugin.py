"""Base classes for table dataset plugins (CLIM-1556).

A table plugin fetches rows from an external system one period at a time, as a raster plugin
fetches grids. It implements two methods:

- `periods(start, end)`: the ordered period ids the source has from *start* to *end*, inclusive,
  in the template's period type (`2026-01` for monthly); either bound may be None, meaning from
  the first or to the last period the source has.
- `fetch_period(period_id)`: the rows of that period, as mappings from column name to value.

Open Geo Service writes the rows through the table store, which checks them against the
template's schema. The template's `ingestion.params` are passed to the constructor as keyword
arguments, beside `schema` and `period_type`, which come from the template itself.

A source that returns records of named fields, a CSV file or a JSON API, subclasses
`FieldsTablePlugin` and implements only `records`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime
from typing import Any

from open_geo_service.shared.time import (
    datetime_to_period_string,
    normalize_period_string,
    parse_period_string_to_datetime,
)
from open_geo_service.tables.schemas import Column, TableSchema


class BaseTablePlugin:
    """A table plugin; subclass it and implement `periods` and `fetch_period`."""

    def __init__(self, *, schema: TableSchema, period_type: str, **params: Any) -> None:
        self.schema = schema
        self.period_type = period_type
        self.params: dict[str, Any] = params

    def periods(self, start: str | None, end: str | None) -> list[str]:
        """The ordered period ids the source has from *start* to *end*, inclusive."""
        raise NotImplementedError("Table plugins must implement periods()")

    def fetch_period(self, period_id: str) -> Iterable[Mapping[str, Any]]:
        """The rows of one period, each naming its columns as the template's schema does."""
        raise NotImplementedError("Table plugins must implement fetch_period()")


class FieldsTablePlugin(BaseTablePlugin):
    """Rows from records of named fields, read once per `periods` call and handed out a period at a time.

    Each schema column is read from the field of the same name, or of the name *columns* maps it
    to; *constants* gives a column one value for every row, such as the unit. Values are read as
    their column's type from their text, so a period may be a date or a period id (`2026-01`,
    `2026-W01`, `2026`) of any period type: the store keeps each row at the start of the
    template's period that holds it. An empty value is a missing one.
    """

    def __init__(
        self,
        *,
        columns: Mapping[str, str] | None = None,
        constants: Mapping[str, Any] | None = None,
        **context: Any,
    ) -> None:
        super().__init__(**context)
        self.columns = dict(columns or {})
        declared = {column.name: column for column in self.schema.columns}
        if unknown := sorted((set(self.columns) | set(constants or {})) - set(declared)):
            raise ValueError(f"columns or constants name columns schema '{self.schema.id}' lacks: {unknown}")
        # Read as a field is, so `period_start: 2020` gives a source without a period field its year.
        self.constants = {
            name: _typed(str(value), declared[name], "constants") for name, value in (constants or {}).items()
        }
        self._by_period: dict[str, list[dict[str, Any]]] = {}

    def records(self, start: date | None, end: date | None) -> Iterable[tuple[str, Mapping[str, Any]]]:
        """Each record from the first day of *start*'s period to the last of *end*'s, and where it is.

        Where it is, such as `CSV line 4`, names it in an error. A source that cannot select by
        period may return more; rows outside the bounds are left out.
        """
        raise NotImplementedError("Fields table plugins must implement records()")

    def periods(self, start: str | None, end: str | None) -> list[str]:
        # Period ids of one type have one width, so they sort and compare as text.
        first = normalize_period_string(start, self.period_type) if start else None
        last = normalize_period_string(end, self.period_type) if end else None
        bounds = [parse_period_string_to_datetime(period).date() if period else None for period in (first, last)]
        self._by_period = self._grouped(self.records(bounds[0], _last_day(bounds[1], self.period_type)))
        return [
            period
            for period in sorted(self._by_period)
            if (first is None or period >= first) and (last is None or period <= last)
        ]

    def fetch_period(self, period_id: str) -> list[dict[str, Any]]:
        return self._by_period.get(period_id, [])

    def _grouped(self, records: Iterable[tuple[str, Mapping[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
        read = [
            (column, self.columns.get(column.name, column.name))
            for column in self.schema.columns
            if column.name not in self.constants
        ]
        period_column = str(self.schema.represented_by("period_start"))
        by_period: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for where, record in records:
            if missing := sorted(set(self.columns.values()) - set(record)):
                raise ValueError(f"{where}: no fields {missing}")
            row = dict(self.constants)
            for column, field in read:
                if field in record:
                    row[column.name] = _typed(record[field], column, where)
            period = row.get(period_column)
            if not isinstance(period, date):
                raise ValueError(f"{where}: no period in '{self.columns.get(period_column, period_column)}'")
            moment = datetime(period.year, period.month, period.day)
            by_period[datetime_to_period_string(moment, self.period_type)].append(row)
        return dict(by_period)


def _last_day(start: date | None, period_type: str) -> date | None:
    from open_geo_service.tables.store import end_of_period

    return end_of_period(start, period_type) if start is not None else None


def _typed(value: object, column: Column, where: str) -> Any:
    """*value* as *column*'s type, read from its text; None for an empty one, or ValueError naming *where*."""
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        if column.type == "string":
            return text
        if column.type == "double":
            return float(text)
        if column.type in ("int32", "int64"):
            return int(text)
        if column.type == "bool":
            return {"true": True, "1": True, "false": False, "0": False}[text.lower()]
        if column.type == "date":
            return parse_period_string_to_datetime(text).date()
        moment = datetime.fromisoformat(text)
        return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
    except (KeyError, ValueError):
        raise ValueError(f"{where}: {column.name} {text!r} is not a {column.type}") from None
