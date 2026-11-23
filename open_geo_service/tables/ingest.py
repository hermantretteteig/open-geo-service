"""Ingest and sync a table dataset from its template's plugin (CLIM-1556).

The plugin returns the rows of one period at a time. They are collected and written through the
table store in one write, which checks every row against the template's schema before anything
is stored: a refused row leaves the table as it was, and the table's file is rewritten once per
run rather than once per period.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from open_geo_service.ingestions.schemas import ArtifactRecord, SyncAction, SyncDetail, SyncKind
from open_geo_service.jobs.models import JobCancelledError
from open_geo_service.shared.time import next_period_string, normalize_period_string
from open_geo_service.tables import store
from open_geo_service.tables.plugin import BaseTablePlugin
from open_geo_service.tables.templates import template_schema


def ingest_table(
    template: dict[str, Any],
    *,
    start: str | None,
    end: str | None,
    publish: bool,
    on_progress: Callable[[int | None, int | None, str | None], None] | None = None,
    is_cancel_requested: Callable[[], bool] | None = None,
    on_update_planned: Callable[[str | None], None] | None = None,
    plugin: BaseTablePlugin | None = None,
    periods: list[str] | None = None,
) -> ArtifactRecord:
    """Fetch the periods from *start* to *end* and write their rows to the template's table dataset.

    Either bound may be None: from the first, or to the last, period the source has. Rows of a
    period the table already holds replace the stored ones. *plugin* and *periods* are a sync's,
    which has asked the source already. Raises ValueError for rows the store refuses.
    """
    plugin = plugin or load_plugin(template)
    if periods is None:
        periods = plugin.periods(start, end)
    if not periods:
        raise ValueError(
            f"data source '{template['id']}' has no periods from {start or 'its first'} to {end or 'its last'}"
        )
    rows: list[Mapping[str, Any]] = []
    for done, period in enumerate(periods):
        if is_cancel_requested is not None and is_cancel_requested():
            raise JobCancelledError("Table ingest cancelled")
        if on_progress is not None:
            on_progress(done, len(periods), f"Fetching {period}")
        rows += plugin.fetch_period(period)
    existing = store.latest_record(str(template["id"]))
    current_end = existing.coverage.temporal.end if existing is not None else None
    if on_update_planned is not None:
        # The end before this update stands only when every fetched period follows it.
        on_update_planned(current_end if current_end is not None and periods[0] > current_end else None)
    if on_progress is not None:
        on_progress(len(periods), len(periods), f"Writing {len(rows)} rows")
    return store.write_rows(
        dataset_id=str(template["id"]),
        rows=rows,
        collection_id=template.get("collection"),
        schema=str(template["schema"]),
        dataset_name=str(template["name"]),
        period_type=str(template["period_type"]),
        publish=publish,
    )


def plan_table_sync(
    template: dict[str, Any], record: ArtifactRecord, end: str | None, plugin: BaseTablePlugin | None = None
) -> SyncDetail:
    """The periods after the table's last one that the source has, through *end* or its last."""
    period_type = str(template["period_type"])
    plugin = plugin or load_plugin(template)
    current_start, current_end = record.coverage.temporal.start, record.coverage.temporal.end
    target = normalize_period_string(end, period_type) if end else None
    periods = plugin.periods(next_period_string(current_end, period_type) if current_end else None, target)
    common: dict[str, Any] = {
        "source_dataset_id": str(template["id"]),
        "sync_kind": SyncKind.TEMPORAL,
        "current_start": current_start,
        "current_end": current_end,
        "target_end_source": "request" if target else "source",
    }
    if not periods:
        return SyncDetail(
            **common,
            action=SyncAction.NO_OP,
            reason="up_to_date",
            message=f"The source has no periods after {current_end}.",
            target_end=target or current_end,
        )
    return SyncDetail(
        **common,
        action=SyncAction.APPEND,
        reason="new_periods",
        message=f"Fetch {len(periods)} new periods, {periods[0]} to {periods[-1]}.",
        target_end=target or periods[-1],
        delta_start=periods[0],
        delta_end=periods[-1],
        periods=periods,
    )


def sync_table(
    template: dict[str, Any],
    record: ArtifactRecord,
    *,
    end: str | None,
    publish: bool,
    on_progress: Callable[[int | None, int | None, str | None], None] | None = None,
) -> tuple[SyncDetail, ArtifactRecord | None]:
    """Fetch and write the periods `plan_table_sync` finds; the record is None when there are none."""
    plugin = load_plugin(template)
    detail = plan_table_sync(template, record, end, plugin)
    if detail.action != SyncAction.APPEND:
        return detail, None
    written = ingest_table(
        template, start=None, end=None, publish=publish, on_progress=on_progress, plugin=plugin, periods=detail.periods
    )
    return detail, written


def load_plugin(template: dict[str, Any]) -> BaseTablePlugin:
    """The template's plugin, given its params, schema and period type."""
    from open_geo_service.shared.dynamic_import import get_dynamic_function

    ingestion = template["ingestion"]
    plugin_class = get_dynamic_function(str(ingestion["plugin"]).strip())
    if not (isinstance(plugin_class, type) and issubclass(plugin_class, BaseTablePlugin)):
        raise ValueError(f"'{ingestion['plugin']}' is not a table plugin, a subclass of BaseTablePlugin")
    return plugin_class(
        schema=template_schema(template), period_type=str(template["period_type"]), **(ingestion.get("params") or {})
    )
