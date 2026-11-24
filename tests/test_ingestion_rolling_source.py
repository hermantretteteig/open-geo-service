"""Rolling source ingestion preserves committed history without refetching it."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import xarray as xr
import zarr

from open_geo_service.data_accessor.services.accessor import open_icechunk_dataset
from open_geo_service.ingestions import processes, services, sync_engine
from open_geo_service.ingestions.schemas import ArtifactVersion, CoverageTemporal, SyncAction
from open_geo_service.openeo import execution as openeo_execution
from open_geo_service.pipelines import runner
from open_geo_service.shared.provisional import FETCHED_AT
from open_geo_service.streaming import BaseDatasetPlugin, normalize_period, orchestrator
from open_geo_service.streaming.store import read_fetched_at


class _RollingPlugin(BaseDatasetPlugin):
    def __init__(self) -> None:
        self.available = ["2026-01-01", "2026-01-02"]
        self.queries: list[tuple[str, str]] = []
        self.fetched: list[str] = []
        # A source with a preliminary product: final values are the day plus 100.
        self.final = False

    async def periods(self, start: str, end: str) -> list[str]:
        self.queries.append((start, end))
        return [period for period in self.available if start <= period <= end]

    def fetch_period(self, period_id: str, bbox: list[float], **params: object) -> xr.Dataset:
        assert period_id in self.available, "Retired source periods must not be fetched"
        self.fetched.append(period_id)
        data = xr.DataArray(
            np.full((2, 2), int(period_id[-2:]) + (100 if self.final else 0), dtype="float32"),
            dims=("y", "x"),
            coords={"y": [3.5, 2.5], "x": [1.5, 2.5]},
        )
        return normalize_period(data, variable="precip", period=period_id)


@pytest.fixture
def rolling_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[_RollingPlugin, dict[str, object], Path]:
    plugin = _RollingPlugin()
    dataset: dict[str, object] = {
        "id": "rolling_precip",
        "name": "Rolling precipitation",
        "variable": "precip",
        "period_type": "daily",
        "ingestion": {"plugin": "example.RollingPlugin"},
    }
    store_path = tmp_path / "rolling.icechunk"
    monkeypatch.setattr(services, "_load_streaming_plugin", lambda *args, **kwargs: plugin)
    monkeypatch.setattr(services.downloader, "get_icechunk_path", lambda _: store_path)
    monkeypatch.setattr(services, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(services, "ARTIFACTS_INDEX_PATH", tmp_path / "artifacts" / "records.json")

    return plugin, dataset, store_path


def test_materialization_stamps_a_declared_release_version(
    rolling_store: tuple[_RollingPlugin, dict[str, object], Path],
) -> None:
    """A release template's declared version reaches the persisted record.

    Planner tests build ArtifactRecord directly, so without this the assignment in
    create_artifact could be dropped and every newly materialized release artifact would
    silently carry version=None while those tests still passed.
    """
    _, dataset, _ = rolling_store
    dataset["sync"] = {"kind": "release", "version": {"value": "R2025A", "authority": "worldpop"}}

    artifact = services.create_artifact(
        dataset=dataset,
        start="2026-01-01",
        end="2026-01-02",
        bbox=[1.0, 2.0, 3.0, 4.0],
        country_code=None,
        overwrite=False,
        publish=False,
    )

    assert artifact.version == ArtifactVersion(value="R2025A", authority="worldpop")
    assert services._load_records()[-1].version == ArtifactVersion(value="R2025A", authority="worldpop")


def test_materialization_leaves_a_temporal_dataset_without_a_version(
    rolling_store: tuple[_RollingPlugin, dict[str, object], Path],
) -> None:
    """A temporal dataset has no release identity, however many periods it appends."""
    _, dataset, _ = rolling_store
    dataset["sync"] = {"kind": "temporal"}

    artifact = services.create_artifact(
        dataset=dataset,
        start="2026-01-01",
        end="2026-01-02",
        bbox=[1.0, 2.0, 3.0, 4.0],
        country_code=None,
        overwrite=False,
        publish=False,
    )

    assert artifact.version is None
    assert services._load_records()[-1].version is None


@pytest.mark.parametrize("request_start", ["2026-01-01", "2026-01-04"])
def test_forward_ingestion_preserves_retired_source_history(
    rolling_store: tuple[_RollingPlugin, dict[str, object], Path], request_start: str
) -> None:
    plugin, dataset, store_path = rolling_store

    def ingest(start: str, end: str) -> services.ArtifactRecord:
        return services.create_artifact(
            dataset=dataset,
            start=start,
            end=end,
            bbox=[1.0, 2.0, 3.0, 4.0],
            country_code=None,
            overwrite=False,
            publish=False,
        )

    ingest("2026-01-01", "2026-01-02")
    plugin.available = ["2026-01-03", "2026-01-04"]
    artifact = ingest(request_start, "2026-01-04")

    assert plugin.queries == [("2026-01-01", "2026-01-02"), ("2026-01-03", "2026-01-04")]
    assert plugin.fetched == ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    assert artifact.coverage.temporal == CoverageTemporal(start="2026-01-01", end="2026-01-04")
    assert artifact.request_scope.start == request_start
    assert services._load_records()[-1].coverage == artifact.coverage
    with open_icechunk_dataset(store_path) as stored:
        assert stored["precip"][:, 0, 0].values.tolist() == [1.0, 2.0, 3.0, 4.0]

    # A gap in the missing delta must still fail before any data is written.
    repo = services.open_or_create_repo(store_path)
    snapshot = repo.lookup_branch("main")
    plugin.available = ["2026-01-06"]
    with pytest.raises(services.HTTPException, match="contiguous sequence beginning at 2026-01-05"):
        ingest("2026-01-06", "2026-01-06")

    # Extending backwards requires history that this rolling source has retired.
    plugin.available = ["2026-01-03", "2026-01-04"]
    with pytest.raises(services.HTTPException, match="Source cannot materialize"):
        ingest("2025-12-31", "2026-01-04")
    assert repo.lookup_branch("main") == snapshot
    assert services._load_records()[-1].artifact_id == artifact.artifact_id
    assert len(plugin.fetched) == 4


@pytest.mark.parametrize("publish", [False, True])
def test_complete_store_without_record_is_registered_without_source_access(
    rolling_store: tuple[_RollingPlugin, dict[str, object], Path],
    monkeypatch: pytest.MonkeyPatch,
    publish: bool,
) -> None:
    plugin, dataset, store_path = rolling_store
    services.create_artifact(
        dataset=dataset,
        start="2026-01-01",
        end="2026-01-02",
        bbox=[1.0, 2.0, 3.0, 4.0],
        country_code=None,
        overwrite=False,
        publish=False,
    )
    services.ARTIFACTS_INDEX_PATH.unlink()
    plugin.available = []
    plugin.queries.clear()
    plugin.fetched.clear()
    published: list[str] = []

    def publish_record(artifact_id: str) -> services.ArtifactRecord:
        published.append(artifact_id)
        return services._load_records()[-1]

    def unexpected_ingest(**kwargs: object) -> None:
        raise AssertionError("Complete stores must not run the fetch orchestrator")

    monkeypatch.setattr(services, "publish_artifact_record", publish_record)
    monkeypatch.setattr(services, "run_streaming_ingest_sync", unexpected_ingest)
    record = services.create_artifact(
        dataset=dataset,
        start="2026-01-02",
        end="2026-01-02",
        bbox=[1.0, 2.0, 3.0, 4.0],
        country_code=None,
        overwrite=False,
        publish=publish,
    )
    assert record.coverage.temporal == CoverageTemporal(start="2026-01-01", end="2026-01-02")
    assert record.request_scope.start == "2026-01-02"
    assert services._load_records()[-1].artifact_id == record.artifact_id
    assert plugin.queries == []
    assert plugin.fetched == []
    assert published == ([record.artifact_id] if publish else [])
    with open_icechunk_dataset(store_path) as stored:
        assert stored["precip"][:, 0, 0].values.tolist() == [1.0, 2.0]


def test_forward_append_progress_counts_only_new_periods(
    rolling_store: tuple[_RollingPlugin, dict[str, object], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin, dataset, _ = rolling_store
    progress: list[tuple[int | None, int | None]] = []

    def report(done: int | None, total: int | None, message: str | None) -> None:
        progress.append((done, total))

    def ingest(end: str, on_progress: Callable[[int | None, int | None, str | None], None] | None = None) -> None:
        services.create_artifact(
            dataset=dataset,
            start="2026-01-01",
            end=end,
            bbox=[1.0, 2.0, 3.0, 4.0],
            country_code=None,
            overwrite=False,
            publish=False,
            on_progress=on_progress,
        )

    ingest("2026-01-02")
    plugin.available = ["2026-01-03"]
    original_fetch = plugin.fetch_period
    at_fetch: list[tuple[int | None, int | None]] = []

    def fetch(period_id: str, bbox: list[float], **params: object) -> xr.Dataset:
        at_fetch.append(progress[-1])
        return original_fetch(period_id, bbox, **params)

    monkeypatch.setattr(plugin, "fetch_period", fetch)
    ingest("2026-01-03", report)
    assert at_fetch == [(0, 1)]
    assert progress[-1] == (1, 1)


def test_overwrite_rematerializes_existing_store(
    rolling_store: tuple[_RollingPlugin, dict[str, object], Path],
) -> None:
    plugin, dataset, store_path = rolling_store
    services.create_artifact(
        dataset=dataset,
        start="2026-01-01",
        end="2026-01-02",
        bbox=[1.0, 2.0, 3.0, 4.0],
        country_code=None,
        overwrite=False,
        publish=False,
    )
    plugin.available.append("2026-01-03")
    plugin.fetched.clear()

    artifact = services.create_artifact(
        dataset=dataset,
        start="2026-01-01",
        end="2026-01-03",
        bbox=[1.0, 2.0, 3.0, 4.0],
        country_code=None,
        overwrite=True,
        publish=False,
    )

    assert plugin.fetched == plugin.available
    assert artifact.coverage.temporal == CoverageTemporal(start="2026-01-01", end="2026-01-03")
    with open_icechunk_dataset(store_path) as stored:
        assert stored["precip"][:, 0, 0].values.tolist() == [1.0, 2.0, 3.0]


@pytest.fixture
def refreshing_store(
    rolling_store: tuple[_RollingPlugin, dict[str, object], Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[_RollingPlugin, dict[str, object], Path, dict[str, datetime]]:
    """The rolling store as a source with a preliminary product, synced on a clock the test sets."""
    plugin, dataset, store_path = rolling_store
    dataset["sync"] = {"kind": "temporal", "execution": "append", "refresh_after": "P10D"}
    clock = {"now": datetime(2026, 1, 2, 12, tzinfo=UTC)}
    monkeypatch.setattr(orchestrator, "utc_now", lambda: clock["now"])
    monkeypatch.setattr(sync_engine, "utc_now", lambda: clock["now"])
    monkeypatch.setattr(services.registry_datasets, "get_dataset", lambda _: dataset)
    monkeypatch.setattr(sync_engine, "_artifact_storage_roots", lambda: (tmp_path.resolve(),))
    monkeypatch.setattr(
        sync_engine,
        "_query_available_periods",
        lambda _, start, end: [period for period in plugin.available if start <= period <= end],
    )
    return plugin, dataset, store_path, clock


def _sync(end: str) -> services.SyncResponse:
    return services.sync_dataset(dataset_id="rolling_precip", end=end, publish=False)


def _pipeline_runs(response: services.SyncResponse) -> list[tuple[str, str, str] | None]:
    """The interval a pipeline following the dataset runs for each update event the sync emits."""
    return [runner.update_interval(event.data) for event in processes.sync_update_events("rolling_precip", response)]


def _stored(store_path: Path) -> tuple[list[float], list[datetime | None]]:
    with open_icechunk_dataset(store_path) as stored:
        values = stored["precip"][:, 0, 0].values.tolist()
    return values, read_fetched_at(store_path) or []


def test_a_preliminary_period_is_fetched_again_once_after_its_final_version_is_expected(
    refreshing_store: tuple[_RollingPlugin, dict[str, object], Path, dict[str, datetime]],
) -> None:
    plugin, dataset, store_path, clock = refreshing_store
    ingested_at = clock["now"]
    artifact = services.create_artifact(
        dataset=dataset,
        start="2026-01-01",
        end="2026-01-02",
        bbox=[1.0, 2.0, 3.0, 4.0],
        country_code=None,
        overwrite=False,
        publish=False,
    )
    assert _stored(store_path) == ([1.0, 2.0], [ingested_at, ingested_at])
    # Each day's final version is expected ten days after it ends: 2026-01-12 and 2026-01-13.
    assert [(p.period, p.final_after) for p in artifact.provisional_periods] == [
        ("2026-01-01", datetime(2026, 1, 12, tzinfo=UTC)),
        ("2026-01-02", datetime(2026, 1, 13, tzinfo=UTC)),
    ]

    # Inside the window nothing is due: preliminary values do not change in place.
    clock["now"] = datetime(2026, 1, 11, tzinfo=UTC)
    response = _sync("2026-01-02")
    assert response.status == "up_to_date"
    assert response.sync_detail is not None and response.sync_detail.action == SyncAction.NO_OP
    assert _pipeline_runs(response) == []
    # The window is read from the template at sync time, so shortening it makes a period due
    # without anything written to the store.
    dataset["sync"] = {**cast(dict[str, object], dataset["sync"]), "refresh_after": "P9D"}
    plan = services.plan_sync_dataset(dataset_id="rolling_precip", end="2026-01-02")
    assert (plan.refresh_start, plan.refresh_end) == ("2026-01-01", "2026-01-01")
    dataset["sync"] = {**cast(dict[str, object], dataset["sync"]), "refresh_after": "P10D"}
    assert plugin.fetched == ["2026-01-01", "2026-01-02"]

    # Once the first day's final version is expected, only that day is replaced.
    clock["now"] = refreshed_at = datetime(2026, 1, 12, 6, tzinfo=UTC)
    plugin.final = True
    response = _sync("2026-01-02")
    detail = response.sync_detail
    assert detail is not None
    assert (detail.action, detail.reason) == (SyncAction.APPEND, "provisional_periods_matured")
    assert detail.message == (
        "Data exists for 2026-01-01 through 2026-01-02. Sync will replace 2026-01-01 with final values."
    )
    assert _stored(store_path) == ([101.0, 2.0], [refreshed_at, ingested_at])
    assert response.dataset is not None
    assert [p.period for p in response.dataset.provisional_periods] == ["2026-01-02"]
    # A pipeline following the dataset reruns from the replaced day, so its final value reaches
    # the destination: from the day before, as for an append, which re-sends the last kept period.
    assert _pipeline_runs(response) == [("2025-12-31", "2026-01-02", "appended")]

    # A sync skipped past the second day's maturity still replaces it at the next one, in the
    # same sync as a new day is added.
    clock["now"] = later = datetime(2026, 1, 20, tzinfo=UTC)
    plugin.available.append("2026-01-03")
    response = _sync("2026-01-03")
    assert response.sync_detail is not None
    assert response.sync_detail.message == (
        "Data exists for 2026-01-01 through 2026-01-02. "
        "Sync will replace 2026-01-02 with final values and add 2026-01-03."
    )
    assert _stored(store_path) == ([101.0, 102.0, 103.0], [refreshed_at, later, later])
    assert response.dataset is not None and response.dataset.provisional_periods == []
    assert _pipeline_runs(response) == [("2026-01-01", "2026-01-03", "appended")]
    # Chunked as `t`, rechunked at the end of each sync, so the refresh and the append above
    # wrote into a chunk of several periods, and the planner reads the array in few requests.
    root = zarr.open_group(services.open_or_create_repo(store_path).readonly_session("main").store, mode="r")
    assert [cast(zarr.Array, root[name]).chunks for name in (FETCHED_AT, "t")] == [(3,), (3,)]

    # Fetched after maturity, no period is due again: each was fetched at most twice.
    assert _sync("2026-01-03").status == "up_to_date"
    assert plugin.fetched == ["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02", "2026-01-03"]


def test_a_store_written_before_fetched_at_gets_it_on_the_next_append(
    refreshing_store: tuple[_RollingPlugin, dict[str, object], Path, dict[str, datetime]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Earlier periods get NaT: unknown, so treated as final and never fetched again."""
    plugin, dataset, store_path, clock = refreshing_store
    with monkeypatch.context() as legacy:
        legacy.setattr(orchestrator, "_stamp_fetched_at", lambda ds, **_: ds)
        legacy.setattr(orchestrator, "ensure_fetched_at", lambda *_, **__: None)
        services.create_artifact(
            dataset=dataset,
            start="2026-01-01",
            end="2026-01-02",
            bbox=[1.0, 2.0, 3.0, 4.0],
            country_code=None,
            overwrite=False,
            publish=False,
        )
    assert read_fetched_at(store_path) is None

    clock["now"] = datetime(2026, 3, 1, tzinfo=UTC)
    plugin.available.append("2026-01-03")
    _sync("2026-01-03")

    assert _stored(store_path) == ([1.0, 2.0, 3.0], [None, None, clock["now"]])
    assert plugin.fetched == ["2026-01-01", "2026-01-02", "2026-01-03"]
    # A coordinate like any other written one, and kept from every reader but those asking for
    # it: openEO loads, aggregation, exports and STAC all open the store as openEO does (CLIM-923).
    with open_icechunk_dataset(store_path, with_fetched_at=True) as stored:
        assert FETCHED_AT in stored.coords
    with openeo_execution._open_artifact(services._load_records()[-1]) as cube:
        assert FETCHED_AT not in cube.variables


def test_append_record_failure_restores_existing_store(
    rolling_store: tuple[_RollingPlugin, dict[str, object], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin, dataset, store_path = rolling_store
    initial = services.create_artifact(
        dataset=dataset,
        start="2026-01-01",
        end="2026-01-02",
        bbox=[1.0, 2.0, 3.0, 4.0],
        country_code=None,
        overwrite=False,
        publish=False,
    )
    plugin.available.append("2026-01-03")

    def fail_record(*args: object, **kwargs: object) -> None:
        raise OSError("record write failed")

    monkeypatch.setattr(services, "_upsert_artifact_record", fail_record)
    with pytest.raises(OSError, match="record write failed"):
        services.create_artifact(
            dataset=dataset,
            start="2026-01-01",
            end="2026-01-03",
            bbox=[1.0, 2.0, 3.0, 4.0],
            country_code=None,
            overwrite=False,
            publish=False,
        )

    assert services._load_records()[-1].artifact_id == initial.artifact_id
    repo = services.open_or_create_repo(store_path)
    assert not [branch for branch in repo.list_branches() if branch.startswith("ogs-ingest-rollback-")]
    with open_icechunk_dataset(store_path) as stored:
        assert stored["precip"][:, 0, 0].values.tolist() == [1.0, 2.0]
