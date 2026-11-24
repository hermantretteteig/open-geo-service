"""Readers always see a whole version of a raster store, across a failed or killed update (CLIM-1328).

An overwrite is built on a temporary branch of the same Icechunk repository and `main` moves to it
in one step; a pyramid rebuild is one commit. The tests update a real store through
``create_artifact`` and read it the way a reader does, through `main`.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
icechunk = pytest.importorskip("icechunk")
pytest.importorskip("topozarr")
zarr = pytest.importorskip("zarr")

from fastapi import HTTPException  # noqa: E402

from open_geo_service import config as api_config  # noqa: E402
from open_geo_service import writer_lease  # noqa: E402
from open_geo_service.data_accessor.services.accessor import open_icechunk_dataset  # noqa: E402
from open_geo_service.data_manager.services import downloader  # noqa: E402
from open_geo_service.ingestions import services  # noqa: E402
from open_geo_service.ingestions.schemas import PublicationStatus  # noqa: E402
from open_geo_service.streaming import BaseDatasetPlugin, normalize_period  # noqa: E402

NY, NX = 101, 75
DAYS = ["2026-01-01", "2026-01-02", "2026-01-03"]


class _ValuePlugin(BaseDatasetPlugin):
    """Every cell of every period holds *value*; *fail_at* raises on that period."""

    def __init__(self, value: float, fail_at: str | None = None) -> None:
        self.value = value
        self.fail_at = fail_at

    async def periods(self, start: str, end: str) -> list[str]:
        return [day for day in DAYS if start <= day <= end]

    def fetch_period(self, period_id: str, bbox: list[float], **params: object) -> Any:
        if period_id == self.fail_at:
            raise RuntimeError("upstream fetch failed")
        data = xr.DataArray(
            np.full((NY, NX), self.value, dtype="float32"),
            dims=("y", "x"),
            coords={"y": np.linspace(4.0, 2.0, NY), "x": np.linspace(1.0, 3.0, NX)},
        )
        return normalize_period(data, variable="tg", period=period_id)


def _use_pyramids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Small enough that the 101x75 grid gets a pyramid, which a first ingest builds by a rewrite."""
    monkeypatch.setattr(downloader, "_PYRAMID_PIXEL_THRESHOLD", 64 * 64)
    monkeypatch.setattr(downloader, "_PYRAMID_TARGET_TILE_SIZE", 24)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    monkeypatch.setattr(services, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(services, "ARTIFACTS_INDEX_PATH", tmp_path / "artifacts" / "records.json")
    state: dict[str, Any] = {"plugin": _ValuePlugin(1.0), "path": tmp_path / "stores" / "tg.icechunk"}
    monkeypatch.setattr(services, "_load_streaming_plugin", lambda *args, **kwargs: state["plugin"])
    monkeypatch.setattr(downloader, "get_icechunk_path", lambda _: state["path"])
    return state


def _ingest(
    state: dict[str, Any], *, overwrite: bool, end: str = DAYS[-1], publish: bool = False
) -> services.ArtifactRecord:
    return services.create_artifact(
        dataset={
            "id": "tg_daily",
            "name": "Daily temperature",
            "variable": "tg",
            "period_type": "daily",
            "ingestion": {"plugin": "example.ValuePlugin"},
        },
        start=DAYS[0],
        end=end,
        bbox=[1.0, 2.0, 3.0, 4.0],
        country_code=None,
        overwrite=overwrite,
        publish=publish,
    )


def _read(path: Path) -> tuple[int, float]:
    """What a reader sees: the number of periods and the value they hold."""
    with open_icechunk_dataset(path) as ds:
        return int(ds.sizes["t"]), float(ds["tg"].mean())


def _repo(path: Path) -> Any:
    return icechunk.Repository.open(icechunk.local_filesystem_storage(str(path)))


def _branches(path: Path) -> set[str]:
    return set(_repo(path).list_branches())


def _published(path: Path) -> str | None:
    try:
        return str(_repo(path).lookup_branch("published"))
    except icechunk.RefNotFoundError:
        return None


@pytest.mark.parametrize("pyramid", [False, True])
def test_an_overwrite_moves_readers_to_the_new_version_in_one_step(
    store: dict[str, Any], monkeypatch: pytest.MonkeyPatch, pyramid: bool
) -> None:
    if pyramid:
        _use_pyramids(monkeypatch)
    _ingest(store, overwrite=False, publish=True)
    before = _published(store["path"])
    reader = open_icechunk_dataset(store["path"])

    store["plugin"] = _ValuePlugin(2.0)
    replaced = _ingest(store, overwrite=True)

    assert _read(store["path"]) == (3, 2.0)
    assert float(reader["tg"].mean()) == 1.0, "a reader keeps the version it opened"
    reader.close()
    assert _branches(store["path"]) == {"main", "published"}
    # Nothing beside the store but the note that unreachable data is due for collection.
    assert sorted(p.name for p in store["path"].parent.iterdir()) == ["tg.icechunk", "tg.icechunk.gc-pending"]

    # The replaced version is kept for the retention period, so the dataset can be rolled back.
    services.publish_artifact_record(replaced.artifact_id, snapshot_id=before)
    assert _read(store["path"]) == (3, 1.0)


def test_a_sync_that_fails_part_way_is_not_a_version_to_roll_back_to(store: dict[str, Any]) -> None:
    record = _ingest(store, overwrite=False, end=DAYS[0], publish=True)
    store["plugin"] = _ValuePlugin(1.0, fail_at=DAYS[2])
    with pytest.raises(Exception, match="upstream fetch failed"):
        _ingest(store, overwrite=False)

    partial = str(_repo(store["path"]).lookup_branch("main"))  # holds DAYS[1], kept for the retry
    assert partial != _published(store["path"])
    with pytest.raises(HTTPException) as refused:
        services.publish_artifact_record(record.artifact_id, snapshot_id=partial)
    assert refused.value.status_code == 404


def test_retention_zero_keeps_only_the_current_version(store: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    # A setting that is not a whole number of days refuses startup rather than stopping cleanup.
    for bad in ("14", 7.5, -1):
        with monkeypatch.context() as patched:
            patched.setattr(api_config, "get_config", lambda bad=bad: {"snapshot_retention_days": bad})
            with pytest.raises(ValueError, match="snapshot_retention_days"):
                api_config.get_snapshot_retention()

    monkeypatch.setattr(services, "_GC_RETENTION", timedelta(0))
    monkeypatch.setattr(api_config, "get_snapshot_retention", lambda: timedelta(0))
    _ingest(store, overwrite=False, publish=True)
    before = _published(store["path"])
    store["plugin"] = _ValuePlugin(2.0)
    replaced = _ingest(store, overwrite=True)

    with pytest.raises(HTTPException) as refused:
        services.publish_artifact_record(replaced.artifact_id, snapshot_id=before)
    assert refused.value.status_code == 404
    # Freed by the pending collection, and the recorded size follows.
    assert services.collect_pending_garbage(store["path"])
    (collected,) = services.list_artifacts().items
    assert collected.size_bytes is not None and replaced.size_bytes is not None
    assert collected.size_bytes < replaced.size_bytes
    assert _read(store["path"]) == (3, 2.0)
    writes = services._records_write_count
    services._refresh_recorded_size(store["path"])
    assert services._records_write_count == writes, "an unchanged size leaves the index, and its caches, alone"

    # A version older than the window is freed by the overwrite itself, and the size counts it gone.
    store["plugin"] = _ValuePlugin(3.0)
    _ingest(store, overwrite=True)
    (overwritten,) = services.list_artifacts().items
    assert overwritten.size_bytes == services.stored_bytes(store["path"])


def test_the_published_branch_decides_what_readers_see(store: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    _use_pyramids(monkeypatch)  # so a sync's normalization is a step of its own to fold in
    unpublished = _ingest(store, overwrite=False, end=DAYS[1])
    assert _published(store["path"]) is None, "an unpublished raster has no published branch"
    assert unpublished.publication.status == PublicationStatus.UNPUBLISHED

    published = _ingest(store, overwrite=False, end=DAYS[1], publish=True)
    assert published.publication.status == PublicationStatus.PUBLISHED
    first = _published(store["path"])
    assert first == str(_repo(store["path"]).lookup_branch("main"))

    synced = _ingest(store, overwrite=False)  # a sync of a published raster moves its branch
    assert synced.publication.status == PublicationStatus.PUBLISHED
    assert _published(store["path"]) == str(_repo(store["path"]).lookup_branch("main")) != first
    assert _read(store["path"]) == (3, 1.0)
    # One whole snapshot per sync, its pyramid levels extended in the same one, above the empty
    # first; the earlier sync's is kept.
    history = list(_repo(store["path"]).ancestry(branch="main"))
    assert len(history) == 3

    # A rollback, which the record follows: it describes the version readers now see.
    rolled_back = services.publish_artifact_record(synced.artifact_id, snapshot_id=first)
    assert _read(store["path"]) == (2, 1.0)
    assert rolled_back.coverage.temporal.end == DAYS[1]
    for refused in ("0" * 20, str(history[-1].id)):  # unknown, and the empty first snapshot
        with pytest.raises(HTTPException) as unknown:
            services.publish_artifact_record(synced.artifact_id, snapshot_id=refused)
        assert unknown.value.status_code == 404

    # A sync that finds the record behind the store re-registers it and publishes `main` again;
    # if its record write fails, readers stay on the version the record describes.
    with monkeypatch.context() as patched:
        patched.setattr(services, "_upsert_artifact_record", _raise)
        with pytest.raises(OSError, match="write failed"):
            _ingest(store, overwrite=False)
    assert _published(store["path"]) == first

    # A writer that registers a rewrite of a published store, as openEO does, publishes it too.
    services.register_artifact_record(rolled_back, publish=False, replace_store=True)
    assert _published(store["path"]) == str(_repo(store["path"]).lookup_branch("main"))
    assert _read(store["path"]) == (3, 1.0)


def test_startup_brings_branch_and_record_into_step(store: dict[str, Any]) -> None:
    # A store from before the branch: published record, no branch, and a tip not marked
    # complete, which publishing marks, so a rollback may return to what readers are served.
    record = _ingest(store, overwrite=False, end=DAYS[1], publish=True)
    _repo(store["path"]).writable_session("main").commit("unmarked", allow_empty=True)
    _repo(store["path"]).delete_branch("published")
    assert services.reconcile_published_branches() == 1
    tip = next(iter(_repo(store["path"]).ancestry(branch="main")))
    assert _published(store["path"]) == str(tip.id) and tip.metadata.get("ogs_complete")
    record = services.list_artifacts().items[0]

    # A crash between moving the branch and writing the record: the record follows the branch.
    def unpublish(records: list[services.ArtifactRecord]) -> None:
        records[0] = record.model_copy(update={"publication": services.ArtifactPublication()})

    services._mutate_records(unpublish)
    assert services.reconcile_published_branches() == 1
    (followed,) = services.list_artifacts().items
    assert followed.publication.status == PublicationStatus.PUBLISHED
    assert services.reconcile_published_branches() == 0

    # A crash after a sync moved the branch, before its record: the record describes the branch.
    synced = _ingest(store, overwrite=False)
    assert synced.publication.snapshot_id == _published(store["path"])

    def crash(records: list[services.ArtifactRecord]) -> None:
        records[:] = [stored for stored in records if stored.artifact_id != synced.artifact_id]

    services._mutate_records(crash)
    assert services.reconcile_published_branches() == 1
    (described,) = services.list_artifacts().items
    assert described.coverage.temporal.end == DAYS[-1]
    assert described.publication.snapshot_id == _published(store["path"])
    assert services.reconcile_published_branches() == 0


@pytest.mark.parametrize("failure", ["fetch", "normalization", "record"])
def test_a_failed_overwrite_leaves_readers_on_the_version_they_had(
    store: dict[str, Any], monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    _use_pyramids(monkeypatch)
    _ingest(store, overwrite=False, publish=True)
    published = _published(store["path"])

    store["plugin"] = _ValuePlugin(2.0, fail_at=DAYS[1] if failure == "fetch" else None)
    if failure == "normalization":
        monkeypatch.setattr(downloader, "write_to_icechunk_store", _raise)
    if failure == "record":
        monkeypatch.setattr(services, "_upsert_artifact_record", _raise)
    with pytest.raises(Exception, match="failed|rolled back"):
        _ingest(store, overwrite=True)

    assert _read(store["path"]) == (3, 1.0)
    assert _published(store["path"]) == published
    assert _branches(store["path"]) == {"main", "published"}
    lock = services._acquire_store_lock(store["path"])
    assert lock.acquire(blocking=False), "the next ingest is not refused"
    lock.release()


def test_a_failed_sync_that_rebuilt_the_store_resets_main(
    store: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sync whose store outgrew a flat layout rebuilds it as a pyramid on `main`; a failure after undoes both."""
    _ingest(store, overwrite=False, end=DAYS[1])
    _use_pyramids(monkeypatch)
    monkeypatch.setattr(services, "_upsert_artifact_record", _raise)

    with pytest.raises(OSError, match="failed"):
        _ingest(store, overwrite=False)

    assert _read(store["path"]) == (2, 1.0)
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(store["path"])))
    assert "multiscales" not in zarr.open_group(repo.readonly_session("main").store, mode="r").attrs, "still flat"
    assert _branches(store["path"]) == {"main"}


def test_cleanup_removes_a_branch_left_undeleted_but_keeps_the_one_a_failed_rollback_names(
    store: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _ingest(store, overwrite=False)
    store["plugin"] = _ValuePlugin(2.0)
    delete_branch = icechunk.Repository.delete_branch
    monkeypatch.setattr(icechunk.Repository, "delete_branch", _raise)

    artifact = _ingest(store, overwrite=True)  # after the record is written: the new version stays
    assert artifact.coverage.temporal.end == DAYS[-1]
    assert _read(store["path"]) == (3, 2.0)

    # During a rollback it is not a failed rollback: the update's own error is raised, `main` is
    # restored, and the branches wait for the next cleanup.
    store["plugin"] = _ValuePlugin(3.0)
    monkeypatch.setattr(services, "_upsert_artifact_record", lambda *args, **kwargs: _raise())
    with pytest.raises(OSError, match="write failed"):
        _ingest(store, overwrite=True)
    assert _read(store["path"]) == (3, 2.0)
    monkeypatch.setattr(icechunk.Repository, "delete_branch", delete_branch)
    assert services.clean_up_interrupted_ingest(store["path"])
    assert _branches(store["path"]) == {"main"}

    # A rollback that cannot reset `main` leaves the version before it on a branch the operator
    # restores from, which no cleanup removes.
    reset = icechunk.Repository.reset_branch
    resets: list[str] = []

    def reset_once(repo: Any, branch: str, snapshot: str) -> None:
        resets.append(snapshot)
        if len(resets) > 1:
            raise OSError("reset failed")
        reset(repo, branch, snapshot)

    monkeypatch.setattr(icechunk.Repository, "reset_branch", reset_once)
    with pytest.raises(RuntimeError, match="rollback could not complete") as failed:
        _ingest(store, overwrite=True)
    (kept,) = _branches(store["path"]) - {"main"}
    assert kept.startswith("ogs-restore-") and kept in str(failed.value)
    assert _read(store["path"]) == (3, 3.0), "main stayed on the attempt"
    monkeypatch.setattr(downloader, "DOWNLOAD_DIR", store["path"].parent)
    services.clean_up_interrupted_ingest(store["path"])
    services.clean_up_interrupted_ingests()
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(store["path"])))
    reset(repo, "main", repo.lookup_branch(kept))
    assert _read(store["path"]) == (3, 2.0)


_KILLED_OVERWRITE = """
import sys, time
from pathlib import Path

import numpy as np
import xarray as xr

from open_geo_service.data_manager.services import downloader
from open_geo_service.ingestions import services
from open_geo_service.streaming import BaseDatasetPlugin, normalize_period

store, artifacts, marker = (Path(arg) for arg in sys.argv[1:4])


class Plugin(BaseDatasetPlugin):
    async def periods(self, start, end):
        return ["2026-01-01", "2026-01-02", "2026-01-03"]

    def fetch_period(self, period_id, bbox, **params):
        if period_id == "2026-01-03":
            marker.write_text("committed two periods of the replacement")
            time.sleep(60)
        data = xr.DataArray(
            np.full((101, 75), 2.0, dtype="float32"),
            dims=("y", "x"),
            coords={"y": np.linspace(4.0, 2.0, 101), "x": np.linspace(1.0, 3.0, 75)},
        )
        return normalize_period(data, variable="tg", period=period_id)


services.ARTIFACTS_DIR = artifacts
services.ARTIFACTS_INDEX_PATH = artifacts / "records.json"
services._load_streaming_plugin = lambda *args, **kwargs: Plugin()
downloader.get_icechunk_path = lambda _: store
services.create_artifact(
    dataset={"id": "tg_daily", "name": "Daily temperature", "variable": "tg", "period_type": "daily",
             "ingestion": {"plugin": "example.Plugin"}},
    start="2026-01-01", end="2026-01-03", bbox=[1.0, 2.0, 3.0, 4.0], country_code=None,
    overwrite=True, publish=False,
)
"""


def test_a_killed_overwrite_leaves_the_store_readable(
    store: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process killed part way runs none of its cleanup; readers still open a whole version."""
    _ingest(store, overwrite=False)
    marker = tmp_path / "fetching"
    writer_lease.release_all()  # the overwrite runs in another process, which must be the writer
    process = subprocess.Popen(
        [sys.executable, "-c", _KILLED_OVERWRITE, str(store["path"]), str(tmp_path / "artifacts"), str(marker)]
    )
    try:
        deadline = time.monotonic() + 60
        while not marker.exists():
            assert process.poll() is None, "the overwrite ended before it could be killed"
            assert time.monotonic() < deadline, "the overwrite never reached its third period"
            time.sleep(0.05)
    finally:
        process.kill()
        process.wait()

    assert _read(store["path"]) == (3, 1.0)
    assert any(branch.startswith("ogs-ingest-replacement-") for branch in _branches(store["path"]))

    # The next writer's startup cleans up, also for a store that is never ingested again, and
    # moves the collection to start then, after everything the killed process wrote.
    marker = store["path"].with_name("tg.icechunk.gc-pending")
    marker.write_text("2000-01-01T00:00:00+00:00")
    monkeypatch.setattr(downloader, "DOWNLOAD_DIR", store["path"].parent)
    assert services.clean_up_interrupted_ingests() == 1
    assert _branches(store["path"]) == {"main"}
    assert marker.read_text() > "2026", "the pending collection starts after what the killed process wrote"
    moved = marker.read_text()
    services.clean_up_interrupted_ingests()
    assert marker.read_text() == moved, "a restart that finds nothing new written leaves it"

    store["plugin"] = _ValuePlugin(2.0)
    _ingest(store, overwrite=True)
    assert _read(store["path"]) == (3, 2.0)


def _raise(*args: object, **kwargs: object) -> Any:
    raise OSError("write failed")
