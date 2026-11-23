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

from open_geo_service import writer_lease  # noqa: E402
from open_geo_service.data_accessor.services.accessor import open_icechunk_dataset  # noqa: E402
from open_geo_service.data_manager.services import downloader  # noqa: E402
from open_geo_service.ingestions import services  # noqa: E402
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


def _ingest(state: dict[str, Any], *, overwrite: bool, end: str = DAYS[-1]) -> services.ArtifactRecord:
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
        publish=False,
    )


def _read(path: Path) -> tuple[int, float]:
    """What a reader of `main` sees: the number of periods and the value they hold."""
    with open_icechunk_dataset(path) as ds:
        return int(ds.sizes["t"]), float(ds["tg"].mean())


def _branches(path: Path) -> set[str]:
    return set(icechunk.Repository.open(icechunk.local_filesystem_storage(str(path))).list_branches())


@pytest.mark.parametrize("pyramid", [False, True])
def test_an_overwrite_moves_readers_to_the_new_version_in_one_step(
    store: dict[str, Any], monkeypatch: pytest.MonkeyPatch, pyramid: bool
) -> None:
    if pyramid:
        _use_pyramids(monkeypatch)
    _ingest(store, overwrite=False)
    reader = open_icechunk_dataset(store["path"])

    store["plugin"] = _ValuePlugin(2.0)
    replaced = _ingest(store, overwrite=True)

    assert _read(store["path"]) == (3, 2.0)
    assert float(reader["tg"].mean()) == 1.0, "a reader keeps the version it opened"
    reader.close()
    assert _branches(store["path"]) == {"main"}
    # Nothing beside the store but the note that the replaced version is due for collection.
    assert sorted(p.name for p in store["path"].parent.iterdir()) == ["tg.icechunk", "tg.icechunk.gc-pending"]

    # Kept for readers until the retention window passes, then freed, and the size follows.
    monkeypatch.setattr(services, "_GC_RETENTION", timedelta(0))
    assert services.collect_pending_garbage(store["path"])
    (collected,) = services.list_artifacts().items
    assert collected.size_bytes is not None and replaced.size_bytes is not None
    assert collected.size_bytes < replaced.size_bytes
    assert _read(store["path"]) == (3, 2.0)


@pytest.mark.parametrize("failure", ["fetch", "normalization", "record"])
def test_a_failed_overwrite_leaves_readers_on_the_version_they_had(
    store: dict[str, Any], monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    _use_pyramids(monkeypatch)
    _ingest(store, overwrite=False)

    store["plugin"] = _ValuePlugin(2.0, fail_at=DAYS[1] if failure == "fetch" else None)
    if failure == "normalization":
        monkeypatch.setattr(downloader, "write_to_icechunk_store", _raise)
    if failure == "record":
        monkeypatch.setattr(services, "_upsert_artifact_record", _raise)
    with pytest.raises(Exception, match="failed|rolled back"):
        _ingest(store, overwrite=True)

    assert _read(store["path"]) == (3, 1.0)
    assert _branches(store["path"]) == {"main"}
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


def test_cleanup_that_fails_after_the_record_is_written_keeps_the_new_version(
    store: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _ingest(store, overwrite=False)
    store["plugin"] = _ValuePlugin(2.0)
    monkeypatch.setattr(services, "_delete_branch", _raise)

    artifact = _ingest(store, overwrite=True)

    assert artifact.coverage.temporal.end == DAYS[-1]
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


def test_a_killed_overwrite_leaves_the_store_readable(store: dict[str, Any], tmp_path: Path) -> None:
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

    store["plugin"] = _ValuePlugin(2.0)
    _ingest(store, overwrite=True)  # removes what the killed one left, then runs
    assert _read(store["path"]) == (3, 2.0)
    assert _branches(store["path"]) == {"main"}


def _raise(*args: object, **kwargs: object) -> Any:
    raise OSError("write failed")
