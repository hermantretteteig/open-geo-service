"""Measure how much disk a stored artifact takes.

Measured once, when an ingest, sync, openEO publish or feature refresh has finished writing, and
recorded on the artifact as `size_bytes`. Never measured on a page load: a store can be hundreds
of thousands of chunk files, and walking them while a heavy job holds the GIL took the overview
page from seconds to many minutes.
"""

from __future__ import annotations

import os
from pathlib import Path


def stored_bytes(path: str | Path) -> int:
    """Bytes the artifact at *path* stores.

    For an Icechunk repository, the bytes of the chunks its branches and tags reach, as Icechunk
    reports them: the versions it keeps, without data awaiting collection or its small metadata
    files. For anything else, the bytes on disk under *path*: a directory walked without
    following symlinks, or a file.

    A path that is missing or unreadable counts as 0 rather than raising, since a size is a
    figure for display and an ingest must not fail over it.
    """
    root = Path(path)
    if (root / "repo").is_file():
        try:
            import icechunk

            repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(root)))
            return int(repo.chunk_storage_stats().non_virtual_bytes())
        except Exception:
            return 0
    try:
        if not root.is_dir():
            return root.stat().st_size if root.is_file() else 0
    except OSError:
        return 0
    total = 0
    stack = [root]
    while stack:
        try:
            entries = list(os.scandir(stack.pop()))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
    return total
