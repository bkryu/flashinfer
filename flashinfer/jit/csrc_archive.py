"""Solid archives of heavy kernel-source families in the installed wheel.

A wheel is a zip that compresses every file on its own, so generated kernel
families made of near-identical expansions (``csrc/cake_sampling``: 18 files of
~3 MB, 87% identical line-for-line) cost their full size per file. The wheel
build (``build_backend._archive_csrc_families``) therefore replaces the heaviest
families under ``flashinfer/data/csrc/`` with one solid ``<family>.tar.xz``
each, which stores the repetition once (roughly 10-25x smaller).

:func:`resolve_csrc_dir` restores a plain directory tree for the JIT: when the
packaged ``csrc`` holds archives it builds ``<workspace>/csrc`` with a symlink
for every plain entry and the archived families extracted next to them, and
``jit_env.FLASHINFER_CSRC_DIR`` points there. Extraction happens once per
archive digest, under a file lock, into a temp dir that is renamed into place.
Source checkouts and editable installs have no archives and are untouched.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import shutil
import tarfile

logger = logging.getLogger("flashinfer.jit")

ARCHIVE_SUFFIX = ".tar.xz"


def archive_family_name(archive: pathlib.Path) -> str:
    return archive.name[: -len(ARCHIVE_SUFFIX)]


def _digest(path: pathlib.Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _link_plain_entries(packaged: pathlib.Path, overlay: pathlib.Path) -> None:
    for entry in packaged.iterdir():
        if entry.name.endswith(ARCHIVE_SUFFIX):
            continue
        dst = overlay / entry.name
        if dst.is_symlink():
            if os.readlink(dst) == str(entry):
                continue
            dst.unlink()
        elif dst.exists():
            # A real directory left by an earlier extraction of a family that is
            # now shipped plain; the plain copy wins.
            shutil.rmtree(dst)
        os.symlink(entry, dst, target_is_directory=entry.is_dir())


def _extract(archive: pathlib.Path, overlay: pathlib.Path) -> None:
    from filelock import FileLock

    family = archive_family_name(archive)
    dst = overlay / family
    marker = overlay / f".{family}.extracted"
    digest = _digest(archive)
    if dst.is_dir() and marker.is_file() and marker.read_text().strip() == digest:
        return
    with FileLock(str(overlay / f".{family}.lock"), thread_local=False):
        if dst.is_dir() and marker.is_file() and marker.read_text().strip() == digest:
            return
        tmp = overlay / f".{family}.tmp-{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        prefix = family + "/"
        with tarfile.open(archive, "r:xz") as tar:
            members = tar.getmembers()
            for m in members:
                if (
                    not m.name.startswith(prefix)
                    or ".." in pathlib.PurePosixPath(m.name).parts
                ):
                    raise ValueError(f"{archive.name}: unexpected member {m.name!r}")
            kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            tar.extractall(tmp, members=members, **kwargs)
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        elif dst.is_dir():
            shutil.rmtree(dst)
        os.replace(tmp / family, dst)
        shutil.rmtree(tmp, ignore_errors=True)
        marker.write_text(digest + "\n")
        logger.info(f"Extracted kernel sources {family} to {dst}")


def resolve_csrc_dir(packaged: pathlib.Path, workspace: pathlib.Path) -> pathlib.Path:
    """The directory the JIT should read kernel sources from.

    ``packaged`` itself when it holds no archives (source checkouts, editable
    installs); otherwise the workspace overlay. Any failure to build the overlay
    (read-only workspace, disk full, corrupt archive) is logged and ``packaged``
    is returned, so plain families keep working and only the archived ones fail
    later with their own "sources not found" errors.
    """
    try:
        archives = (
            sorted(packaged.glob("*" + ARCHIVE_SUFFIX)) if packaged.is_dir() else []
        )
    except OSError:
        archives = []
    if not archives:
        return packaged
    overlay = workspace / "csrc"
    try:
        overlay.mkdir(parents=True, exist_ok=True)
        _link_plain_entries(packaged, overlay)
        for archive in archives:
            _extract(archive, overlay)
    except Exception as e:  # noqa: BLE001 - degrade to the packaged tree
        logger.warning(
            f"Could not materialize archived kernel sources under {overlay} ({e}); "
            f"using {packaged} (archived families will be unavailable)."
        )
        return packaged
    return overlay
