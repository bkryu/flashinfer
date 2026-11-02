"""The wheel's solid kernel-source archives are materialized into a workspace overlay."""

import io
import lzma
import os
import tarfile
from pathlib import Path

import pytest

from flashinfer.jit.csrc_archive import ARCHIVE_SUFFIX, resolve_csrc_dir


def _make_archive(csrc: Path, family: str, files: dict[str, str]) -> Path:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for rel, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(f"{family}/{rel}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    out = csrc / f"{family}{ARCHIVE_SUFFIX}"
    out.write_bytes(lzma.compress(buf.getvalue()))
    return out


def _packaged(tmp_path: Path) -> Path:
    csrc = tmp_path / "pkg" / "data" / "csrc"
    (csrc / "plain_family").mkdir(parents=True)
    (csrc / "plain_family" / "kernel.cu").write_text("plain")
    (csrc / "top_level.cuh").write_text("header")
    return csrc


def test_no_archives_returns_packaged_dir(tmp_path):
    csrc = _packaged(tmp_path)
    assert resolve_csrc_dir(csrc, tmp_path / "ws") == csrc
    assert not (tmp_path / "ws").exists()


def test_archives_are_extracted_and_plain_entries_linked(tmp_path):
    csrc = _packaged(tmp_path)
    _make_archive(csrc, "fam", {"a/k1.cu": "k1", "k2.cuh": "k2"})
    ws = tmp_path / "ws"
    overlay = resolve_csrc_dir(csrc, ws)
    assert overlay == ws / "csrc"
    assert (overlay / "fam" / "a" / "k1.cu").read_text() == "k1"
    assert (overlay / "fam" / "k2.cuh").read_text() == "k2"
    assert not (overlay / "fam").is_symlink()
    assert (overlay / "plain_family").is_symlink()
    assert (overlay / "plain_family" / "kernel.cu").read_text() == "plain"
    assert (overlay / "top_level.cuh").is_symlink()
    assert not (overlay / f"fam{ARCHIVE_SUFFIX}").exists()


def test_extraction_is_cached_by_digest_and_refreshed_on_change(tmp_path):
    csrc = _packaged(tmp_path)
    archive = _make_archive(csrc, "fam", {"k.cu": "v1"})
    ws = tmp_path / "ws"
    overlay = resolve_csrc_dir(csrc, ws)
    extracted = overlay / "fam" / "k.cu"
    first_ino = extracted.stat().st_ino
    resolve_csrc_dir(csrc, ws)
    assert extracted.stat().st_ino == first_ino  # same digest: not re-extracted
    archive.unlink()
    _make_archive(csrc, "fam", {"k.cu": "v2"})
    resolve_csrc_dir(csrc, ws)
    assert extracted.read_text() == "v2"


def test_member_outside_family_is_rejected_and_degrades_to_packaged(tmp_path):
    csrc = _packaged(tmp_path)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo("other/evil.cu")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    (csrc / f"fam{ARCHIVE_SUFFIX}").write_bytes(lzma.compress(buf.getvalue()))
    ws = tmp_path / "ws"
    assert resolve_csrc_dir(csrc, ws) == csrc
    assert not (ws / "csrc" / "other").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_unwritable_workspace_degrades_to_packaged(tmp_path):
    csrc = _packaged(tmp_path)
    _make_archive(csrc, "fam", {"k.cu": "v"})
    ws = tmp_path / "ro"
    ws.mkdir()
    ws.chmod(0o500)
    try:
        assert resolve_csrc_dir(csrc, ws) == csrc
    finally:
        ws.chmod(0o700)
