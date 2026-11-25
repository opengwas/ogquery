"""The LD Test Panel, its Test Store and their loader are opengwasdb's bytes (ogquery#2).

`data/SOURCE` names the opengwasdb commit they came from and, for each file, its git blob id
there. A blob id is the SHA-1 of the file's bytes under git's `blob <size>\\0` header, so the
copy is held to that commit byte for byte without fetching it, and no file here can be edited to
make a test pass: opengwasdb regenerates them, and a refresh rewrites SOURCE with them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / "data" / "SOURCE"


def listed() -> dict[str, str]:
    """Each file SOURCE lists, by its path from the repository root, and its blob id."""
    files: dict[str, str] = {}
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if line.startswith("file "):
            _, blob, path = line.split()
            assert path not in files, f"{path} is listed twice"
            files[path] = blob
    return files


def blob_id(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def test_source_names_the_commit_and_the_files() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    assert "\ncommit c8ebe4ca2e7697f3b723fd89ebb645bb1d221fd4\n" in text
    files = listed()
    assert "tests/ld/testdata.py" in files
    assert "tests/ld/data/test-panel/store/zarr.json" in files
    assert "tests/ld/data/test-panel/reference/r.npz" in files
    assert len(files) > 60


def test_every_listed_file_is_the_source_commit_s_bytes() -> None:
    changed = {path for path, blob in listed().items() if blob_id(ROOT / path) != blob}
    assert changed == set(), f"edited, or not this commit's: {sorted(changed)}"


def test_every_file_under_data_is_listed() -> None:
    ours = {"tests/ld/data/SOURCE", "tests/ld/data/.gitattributes"}
    on_disk = {
        p.relative_to(ROOT).as_posix()
        for p in (HERE / "data").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    assert on_disk - ours == {p for p in listed() if p.startswith("tests/ld/data/")}
