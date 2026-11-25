"""The LD Test Panel, its Test Stores and their loader are opengwasdb's bytes (ogquery#2).

`data/SOURCE` names the opengwasdb commit the copies came from and, for each file, its git blob
id there. A blob id is the SHA-1 of the file's bytes under git's `blob <size>\\0` header, so a
copy is held to that commit byte for byte without fetching it, and no file here can be edited to
make a test pass. The K = 0 Test Store is generated, not copied: SOURCE records the commit and
command that wrote it, the blob ids of what it read there, and pins what it wrote here.

Everything is checked against SOURCE itself, so a refresh to a later commit rewrites SOURCE and
the files and changes no test.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / "data" / "SOURCE"
#: The files under `data/` that are ogquery's own, not opengwasdb's.
OURS = frozenset({"tests/ld/data/SOURCE", "tests/ld/data/.gitattributes"})
COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass
class Source:
    """SOURCE, read: its fields, and its lines of each kind as {path: blob id}."""

    fields: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)


def read() -> Source:
    found = Source()
    kinds = {"file": found.files, "input": found.inputs, "output": found.outputs}
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, _, value = line.partition(" ")
        if key in kinds:
            blob, path = value.split()
            assert COMMIT.fullmatch(blob), line
            assert path not in kinds[key], f"{path} is listed twice as {key}"
            kinds[key][path] = blob
        else:
            assert key not in found.fields, f"{key} is given twice"
            found.fields[key] = value
    return found


def blob_id(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def test_source_names_a_commit_a_repository_and_its_files() -> None:
    source = read()
    assert COMMIT.fullmatch(source.fields["commit"]), source.fields["commit"]
    assert source.fields["repository"].startswith("https://"), source.fields["repository"]
    assert source.files, "SOURCE lists the files it copied"


def test_every_copied_file_is_the_source_commit_s_bytes() -> None:
    changed = {path for path, blob in read().files.items() if blob_id(ROOT / path) != blob}
    assert changed == set(), f"edited, or not this commit's: {sorted(changed)}"


def test_the_generated_store_names_how_it_was_made_and_holds_what_it_wrote() -> None:
    source = read()
    generated = source.fields["generated"]
    assert COMMIT.fullmatch(source.fields["generated-commit"])
    assert source.fields["generated-command"].endswith(generated), "it wrote where it is"
    assert source.inputs, "the files it read are named"
    assert source.outputs and all(path.startswith(generated + "/") for path in source.outputs)
    changed = {path for path, blob in source.outputs.items() if blob_id(ROOT / path) != blob}
    assert changed == set(), f"edited since it was generated: {sorted(changed)}"


def test_an_input_copied_here_is_the_copy_s_own_bytes() -> None:
    # The generation read a file of the same commit the copies came from: where both name it,
    # they name one blob.
    source = read()
    if source.fields["generated-commit"] == source.fields["commit"]:
        shared = source.inputs.keys() & source.files.keys()
        assert shared, "the loader and the reference r are both read and copied"
        assert {p: source.inputs[p] for p in shared} == {p: source.files[p] for p in shared}


def test_every_file_under_data_is_listed() -> None:
    source = read()
    on_disk = {
        p.relative_to(ROOT).as_posix()
        for p in (HERE / "data").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    listed = {p for p in source.files if p.startswith("tests/ld/data/")} | set(source.outputs)
    assert on_disk - OURS == listed
