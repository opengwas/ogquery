"""The LD Test Panel, its Test Stores and their loader are opengwasdb's bytes (ogquery#2).

`data/SOURCE` names the opengwasdb commit the copies came from and, for each file, its git blob
id there. A blob id is the SHA-1 of the file's bytes under git's `blob <size>\\0` header, so a
copy is held to that commit byte for byte without fetching it, and no file here can be edited to
make a test pass. Three things are generated, not copied: the K = 0 Test Store, the Test Panel's
Genotype Store and opengwasdb's answers on it (ogquery#11). For each, a `generated` block records
the commit and command that wrote it and the blob ids of what it read, and pins what it wrote.

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
#: The fields of a `generated` block, which its `generated` line opens.
BLOCK_FIELDS = frozenset({"generated-commit", "generated-command"})


@dataclass
class Generated:
    """One `generated` block: where it wrote, its fields, and its input and output lines."""

    path: str
    fields: dict[str, str] = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)


@dataclass
class Source:
    """SOURCE, read: its fields, its copies as {path: blob id}, and its generated blocks."""

    fields: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    generated: list[Generated] = field(default_factory=list)


def read() -> Source:
    found = Source()
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, _, value = line.partition(" ")
        block = found.generated[-1] if found.generated else None
        if key == "generated":
            assert value not in {g.path for g in found.generated}, f"{value} generated twice"
            found.generated.append(Generated(value))
        elif key in ("file", "input", "output"):
            blob, path = value.split()
            assert COMMIT.fullmatch(blob), line
            if key == "file":
                lines = found.files
            else:
                assert block is not None, f"{line} is outside a generated block"
                lines = block.inputs if key == "input" else block.outputs
            assert path not in lines, f"{path} is listed twice as {key}"
            lines[path] = blob
        elif key in BLOCK_FIELDS:
            assert block is not None, f"{line} is outside a generated block"
            assert key not in block.fields, f"{key} is given twice in {block.path}"
            block.fields[key] = value
        else:
            assert block is None, f"{key} comes after the generated blocks"
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


def test_each_generated_store_names_how_it_was_made_and_holds_what_it_wrote() -> None:
    source = read()
    assert {g.path for g in source.generated} == {
        "tests/ld/data/test-panel/store-k0",
        "tests/ld/data/test-panel/genotype-store",
        "tests/ld/data/test-panel/genotype-answers",
    }
    for block in source.generated:
        assert block.fields.keys() == BLOCK_FIELDS, block.path
        assert COMMIT.fullmatch(block.fields["generated-commit"]), block.path
        assert block.fields["generated-command"].endswith(block.path), "it wrote where it is"
        assert block.inputs, f"{block.path}: the files it read are named"
        assert block.outputs, block.path
        assert all(path.startswith(block.path + "/") for path in block.outputs), block.path
        changed = {path for path, blob in block.outputs.items() if blob_id(ROOT / path) != blob}
        assert changed == set(), f"edited since it was generated: {sorted(changed)}"


def test_an_input_held_here_is_the_bytes_the_generation_read() -> None:
    # Where a generation read a file this repository holds -- a copy, a store it read, the
    # questions it asked -- it read these bytes: the Genotype Store pins the pgen beside it, and
    # the answers are the answers to these questions only.
    source = read()
    held: dict[str, set[str]] = {}
    for block in source.generated:
        here = {path: blob for path, blob in block.inputs.items() if (ROOT / path).is_file()}
        assert here, f"{block.path}: it read at least one file this repository holds"
        changed = {path for path, blob in here.items() if blob_id(ROOT / path) != blob}
        assert changed == set(), f"{block.path} read other bytes: {sorted(changed)}"
        copied = here.keys() & source.files.keys()
        assert {p: here[p] for p in copied} == {p: source.files[p] for p in copied}
        held[block.path] = set(here)
    panel = "tests/ld/data/test-panel/"
    pinned = {panel + name for name in ("panel.pgen", "panel.psam", "panel.pvar")}
    assert pinned <= held[panel + "genotype-store"], "the store's pinned inputs are here"
    questions = {"tests/ld/genotype_answers.py", "tests/ld/contract/support.py"}
    assert questions <= held[panel + "genotype-answers"], "and the questions, and their ALIDs"


def test_every_file_under_data_is_listed() -> None:
    source = read()
    on_disk = {
        p.relative_to(ROOT).as_posix()
        for p in (HERE / "data").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    listed = {p for p in source.files if p.startswith("tests/ld/data/")}
    listed |= {path for block in source.generated for path in block.outputs}
    assert on_disk - OURS == listed
