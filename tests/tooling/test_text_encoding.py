"""Every text file the package, its scripts and its tests read or write names its encoding.

Without one, Python uses the locale's, which on Windows is cp1252: a UTF-8 file then reads as
other text, silently. CI run 45664832290 of opengwas/ogquery#2 asked a test a different
chromosome label that way. Ruff's `PLW1514` would catch only some of these (it is a preview rule
in ruff 0.17, and it cannot see that `(directory / name).open()` is a path), so this scans the
source: `open`, and `.read_text`, `.write_text` and `.open` called as a path's, in text mode,
must pass `encoding`.

`tests/ld/testdata.py` is opengwasdb's file, held byte for byte to the commit
`tests/ld/data/SOURCE` names, and is not edited here; every file it reads is ASCII, which this
also holds, so the locale cannot change what it reads.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCANNED = ("ogquery", "scripts", "tests")
#: opengwasdb's, byte for byte (tests/ld/data/SOURCE).
NOT_OURS = frozenset({"tests/ld/testdata.py"})
#: What `NOT_OURS` reads, which must stay ASCII while it reads them with the locale's encoding.
READ_BY_NOT_OURS = (
    "tests/ld/data/test-panel/manifest.json",
    "tests/ld/data/test-panel/variants.tsv",
    "tests/ld/data/test-panel/panel.pvar",
    "tests/ld/data/test-panel/gwas/z.tsv",
    "tests/ld/data/test-panel/plink2/clump_r2_0.01.clumps",
    "tests/ld/data/test-panel/plink2/clump_r2_0.001.clumps",
)


def _mode(call: ast.Call, position: int) -> str | None:
    """The call's mode as a string literal, positional at `position` or `mode=`; None if absent."""
    found: ast.expr | None = call.args[position] if len(call.args) > position else None
    for keyword in call.keywords:
        if keyword.arg == "mode":
            found = keyword.value
    if isinstance(found, ast.Constant) and isinstance(found.value, str):
        return found.value
    return None if found is None else "?"


def unnamed(source: str) -> list[int]:
    """The lines of `source` that read or write text without naming its encoding."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or any(k.arg == "encoding" for k in node.keywords):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "open":
            mode = _mode(node, 1)
        elif isinstance(func, ast.Attribute) and func.attr in ("read_text", "write_text"):
            mode = "t"
        elif isinstance(func, ast.Attribute) and func.attr == "open" and (
            not node.args or isinstance(node.args[0], ast.Constant)
        ) and all(k.arg in ("mode", "newline", "buffering", "errors") for k in node.keywords):
            mode = _mode(node, 0)
        else:
            continue
        if mode is None or "b" not in mode:
            lines.append(node.lineno)
    return lines


def test_the_scan_sees_what_it_should() -> None:
    source = "\n".join(
        [
            "open(p)",
            "open(p, 'rb')",
            "open(p, encoding='utf-8')",
            "path.read_text()",
            "path.read_text(encoding='utf-8')",
            "path.write_text(text)",
            "(d / n).open(newline='')",
            "(d / n).open('rb')",
            "Sidecar.open(store_path)",
            "store.open()",
        ]
    )
    assert unnamed(source) == [1, 4, 6, 7, 10]


def test_every_text_read_and_write_names_its_encoding() -> None:
    found = {}
    for top in SCANNED:
        for path in sorted((ROOT / top).rglob("*.py")):
            name = path.relative_to(ROOT).as_posix()
            if name in NOT_OURS or ".pixi" in path.parts:
                continue
            lines = unnamed(path.read_text(encoding="utf-8"))
            if lines:
                found[name] = lines
    assert found == {}, f"text read or written in the locale's encoding: {found}"


def test_what_opengwasdb_s_loader_reads_is_ascii() -> None:
    for name in READ_BY_NOT_OURS:
        assert (ROOT / name).read_bytes().isascii(), name
