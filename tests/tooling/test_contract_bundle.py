"""`scripts/contract_bundle.py`: the committed bundle is the opengwasdb commit `SOURCE` names.

The comparison itself, without the network; CI's `contract bundle` job runs `check`
against GitHub.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from script_modules import load_script

bundle = load_script("contract_bundle")


def _copy(tmp_path: Path, name: str) -> Path:
    target = tmp_path / name
    target.mkdir()
    for item in bundle.DATA:
        source = bundle.BUNDLE / item
        if source.is_dir():
            shutil.copytree(source, target / item)
        else:
            shutil.copy2(source, target / item)
    return target


def test_source_names_opengwasdb_and_a_full_commit():
    source = bundle.read_source()
    assert source["repository"].endswith("opengwas/opengwasdb.git")
    assert len(source["commit"]) == 40 and int(source["commit"], 16) >= 0
    assert source["path"] == bundle.UPSTREAM_PATH


def test_a_copy_of_the_bundle_agrees_with_it(tmp_path: Path):
    assert bundle.differences(bundle.BUNDLE, _copy(tmp_path, "upstream")) == []


def test_a_changed_missing_or_extra_file_is_named(tmp_path: Path):
    upstream = _copy(tmp_path, "upstream")
    ours = _copy(tmp_path, "ours")
    changed = ours / "expected" / "dense_ranges.json"
    changed.write_bytes(changed.read_bytes() + b" ")
    (ours / "VERSION").unlink()
    (ours / "releases" / "stray.txt").write_text("x", encoding="utf-8")
    assert bundle.differences(ours, upstream) == [
        "missing here: VERSION",
        "not in opengwasdb's bundle: releases/stray.txt",
        "differs: expected/dense_ranges.json",
    ]
