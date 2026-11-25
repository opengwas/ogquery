"""`scripts/contract_bundle.py`: the committed bundle is the opengwasdb commit `SOURCE` names.

The comparison itself, without the network; CI's `contract bundle` job runs `check`
against GitHub.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
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


def _git(repo: Path, *args: str) -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _upstream(tmp_path: Path, *, drop: str | None = None) -> tuple[Path, str]:
    """A local git repository holding the bundle's data at SOURCE's path, less `drop`."""
    repo = tmp_path / "opengwasdb"
    place = repo / bundle.read_source()["path"]
    place.parent.mkdir(parents=True)
    shutil.move(_copy(tmp_path, "data"), place)
    (place / "VERSION").write_text("bundle 2\nformat_version 0.3.0\n", encoding="utf-8")
    if drop:
        shutil.rmtree(place / drop) if (place / drop).is_dir() else (place / drop).unlink()
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "bundle")
    return repo, _git(repo, "rev-parse", "HEAD")


def _local_bundle(tmp_path: Path) -> Path:
    local = _copy(tmp_path, "local")
    shutil.copy2(bundle.SOURCE, local / "SOURCE")
    return local


def _contents(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_refresh_takes_the_commit_and_records_it(tmp_path: Path):
    repo, commit = _upstream(tmp_path)
    local = _local_bundle(tmp_path)
    bundle.refresh(commit, str(repo), bundle=local)
    assert (local / "VERSION").read_text(encoding="utf-8").startswith("bundle 2")
    assert bundle.read_source(local / "SOURCE")["commit"] == commit
    assert not list(local.glob(".refresh-*")), "the staging directory is left behind"


def test_refresh_of_an_incomplete_bundle_changes_nothing(tmp_path: Path):
    repo, commit = _upstream(tmp_path, drop="expected")
    local = _local_bundle(tmp_path)
    before = _contents(local)
    with pytest.raises(bundle.BundleError, match="lacks"):
        bundle.refresh(commit, str(repo), bundle=local)
    assert _contents(local) == before


@pytest.mark.parametrize(
    "commit", ["main", "v0.1.0", "69c29452", "69C2945229070EDE4CCF8CED0520B281A87A0D68"]
)
def test_a_commit_that_is_not_a_full_sha_is_refused(commit: str):
    with pytest.raises(bundle.BundleError, match="full 40-hex"):
        bundle.require_full_sha(commit)
    assert bundle.main(["refresh", "--commit", commit]) == 1
