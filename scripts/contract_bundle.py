#!/usr/bin/env python3
"""The contract bundle is opengwasdb's: check it, or refresh it from a commit.

opengwasdb builds the Store Reader contract bundle (its `tests/store_reader_contract/`,
`generate.py`), because only it has the builders; ogquery reads it and never writes it.
`tests/contract/SOURCE` names the opengwasdb commit the committed copy came from. The
bundle's data -- `releases/`, `expected/`, `VERSION` and `.gitattributes` -- is that
commit's, byte for byte. The rest of `tests/contract/` is ogquery's: `contract_reader.py`
and the test import `ogquery`, and the README says where the bundle comes from.

    python scripts/contract_bundle.py check
    python scripts/contract_bundle.py refresh --commit <opengwasdb sha>

`check` fetches the named commit and fails on any file that differs, is missing or is
extra. `refresh` replaces the data with another commit's and rewrites `SOURCE`; review
the diff, run the tests, and raise nothing here: the bundle's own `VERSION` says what
changed. `--repository` takes a local opengwasdb clone instead of GitHub.

Standard library and git only.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "tests" / "contract"
SOURCE = BUNDLE / "SOURCE"
#: Where the bundle lives in opengwasdb.
UPSTREAM_PATH = "tests/store_reader_contract"
#: The bundle's data: what opengwasdb's generator writes, and what ogquery never edits.
DATA = ("releases", "expected", "VERSION", ".gitattributes")
DEFAULT_REPOSITORY = "https://github.com/opengwas/opengwasdb.git"


class BundleError(Exception):
    """The committed bundle is not the one `SOURCE` names, or it cannot be fetched."""


def read_source(path: Path = SOURCE) -> dict[str, str]:
    """`SOURCE`: `key value` lines, `#` comments ignored."""
    fields = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            key, _, value = line.partition(" ")
            fields[key] = value.strip()
    missing = {"repository", "commit", "path"} - set(fields)
    if missing:
        raise BundleError(f"{path} does not name {sorted(missing)}")
    return fields


def fetch(repository: str, commit: str, into: Path) -> Path:
    """The bundle directory at `commit` of `repository`, checked out under `into`."""

    def git(*args: str) -> None:
        done = subprocess.run(["git", "-C", str(into), *args], capture_output=True, text=True)
        if done.returncode:
            raise BundleError(f"git {' '.join(args)}: {done.stderr.strip()}")

    git("init", "-q")
    git("config", "core.autocrlf", "false")
    git("fetch", "-q", "--depth", "1", repository, commit)
    git("checkout", "-q", "FETCH_HEAD", "--", UPSTREAM_PATH)
    return into / UPSTREAM_PATH


def _files(directory: Path) -> dict[str, bytes]:
    out = {}
    for name in DATA:
        path = directory / name
        if path.is_file():
            out[name] = path.read_bytes()
        elif path.is_dir():
            for file in sorted(path.rglob("*")):
                if file.is_file():
                    out[file.relative_to(directory).as_posix()] = file.read_bytes()
    return out


def differences(committed: Path, upstream: Path) -> list[str]:
    """Every data file that differs between the two bundles; none when they agree."""
    ours, theirs = _files(committed), _files(upstream)
    out = [f"missing here: {name}" for name in sorted(set(theirs) - set(ours))]
    out += [f"not in opengwasdb's bundle: {name}" for name in sorted(set(ours) - set(theirs))]
    both = sorted(set(ours) & set(theirs))
    out += [f"differs: {name}" for name in both if ours[name] != theirs[name]]
    return out


def check(repository: str | None = None) -> list[str]:
    source = read_source()
    with tempfile.TemporaryDirectory() as scratch:
        upstream = fetch(repository or source["repository"], source["commit"], Path(scratch))
        return differences(BUNDLE, upstream)


def refresh(commit: str, repository: str | None = None) -> None:
    source = read_source()
    with tempfile.TemporaryDirectory() as scratch:
        upstream = fetch(repository or source["repository"], commit, Path(scratch))
        for name in DATA:
            target = BUNDLE / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            if (upstream / name).is_dir():
                shutil.copytree(upstream / name, target)
            else:
                shutil.copy2(upstream / name, target)
    text = SOURCE.read_text(encoding="utf-8")
    SOURCE.write_text(text.replace(f"commit {source['commit']}", f"commit {commit}"), "utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="the committed bundle is SOURCE's commit's")
    refreshing = commands.add_parser("refresh", help="take the bundle from another commit")
    refreshing.add_argument("--commit", required=True, help="a full opengwasdb commit sha")
    for command in commands.choices.values():
        command.add_argument("--repository", default=None, help="a URL or a local clone")
    args = parser.parse_args(argv)
    try:
        if args.command == "refresh":
            refresh(args.commit, args.repository)
            print(f"tests/contract now holds the bundle of opengwasdb {args.commit}")
            return 0
        found = check(args.repository)
    except BundleError as error:
        print(f"contract-bundle: {error}", file=sys.stderr)
        return 1
    for line in found:
        print(line, file=sys.stderr)
    if found:
        print("the committed bundle is not the one tests/contract/SOURCE names", file=sys.stderr)
        return 1
    print(f"ok: tests/contract is opengwasdb {read_source()['commit']}'s bundle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
