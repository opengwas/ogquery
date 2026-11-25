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
changed. A commit is always a full 40-hex sha, never a branch or tag, which would move.
`refresh` assembles the new data beside the old and swaps it in only once every file is
there, so a commit that lacks part of the bundle leaves the committed copy untouched.
`--repository` takes a local opengwasdb clone instead of GitHub; the bundle's place in
it is `SOURCE`'s `path`.

Standard library and git only.
"""

from __future__ import annotations

import argparse
import re
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
#: A commit as `SOURCE` records it: the full sha, which names one tree for ever.
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


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
    require_full_sha(fields["commit"])
    return fields


def require_full_sha(commit: str) -> str:
    """`commit`, refused unless it is a full 40-hex sha: a branch, tag or short sha can
    name another tree tomorrow, and `SOURCE` must not."""
    if not FULL_SHA.match(commit):
        raise BundleError(f"{commit!r} is not a full 40-hex commit sha")
    return commit


def fetch(repository: str, commit: str, into: Path, path: str = UPSTREAM_PATH) -> Path:
    """The bundle directory `path` at `commit` of `repository`, checked out under `into`."""

    def git(*args: str) -> None:
        done = subprocess.run(["git", "-C", str(into), *args], capture_output=True, text=True)
        if done.returncode:
            raise BundleError(f"git {' '.join(args)}: {done.stderr.strip()}")

    into.mkdir(parents=True, exist_ok=True)
    git("init", "-q")
    git("config", "core.autocrlf", "false")
    git("fetch", "-q", "--depth", "1", repository, commit)
    git("checkout", "-q", "FETCH_HEAD", "--", path)
    return into / path


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
        upstream = fetch(
            repository or source["repository"], source["commit"], Path(scratch), source["path"]
        )
        return differences(BUNDLE, upstream)


def refresh(commit: str, repository: str | None = None, bundle: Path = BUNDLE) -> None:
    """Replace `bundle`'s data with `commit`'s, and record `commit` in its `SOURCE`."""
    require_full_sha(commit)
    source = read_source(bundle / "SOURCE")
    with tempfile.TemporaryDirectory() as scratch:
        upstream = fetch(
            repository or source["repository"], commit, Path(scratch) / "fetch", source["path"]
        )
        install(upstream, bundle)
    text = (bundle / "SOURCE").read_text(encoding="utf-8")
    (bundle / "SOURCE").write_text(
        text.replace(f"commit {source['commit']}", f"commit {commit}"), encoding="utf-8"
    )


def install(upstream: Path, bundle: Path) -> None:
    """Copy `upstream`'s data into `bundle`, all or nothing.

    Every data entry is first copied into a staging directory beside `bundle`; a missing
    entry raises before anything in `bundle` is touched. Only then is each old entry moved
    aside and the new one moved in, on the same file system.
    """
    missing = [name for name in DATA if not (upstream / name).exists()]
    if missing:
        raise BundleError(f"{upstream} lacks {missing}: not a contract bundle")
    staging = Path(tempfile.mkdtemp(prefix=".refresh-", dir=bundle))
    old = staging / "old"
    try:
        for name in DATA:
            if (upstream / name).is_dir():
                shutil.copytree(upstream / name, staging / name)
            else:
                shutil.copy2(upstream / name, staging / name)
        old.mkdir()
        for name in DATA:
            if (bundle / name).exists():
                (bundle / name).replace(old / name)
            (staging / name).replace(bundle / name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


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
            refresh(require_full_sha(args.commit), args.repository)
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
