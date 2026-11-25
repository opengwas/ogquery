"""The package version comes from the git tag.

Ported from opengwasdb's `tests/test_version.py` (opengwas/opengwasdb#333) for
opengwas/ogquery#1.

Every merge to `main` is tagged `vX.Y.Z`, and `hatch-vcs` turns the tag into
the distribution's version at build time. Two things can therefore be wrong and
are checked here separately:

* **The reader.** `ogquery.version` reports what the installed distribution
  says, and `0.0.0+unknown` when nothing is installed. A synthetic
  `*.dist-info` on `sys.path` proves the first without depending on this
  checkout's environment; the second is the fallback that replaced a literal
  that could be mistaken for a release.
* **The tag rule.** The configured `git describe` command, run in a throwaway
  repository, picks a `v*` tag and ignores a stray one, and the configured
  `tag_regex` turns the tag on `HEAD` into its version.

The build itself is not run here: it needs the build backend from the network.
The build → metadata step is checked where a wheel is built, by
`scripts/smoke_test_installed.py --version`, which requires the installed
package to report the version the release was built as.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any

from packaging.version import Version

import ogquery
from ogquery import version as version_module

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _raw_options() -> Any:
    """`[tool.hatch.version.raw-options]` as TOML parses it.

    `tomllib` gives a nested tree of `str`, `list` and `dict`, so the value is
    `Any`; the tests below index into it and assert what they find, which is
    what a table of options needs.
    """
    config = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return config["tool"]["hatch"]["version"]["raw-options"]


def _describe_command() -> list[str]:
    return list(_raw_options()["scm"]["git"]["describe_command"])


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


# ── The reader ──────────────────────────────────────────────────────────────


def test_the_package_reports_the_installed_distribution_version() -> None:
    from importlib.metadata import version as installed_metadata_version

    assert ogquery.__version__ == installed_metadata_version("ogquery")
    assert version_module.installed_version() == installed_metadata_version("ogquery")


def test_the_reader_reports_a_distribution_that_did_not_exist_at_import_time(
    tmp_path: Path, monkeypatch
) -> None:
    """The version is read, not compiled in: a distribution found on `sys.path`
    is what is reported, whatever this checkout's own metadata says."""
    dist_info = tmp_path / "ogquery-9.9.9.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: ogquery\nVersion: 9.9.9\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    assert version_module.installed_version() == "9.9.9"


def test_the_fallback_is_a_version_no_release_can_claim(monkeypatch) -> None:
    def missing(_name: str) -> str:
        raise PackageNotFoundError(_name)

    monkeypatch.setattr(version_module, "version", missing)
    assert version_module.installed_version() == version_module.FALLBACK_VERSION
    assert version_module.FALLBACK_VERSION == "0.0.0+unknown"


def test_the_fallback_parses_and_sorts_below_every_release() -> None:
    """`0.0.0+unknown` is valid PEP 440 and obviously not a release.

    opengwasdb's fallback was once `"0.3.0"`, which a reader would take for a
    real build; the local segment says the version is unknown.
    """
    fallback = Version(version_module.FALLBACK_VERSION)
    assert fallback.public == "0.0.0"
    assert fallback.local == "unknown"
    assert fallback < Version("0.0.1")


# ── The tag rule ────────────────────────────────────────────────────────────


def test_pyproject_takes_the_version_from_tags_with_the_same_fallback() -> None:
    """The build config and the runtime fallback are one value, not two.

    A wheel built without git metadata carries `fallback_version`; a process
    without installed metadata reads `FALLBACK_VERSION`. If they disagreed, the
    same unknown build would report two versions (opengwasdb#333).
    """
    config = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = config["project"]
    assert project["dynamic"] == ["version"]
    assert "version" not in project, "a static version shadows the tag"

    hatch_version = config["tool"]["hatch"]["version"]
    assert hatch_version["source"] == "vcs"
    options = hatch_version["raw-options"]
    assert options["fallback_version"] == version_module.FALLBACK_VERSION
    assert options["tag_regex"] == r"^v(?P<version>.*)$"
    # setuptools-scm's default node-and-date, as opengwasdb: a clean checkout
    # carries the commit alone (`+g1a2b3c4`), so the version is stable for as
    # long as the code is; only a dirty tree gains a date.
    assert options["local_scheme"] == "node-and-date"


def test_git_describe_is_restricted_to_release_tags() -> None:
    """Parsing `v*` is not enough: `git describe` must not *select* another tag.

    A stray `99.0.0` tag on `HEAD` made the build fail with "Can't parse version
    from tag '99.0.0'" (spec review, finding 3): `tag_regex` restricts parsing
    only, and the default describe glob accepts it. The configured command is
    run here, in a throwaway repository, because that command is what decides.
    """
    command = _describe_command()
    assert "--match" in command
    assert command[command.index("--match") + 1] == "v*"


def test_the_configured_describe_ignores_a_stray_tag(tmp_path: Path) -> None:
    root = _tagged_repository(tmp_path)

    described = subprocess.run(
        _describe_command(), cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()

    assert described.startswith("v0.2.0-1-g"), described
    assert "99.0.0" not in described


def test_without_its_match_the_command_would_take_the_stray_tag(
    tmp_path: Path,
) -> None:
    """The negative that makes the test above mean something."""
    root = _tagged_repository(tmp_path)
    unfiltered = [arg for arg in _describe_command() if arg not in {"--match", "v*"}]

    described = subprocess.run(
        unfiltered, cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()

    assert described.startswith("99.0.0-0-g"), described


def test_the_tag_on_head_parses_to_its_version(tmp_path: Path) -> None:
    """A tag on `HEAD` is the version, through the configured regex."""
    root = _tagged_repository(tmp_path)
    exact = _git(root, "describe", "--tags", "--exact-match", "v0.2.0")
    match = re.compile(str(_raw_options()["tag_regex"])).match(exact)
    assert match is not None
    assert match["version"] == "0.2.0"


def _tagged_repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    (root / "file.txt").write_text("one\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "one")
    _git(root, "tag", "--annotate", "v0.2.0", "--message", "Release v0.2.0")
    (root / "file.txt").write_text("two\n", encoding="utf-8")
    _git(root, "commit", "-q", "-am", "two")
    _git(root, "tag", "99.0.0")
    return root
