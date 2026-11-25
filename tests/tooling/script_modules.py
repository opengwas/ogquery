"""Load a module from `scripts/` so it can be tested (from opengwasdb, opengwas/opengwasdb#333).

The scripts under `scripts/` are entry points, not part of the installed
package, so a test loads them by path. Kept in one place so the callers cannot drift.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load_script(name: str, directory: Path = SCRIPTS) -> ModuleType:
    """The script `directory`/`name`.py as a module. `directory` is on `sys.path` while it loads,
    as it is when the script is run, so a script may import a sibling by name."""
    path = directory / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so that anything the module resolves through
    # its own namespace -- a dataclass reads `sys.modules[cls.__module__]` --
    # finds it. Without this a script defining a dataclass cannot be loaded.
    sys.modules[name] = module
    sys.path.insert(0, str(directory))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(directory))
    return module


def install_stub(
    bin_dir: Path, name: str, body: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stub command `name` on `PATH`, running `body` with this interpreter.

    On Linux and macOS it is an executable script with this Python on its shebang line;
    on Windows, which runs no shebang, it is `name.py` with a `name.cmd` beside it that
    runs it, found through PATHEXT. `PATH` is joined with `os.pathsep`, so the stub is
    found first on either.
    """
    import os

    bin_dir.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        (bin_dir / f"{name}.py").write_text(body, encoding="utf-8")
        (bin_dir / f"{name}.cmd").write_text(
            f'@"{sys.executable}" "%~dp0{name}.py" %*\r\n', encoding="utf-8"
        )
    else:
        (bin_dir / name).write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
