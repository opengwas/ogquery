"""Paths as Windows answers two questions about them, for the tests that hold a read to it on any
platform (opengwas/ogquery#2). A test hands one of these to the one module under test, in place
of its `Path` or a path it reads, so nothing else in the process sees it.

- `BackslashRelative`: a path whose `relative_to` is spelt with `\\`, as a Windows checkout's is.
- `StatShifted`: a path whose `stat()` reports another ctime than the opened file's `fstat()`,
  for a file under an ALID companion's generation, as Windows with Python 3.14 disagreed in CI
  run 45664832290.
"""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath
from typing import Any

#: The platform's concrete path class, which a subclass must extend (PosixPath or WindowsPath).
_Concrete: Any = type(Path())


class BackslashRelative(_Concrete):
    def relative_to(self, *other: Any, **kwargs: Any) -> Any:
        return PureWindowsPath(*super().relative_to(*other, **kwargs).parts)


class _Shifted:
    """A stat result whose ctime is one nanosecond on."""

    def __init__(self, stat: os.stat_result) -> None:
        self._stat = stat

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stat, name)

    @property
    def st_ctime_ns(self) -> int:
        return self._stat.st_ctime_ns + 1


class StatShifted(_Concrete):
    def stat(self, *, follow_symlinks: bool = True) -> Any:
        found = super().stat(follow_symlinks=follow_symlinks)
        return _Shifted(found) if self.parent.parent.name == "alid_mmap" else found
