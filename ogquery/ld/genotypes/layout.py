"""The Genotype Store sidecar's own layout, as a read opens it: root kind, arrays and relative
input pins (opengwas/ogld#91).

From opengwasdb `964ac1b2`'s `opengwasdb/ld/genotypes/layout.py` (opengwas/ogquery#11; ADR 0002,
amended): the names a read reaches. What a writer records -- `input_pin` and `fileset_record` --
stays in opengwasdb, whose writer and validation import the rest from here. The root key is the
kind's own name, `ogquery.ld.query.kind.KEY`, not a second spelling of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ogquery.ld.format import InputError
from ogquery.ld.query.kind import KEY
from ogquery.ld.store import layout

__all__ = ["ARRAYS", "KEY", "VERSION", "Fileset", "check_input", "mapped_arrays", "open_filesets"]

VERSION = "0.1.0"
ARRAYS = {
    "pgen_row": "<i8",
    "fileset": "<i4",
    "projection": "<f8",
    "norm": "<f8",
    "sign": "<f8",
    "basis": "<f8",
}


def check_input(root: Path, pin: dict[str, Any], *, hashes: bool = False) -> Path:
    path = root / str(pin["path"])
    try:
        stat = path.stat()
    except OSError as missing:
        raise InputError(
            f"{path}: recorded genotype input is missing or unreadable: {missing}"
        ) from missing
    for key, actual in (("size", stat.st_size), ("mtime_ns", stat.st_mtime_ns)):
        if actual != pin[key]:
            raise InputError(f"{path}: {key} changed: recorded {pin[key]}, found {actual}")
    if hashes and layout.file_sha256(path) != pin["sha256"]:
        raise InputError(f"{path}: SHA-256 changed from the recorded genotype input hash")
    return path


@dataclass(frozen=True)
class Fileset:
    """Persisted sample selection, without reparsing a mutable psam at query time."""

    pgen: Path
    psam: Path
    subset: npt.NDArray[np.uint32]
    order: npt.NDArray[np.int64] | None
    samples: int
    panel: str


def open_filesets(root: Path, records: list[dict[str, Any]]) -> list[Fileset]:
    out = []
    for record in records:
        paths = {name: check_input(root, record[name]) for name in ("pgen", "psam", "pvar")}
        out.append(
            Fileset(
                paths["pgen"],
                paths["psam"],
                np.asarray(record["subset"], dtype=np.uint32),
                None if record["order"] is None else np.asarray(record["order"], dtype=np.int64),
                record["samples"],
                record["panel"],
            )
        )
    return out


def mapped_arrays(root: Path, metadata: dict[str, Any]) -> dict[str, npt.NDArray[Any]]:
    out = {}
    for name, dtype in ARRAYS.items():
        path = root / "genotypes" / f"{name}.npy"
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as broken:
            raise InputError(
                f"{path}: missing or unreadable Genotype Store array: {broken}"
            ) from broken
        if array.dtype != np.dtype(dtype) or list(array.shape) != metadata[name]["shape"]:
            raise InputError(f"{path}: array dtype or shape differs from its recorded descriptor")
        out[name] = array
    return out
