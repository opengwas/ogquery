"""A 0.2-shaped copy of a contract-bundle release, and a Variant Table reader for it.

ogquery reads format 0.3.0 alone and a 0.2.x release only with a reader for its Store
Variant Table (`open_release(path, legacy=...)`), which opengwasdb passes. Its tests
need a 0.2 release without opengwasdb, so this makes one from a bundle release the way
opengwasdb's `tests/format_0_2_3.py` does (`as_0_2_3`), less what only opengwasdb's
readers see:

- the ALID sidecar's four files, the position sidecar's three and the `source_alid`
  sidecar's three are deleted (ADR 0105 §4, §6, §8);
- `variant_alid_bytes.npy`/`variant_alid_rows.npy`, the 0.2 series' fixed-width ALID
  index, are written as `write_fixed_width_alid_index` (a copy of opengwasdb's);
- `variant_offsets.npy` is written, one entry per row. A 0.2 build stores each row's BGZF
  virtual offset there; the Store Reader only hands an entry back to the table's
  `line_at`, so `GzipVariantTable` stores the row's index instead and reads it back;
- each manifest's `format_version` and each Indexed Variant Subset's
  `source_format_version` are re-spelt, byte for byte otherwise.

`index.sqlite` and the Top-Hit tiers' `abs_z` and `p_value`, which a 0.2 build also
wrote, are not restored: the Store Reader reads neither (opengwasdb's validator does).
A Hybrid is refused: its 0.2 Dense Component carries its own copy of the shared rows,
which only opengwasdb's writer can give back (`with_dense_copies`).

`GzipVariantTable` is a `LegacyVariantTable` in the standard library: a BGZF file is a
series of gzip members, which `gzip` reads whole. It answers a range by scanning, which
is all a fixture of a few hundred rows needs; opengwasdb's reader seeks with tabix.
"""

from __future__ import annotations

import gzip
import json
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from ogquery.alid import ALID_DTYPE, is_indexable_alid
from ogquery.alid_sidecar import ALID_SIDECAR_FILENAMES, AlidSidecar
from ogquery.position_sidecar import POSITION_SIDECAR_FILENAMES
from ogquery.source_alid_sidecar import SOURCE_ALID_SIDECAR_FILENAMES
from ogquery.variant_axis import variant_alid_bytes_path, variant_alid_rows_path


def write_fixed_width_alid_index(directory: Path, alids: list[str]) -> None:
    """The 0.2 series' ALID index for `alids` (one per row), as opengwasdb `e93fba8f` wrote it."""
    indexable = [(alid, i) for i, alid in enumerate(alids) if is_indexable_alid(alid)]
    alid_bytes = np.array([alid for alid, _ in indexable], dtype=ALID_DTYPE)
    row_indices = np.array([i for _, i in indexable], dtype="int32")
    sort_order = np.argsort(alid_bytes, kind="stable")
    np.save(variant_alid_bytes_path(directory), alid_bytes[sort_order])
    np.save(variant_alid_rows_path(directory), row_indices[sort_order])


def as_0_2(release: Path, version: str = "0.2.3") -> Path:
    """The 0.3.0 Dense or Ragged release at `release`, turned into a `version` one in place."""
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    if manifest["primary_layout"] == "hybrid":
        raise ValueError(f"{release} is a Hybrid, whose 0.2 copy only opengwasdb can write")
    sidecar = AlidSidecar.open(release)
    assert sidecar is not None, f"{release} has no ALID sidecar: not a 0.3.0 release"
    alids = sidecar.alids(np.arange(sidecar.n_rows)).tolist()
    del sidecar
    for name in (*ALID_SIDECAR_FILENAMES, *POSITION_SIDECAR_FILENAMES):
        (release / name).unlink()
    for name in SOURCE_ALID_SIDECAR_FILENAMES:
        (release / name).unlink()
    write_fixed_width_alid_index(release, alids)
    np.save(release / "variant_offsets.npy", np.arange(len(alids), dtype=np.uint64))
    _restamp(release / "manifest.json", "format_version", version)
    for subset in sorted((release / "data.zarr" / "indexed_subsets").glob("*/zarr.json")):
        _restamp(subset, "source_format_version", version)
    return release


def copy_as_0_2(release: Path, into: Path, version: str = "0.2.3") -> Path:
    """A `version` copy of the bundle release at `release`, under `into`."""
    return as_0_2(Path(shutil.copytree(release, into / release.name)), version)


def _restamp(path: Path, key: str, version: str) -> None:
    text = path.read_text(encoding="utf-8")
    stamp = f'"{key}": "0.3.0"'
    assert text.count(stamp) == 1, f"{path} does not record {key} 0.3.0 once"
    path.write_text(text.replace(stamp, f'"{key}": "{version}"'), encoding="utf-8")


class GzipVariantTable:
    """A `LegacyVariantTable` over a release's `variants.tsv.gz`, read with `gzip`."""

    def __init__(self, release: Path, *, aliases: bool) -> None:
        with gzip.open(release / "variants.tsv.gz", "rt", encoding="utf-8", newline="") as f:
            self._rows = [line for line in f if not line.startswith("#")]
        self.aliases = aliases
        self.reads: list[str] = []

    def fetch(self, chromosome: str, start: int, end: int) -> Iterable[str]:
        """Rows of `chromosome` whose 1-based position lies in the 0-based half-open
        `[start, end)`, as tabix answers on the table's position column."""
        self.reads.append("fetch")
        for row in self._rows:
            fields = row.split("\t", 2)
            if fields[0] == chromosome and start < int(fields[1]) <= end:
                yield row

    def line_at(self, offset: int) -> str:
        self.reads.append("line_at")
        return self._rows[int(offset)]

    def rows(self) -> Iterator[str]:
        self.reads.append("rows")
        return iter(self._rows)

    def alias_index(self, alias: str) -> int | None:
        """None: `variant_aliases` was empty in every release measured (ADR 0102 §1)."""
        self.reads.append("alias_index")
        return None

    def close(self) -> None:
        pass
