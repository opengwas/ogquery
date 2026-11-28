"""A 0.4.0-shaped copy of a contract-bundle release, written to spec §4.0 and §21.5.

The bundle's releases are format 0.3.0 and are opengwasdb's, read only; the 0.4.0 bundle is
built by opengwasdb's integration branch (ogquery#18). Until it lands, ogquery's tests need
a 0.4.0 release, so this makes one from a 0.3.0 release, the way `format_0_2` makes a 0.2 one:

- the `variant_alid_*` sidecar is replaced by the `variant_cpra_*` one, and the
  `variant_source_alid_*` sidecar by the `variant_source_cpra_*` one (§21.5), each laid out
  as its 0.3.0 original (`ogquery.alid_sidecar.encode_alids`, `encode_source_alids`);
- every SNV's key is respelt REF:ALT with **ALT the allele its statistics were signed for**
  (0.3.0's A1, the effect allele), so the stored values are the ones the 0.3.0 release holds
  and every read must give them back unchanged; an indel keeps its reference-first key,
  `A1` as REF. Every new CPRA is asserted normal by shape (`trim_alleles` changes nothing);
- `variant_rsid_*` is deleted: 0.4.0's rsID index is #524's and no layout is decided, so the
  reader opens none (a test supplies the one thing it needs, rsID names, by a stand-in);
- each manifest (a Hybrid's Dense Component's too) is restamped 0.4.0 and gains
  `cpra_normal_form`, `provenance.reference_sequence` and `provenance.admission` (§2); each
  Indexed Variant Subset's `source_format_version` is restamped;
- `variants.tsv.gz` is rewritten with 0.4.0's columns (`ref`, `alt`, `cpra`, `source_cpra`
  where 0.3's `effect_allele`, `other_allele`, `alid`, `source_alid` stood), plain gzip, which
  the tests' table reader reads as it reads a BGZF file.

`as_0_4(release, keys=..., resign=...)` can also **re-sign** rows, so a release carries values
a 0.3.0 release does not: `keys` names each row's CPRA, and every row in `resign` has its stored
z negated and its EAF complemented, as a build signs a statistic for ALT when the source's
effect allele was REF (§5, §9). That is only possible on a release whose `z` is `int16_fixed`
and whose `eaf` is `float32` (`dense_ranges`), where the stored planes are what is written.

One limit the tests accept: the rows are not re-sorted by (REF, ALT) as §4.0 sorts a
Variant Table, so rows at one position may stand in 0.3.0's allele order (no read depends on
it: the CPRA sidecar sorts its own keys).
"""

from __future__ import annotations

import gzip
import json
import shutil
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from format_0_2 import _alids, _restamp

from ogquery.alid_sidecar import AlidSidecar, encode_alids
from ogquery.cpra import trim_alleles
from ogquery.cpra_sidecar import CPRA_SIDECAR_FILENAMES, SOURCE_CPRA_SIDECAR_FILENAMES
from ogquery.npy_sidecar import release_maps
from ogquery.source_alid_sidecar import SourceAlidSidecar, encode_source_alids

#: A sequence digest as the manifest records one: 64 lower-case hex characters.
DIGEST = "263979d2475a76b5396c58dbeee0138d3548cf5011f53f2189662466c789e258"


def normal_form_entry(assembly: str = "GRCh38", digest: str = DIGEST) -> dict[str, Any]:
    """A 0.4.0 manifest's `cpra_normal_form` entry (spec §2)."""
    return {
        "alleles": "ref_alt",
        "signed_for": "alt",
        "indels": "left_aligned",
        "sequence": {"assembly": assembly, "digest": digest},
    }


def reference_sequence_entry(assembly: str = "GRCh38", digest: str = DIGEST) -> dict[str, Any]:
    """A 0.4.0 manifest's `provenance.reference_sequence` (spec §2): seven keys."""
    return {
        "assembly": assembly,
        "digest": digest,
        "sha256": "ab" * 32,
        "format": "ogquery-byte-genome",
        "format_version": "1.0",
        "path": "/reference/GRCh38.byte-genome",
        "source": {"fasta": "GRCh38.primary.fa.gz", "sha256": "cd" * 32},
    }


def admission_entry() -> dict[str, Any]:
    """A 0.4.0 manifest's `provenance.admission` (spec §2)."""
    return {"rules": "admission/17", "dropped": {"ref_mismatch": 0, "contig_not_primary": 3}}


def cpra_of(alid: str) -> str:
    """The 0.4.0 key of the 0.3.0 row `alid`: an SNV REF:ALT with ALT its A1, an indel as it was."""
    chrom, pos, a1, a2 = alid.split(":")
    if len(a1) == len(a2):
        chrom, pos, ref, alt = chrom, pos, a2, a1
    else:
        ref, alt = a1, a2
    assert trim_alleles(int(pos), ref, alt) == (int(pos), ref, alt), f"{alid} is not normal"
    return f"{chrom}:{pos}:{ref}:{alt}"


def as_0_4(
    release: Path,
    *,
    keys: Sequence[str] | None = None,
    resign: Collection[int] = (),
) -> Path:
    """The 0.3.0 release at `release`, rewritten as a 0.4.0 one in place (module docstring).

    `keys` gives each row's CPRA where the default (`cpra_of`) is not wanted, and `resign` the
    rows whose stored z and EAF are re-signed for ALT."""
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == "0.3.0", f"{release} is not a 0.3.0 release"
    alids = _alids(release)
    sources = _sources(release, alids)
    cpras = [cpra_of(alid) for alid in alids] if keys is None else list(keys)
    assert len(cpras) == len(alids)
    for cpra in cpras:
        _, pos, ref, alt = cpra.split(":")
        assert trim_alleles(int(pos), ref, alt) == (int(pos), ref, alt), f"{cpra} is not normal"
    renamed = dict(zip(alids, cpras, strict=True))
    values = [None if s is None else renamed.get(s, s) for s in sources]
    encoded = encode_alids(cpras)
    for name, array in zip(
        CPRA_SIDECAR_FILENAMES,
        (encoded.utf8, encoded.offsets, encoded.keys, encoded.order),
        strict=True,
    ):
        np.save(release / name, array)
    source = encode_source_alids(values, cpras)
    for name, array in zip(
        SOURCE_CPRA_SIDECAR_FILENAMES, (source.state, source.utf8, source.offsets), strict=True
    ):
        np.save(release / name, array)
    for name in (*AlidSidecar.FILENAMES, *SourceAlidSidecar.FILENAMES):
        (release / name).unlink()
    for name in ("variant_rsid_bytes.npy", "variant_rsid_rows.npy"):
        (release / name).unlink()
    _write_table(release, cpras, values)
    if resign:
        _resign(release, resign)
    _restamp_manifest(release / "manifest.json")
    if (release / "dense" / "manifest.json").exists():
        _restamp_manifest(release / "dense" / "manifest.json")
    for subset in sorted((release / "data.zarr" / "indexed_subsets").glob("*/zarr.json")):
        _restamp(subset, "source_format_version", "0.4.0")
    return release


def _write_table(release: Path, cpras: list[str], sources: list[str | None]) -> None:
    """`variants.tsv.gz` as 0.4.0 spells it: the table's columns keep their positions, and
    `ref`, `alt`, `cpra` and `source_cpra` stand where 0.3's allele, ALID and source columns did."""
    with gzip.open(release / "variants.tsv.gz", "rt", encoding="utf-8", newline="") as f:
        old = [line.rstrip("\n").split("\t") for line in f if not line.startswith("#")]
    lines = ["#chromosome\tposition\tvariant_index\tref\talt\tcpra\trsid\tsource_cpra"]
    for fields, cpra, source in zip(old, cpras, sources, strict=True):
        chrom, pos, index, _a1, _a2, _alid, rsid, _source = fields
        c, p, ref, alt = cpra.split(":")
        assert (c, p) == (chrom, pos), "a row keeps its position"
        lines.append("\t".join([chrom, pos, index, ref, alt, cpra, rsid, source or "."]))
    with gzip.open(release / "variants.tsv.gz", "wt", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")


def _resign(release: Path, rows: Collection[int]) -> None:
    """Negate the stored z and complement the stored EAF of `rows`, for a Dense release whose
    planes are written as they read (module docstring)."""
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    encoding = manifest["encoding"]
    assert encoding["z"]["kind"] == "int16_fixed" and encoding["eaf"]["kind"] == "float32", (
        "only a release with int16_fixed z and float32 eaf can be re-signed as stored"
    )
    group = zarr.open_group(str(release / "data.zarr"), mode="r+")
    index = sorted(rows)
    z, eaf = group["z"], group["eaf"]
    codes = z[:]
    assert int(codes[index].min()) > -32768, "a z code is not the missing marker"
    codes[index] = -codes[index]
    z[:] = codes
    frequencies = eaf[:]
    frequencies[index] = np.float32(1) - frequencies[index]
    eaf[:] = frequencies


def copy_as_0_4(release: Path, into: Path) -> Path:
    """A 0.4.0 copy of the bundle release at `release`, under `into`."""
    return as_0_4(Path(shutil.copytree(release, into / release.name)))


def cpras(release: Path) -> list[str]:
    """Every row's CPRA, read from a 0.4.0 release's CPRA sidecar."""
    from ogquery.cpra_sidecar import CpraSidecar

    sidecar = CpraSidecar.open(release)
    assert sidecar is not None, f"{release} has no CPRA sidecar"
    out = [str(c) for c in sidecar.alids(np.arange(sidecar.n_rows)).tolist()]
    release_maps(sidecar)
    return out


def _sources(release: Path, alids: list[str]) -> list[str | None]:
    """Each row's 0.3.0 `source_alid`, from its sidecar."""
    sidecar = AlidSidecar.open(release)
    sources = SourceAlidSidecar.open(release)
    assert sidecar is not None and sources is not None, f"{release} has no 0.3.0 sidecars"
    values: list[str | None] = sources.values(np.arange(len(alids)), sidecar.alids).tolist()
    release_maps(sidecar)
    release_maps(sources)
    return values


def _restamp_manifest(path: Path) -> None:
    _restamp(path, "format_version", "0.4.0")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assembly = manifest["reference_assembly"]
    manifest["cpra_normal_form"] = normal_form_entry(assembly)
    manifest["provenance"]["reference_sequence"] = reference_sequence_entry(assembly)
    manifest["provenance"]["admission"] = admission_entry()
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
