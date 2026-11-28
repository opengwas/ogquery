"""A Genotype Store answers on X, Y and MT as it does on an autosome (opengwas/opengwasdb#519).

The HGDP+1kGP X, Y and MT builds are Genotype Stores (og-dev-hub ADR 0006). Their dosages follow
the engine's ploidy rule (ADR 0109, opengwasdb's): a male's one copy of non-PAR X, of Y and of MT
is stored 0/2, so r is plain Pearson r on the dosages and needs no sex. The reader holds no sex and
no chromosome list: it reads dosages, a norm and a projection per variant, so these tests write a
small store of each chromosome (`write_store`) and hold `matrix` and `proxies` to literal r.

Eight samples, dosages 0 or 2 (a haploid copy written as a homozygote), over three variants:

    a = 2 2 2 2 0 0 0 0      b = 2 2 2 0 0 0 0 0      c = 0 0 0 0 2 2 2 2   (ALT dosages)

Pearson r of two 0/1 vectors is phi: r(a, b) = 12 / sqrt(240) = 0.7746, which at the store's scale
of 511 is the code 396; r(a, c) = -1 (code -511); r(b, c) = -396 / 511 (ALT dosages). A1-signed,
with the alleles spelled as the build spells them (sorted, A1 first): `a` is A>C so its A1 is
REF and its sign -1; `b` is C>A so its A1 is ALT and its sign +1; `c` is A>C, sign -1.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import zarr
from spelling import sorted_identity
from zarr.codecs import ZstdCodec
from zarr.storage import LocalStore

import ogquery.ld

pytestmark = pytest.mark.genotype

TEST_STORE = Path(__file__).resolve().parent / "data" / "test-panel" / "genotype-store"
N = 8
SCALE = 511
CODE = 396  # round(12 / sqrt(240) * 511): 0.7746 * 511 = 395.8

#: ALT dosages of a, b, c over the eight samples (0/2: a haploid copy is a homozygote).
DOSAGES = np.array(
    [[2, 2, 2, 2, 0, 0, 0, 0], [2, 2, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 2, 2, 2]],
    dtype=np.float64,
)
#: REF and ALT of a, b, c.
ALLELES = (("A", "C"), ("C", "A"), ("A", "C"))
#: Where each chromosome's three variants lie: PAR1 is X:10,001-2,781,479, so these are non-PAR.
POSITIONS = {"X": (30_000_000, 30_100_000, 30_200_000), "Y": (15_000_000, 15_100_000, 15_200_000),
             "MT": (3_000, 3_500, 4_000)}
# A1-signed: r(a,b) = (-)(+) 396/511; r(a,c) = (-)(-)(-1) = -1; r(b,c) = (+)(-)(-396/511)
EXPECTED_R = np.array([
    [1.0, -CODE / SCALE, -1.0],
    [-CODE / SCALE, 1.0, CODE / SCALE],
    [-1.0, CODE / SCALE, 1.0],
])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pin(store: Path, path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": os.path.relpath(path, store), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": _sha256(path)}


def _write_pgen(path: Path) -> None:
    import pgenlib

    with pgenlib.PgenWriter(str(path).encode(), N, len(DOSAGES), False) as writer:
        for row in DOSAGES:
            writer.append_alleles(np.repeat((row // 2).astype(np.int32), 2))


def write_store(tree: Path, chrom: str, sex: str = "1") -> Path:
    """A Genotype Store of a, b and c on `chrom`, beside the pgen, psam and pvar it pins."""
    tree.mkdir(parents=True, exist_ok=True)
    positions = POSITIONS[chrom]
    pgen, psam, pvar = tree / "panel.pgen", tree / "panel.psam", tree / "panel.pvar"
    _write_pgen(pgen)
    # Every sample is a male by default: a Y Store holds males only, and a male's X, Y and MT are
    # 0/2. `sex` lets a test say otherwise, to show that no answer reads it.
    psam.write_text("#IID\tSEX\n" + "".join(f"s{i}\t{sex}\n" for i in range(N)), encoding="utf-8")
    identities = [sorted_identity(chrom, p, ref, alt)
                  for p, (ref, alt) in zip(positions, ALLELES, strict=True)]
    pvar.write_text(
        "#CHROM\tPOS\tID\tREF\tALT\tFILTER\n"
        + "".join(f"{chrom}\t{p}\t{i.alid}\t{ref}\t{alt}\t.\n"
                  for p, i, (ref, alt) in zip(positions, identities, ALLELES, strict=True)),
        encoding="utf-8",
    )
    store = tree / "genotype-store"
    shutil.copytree(TEST_STORE, store)
    for name in ("variants", "genotypes"):
        shutil.rmtree(store / name)
    for name in ("pgen", "psam", "pvar"):  # fixed mtimes: an open refuses a changed one
        os.utime(tree / f"panel.{name}", ns=(1_795_726_371_000_000_000,) * 2)
    _write_variants(store, chrom, positions, identities)
    _write_arrays(store, identities)
    document = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    record = document["attributes"]["genotype_store"]
    record.pop("pc_diagnostic", None)
    record["n_variants"] = len(DOSAGES)
    record["panel"] = {"id": "ploidy", "ancestry": "EUR", "n": N, "samples_sha256": None}
    record["filesets"] = [{
        "order": None, "panel": "ploidy", "samples": N, "subset": list(range(N)),
        **{name: _pin(store, tree / f"panel.{name}") for name in ("pgen", "psam", "pvar")},
    }]
    record["inputs"] = []
    record["arrays"] = {
        name: {"shape": list(np.load(store / "genotypes" / f"{name}.npy", mmap_mode="r").shape),
               "sha256": _sha256(store / "genotypes" / f"{name}.npy")}
        for name in ("basis", "fileset", "norm", "pgen_row", "projection", "sign")
    }
    record["adjustment"] = {"covariates": [], "k": 0, "k_target": 0, "k_min": None,
                            "pc_file": None, "pc_check": None}
    record["counts"] = {"alids_left_out": 0}
    record["variants_dropped"] = {"duplicate_reversed_indel": 0, "filter": 0, "info": 0,
                                  "mac": 0, "zero_variance": 0}
    record["variant_filters"] = {
        "filter": ["PASS", "."], "mac_min": 10, "info_min": None, "autosomes_only": False,
        "ploidy": {
            "chromosomes": [chrom], "sex_source": "psam SEX (1 male, 2 female)",
            "males": N, "females": 0, "male_x_coding": "0/2",
            "x_pars_grch38": [[10001, 2781479], [155701383, 156030895]], "mac_min_haploid": 10,
        },
    }
    document["attributes"].pop("alid_mmap", None)
    (store / "zarr.json").write_text(json.dumps(document), encoding="utf-8")
    return store


def _column(group: zarr.Group, name: str, data: npt.ArrayLike, dtype: str) -> None:
    group.create_array(name, data=np.asarray(data, dtype=dtype), chunks=(len(DOSAGES),),
                       fill_value=np.nan if dtype == "float32" else 0,
                       compressors=ZstdCodec(level=3))


def _write_variants(store: Path, chrom: str, positions: tuple[int, ...], identities: Any) -> None:
    root = zarr.open_group(LocalStore(str(store)), mode="r+")
    group = root.create_group("variants")
    group.attrs["chromosomes"] = [{"name": chrom, "row_start": 0, "row_stop": len(positions)}]
    _column(group, "position", positions, "int64")
    _column(group, "a1_frequency", [0.5, 0.375, 0.5], "float32")
    for name in ("alid", "a1", "a2"):
        group.create_array(
            name, data=np.array([getattr(i, name) for i in identities], dtype=np.str_),
            chunks=(len(positions),), compressors=ZstdCodec(level=3),
        )
    order = sorted(range(len(identities)), key=lambda i: identities[i].alid.encode())
    keys = np.zeros((len(order), 32), dtype=np.uint8)
    for at, i in enumerate(order):
        encoded = identities[i].alid.encode()
        keys[at, : len(encoded)] = list(encoded)
    index = group.create_group("alid_index")
    index.create_array("alid", data=keys.reshape(-1), compressors=ZstdCodec(level=3))
    index.create_array("row", data=np.array(order, dtype=np.int32), fill_value=-1,
                       compressors=ZstdCodec(level=3))


def _write_arrays(store: Path, identities: Any) -> None:
    arrays = store / "genotypes"
    arrays.mkdir()
    basis = np.full((N, 1), 1 / np.sqrt(N))
    projection = DOSAGES @ basis
    norm = np.sqrt((DOSAGES**2).sum(axis=1) - (projection**2).sum(axis=1))
    sign = np.array([1.0 if i.a1_is_alt else -1.0 for i in identities])
    for name, array in {
        "basis": basis, "projection": projection, "norm": norm, "sign": sign,
        "pgen_row": np.arange(len(DOSAGES), dtype="<i8"),
        "fileset": np.zeros(len(DOSAGES), dtype="<i4"),
    }.items():
        np.save(arrays / f"{name}.npy", array)


@pytest.fixture(params=["X", "Y", "MT"])
def opened(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    chrom = request.param
    store = write_store(tmp_path, chrom)
    with ogquery.ld.open(store) as reader:
        reader.chrom = chrom  # type: ignore[attr-defined]
        yield reader


def test_a_genotype_store_answers_r_on_x_y_and_mt_with_literal_values(opened: Any) -> None:
    chrom = opened.chrom
    alids = [f"{chrom}:{p}:A:C" for p in POSITIONS[chrom]]
    found = opened.matrix(alids)
    np.testing.assert_array_equal(found.r, EXPECTED_R)


def test_no_answer_reads_the_sexes(tmp_path: Path) -> None:
    """r is from dosages alone: the same dosages with every sample female answer the same."""
    male = ogquery.ld.open(write_store(tmp_path / "male", "X"))
    female = ogquery.ld.open(write_store(tmp_path / "female", "X", sex="2"))
    alids = [f"X:{p}:A:C" for p in POSITIONS["X"]]
    np.testing.assert_array_equal(male.matrix(alids).r, female.matrix(alids).r)
    male.close()
    female.close()


def test_a_genotype_store_answers_proxies_on_x_y_and_mt_with_literal_values(opened: Any) -> None:
    """At r² ≥ 0.5 every pair is a proxy of the other: r² = 0.6005 (396/511 squared) for a-b and
    b-c, 1 for a-c. Each query's proxies come in the order the store answers them."""
    chrom = opened.chrom
    a, b, c = (f"{chrom}:{p}:A:C" for p in POSITIONS[chrom])
    found = opened.proxies([a, b, c], r2=0.5, kb=1000)
    assert found.query.tolist() == [a, a, b, b, c, c]
    assert found.proxy.tolist() == [c, b, a, c, a, b]
    np.testing.assert_array_equal(
        found.r, [-1.0, -CODE / SCALE, -CODE / SCALE, CODE / SCALE, -1.0, CODE / SCALE]
    )
    np.testing.assert_allclose(
        found.r2, [1.0, (CODE / SCALE) ** 2, (CODE / SCALE) ** 2, (CODE / SCALE) ** 2, 1.0,
                   (CODE / SCALE) ** 2],
    )
    step = 500 if chrom == "MT" else 100_000  # the variants lie a step apart
    assert found.distance.tolist() == [2 * step, step, step, step, 2 * step, step]
    assert found.absent == ()
