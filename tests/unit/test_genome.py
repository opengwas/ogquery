"""The byte genome: written from contig sequences, memory-mapped and read with numpy alone
(opengwas/ogquery#14, ADR 0003, opengwasdb ADR 0108 §7 and §9)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

from ogquery.cpra import ContigNotHeld, PositionOutsideContig
from ogquery.genome import (
    ByteGenome,
    ByteGenomeMismatch,
    RefusedContig,
    UnreadableByteGenome,
    sidecar_path,
    write_byte_genome,
)

#: The digest of contig `1` = `GATCAAAA` and `MT` = `ACGT`, from ADR 0108 §9's definition,
#: computed by the shell (opengwasdb's `test_the_digest_is_of_the_sequence_not_the_file`):
#: printf '1\t8\t<sha256 GATCAAAA>\nMT\t4\t<sha256 ACGT>\n' | sha256sum
DIGEST_1_MT = "9290091f6ee828a6013296a1b5c2bb577dcd7aae644c5cfec11677449b97b55f"
PROVENANCE = {"source": "test.fa", "source_sha256": "0" * 64, "derived_by": "pytest"}

Contigs = Iterable[tuple[str, Iterable[str | bytes]]]


def _write(path: Path, contigs: Contigs) -> Path:
    return write_byte_genome(path, contigs, assembly="GRCh38", provenance=PROVENANCE)


@pytest.fixture
def genome_path(tmp_path: Path) -> Path:
    # Soft-masked and chunked unevenly, as a FASTA's lines arrive: neither may show.
    return _write(
        tmp_path / "g.bytes",
        [("1", ["gatc", b"aA", "", "aa"]), ("2", ["NNRY", "acgt"]), ("MT", [b"ACGT"])],
    )


def test_a_genome_round_trips_through_fetch_and_bases(genome_path: Path) -> None:
    with ByteGenome.open(genome_path) as genome:
        assert genome.contigs == ("1", "2", "MT")
        assert [genome.length(c) for c in genome.contigs] == [8, 8, 4]
        assert genome.assembly == "GRCh38"
        assert genome.provenance == PROVENANCE
        assert genome.fetch("1", 0, 8) == "GATCAAAA"
        assert genome.fetch("chr1", 3, 5) == "CA"  # a label is read as an identity reads it
        assert genome.fetch("1", 8, 8) == ""
        # Bases are stored as the source has them, upper-cased: never rewritten.
        assert genome.fetch("2", 0, 8) == "NNRYACGT"
        assert genome.fetch("chrM", 0, 4) == "ACGT"
        got = genome.bases("1", np.array([1, 4, 8, 4], dtype=np.int64))
        assert got.dtype == np.uint8
        assert got.tobytes() == b"GCAC"
        assert genome.bases("MT", np.array([4, 1], dtype=np.int32)).tobytes() == b"TA"
        assert genome.bases("2", np.array([], dtype=np.int64)).shape == (0,)
    data = genome_path.read_bytes()
    assert data == b"GATCAAAANNRYACGTACGT"  # raw uint8, no separators
    sidecar = json.loads(sidecar_path(genome_path).read_text(encoding="utf-8"))
    assert sidecar["format_version"] == "1.0"
    assert sidecar["data_sha256"] == hashlib.sha256(data).hexdigest()
    assert [(c["label"], c["offset"], c["length"]) for c in sidecar["contigs"]] == [
        ("1", 0, 8),
        ("2", 8, 8),
        ("MT", 16, 4),
    ]
    assert sidecar["contigs"][0]["sha256"] == hashlib.sha256(b"GATCAAAA").hexdigest()


def test_fetch_and_bases_refuse_what_the_genome_does_not_hold(genome_path: Path) -> None:
    with ByteGenome.open(genome_path) as genome:
        with pytest.raises(ContigNotHeld, match="22"):
            genome.fetch("22", 0, 1)
        with pytest.raises(ContigNotHeld):
            genome.fetch("HLA-A", 0, 1)
        with pytest.raises(ContigNotHeld):
            genome.bases("X", np.array([1]))
        with pytest.raises(PositionOutsideContig, match="8"):
            genome.fetch("1", 7, 9)
        with pytest.raises(PositionOutsideContig):
            genome.fetch("1", -1, 2)
        with pytest.raises(PositionOutsideContig):
            genome.fetch("1", 5, 4)
        # 1-based: 0 is before the contig, and 9 past contig 1's 8 bases.
        with pytest.raises(PositionOutsideContig):
            genome.bases("1", np.array([1, 0]))
        with pytest.raises(PositionOutsideContig):
            genome.bases("1", np.array([9, 1]))
        with pytest.raises(TypeError, match="integer"):
            genome.bases("1", np.array([1.0]))


@pytest.mark.parametrize(
    ("contigs", "match"),
    [
        ([("1", ["A"]), ("chr2", ["A"])], "canonical"),
        ([("1", ["A"]), ("23", ["A"])], "canonical"),
        ([("1", ["A"]), ("1_KI270706v1_random", ["A"])], "primary"),
        ([("1", ["A"]), ("HLA-A", ["A"])], "primary"),
        ([("1", ["A"]), ("1", ["C"])], "twice"),
        ([("2", ["A"]), ("1", ["A"])], "order"),
        ([("X", ["A"]), ("22", ["A"])], "order"),
        ([("1", ["A"]), ("2", [])], "empty"),
        ([("1", ["A"]), ("2", ["", b""])], "empty"),
        ([("1", ["AC-T"])], "'-'"),
        ([("1", ["AC\nT"])], "base"),
        ([("1", ["ACÄT"])], "base"),
        ([], "no contig"),
    ],
)
def test_the_writer_refuses_and_leaves_nothing_behind(
    tmp_path: Path, contigs: Contigs, match: str
) -> None:
    with pytest.raises(RefusedContig, match=match):
        _write(tmp_path / "g.bytes", contigs)
    assert list(tmp_path.iterdir()) == []


def test_the_writer_never_overwrites(genome_path: Path) -> None:
    before = genome_path.read_bytes()
    with pytest.raises(FileExistsError):
        _write(genome_path, [("1", ["C"])])
    assert genome_path.read_bytes() == before
    sidecar_path(genome_path).unlink()
    with pytest.raises(FileExistsError):  # the data file alone is enough to refuse
        _write(genome_path, [("1", ["C"])])


def test_a_sidecar_alone_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "g.bytes"
    sidecar_path(path).write_text("not mine", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _write(path, [("1", ["C"])])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["g.bytes.json"]
    assert sidecar_path(path).read_text(encoding="utf-8") == "not mine"


@pytest.mark.skipif(sys.platform == "win32", reason="a symlink needs a privilege on Windows")
@pytest.mark.parametrize("which", ["data", "sidecar"])
def test_a_dangling_symlink_is_never_overwritten(tmp_path: Path, which: str) -> None:
    path = tmp_path / "g.bytes"
    link = path if which == "data" else sidecar_path(path)
    link.symlink_to(tmp_path / "nowhere")
    with pytest.raises(FileExistsError):
        _write(path, [("1", ["C"])])
    assert link.is_symlink() and not (tmp_path / "nowhere").exists()


@pytest.mark.parametrize("which", ["data", "sidecar"])
def test_a_name_taken_while_writing_is_never_overwritten(
    genome_path: Path, which: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check before writing is a courtesy; the publish itself must refuse a name that
    appeared since (another writer), so it may not be `exists()` then `os.replace`."""
    before = genome_path.read_bytes(), sidecar_path(genome_path).read_bytes()
    if which == "sidecar":
        genome_path.unlink()  # only the sidecar is taken
    monkeypatch.setattr(os.path, "lexists", lambda _: False)
    monkeypatch.setattr(Path, "exists", lambda self, **_: False)
    with pytest.raises(FileExistsError):
        _write(genome_path, [("1", ["C"])])
    monkeypatch.undo()
    assert sidecar_path(genome_path).read_bytes() == before[1]
    if which == "data":
        assert genome_path.read_bytes() == before[0]
    else:
        assert not genome_path.exists()  # the data it published is withdrawn
    assert not [p for p in genome_path.parent.iterdir() if p.name.endswith(".tmp")]


@pytest.mark.skipif(sys.platform == "win32", reason="Windows refuses to replace an open file")
def test_open_maps_the_file_it_measured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A data file replaced between measuring and mapping is never mapped as a prefix."""
    path = _write(tmp_path / "g.bytes", [("1", ["GATCAAAA"]), ("MT", ["ACGT"])])
    memmap = np.memmap

    def replaced_first(*args: object, **kwargs: object) -> object:
        larger = tmp_path / "larger"
        larger.write_bytes(b"T" * 13)
        os.replace(larger, path)
        return memmap(*args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(np, "memmap", replaced_first)
    with ByteGenome.open(path) as genome:
        assert genome.fetch("1", 0, 8) == "GATCAAAA"


def test_the_digest_is_computed_from_the_bytes_by_adr_0108_s9(tmp_path: Path) -> None:
    path = _write(tmp_path / "g.bytes", [("1", ["GATC", "aaaa"]), ("MT", ["acgt"])])
    with ByteGenome.open(path) as genome:
        assert genome.digest() == DIGEST_1_MT
        genome.verify()


def test_the_digest_is_never_the_recorded_one(tmp_path: Path) -> None:
    path = _write(tmp_path / "g.bytes", [("1", ["GATCAAAA"]), ("MT", ["ACGT"])])
    _edit_sidecar(path, lambda s: s.update(sequence_digest="f" * 64))
    with ByteGenome.open(path) as genome:
        assert genome.digest() == DIGEST_1_MT
        with pytest.raises(ByteGenomeMismatch, match="sequence digest"):
            genome.verify()


def test_verify_catches_one_flipped_byte(tmp_path: Path) -> None:
    path = _write(tmp_path / "g.bytes", [("1", ["GATCAAAA"]), ("MT", ["ACGT"])])
    data = bytearray(path.read_bytes())
    data[5] = ord("C")  # GATCACAA: one base of contig 1
    path.write_bytes(bytes(data))
    with ByteGenome.open(path) as genome:
        assert genome.fetch("1", 5, 6) == "C"
        assert genome.digest() != DIGEST_1_MT
        with pytest.raises(ByteGenomeMismatch, match="contig 1") as raised:
            genome.verify()
    message = str(raised.value)
    assert "sequence digest" in message and "data file" in message
    assert "contig MT" not in message


def test_verify_catches_a_recorded_contig_hash_that_is_wrong(tmp_path: Path) -> None:
    path = _write(tmp_path / "g.bytes", [("1", ["GATCAAAA"]), ("MT", ["ACGT"])])
    _edit_sidecar(path, lambda s: s["contigs"][1].update(sha256="0" * 64))
    with ByteGenome.open(path) as genome, pytest.raises(ByteGenomeMismatch, match="contig MT"):
        genome.verify()


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda s: s.update(format_version="2.0"), "format version"),
        (lambda s: s.update(format="something-else"), "not a byte genome"),
        (lambda s: s["contigs"][1].update(offset=7), "tile"),
        (lambda s: s["contigs"][0].update(offset=1), "tile"),
        (lambda s: s["contigs"][1].update(length=5), "tile|size"),
        (lambda s: [s["contigs"][0].update(label="X"), s["contigs"][1].update(label="2")], "order"),
        (lambda s: s["contigs"][1].update(label="1"), "twice"),
        (lambda s: s["contigs"][1].update(label="chrMT"), "canonical"),
        (lambda s: s.pop("data_sha256"), "data_sha256"),
        (lambda s: s["contigs"][0].update(length=True), "length"),
        (lambda s: s.update(sequence_digest=""), "sequence_digest"),
        (lambda s: s.update(sequence_digest=DIGEST_1_MT.upper()), "sequence_digest"),
        (lambda s: s.update(data_sha256=s["data_sha256"][:63]), "data_sha256"),
        (lambda s: s.update(data_sha256=s["data_sha256"][:63] + "g"), "data_sha256"),
        (lambda s: s["contigs"][1].update(sha256="sha256:" + "0" * 57), "sha256"),
    ],
)
def test_open_refuses_a_sidecar_it_cannot_trust(tmp_path: Path, edit: object, match: str) -> None:
    path = _write(tmp_path / "g.bytes", [("1", ["GATCAAAA"]), ("MT", ["ACGT"])])
    _edit_sidecar(path, edit)  # type: ignore[arg-type]
    with pytest.raises(UnreadableByteGenome, match=match):
        ByteGenome.open(path)


@pytest.mark.parametrize("data", [b"GATCAAAAACG", b"GATCAAAAACGTA", b""])
def test_open_refuses_a_data_file_of_the_wrong_size(tmp_path: Path, data: bytes) -> None:
    path = _write(tmp_path / "g.bytes", [("1", ["GATCAAAA"]), ("MT", ["ACGT"])])
    path.write_bytes(data)
    with pytest.raises(UnreadableByteGenome, match="12 bases"):
        ByteGenome.open(path)


def test_open_refuses_a_missing_sidecar(tmp_path: Path) -> None:
    path = _write(tmp_path / "g.bytes", [("1", ["A"])])
    sidecar_path(path).unlink()
    with pytest.raises(UnreadableByteGenome, match="sidecar"):
        ByteGenome.open(path)


def test_a_closed_genome_refuses_to_read(genome_path: Path) -> None:
    genome = ByteGenome.open(genome_path)
    genome.close()
    with pytest.raises(ValueError, match="closed"):
        genome.fetch("1", 0, 1)


def _edit_sidecar(path: Path, edit: object) -> None:
    sidecar = sidecar_path(path)
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    edit(document)  # type: ignore[operator]
    sidecar.write_text(json.dumps(document), encoding="utf-8")
