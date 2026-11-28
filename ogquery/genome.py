"""The byte genome: the release assembly's sequence as one memory-mapped file of uint8
(opengwas/ogquery#14, ADR 0003; opengwasdb ADR 0108 §7 and §9).

CPRA formation reads REF from it (`ogquery.cpra`), and builds, ogquery and the Data API
read the same file, so all three spell a CPRA from one byte-identical reference. It is
numpy and the standard library alone: the FASTA it is derived from is read by the engine
(opengwasdb's `FastaSequence`, through pysam), which hands `write_byte_genome` each
contig's sequence.

The format (ADR 0003):

- **The data file** is raw uint8, each primary contig's upper-cased bases as their ASCII
  bytes, concatenated in the variant order (1-22, X, Y, MT) with no separator. A base is
  stored as the source has it, upper-cased: A, C, G, T, N or another IUPAC code, never
  rewritten.
- **The sidecar**, `<data file>.json` (`sidecar_path`), names the format and its version,
  the assembly, each contig (label, byte offset, length, the SHA-256 of its sequence), the
  sequence digest (opengwasdb ADR 0108 §9), the data file's SHA-256, and the provenance the
  writer's caller supplies.

`ByteGenome.open` refuses an unknown format version, a data file whose size is not its
contigs' total, and offsets that do not tile the file. `ByteGenome.digest` computes the
digest from the bytes, never from the sidecar; `ByteGenome.verify` compares what it
computes with what the sidecar records.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Self

import numpy as np
import numpy.typing as npt

from ogquery.cpra import ContigNotHeld, PositionOutsideContig
from ogquery.identity import (
    VariantNormalisationError,
    chromosome_sort_key,
    is_primary_contig,
    normalise_chromosome,
)

#: The sidecar's `format`: what the file is.
FORMAT_NAME: Final = "ogquery-byte-genome"
#: The one sidecar `format_version` this reader implements; any other is refused.
FORMAT_VERSION: Final = "1.0"
#: The sidecar sits beside the data file, named `<data file><SIDECAR_SUFFIX>`.
SIDECAR_SUFFIX: Final = ".json"
#: How many bytes a hash reads at a time.
_HASH_CHUNK = 1 << 24
#: A SHA-256 as a sidecar or a manifest records one: 64 lower-case hex characters.
SHA256_HEX: Final = re.compile(r"[0-9a-f]{64}")
#: The IUPAC nucleotide codes, upper-cased: every base a byte genome stores.
IUPAC_BASES: Final = b"ACGTUNRYSWKMBDHV"


class ByteGenomeError(ValueError):
    """A byte genome that cannot be written, read or trusted."""


class RefusedContig(ByteGenomeError):
    """`write_byte_genome` was handed a contig the format cannot hold."""


class UnreadableByteGenome(ByteGenomeError):
    """`ByteGenome.open` refuses a sidecar or data file it cannot trust."""


class ByteGenomeMismatch(ByteGenomeError):
    """`ByteGenome.verify` computed something other than what the sidecar records."""


def sidecar_path(path: str | Path) -> Path:
    """The sidecar of the data file ``path``: ``<path>.json``."""
    path = Path(path)
    return path.with_name(path.name + SIDECAR_SUFFIX)


@dataclass(frozen=True, slots=True)
class _Contig:
    label: str
    offset: int
    length: int
    sha256: str


def sequence_digest(contigs: Iterable[tuple[str, int, str]]) -> str:
    """opengwasdb ADR 0108 §9's sequence digest over ``(label, length, sha256)`` per contig,
    in the variant order: the SHA-256, in lower-case hex, of one ASCII line per contig, its
    label, a tab, its length in bases, a tab, the lower-case hex SHA-256 of its upper-cased
    sequence, and a newline (as opengwasdb's `FastaSequence.digest`)."""
    lines = hashlib.sha256()
    for label, length, sha256 in contigs:
        lines.update(f"{label}\t{length}\t{sha256}\n".encode("ascii"))
    return lines.hexdigest()


def write_byte_genome(
    path: str | Path,
    contigs: Iterable[tuple[str, Iterable[str | bytes]]],
    *,
    assembly: str,
    provenance: Mapping[str, Any],
) -> Path:
    """Write the byte genome ``path`` and its sidecar from ``contigs``; returns ``path``.

    ``contigs`` yields ``(label, chunks)`` in the variant order, each label a canonical
    primary contig (`normalise_chromosome`, `is_primary_contig`: ``1``, not ``chr1``), and
    each contig's sequence as chunks of bases, so a whole genome is never held in memory.
    Bases are upper-cased and otherwise stored as given; a byte that is not an IUPAC
    nucleotide code is refused. A non-canonical or non-primary label, a contig given twice
    or out of order, and an empty contig raise `RefusedContig`.

    ``provenance`` (the source FASTA's path and checksum, who derived it and when) is
    written as given, and must be JSON. Both files are written under temporary names and
    published under their own by a hard link, which refuses a name already taken, the data
    file first, so a sidecar exists only beside a complete data file; nothing is left
    behind on a refusal. An existing ``path`` or sidecar, a dangling symlink included, is
    never overwritten (`FileExistsError`), even one another writer creates meanwhile.
    """
    path = Path(path)
    sidecar = sidecar_path(path)
    for existing in (path, sidecar):
        if os.path.lexists(existing):  # a courtesy before 3 GB is written; `_publish` decides
            raise FileExistsError(f"{existing}: a byte genome is never overwritten")
    provenance = dict(provenance)
    temporaries: list[Path] = []
    try:
        data_tmp = _temporary(path)
        temporaries.append(data_tmp)
        written = _write_contigs(data_tmp, contigs)
        document = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "assembly": assembly,
            "sequence_digest": sequence_digest(
                (c.label, c.length, c.sha256) for c in written.contigs
            ),
            "data_sha256": written.data_sha256,
            "contigs": [
                {"label": c.label, "offset": c.offset, "length": c.length, "sha256": c.sha256}
                for c in written.contigs
            ],
            "provenance": provenance,
        }
        text = json.dumps(document, indent=2, allow_nan=False) + "\n"
        sidecar_tmp = _temporary(sidecar)
        temporaries.append(sidecar_tmp)
        with sidecar_tmp.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        _publish(data_tmp, path)
        temporaries.remove(data_tmp)
        temporaries.append(path)  # until its sidecar lands, a data file alone is unfinished
        _publish(sidecar_tmp, sidecar)
        temporaries.clear()
    finally:
        for leftover in temporaries:
            leftover.unlink(missing_ok=True)
    return path


@dataclass(frozen=True, slots=True)
class _Written:
    contigs: tuple[_Contig, ...]
    data_sha256: str


def _publish(temporary: Path, final: Path) -> None:
    """Give ``temporary`` the name ``final`` without ever replacing a file of that name.

    `os.link` fails if the name exists, a dangling symlink included, in one step, where a
    check followed by `os.replace` would let a second writer in between."""
    try:
        os.link(temporary, final)
    except FileExistsError as error:
        raise FileExistsError(f"{final}: a byte genome is never overwritten") from error
    temporary.unlink()


def _temporary(path: Path) -> Path:
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    return Path(name)


def _write_contigs(target: Path, contigs: Iterable[tuple[str, Iterable[str | bytes]]]) -> _Written:
    written: list[_Contig] = []
    whole = hashlib.sha256()
    offset = 0
    with target.open("wb") as stream:
        for label, chunks in contigs:
            _require_next_label(label, written)
            contig = hashlib.sha256()
            length = 0
            for chunk in chunks:
                bases = _bases_of(label, chunk)
                stream.write(bases)
                contig.update(bases)
                whole.update(bases)
                length += len(bases)
            if not length:
                raise RefusedContig(f"contig {label} is empty: a byte genome holds no empty contig")
            written.append(_Contig(label, offset, length, contig.hexdigest()))
            offset += length
        if not written:
            raise RefusedContig("no contig was given: a byte genome holds at least one")
        stream.flush()
        os.fsync(stream.fileno())
    return _Written(tuple(written), whole.hexdigest())


def _require_next_label(label: str, written: list[_Contig]) -> None:
    if not is_primary_contig(label):
        raise RefusedContig(
            f"contig {label!r} is not a primary contig (1-22, X, Y, MT): a byte genome holds "
            "only the primary assembly"
        )
    canonical = normalise_chromosome(label)
    if label != canonical:
        raise RefusedContig(
            f"contig {label!r} is not a canonical label: name it {canonical!r} "
            "(`normalise_chromosome`)"
        )
    if any(c.label == label for c in written):
        raise RefusedContig(f"contig {label} is given twice")
    if written and chromosome_sort_key(label) <= chromosome_sort_key(written[-1].label):
        raise RefusedContig(
            f"contig {label} comes after {written[-1].label}: contigs are written in the "
            "variant order (1-22, X, Y, MT)"
        )


def _bases_of(label: str, chunk: str | bytes) -> bytes:
    if isinstance(chunk, str):
        try:
            chunk = chunk.encode("ascii")
        except UnicodeEncodeError as error:
            raise RefusedContig(
                f"contig {label} holds {chunk[error.start : error.end]!r}, which is not an "
                "IUPAC base"
            ) from error
    bases = chunk.upper()
    foreign = bases.translate(None, IUPAC_BASES)
    if foreign:
        shown = "".join(sorted(set(foreign.decode("ascii", "backslashreplace"))))
        raise RefusedContig(f"contig {label} holds {shown!r}, which is not an IUPAC base")
    return bases


class ByteGenome:
    """A byte genome, memory-mapped read-only: the `ReferenceSequence` CPRA formation reads.

    Contigs are named by their canonical labels, and a label is read as an identity reads
    it (`chr1` is contig ``1``, `chrM` is ``MT``). `close` (or a ``with`` block) releases
    the map; nothing it returns holds a view of it. A handle is safe to share across a
    fork, as the page cache is.
    """

    def __init__(self, path: Path, data: np.memmap[Any, np.dtype[np.uint8]], sidecar: _Sidecar):
        self.path = path
        self._data: np.memmap[Any, np.dtype[np.uint8]] | None = data
        self._bytes: npt.NDArray[np.uint8] | None = data.view(np.ndarray)
        self._sidecar = sidecar
        self._contigs = {c.label: c for c in sidecar.contigs}
        self._computed: _Computed | None = None

    @classmethod
    def open(cls, path: str | Path) -> ByteGenome:
        """Open the byte genome ``path``, refusing (`UnreadableByteGenome`) a missing or
        malformed sidecar, an unknown format version, offsets that do not tile the data
        file, and a data file whose size is not its contigs' total."""
        path = Path(path)
        sidecar = _read_sidecar(path)
        total = sum(c.length for c in sidecar.contigs)
        with path.open("rb") as stream:  # one open file is measured and mapped
            size = os.fstat(stream.fileno()).st_size
            if size != total:
                raise UnreadableByteGenome(
                    f"{path}: the data file's size, {size} bytes, is not the {total} bases its "
                    "sidecar's contigs hold; it is truncated or not this sidecar's"
                )
            data = np.memmap(stream, dtype=np.uint8, mode="r", shape=(size,))
        return cls(path, data, sidecar)

    def close(self) -> None:
        """Release the map; a closed genome refuses to read."""
        self._data = None
        self._bytes = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def assembly(self) -> str:
        """The assembly the sidecar names."""
        return self._sidecar.assembly

    @property
    def provenance(self) -> dict[str, Any]:
        """The provenance the writer's caller recorded, as recorded (a copy)."""
        return copy.deepcopy(self._sidecar.provenance)

    @property
    def contigs(self) -> tuple[str, ...]:
        """The contigs held, by label, in the variant order."""
        return tuple(c.label for c in self._sidecar.contigs)

    def length(self, chrom: str) -> int:
        return self._contig(chrom).length

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Bases ``[start, end)`` (0-based, half-open) of ``chrom``, upper-cased.

        Raises `ContigNotHeld` for a contig the genome does not hold and
        `PositionOutsideContig` for a span that leaves it."""
        contig = self._contig(chrom)
        if not 0 <= start <= end <= contig.length:
            raise PositionOutsideContig(
                f"{chrom}:{start + 1}-{end}: outside contig {contig.label}, which is "
                f"{contig.length} bases in {self.path}"
            )
        data = self._view()
        return data[contig.offset + start : contig.offset + end].tobytes().decode("ascii")

    def bases(self, chrom: str, positions: npt.ArrayLike) -> npt.NDArray[np.uint8]:
        """The base at each 1-based position of ``chrom``, as its ASCII byte (uint8).

        One vectorised gather, for a builder looking up REF at millions of sites. Raises
        `ContigNotHeld` and `PositionOutsideContig` as `fetch` does, for the whole array;
        a position that is not an integer is a `TypeError`."""
        contig = self._contig(chrom)
        wanted = np.asarray(positions)
        if wanted.dtype.kind not in "iu":
            raise TypeError(f"positions must be integers, not {wanted.dtype}")
        data = self._view()
        if not wanted.size:
            return np.empty(wanted.shape, dtype=np.uint8)
        low, high = int(wanted.min()), int(wanted.max())
        if low < 1 or high > contig.length:
            bad = low if low < 1 else high
            raise PositionOutsideContig(
                f"{chrom}:{bad}: outside contig {contig.label}, which is {contig.length} bases "
                f"in {self.path} (positions are 1-based)"
            )
        return data[wanted.astype(np.int64, copy=False) + (contig.offset - 1)]

    def digest(self) -> str:
        """The sequence digest of opengwasdb ADR 0108 §9, computed from the bytes.

        The SHA-256, in lower-case hex, of one ASCII line per contig, in the variant order:
        its label, a tab, its length in bases, a tab, the lower-case hex SHA-256 of its
        sequence, and a newline; the same as `FastaSequence.digest` gives for the FASTA the
        genome was written from. Read once per handle, then cached: about 3 GB for GRCh38.
        The sidecar's recorded digest is never returned; `verify` compares the two."""
        return self._compute().sequence_digest

    def verify(self) -> None:
        """Compare what the bytes give with what the sidecar records: each contig's
        SHA-256, the sequence digest and the data file's SHA-256. Raises
        `ByteGenomeMismatch` naming every difference."""
        computed = self._compute()
        recorded = self._sidecar
        differences = [
            f"contig {c.label}: sha256 {sha} computed, {c.sha256} recorded"
            for c, sha in zip(recorded.contigs, computed.contig_sha256, strict=True)
            if sha != c.sha256
        ]
        if computed.sequence_digest != recorded.sequence_digest:
            differences.append(
                f"sequence digest: {computed.sequence_digest} computed, "
                f"{recorded.sequence_digest} recorded"
            )
        if computed.data_sha256 != recorded.data_sha256:
            differences.append(
                f"data file: sha256 {computed.data_sha256} computed, "
                f"{recorded.data_sha256} recorded"
            )
        if differences:
            raise ByteGenomeMismatch(
                f"{self.path} is not what its sidecar records: " + "; ".join(differences)
            )

    def _view(self) -> npt.NDArray[np.uint8]:
        if self._bytes is None:
            raise ValueError(f"{self.path}: the byte genome is closed")
        return self._bytes

    def _contig(self, chrom: str) -> _Contig:
        try:
            contig = self._contigs.get(normalise_chromosome(chrom))
        except VariantNormalisationError:
            contig = None
        if contig is None:
            raise ContigNotHeld(
                f"{self.path} holds no primary contig {chrom!r}; it holds {', '.join(self.contigs)}"
            )
        return contig

    def _compute(self) -> _Computed:
        if self._computed is None:
            data = self._view()
            whole = hashlib.sha256()
            contig_sha256 = []
            for c in self._sidecar.contigs:
                contig = hashlib.sha256()
                for start in range(c.offset, c.offset + c.length, _HASH_CHUNK):
                    chunk = data[start : min(start + _HASH_CHUNK, c.offset + c.length)]
                    contig.update(chunk)
                    whole.update(chunk)
                contig_sha256.append(contig.hexdigest())
            digest = sequence_digest(
                (c.label, c.length, sha)
                for c, sha in zip(self._sidecar.contigs, contig_sha256, strict=True)
            )
            self._computed = _Computed(tuple(contig_sha256), digest, whole.hexdigest())
        return self._computed


@dataclass(frozen=True, slots=True)
class _Computed:
    contig_sha256: tuple[str, ...]
    sequence_digest: str
    data_sha256: str


@dataclass(frozen=True, slots=True)
class _Sidecar:
    assembly: str
    contigs: tuple[_Contig, ...]
    sequence_digest: str
    data_sha256: str
    provenance: dict[str, Any]


def _read_sidecar(path: Path) -> _Sidecar:
    sidecar = sidecar_path(path)
    try:
        document = json.loads(sidecar.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise UnreadableByteGenome(
            f"{path}: no sidecar {sidecar.name} beside it; a byte genome is read only with its "
            "sidecar, which `write_byte_genome` writes last"
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UnreadableByteGenome(f"{sidecar}: not a JSON sidecar ({error})") from error
    if not isinstance(document, dict) or document.get("format") != FORMAT_NAME:
        raise UnreadableByteGenome(f"{sidecar}: not a byte genome's sidecar ({FORMAT_NAME})")
    version = document.get("format_version")
    if version != FORMAT_VERSION:
        raise UnreadableByteGenome(
            f"{sidecar}: format version {version!r}, which this reader does not implement "
            f"(it reads {FORMAT_VERSION})"
        )
    assembly = _field(sidecar, document, "assembly", str)
    digest = _checksum(sidecar, document, "sequence_digest")
    data_sha256 = _checksum(sidecar, document, "data_sha256")
    provenance = _field(sidecar, document, "provenance", dict)
    entries = _field(sidecar, document, "contigs", list)
    contigs: list[_Contig] = []
    offset = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise UnreadableByteGenome(f"{sidecar}: contig entry {entry!r} is not an object")
        contig = _Contig(
            label=_field(sidecar, entry, "label", str),
            offset=_field(sidecar, entry, "offset", int),
            length=_field(sidecar, entry, "length", int),
            sha256=_checksum(sidecar, entry, "sha256"),
        )
        try:
            _require_next_label(contig.label, contigs)
        except RefusedContig as error:
            raise UnreadableByteGenome(f"{sidecar}: {error}") from error
        if contig.length < 1:
            raise UnreadableByteGenome(
                f"{sidecar}: contig {contig.label} has length {contig.length}"
            )
        if contig.offset != offset:
            raise UnreadableByteGenome(
                f"{sidecar}: contig {contig.label} starts at byte {contig.offset}, but the "
                f"contigs before it end at {offset}; offsets must tile the data file in order"
            )
        contigs.append(contig)
        offset += contig.length
    if not contigs:
        raise UnreadableByteGenome(f"{sidecar}: no contig")
    return _Sidecar(assembly, tuple(contigs), digest, data_sha256, provenance)


def _field[T](sidecar: Path, document: dict[str, Any], key: str, kind: type[T]) -> T:
    value = document.get(key)
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise UnreadableByteGenome(
            f"{sidecar}: {key!r} is {value!r}, not the {kind.__name__} the format records"
        )
    return value


def _checksum(sidecar: Path, document: dict[str, Any], key: str) -> str:
    """A SHA-256 the sidecar records: 64 lower-case hex characters, or the sidecar is
    refused. Whether it is the right one is `verify`'s question."""
    value = _field(sidecar, document, key, str)
    if not SHA256_HEX.fullmatch(value):
        raise UnreadableByteGenome(
            f"{sidecar}: {key!r} is {value!r}, not a SHA-256 in 64 lower-case hex characters"
        )
    return value
