"""The Partner Rows' frames and their group (partner-rows §5.1, §5.2; ogld#67).

The Partner Rows hold, for each variant, every partner within `far_bp` whose code reaches the
Partner Floor within `near_bp` or the far code floor beyond it, in both variants' rows. They are
grouped `ROWS_PER_FRAME` rows to a zstd frame, and one chromosome's frames are the bytes of a
single-chunk `uint8` array, so a row is one positional read and one decompress (partner-rows §5.4).

This module owns both directions of that codec and the group around it:

- `encode_frame` and `decode_frame` write and read a frame's byte layout (§5.2): k, the rows'
  lengths, the partner deltas and the codes, the two arrays byte-shuffled, as one zstd frame at
  level 3 with the content checksum and the content size set in its header. `Frame` is a decoded
  frame, and `Frame.row` gives one row's partners and codes. opengwas/ogld#68 validates frames
  through them and opengwas/ogld#70 reads rows with them; neither parses a payload of its own.
- `FramesGroup` is one chromosome's `partners/<chrom>/` group: its two arrays, its attributes and
  the chunk file they point at. A frame is compressed on a bounded pool and appended to the chunk
  file as it is finished, in frame order, so the bytes do not depend on the pool's size or on the
  order its workers finish in. The array's shape is not known until the chromosome ends, so the
  group is a building file until `close`, which creates the metadata at its final shape and puts
  the file where that metadata says its chunk is.

A frame is never read short: a checksum that fails, a payload that is not exactly
4 + 4k + 6E bytes, or a k that is not the rows the frame is due raises `ValueError` naming the
chromosome and the frame (§5.4). A caller that got rows from a frame got all of them.
"""

from __future__ import annotations

import importlib
import threading
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import zarr

from ogquery.ld.format import Codes, InputError, REncoding
from ogquery.ld.format.geometry import indptr_of
from ogquery.ld.store import layout
from ogquery.ld.store.location import FrameSource, frame_source

#: A frame's zstd level (§5.2), recorded in the group's `zstd_level`.
ZSTD_LEVEL: Final = 3

#: The payload's arrays' wire dtypes (§5.2), little-endian whatever this machine is: the field
#: widths are bytes on disk. The codes are the encoding's container, which every admitted encoding
#: shares (ADR 0090), so their wire dtype is `REncoding.dtype` with a little-endian order.
_LITTLE_ROW: Final = np.dtype("<u4")
_LITTLE_DELTA: Final = np.dtype("<i4")
_LITTLE_CODES: Final = REncoding.dtype.newbyteorder("<")

# numcodecs is a runtime dependency without type stubs, as in decoding and build.inputs.
_NUMCODECS = importlib.import_module("numcodecs")

#: One zstd frame at level 3 with its content checksum, which is what a frame's bytes are (§5.2).
_ZSTD: Final = _NUMCODECS.Zstd(level=ZSTD_LEVEL, checksum=True)


@dataclass(frozen=True)
class Frame:
    """One decoded frame: the rows [first, first + rows) of chromosome `chrom` (§5.2).

    `indptr` holds the frame's entry offsets, `rows + 1` of them, so row i's partners are
    `partners[indptr[i]:indptr[i + 1]]`, ascending and each its chromosome-local row. `number` is
    the frame's own index, so a refusal can name it.
    """

    chrom: str
    number: int
    first: int
    indptr: npt.NDArray[np.int64]
    partners: npt.NDArray[np.int32]
    codes: Codes

    @property
    def rows(self) -> int:
        """The rows this frame holds, its k (§5.2)."""
        return len(self.indptr) - 1

    @property
    def entries(self) -> int:
        """The entries it holds, E."""
        return int(self.indptr[-1])

    def row(self, local_row: int) -> tuple[npt.NDArray[np.int32], Codes]:
        """The partners of chromosome-local row `local_row` and their codes, or a refusal naming
        the frame when the row is not one of its own."""
        if not self.first <= local_row < self.first + self.rows:
            raise ValueError(
                f"row {local_row} is not in frame {self.number} of chromosome {self.chrom}, "
                f"which holds rows {self.first}–{self.first + self.rows - 1} (partner-rows §5.2)"
            )
        at = local_row - self.first
        span = slice(int(self.indptr[at]), int(self.indptr[at + 1]))
        return self.partners[span], self.codes[span]


def _where(chrom: str, number: int) -> str:
    return f"frame {number} of chromosome {chrom}"


def _unshuffled(raw: bytes, dtype: np.dtype[Any]) -> npt.NDArray[Any]:
    """The elements of `raw`'s planes, in order: the inverse of `_shuffled` (§5.2)."""
    width = dtype.itemsize
    planes = np.frombuffer(raw, "u1").reshape(width, -1)
    return planes.T.copy().ravel().view(dtype)


def _row_of_planes(
    raw: bytes, *, offset: int, elements: int, start: int, count: int, dtype: np.dtype[Any]
) -> npt.NDArray[Any]:
    """Elements [start, start + count) of the byte-shuffled array at `offset`, gathered from its
    planes: the inverse of `_shuffled` (§5.2) for one row of a frame, without unshuffling the rows
    a reader did not ask for. `elements` is the array's whole length, which is each plane's stride.
    """
    planes = np.empty((dtype.itemsize, count), dtype=np.uint8)
    for byte in range(dtype.itemsize):
        at = offset + byte * elements + start
        planes[byte] = np.frombuffer(raw[at : at + count], dtype=np.uint8)
    return planes.T.copy().ravel().view(dtype)


#: A zstd frame's magic number, little-endian as it lies on disk (RFC 8878 §3.1.1).
ZSTD_MAGIC: Final = b"\x28\xb5\x2f\xfd"


def _header(raw: bytes, where: str) -> None:
    """Refuse a frame whose zstd header does not declare its content checksum and its content size
    (partner-rows §5.2; RFC 8878 §3.1.1.1).

    The checksum and the content size are what make a frame verifiable: a decoder that verifies
    neither accepts a damaged frame as data, and configuring the compressor with them does not
    require an incoming frame to carry them. So both are read from the header descriptor and
    refused when absent, before anything is decoded.
    """
    if len(raw) < 5 or not raw.startswith(ZSTD_MAGIC):
        raise ValueError(
            f"{where} does not begin a zstd frame: its first bytes are "
            f"{raw[:5].hex() or 'none'} (partner-rows §5.2)"
        )
    descriptor = raw[4]
    size_flag, single_segment = descriptor >> 6, descriptor & 0b0010_0000
    if not descriptor & 0b0000_0100:
        raise ValueError(
            f"{where} declares no content checksum: a frame's header carries one, or the frame "
            "cannot be verified (partner-rows §5.2)"
        )
    if size_flag == 0 and not single_segment:
        raise ValueError(
            f"{where} declares no content size: a frame's header carries one, or its payload "
            "cannot be checked against it (partner-rows §5.2)"
        )


def _payload(raw: bytes, where: str) -> bytes:
    """`raw`'s decompressed payload, or a refusal naming `where` when its zstd frame or its
    checksum fails (§5.4)."""
    _header(raw, where)
    try:
        return bytes(_ZSTD.decode(raw))
    except Exception as failed:
        raise ValueError(
            f"{where} does not decompress: its zstd frame is damaged, or its content checksum "
            f"fails ({failed}). A frame that cannot be checked is never read as a short row "
            "(partner-rows §5.4)"
        ) from failed


def _counted(payload: bytes, where: str, rows: int) -> tuple[int, npt.NDArray[np.int64]]:
    """The payload's k and lengths, refused when k is not `rows` or the payload's length is not
    exactly 4 + 4k + 6E (§5.2)."""
    if len(payload) < 4:
        raise ValueError(
            f"{where} holds {len(payload)} bytes: too few for its k (partner-rows §5.2)"
        )
    k = int(np.frombuffer(payload, _LITTLE_ROW, 1)[0])
    if k != rows:
        raise ValueError(f"{where} holds {k} rows, not the {rows} it is due (partner-rows §5.2)")
    if len(payload) < 4 + 4 * k:
        raise ValueError(
            f"{where} holds {len(payload)} bytes: too few for k = {k} rows' lengths "
            "(partner-rows §5.2)"
        )
    lengths = np.frombuffer(payload, _LITTLE_ROW, k, 4).astype(np.int64)
    entries = int(lengths.sum())
    if len(payload) != 4 + 4 * k + 6 * entries:
        raise ValueError(
            f"{where}'s payload is {len(payload)} bytes, and k = {k} with {entries} entries "
            f"needs exactly {4 + 4 * k + 6 * entries} = 4 + 4k + 6E (partner-rows §5.2)"
        )
    return k, lengths


def _partners(
    first: int, lengths: npt.NDArray[np.int64], deltas: npt.NDArray[np.int64]
) -> npt.NDArray[np.int32]:
    """The deltas as each row's chromosome-local partners: a cumulative sum reset at each row's
    first entry (§5.2)."""
    partners = np.cumsum(deltas, dtype=np.int64)
    nonempty = np.flatnonzero(lengths)
    starts = np.cumsum(lengths)[nonempty] - lengths[nonempty]
    before = np.where(starts > 0, partners[np.maximum(starts - 1, 0)], 0)
    partners += np.repeat(first + nonempty - before, lengths[nonempty])
    bounds = np.iinfo(np.int32)
    if len(partners) and (partners.min() < 0 or partners.max() > bounds.max):
        raise ValueError(
            "a Partner Rows frame decodes to a partner that is not a row (partner-rows §5.2)"
        )
    return partners.astype(np.int32)


@dataclass(frozen=True)
class FramePayload:
    """One frame's decompressed payload, with the rows' lengths that address it (§5.2).

    A reader keeps this rather than every decoded row: the decompress is the expensive half of a
    frame, and a row's own bytes are a gather from each of the payload's planes. `row` does that
    gather and decodes one row (the row-local unshuffle, §7), so reading one row of a frame costs
    one row's work and not the other fifteen; `decode` unshuffles the whole payload and gives the
    `Frame` that validation and the frames' own tests read.
    """

    chrom: str
    number: int
    first: int
    lengths: npt.NDArray[np.int64]
    payload: bytes

    @classmethod
    def parse(cls, raw: bytes, *, chrom: str, number: int, first: int, rows: int) -> FramePayload:
        """The payload `raw` holds — one zstd frame, its checksum verified — checked as the frame
        of rows [first, first + `rows`) of chromosome `chrom` (§5.2, §5.4)."""
        where = _where(chrom, number)
        held = _payload(raw, where)
        _, lengths = _counted(held, where, rows)
        return cls(chrom, number, first, lengths.astype(np.int64), held)

    @property
    def rows(self) -> int:
        """The rows this frame holds, its k (§5.2)."""
        return len(self.lengths)

    @property
    def entries(self) -> int:
        """The entries it holds, E."""
        return int(self.lengths.sum())

    @property
    def bytes(self) -> int:
        """What keeping it holds: the payload and the offsets into it."""
        return len(self.payload) + int(self.lengths.nbytes)

    def decode(self) -> Frame:
        """The whole frame, every row's partners and codes (§5.2): one unshuffle of each plane."""
        head = 4 + 4 * self.rows
        entries = self.entries
        deltas = _unshuffled(self.payload[head : head + 4 * entries], _LITTLE_DELTA)
        codes = _unshuffled(self.payload[head + 4 * entries :], _LITTLE_CODES)
        return Frame(
            self.chrom,
            self.number,
            self.first,
            indptr_of(self.lengths),
            _partners(self.first, self.lengths, deltas.astype(np.int64)),
            codes.astype(REncoding.dtype),
        )

    def row(self, local_row: int) -> tuple[npt.NDArray[np.int32], Codes]:
        """Row `local_row`'s partners and codes, decoded from the payload's planes alone (§5.2).

        The deltas are gathered from their planes for this row's entries and cumulated from its own
        row, so the rest of the frame is never unshuffled: one row costs one row's bytes.
        """
        if not self.first <= local_row < self.first + self.rows:
            raise ValueError(
                f"row {local_row} is not in frame {self.number} of chromosome {self.chrom}, "
                f"which holds rows {self.first}–{self.first + self.rows - 1} (partner-rows §5.2)"
            )
        at = local_row - self.first
        start = int(self.lengths[:at].sum())
        count = int(self.lengths[at])
        if not count:
            return np.empty(0, np.int32), np.empty(0, REncoding.dtype)
        head = 4 + 4 * self.rows
        entries = self.entries
        deltas = _row_of_planes(
            self.payload,
            offset=head,
            elements=entries,
            start=start,
            count=count,
            dtype=_LITTLE_DELTA,
        ).astype(np.int64)
        codes = _row_of_planes(
            self.payload,
            offset=head + 4 * entries,
            elements=entries,
            start=start,
            count=count,
            dtype=_LITTLE_CODES,
        ).astype(REncoding.dtype)
        partners = np.cumsum(deltas) + local_row
        bounds = np.iinfo(np.int32)
        if partners.min() < 0 or partners.max() > bounds.max:
            raise ValueError(
                f"frame {self.number} of chromosome {self.chrom} decodes to a partner that is not "
                "a row (partner-rows §5.2)"
            )
        return partners.astype(np.int32), codes


# ---------------------------------------------------------------- reading (§7)

#: One row decoded: its partners, chromosome-local and ascending, and their codes (§5.2).
_Row = tuple[npt.NDArray[np.int32], Codes]

#: The frames a reader keeps unless its caller says otherwise, in bytes (§7): 256 MiB, which holds
#: about 3,400 of a panel store's frames (a median frame's payload is some 77 KB) and all 58 of the
#: Test Store's. It is the frame cache's own budget, beside `location.CACHE_BYTES`, which
#: keeps the Near Band's inner chunks for `matrix` and `region`.
FRAME_CACHE_BYTES: Final = 1 << 28


class FrameCache:
    """Decoded frames kept by one reader, byte-bounded, the least recently used dropped first
    (partner-rows §7).

    It is separate from `location.ChunkCache`, which keeps the Near Band's inner chunks for
    `matrix` and `region`: a frame is not a chunk of a Zarr array, and a request that reads one
    never reads the other. Neighbouring leads often share a frame, and repeated Analyses on one
    reader hit it, so a frame decoded once serves every row taken from it.

    One lock guards lookup, insertion and eviction, so the count stays the sum of what is kept and
    a frame another thread kept first is the one given back. Decoding runs outside it: two threads
    that miss the same frame may each decode it and keep one copy, which costs a duplicate decode
    and never a different row (partner-rows §9.2 needs no single-flight for that).
    """

    def __init__(self, budget: int | None = None) -> None:
        if budget is not None and (
            isinstance(budget, bool) or not isinstance(budget, int) or budget < 0
        ):
            raise InputError(f"frame_cache_bytes is a count of bytes, an int ≥ 0, not {budget!r}")
        self._budget = FRAME_CACHE_BYTES if budget is None else budget
        self._bytes = 0
        self._kept: OrderedDict[tuple[str, int], tuple[FramePayload, int]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def budget(self) -> int:
        """The bytes this cache keeps at most (§7)."""
        return self._budget

    @property
    def held(self) -> int:
        """The bytes it keeps now, as it counts them: never more than the budget."""
        with self._lock:
            return self._bytes

    def take(self, chrom: str, number: int) -> FramePayload | None:
        """Frame `number` of `chrom` if it is kept, marked as just used; None otherwise."""
        key = (chrom, number)
        with self._lock:
            kept = self._kept.get(key)
            if kept is None:
                return None
            self._kept.move_to_end(key)
        return kept[0]

    def keep(self, frame: FramePayload) -> FramePayload:
        """Keep `frame` in the budget, or the copy another thread kept first, and give it back.

        The least recently used frames are dropped until the count is within the budget, a frame
        larger than the whole budget included: a cache keeps no more than it says it does.
        """
        key = (frame.chrom, frame.number)
        counted = frame.bytes
        with self._lock:
            kept = self._kept.get(key)
            if kept is not None:
                self._kept.move_to_end(key)
                return kept[0]
            self._kept[key] = (frame, counted)
            self._bytes += counted
            while self._bytes > self._budget and self._kept:
                _, (_, dropped) = self._kept.popitem(last=False)
                self._bytes -= dropped
            return frame


def _member(group: zarr.Group, name: str) -> zarr.Array[Any]:
    """The array `name` of a Partner Rows group, refused naming it when the group holds none."""
    path = f"{group.path}/{name}"
    try:
        member = group[name]
    except KeyError:
        raise ValueError(
            f"{path} is missing: a Partner Rows group holds it (partner-rows §5.1)"
        ) from None
    if not isinstance(member, zarr.Array):
        raise ValueError(
            f"{path} is not an array: a Partner Rows group holds two (partner-rows §5.1)"
        )
    return member


def _attribute(attributes: Mapping[str, Any], name: str, path: str) -> int:
    """One of a Partner Rows group's integer attributes, refused naming it when it is not there."""
    value = attributes.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{path} records no {name}: a Partner Rows group holds it (partner-rows §5.3)"
        )
    return value


def _offsets(index: zarr.Array[Any], frames: int, size: int, chrom: str) -> npt.NDArray[np.uint64]:
    """A chromosome's frame offsets, read whole and checked before any frame is (partner-rows §7).

    They are the group's `frames` + 1 byte offsets of one chunk: 0 first, strictly ascending, and
    ending at the `frames_bytes` the group records, so a frame's range is inside the file the
    metadata names. An offsets chunk the store lost reads as its fill value, which the last of these
    checks refuses.
    """
    path = layout.partners_path(chrom)
    if index.shape != (frames + 1,) or index.chunks != index.shape:
        raise ValueError(
            f"{path}/frames_idx has shape {index.shape} and chunks {index.chunks}, not one chunk "
            f"of the {frames + 1} offsets of {frames} frames (partner-rows §5.1)"
        )
    offsets = np.asarray(index[:], dtype=np.uint64)
    if (offsets > size).any() or int(offsets[0]) != 0 or int(offsets[-1]) != size:
        raise ValueError(
            f"chromosome {chrom}'s frames_idx runs from {int(offsets[0])} to {int(offsets[-1])}, "
            f"and the group's frames_bytes is {size}: the offsets must start at 0, ascend and end "
            "at the file's length (partner-rows §5.2)"
        )
    # A group holds at least one frame, so the offsets have at least one step.
    steps = np.diff(offsets.astype(np.int64))
    if (steps <= 0).any():
        raise ValueError(
            f"chromosome {chrom}'s frames_idx does not ascend at offset "
            f"{int(np.flatnonzero(steps <= 0)[0])}: every frame holds bytes (partner-rows §5.2)"
        )
    return offsets


def _counts(group: zarr.Group, path: str) -> tuple[int, int, int, int]:
    """A Partner Rows group's rows, rows a frame, frames and chunk-file bytes (§5.3), refused when
    they cannot describe one: a group holds at least one row in a frame of at least one, and
    ⌈rows / rows_per_frame⌉ frames for them.
    """
    attributes = group.attrs
    rows = _attribute(attributes, "rows", path)
    per_frame = _attribute(attributes, "rows_per_frame", path)
    frames = _attribute(attributes, "frames", path)
    size = _attribute(attributes, "frames_bytes", path)
    if rows < 1 or per_frame < 1:
        raise ValueError(
            f"{path} holds {rows} rows in frames of {per_frame}: a Partner Rows group holds at "
            "least one row, in a frame of at least one (partner-rows §5.2)"
        )
    due = -(-rows // per_frame)
    if frames != due:
        raise ValueError(
            f"{path} records {frames} frames for {rows} rows in frames of {per_frame}: it holds "
            f"{due} (partner-rows §5.2)"
        )
    return rows, per_frame, frames, size


def _one_chunk(array: zarr.Array[Any], size: int, path: str) -> zarr.Array[Any]:
    """The `frames` array, refused unless it is one chunk of the `size` bytes the group records:
    a frame's byte range is an offset into that one file (§5.1)."""
    if array.shape != (size,) or array.chunks != array.shape or array.dtype != np.uint8:
        raise ValueError(
            f"{path}/{layout.FRAMES} is shape {array.shape}, chunks {array.chunks} and "
            f"{array.dtype}, not one chunk of the {size} bytes of frames the group records "
            "(partner-rows §5.1)"
        )
    return array


class PartnerFrames:
    """One chromosome's Partner Rows, opened to be read a row or a frame at a time (§5.1, §7).

    Opening reads what a chromosome's first use needs and nothing else: the group's attributes and
    the `frames_idx` chunk whole (§7 step 1), whose offsets are checked against the group's `frames`
    and `frames_bytes` before any frame is read, and the `frames` array's metadata, which names the
    one chunk file the frames are byte ranges of.

    A frame is one read through `location`'s frame source — `os.pread` on a local file's kept
    descriptor, one Range request for a store that is not on this disk — and one
    `FramePayload.parse`, the store's one frame decoder, so a frame that fails its header, its
    checksum, its k or its payload length is refused naming the chromosome and the frame, and never
    read as a short row (§5.4). Each frame parsed is kept in the reader's `FrameCache`, so a row of
    a frame another row already fetched costs no read, and a row costs only its own gather from the
    payload's planes (§7).

    A source is not fork-safe and is not inherited across `fork` (its lock and descriptor are not):
    a process worker opens its own reader, and so its own sources (partner-rows §9; ogld#71).
    """

    def __init__(self, group: zarr.Group, chrom: str, cache: FrameCache) -> None:
        path = layout.partners_path(chrom)
        self._chrom = chrom
        self._cache = cache
        self._rows, self._per_frame, self._frames, size = _counts(group, path)
        self._offsets = _offsets(_member(group, layout.FRAMES_INDEX), self._frames, size, chrom)
        self._array = _one_chunk(_member(group, layout.FRAMES), size, path)
        self._source: FrameSource = frame_source(self._array)

    @property
    def rows(self) -> int:
        """The rows this chromosome's Partner Rows hold, the group's `rows` (§5.3)."""
        return self._rows

    def frame(self, number: int) -> FramePayload:
        """Frame `number`: from the cache, or one byte-range read and one decompress (§7)."""
        if not 0 <= number < self._frames:
            raise ValueError(
                f"frame {number} is not one of chromosome {self._chrom}'s {self._frames} frames "
                "(partner-rows §5.2)"
            )
        kept = self._cache.take(self._chrom, number)
        if kept is not None:
            return kept
        return self._cache.keep(self._read(number, number)[0])

    def close(self) -> None:
        """Release the chunk file this chromosome's frames are read from (§7): a local store's
        descriptor, closed by the reader that owns it."""
        self._source.close()

    def row(self, local_row: int) -> tuple[npt.NDArray[np.int32], Codes]:
        """Row `local_row`'s partners and codes, decoded from the frame that holds it (§7 step 3).

        The row is gathered from the frame's planes and its deltas cumulated, so a row costs one
        row's work: the frame's other rows are never unshuffled to answer it. The arrays are new
        ones, owned by the caller, and the cached payload they came from is not changed by them.
        """
        if not 0 <= local_row < self._rows:
            raise ValueError(
                f"row {local_row} is not in chromosome {self._chrom}, which holds {self._rows} "
                "rows (partner-rows §5.2)"
            )
        return self.frame(local_row // self._per_frame).row(local_row)

    def rows_of(
        self, local_rows: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32], Codes]:
        """Each of `local_rows`' rows, back to back, as (indptr, partners, codes), in the order
        asked for and with a repeated row sliced again (§7).

        The frames the rows need that are not kept are read in runs of physically adjacent frames,
        one byte-range read a run: frames that do not touch are never coalesced, because the bytes
        between them would be read for nothing (§9.2).
        """
        at = np.asarray(local_rows, dtype=np.int64)
        outside = at[(at < 0) | (at >= self._rows)]
        if len(outside):
            raise ValueError(
                f"rows {outside[:5].tolist()} are not in chromosome {self._chrom}, which holds "
                f"{self._rows} rows (partner-rows §5.2)"
            )
        rows = self._decoded(at)
        indptr = indptr_of(np.array([len(partners) for partners, _ in rows], dtype=np.int64))
        if not rows:
            return indptr, np.empty(0, np.int32), np.empty(0, REncoding.dtype)
        return (
            indptr,
            np.concatenate([partners for partners, _ in rows]),
            np.concatenate([codes for _, codes in rows]),
        )

    def _decoded(self, at: npt.NDArray[np.int64]) -> list[_Row]:
        """Each of `at`'s rows decoded, in that order (§7): from the cache where it holds the
        frame, and from the runs of frames just read where it does not."""
        pieces, missing = self._from_cache(at)
        self._from_runs(at, pieces, missing)
        rows = [piece for piece in pieces if piece is not None]
        assert len(rows) == len(at), "every row asked for is decoded"
        return rows

    def _from_cache(
        self, at: npt.NDArray[np.int64]
    ) -> tuple[list[_Row | None], dict[int, list[int]]]:
        """Each row's slice of its frame where the cache holds it, and the rows it does not, by the
        frame each needs."""
        pieces: list[_Row | None] = [None] * len(at)
        missing: dict[int, list[int]] = {}
        which = at // self._per_frame
        for i, (row, number) in enumerate(zip(at.tolist(), which.tolist(), strict=True)):
            kept = self._cache.take(self._chrom, number)
            if kept is None:
                missing.setdefault(number, []).append(i)
            else:
                pieces[i] = kept.row(row)
        return pieces, missing

    def _from_runs(
        self, at: npt.NDArray[np.int64], pieces: list[_Row | None], missing: dict[int, list[int]]
    ) -> None:
        """Decode the rest from the runs of adjacent frames just read, then offer those payloads to
        the cache: a run is decoded while it is owned, so no budget can evict it between the read
        and the row, and only one run's payloads are held at a time beside the result (§9.2)."""
        for run in _runs(sorted(missing)):
            held = self._read(run[0], run[-1])
            for number, payload in zip(run, held, strict=True):
                for i in missing[number]:
                    pieces[i] = payload.row(int(at[i]))
            for payload in held:
                self._cache.keep(payload)

    def _read(self, first: int, last: int) -> list[FramePayload]:
        """Frames [first, last] as one read of the chunk file, parsed and handed back unkept (§9.2).

        They are physically adjacent, so one read serves them all; the bytes of a frame are its own
        slice of what was read, and each is parsed on its own, so a frame that fails its header,
        its checksum, its k or its payload length is refused naming itself. The caller owns the
        payloads it gets: a bulk read decodes its rows from them before the cache is offered any,
        so a run is never read twice for want of a budget.
        """
        start, stop = int(self._offsets[first]), int(self._offsets[last + 1])
        try:
            raw = self._source.read(start, stop)
        except ValueError as unread:
            raise ValueError(f"{_run(first, last, self._chrom)}: {unread}") from unread
        if len(raw) != stop - start:
            raise ValueError(
                f"{_run(first, last, self._chrom)} holds {len(raw)} of the {stop - start} bytes at "
                f"[{start}, {stop}): a short read is a damaged chunk file, never a short row "
                "(partner-rows §5.4)"
            )
        return [
            FramePayload.parse(
                raw[int(self._offsets[number]) - start : int(self._offsets[number + 1]) - start],
                chrom=self._chrom,
                number=number,
                first=number * self._per_frame,
                rows=min(self._per_frame, self._rows - number * self._per_frame),
            )
            for number in range(first, last + 1)
        ]


def _runs(numbers: list[int]) -> Iterator[list[int]]:
    """`numbers`, ascending and distinct, as the runs of physically adjacent frames among them:
    each a read of its own, and none bridging a frame that was not asked for (§9.2)."""
    run: list[int] = []
    for number in [*numbers, -1]:
        if run and number != run[-1] + 1:
            yield run
            run = []
        if number >= 0:
            run.append(number)


def _run(first: int, last: int, chrom: str) -> str:
    """A frame, or the run of frames one read served, as a refusal names it."""
    return _where(chrom, first) if first == last else f"frames {first}–{last} of chromosome {chrom}"


def open_partner_frames(root: zarr.Group, chrom: str, cache: FrameCache) -> PartnerFrames:
    """Chromosome `chrom`'s Partner Rows of `root`, opened for reading, refused naming the group
    when the store holds none for it (partner-rows §5.1)."""
    path = layout.partners_path(chrom)
    try:
        member = root[path]
    except KeyError:
        raise ValueError(
            f"{path} is missing: this LD Store holds no Partner Rows for chromosome {chrom} "
            "(partner-rows §5.1)"
        ) from None
    if not isinstance(member, zarr.Group):
        raise ValueError(
            f"{path} is not a group: a Partner Rows component is one (partner-rows §5.1)"
        )
    return PartnerFrames(member, chrom, cache)


