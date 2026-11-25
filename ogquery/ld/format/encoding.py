"""How r is stored, as a reader decodes it (ld-store §3).

An LD Store records its encoding (ld-store §7), and `REncoding` is that record as a value: the one
owner of the dtype, the scale, the reserved code and the rounding, and so of which records a
reader implements and of dequantising (ADR 0090). The reader decodes with the encoding the store
records, so no component can decode at another store's scale.

The build side of the encoding -- quantising, the step, the far code floor and Partner Floor, the
masks a build and a Genotype Store keep pairs by, and t_far itself -- stays in opengwasdb, which
extends this class (ogquery#2; ADR 0002). A reader uses the t_far and floors a store records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, TypeGuard

import numpy as np
import numpy.typing as npt

from ogquery.ld.format.provenance import Encoding
from ogquery.ld.format.semantics import InputError

#: An array of codes, in the dtype every admitted encoding's container has (ADR 0090).
type Codes = npt.NDArray[np.int16]


@dataclass(frozen=True)
class REncoding:
    """One LD Store's encoding of r: r̂ = code / `scale`, in an int16 container (ld-store §3).

    Every admitted encoding shares the container (owner, 10 November 2026; ADR 0090): int16
    codes, −32768 reserved and every `r` array's fill value, and rounding half to even. Only
    the scale differs, a whole number from 1 to 32767, so the scale is symmetric: negating a
    code never overflows, r = ±1 is exact, and a lost chunk still reads as a code that is never
    a correlation. The HGDP+1kGP stores are × 511 and UK Biobank EUR × 2047. There is no
    default: a build is given its scale and a store records it.
    """

    scale: int

    #: The container's dtype, the same for every admitted encoding.
    dtype: ClassVar[np.dtype[np.int16]] = np.dtype(np.int16)
    #: Never a correlation. Every `r` array's fill value, so a chunk that was never written
    #: cannot read as r = 0 (ld-store §3, §6).
    reserved_code: ClassVar[int] = int(np.iinfo(np.int16).min)
    #: The largest scale the container holds symmetrically: the other end from the reserved code.
    largest_scale: ClassVar[int] = int(np.iinfo(np.int16).max)
    #: How r × scale becomes a code: `numpy.rint`, which rounds half to even.
    rounding: ClassVar[str] = "half_to_even"

    def __post_init__(self) -> None:
        if not self.admits(self.scale):
            raise InputError(
                f"an r scale is a whole number from 1 to {self.largest_scale}, not "
                f"{self.scale!r} (ld-store §3)"
            )

    @classmethod
    def admits(cls, scale: object) -> bool:
        """Whether `scale` is one the container holds: a whole number from 1 to 32767."""
        return _whole(scale) and 1 <= scale <= cls.largest_scale

    @classmethod
    def of(cls, record: object) -> REncoding:
        """The encoding a store's `encoding` record names (ld-store §7), or an `InputError` naming
        the record when this package does not implement it (ld-store §13): another dtype, reserved
        code or rounding, a scale outside 1–32767, or a key it does not know."""
        if not cls._implements(record):
            raise InputError(
                f"the encoding {record!r} is not one ogquery.ld implements: exactly the keys "
                f"{sorted(Encoding.__required_keys__)}, an int16 container with "
                f"{cls.reserved_code} reserved, rounded {cls.rounding}, at a whole scale from 1 to "
                f"{cls.largest_scale} (ld-store §3, §13)"
            )
        return cls(record["r_scale"])

    @classmethod
    def _implements(cls, record: object) -> TypeGuard[dict[str, Any]]:
        # Exactly the record's keys, as `Encoding` types them: a key this package does not know
        # could change what a code means, so a record with one is refused, not ignored.
        if not isinstance(record, dict) or set(record) != Encoding.__required_keys__:
            return False
        container = (record["r_dtype"], record["r_reserved_code"], record["rounding"])
        expected = (cls.dtype.name, cls.reserved_code, cls.rounding)
        return container == expected and cls.admits(record["r_scale"])

    def holds(self, codes: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """Where `codes` are codes of this encoding, |code| ≤ scale: the reserved code never is,
        and neither is a code past the scale, which would decode to |r̂| > 1 (ld-store §12)."""
        # Compared in the codes' own dtype: every scale fits int16, and a validation block of 8 Mi
        # codes would otherwise hold a 64 MiB copy (ld-store §12).
        values = np.asarray(codes)
        return np.asarray((values >= -self.scale) & (values <= self.scale))

    def require_codes(self, codes: npt.ArrayLike) -> None:
        """Raise ValueError naming the values of `codes` this encoding never writes. The
        reserved code −32768 is what an unwritten chunk reads as, so taking it for a number is
        exactly the silent zero it exists to prevent; a code past the scale would be an r beyond
        ±1. The reader refuses both before it decodes, and the writer before it stores."""
        values = np.asarray(codes)
        held = self.holds(values)
        if not held.all():
            found = values[~held].ravel()[:5].tolist()
            raise ValueError(
                f"codes {found} are not codes of r at × {self.scale}: {self.reserved_code} is "
                "reserved for an unwritten chunk, and any other is past the scale (ld-store §3)"
            )

    def dequantise(self, codes: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """The Stored r of each code: code / scale (ld-store §3). A code this encoding never writes
        raises, naming it (`require_codes`)."""
        values = np.asarray(codes)
        self.require_codes(values)
        return values.astype(np.float64) / self.scale


def _whole(value: object) -> TypeGuard[int]:
    """A whole number as JSON holds one: an int, never a bool or a float."""
    return isinstance(value, int) and not isinstance(value, bool)


