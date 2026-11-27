"""How r is stored, as a reader decodes it (ld-store §3).

An LD Store records its encoding (ld-store §7), and `REncoding` is that record as a value: the one
owner of the dtype, the scale, the reserved code and the rounding, and so of which records a
reader implements and of dequantising (ADR 0090). The reader decodes with the encoding the store
records, so no component can decode at another store's scale.

A Genotype Store's reader computes r and so quantises it, with the encoding the store records, and
keeps a pair as the store would have: `quantise`, `stored_mask` and `partner_mask` are here for it,
the one copy the builder, the Partner Rows emitter and that reader judge with (opengwas/ogquery#11;
ADR 0002, amended). The rest of the build side -- the record a build writes, the step, the far code
floor and Partner Floor, and t_far itself -- stays in opengwasdb, which extends this class. A reader
uses the t_far, floors and distances a store records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Final, TypeGuard

import numpy as np
import numpy.typing as npt

from ogquery.ld.format.provenance import Encoding
from ogquery.ld.format.semantics import InputError

#: An array of codes, in the dtype every admitted encoding's container has (ADR 0090).
type Codes = npt.NDArray[np.int16]

# A dot product of standardised vectors can land a few ulps past ±1. Further than this is
# not rounding error but a builder fault, and clamping it would store a plausible r = ±1.
_R_SLACK: Final = 1e-6


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

    def quantise(self, r: npt.ArrayLike) -> Codes:
        """The code of each r: clamp(rint(r × scale), −scale, +scale) (ld-store §3).

        `numpy.rint` rounds half to even, as the spec requires. A value that is not finite,
        or whose magnitude exceeds 1 by more than float error, raises: r is undefined only
        for a zero-variance variant, which the builder drops (ld-store §3), so such a value means
        the computation went wrong, and a clamped code would hide it.
        """
        values = np.asarray(r, dtype=np.float64)
        bad = ~np.isfinite(values) | (np.abs(values) > 1 + _R_SLACK)
        if bad.any():
            raise ValueError(
                f"not a correlation, so it has no code: {values[bad].ravel()[:5].tolist()} "
                "(ld-store §3)"
            )
        codes = np.clip(np.rint(values * self.scale), -self.scale, self.scale)
        return codes.astype(self.dtype)

    def stored_mask(
        self,
        distance: npt.ArrayLike,
        codes: npt.ArrayLike,
        *,
        near_bp: int,
        far_bp: int,
        far_code_floor: int,
    ) -> npt.NDArray[np.bool_]:
        """Whether each pair is Stored at its distance, from its code (ld-store §8).

        Every pair within `near_bp` is Stored; beyond it, one is Stored exactly when it is
        within `far_bp` and its |code| reaches `far_code_floor`. The builder, the Partner Rows
        emitter and the Genotype Store reader all judge with this one predicate, so the Bound
        and Not Measured cannot drift between them (owner, 21 November 2026).
        """
        gap = np.abs(np.asarray(distance, dtype=np.int64))
        magnitude = np.abs(np.asarray(codes, dtype=np.int32))
        near = gap <= near_bp
        far = (gap <= far_bp) & (magnitude >= far_code_floor)
        return near | far

    def partner_mask(
        self,
        distance: npt.ArrayLike,
        codes: npt.ArrayLike,
        *,
        near_bp: int,
        far_bp: int,
        partner_code_floor: int,
        far_code_floor: int,
    ) -> npt.NDArray[np.bool_]:
        """Whether each pair is in a Partner Rows row, from its code (partner-rows §3, §7).

        A pair within `near_bp` is kept when its |code| reaches the Partner Floor, and a pair
        beyond it, out to `far_bp`, when its |code| reaches the far code floor. This is the
        floor choice the Partner Rows emitter makes and the Genotype Store reader answers
        `proxies` and `clump` with, held once (owner, 21 November 2026).

        Pass each pair's real `distance`. The Partner Rows emitter instead passes its two
        already-split groups' boundary: `near_bp` for the Near Band's codes, every one of which
        is within `near_bp`, and `far_bp` for the Far Pairs, every one of which is beyond
        `near_bp` and within `far_bp`. The answer is the same for every such pair, and
        opengwasdb's `tests/ld/test_pair_masks.py` holds the emitter's bytes to this rule.
        """
        gap = np.abs(np.asarray(distance, dtype=np.int64))
        magnitude = np.abs(np.asarray(codes, dtype=np.int32))
        floor = np.where(gap <= near_bp, partner_code_floor, far_code_floor)
        return (gap <= far_bp) & (magnitude >= floor)


def _whole(value: object) -> TypeGuard[int]:
    """A whole number as JSON holds one: an int, never a bool or a float."""
    return isinstance(value, int) and not isinstance(value, bool)


