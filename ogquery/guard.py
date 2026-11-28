"""The index-space guard: an index is scoped to the release and axis it came from (ADR 0109 §2).

`variant_index` is release-scoped (opengwasdb ADR 0002) and so is `analysis_index`: a splice
renumbers both. An index cached from one release is still in range in another and names
whichever Analysis or variant now holds it, so no range check can notice. Two Stores can even
share a `release_id`, and a Hybrid's panel rows and its shared axis are two numberings of one
release. So what an index is scoped to is an `IndexSpace`: the Store and release it belongs to
(`ReleaseIdentity`) and the axis it numbers (`VARIANT`, `PANEL`, `ANALYSIS`).

Every entry point an integer can reach is told the space the integer came from
(``expect_space=``) and refuses a mismatch, naming both. A query result carries the spaces of
its two index columns (`Associations.variant_space`, `.analysis_space`), so `resolve` can do the
same. Nothing here defaults: an id that is missing, empty or the string ``'None'`` names no
release, and refuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

#: The axes an index numbers. A Hybrid's shared variant axis is `VARIANT`; the rows of its
#: Dense Component's panel are `PANEL`, a different numbering of the same release.
VARIANT = "variant"
PANEL = "variant:panel"
ANALYSIS = "analysis"
SPACES = (VARIANT, PANEL, ANALYSIS)


class ReleaseIdUnknown(ValueError):
    """A reader could not learn which release it reads, or was given an id that names none:
    no `manifest.json` that describes its store is reachable, or the id is missing, empty or
    ``'None'``. It refuses rather than guess, since its indices would be scoped to nothing."""


def identifier(value: Any, what: str) -> str:
    """`value` as an identifier a release can be held to: a non-empty string that is not
    the text of a missing value. The one validator of `store_id` and `release_id`."""
    if not isinstance(value, str) or not value.strip() or value.strip() in ("None", "null"):
        raise ReleaseIdUnknown(f"{what} {value!r} names nothing: an identifier is required")
    return value


@dataclass(frozen=True)
class ReleaseIdentity:
    """The Store and the release an index belongs to; a `release_id` alone is not enough,
    as two Stores may use the same one."""

    store_id: str
    release_id: str

    def __post_init__(self) -> None:
        identifier(self.store_id, "store_id")
        identifier(self.release_id, "release_id")

    def __str__(self) -> str:
        return f"release {self.release_id!r} of store {self.store_id!r}"


@dataclass(frozen=True)
class IndexSpace:
    """What an integer index is scoped to: a release's identity and the axis it numbers."""

    identity: ReleaseIdentity
    space: str

    def __post_init__(self) -> None:
        if self.space not in SPACES:
            raise ValueError(f"unknown index space {self.space!r}; one of {SPACES}")

    @property
    def release_id(self) -> str:
        return self.identity.release_id

    def __str__(self) -> str:
        return f"the {self.space} index of {self.identity}"


class CellSpaces(NamedTuple):
    """The two spaces a reader's cells are numbered in: its rows (variants) and columns."""

    variant: IndexSpace
    analysis: IndexSpace


class ReleaseMismatchError(ValueError):
    """An index, or a result holding indices, belongs to another index space than the one
    asked to interpret it. It would name a different Analysis or variant, silently."""

    def __init__(self, owner: IndexSpace, expected: IndexSpace | None, what: str) -> None:
        self.owner = owner
        self.expected = expected
        self.what = what
        self.release_id = owner.release_id
        if expected is None:
            text = (
                f"{what} carries no index space, so its indices cannot be tied to "
                f"{owner}; they are scoped to a release and an axis, and a splice renumbers them"
            )
        else:
            text = (
                f"{what} is {expected} but this is {owner}: release_id "
                f"{expected.release_id!r} is not {owner.release_id!r}"
                if expected.release_id != owner.release_id
                else f"{what} is {expected} but this is {owner}"
            ) + "; an index cached from another space names a different Analysis or variant here"
        super().__init__(text)


def require_space(owner: IndexSpace, expected: IndexSpace | None, what: str) -> None:
    """Refuse unless `expected` is `owner`: the release, the Store and the axis."""
    if expected != owner:
        raise ReleaseMismatchError(owner, expected, what)


def require_cells(owner: CellSpaces, expected: CellSpaces | None, what: str) -> None:
    """`require_space`, for both columns of a reader's cells."""
    if expected is None:
        raise ReleaseMismatchError(owner.variant, None, what)
    require_space(owner.variant, expected.variant, what)
    require_space(owner.analysis, expected.analysis, what)
