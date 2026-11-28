"""The release_id guard: an index is scoped to the release it came from (ADR 0109 §2).

`variant_index` is release-scoped (opengwasdb ADR 0002) and so is `analysis_index`: a splice
renumbers both. An index cached from one release is still in range in another and names
whichever Analysis or variant now holds it, so no range check can notice. Every entry point
an integer can reach is therefore told the `release_id` the integer came from
(``expect_release_id=``) and refuses a mismatch, naming both ids, and a query result carries
the `release_id` it was read from so `resolve` can do the same.
"""

from __future__ import annotations


class ReleaseMismatchError(ValueError):
    """An index, or a result holding indices, was read from another release than the one
    asked to interpret it. It would name a different Analysis or variant, silently."""

    def __init__(self, release_id: str, expected: str | None, what: str) -> None:
        self.release_id = release_id
        self.expected = expected
        self.what = what
        if expected is None:
            text = (
                f"{what} carries no release_id, so its indices cannot be tied to release "
                f"{release_id!r}; they are release-scoped and a splice renumbers them"
            )
        else:
            text = (
                f"{what} was read from release {expected!r} but this is release "
                f"{release_id!r}; an index is scoped to its release, and an index cached "
                "from another names a different Analysis or variant here"
            )
        super().__init__(text)


def require_release(release_id: str, expected: str | None, what: str) -> None:
    """Refuse unless `expected` is `release_id`, the release that holds the index."""
    if expected != release_id:
        raise ReleaseMismatchError(release_id, expected, what)
