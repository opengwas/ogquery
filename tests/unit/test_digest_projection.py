"""The projected question set (opengwasdb ADR 0109 §8b, ogquery#13).

A release is asked a reference release's questions, and its answers are projected onto the
reference's index space before any limit. The four seams:

1. on the contract releases, projected answers for a release equal its own;
2. a release with two Analyses swapped answers identically under the projection and
   differently without it;
3. an Analysis the release does not hold is refused, never answered empty;
4. a limit cannot hide a hit another Analysis displaced.

The contract bundle is read, never written. The swapped release is the contract release read
through a reader that numbers two of its Analyses the other way round (`swapped`): every
index an answer holds, and every list in index order, is as a release built with those two
exchanged would give. It changes nothing on disk.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ogquery import digests
from ogquery.analyses import UnknownAnalysisError
from ogquery.associations import Associations

RELEASES = Path(__file__).parents[1] / "contract" / "releases"
COMPLETED = ("dense_completed", "ragged_completed", "hybrid_completed", "hybrid_overflow_only")
SWAPPED = ("dense_completed", "ragged_completed", "hybrid_completed")


def swapped(path: Path, a: int = 0, b: int = 1) -> Any:
    """The release at `path`, read as if its Analyses `a` and `b` had exchanged indices."""
    q = digests.open_query(path)
    base = type(q)
    by_id = {row["analysis_id"]: i for i, row in q.analyses_table().items()}

    def flip(index: Any) -> Any:
        index = np.asarray(index)
        return np.where(index == a, b, np.where(index == b, a, index))

    def associations(name: str) -> Any:
        def ask(self: Any, *args: Any, **kwargs: Any) -> Associations:
            found = getattr(base, name)(self, *args, **kwargs)
            return replace(found, analysis_index=flip(found.analysis_index))

        return ask

    def analyses_table(self: Any) -> dict[int, dict[str, str]]:
        table = base.analyses_table(self)
        return {i: {**table[int(flip(i))], "analysis_index": str(i)} for i in table}

    def resolve(self: Any, result: Any, **kwargs: Any) -> Any:
        unflipped = replace(result, analysis_index=flip(result["analysis_index"]))
        return base.resolve(self, unflipped, **kwargs)

    def renumbered(answer: dict[str, np.ndarray], whole: bool) -> dict[str, np.ndarray]:
        """A Rho answer that lists Analyses, in the order of the exchanged indices."""
        order = np.argsort([int(flip(by_id[x])) for x in answer["analysis_id"]], kind="stable")
        if not whole:
            return {name: values[order] for name, values in answer.items()}
        both = np.ix_(order, order)
        return {
            "analysis_id": answer["analysis_id"][order],
            "rho": answer["rho"][both],
            "n_null": answer["n_null"][both],
        }

    def rho_row(self: Any, analysis_id: str) -> dict[str, np.ndarray]:
        return renumbered(base.rho_row(self, analysis_id), False)

    def rho_matrix(self: Any, ids: Any = None) -> dict[str, np.ndarray]:
        answer = base.rho_matrix(self, ids)
        return (
            answer
            if ids is not None or not len(answer["analysis_id"])
            else renumbered(answer, True)
        )

    namespace: dict[str, Any] = {
        name: associations(name)
        for name in (
            "analysis",
            "phewas",
            "range_phewas",
            "lookup",
            "top_hits",
            "range_by_analysis",
        )
        if hasattr(base, name)
    }
    namespace.update(analyses_table=analyses_table, resolve=resolve)
    if hasattr(base, "rho_matrix"):
        namespace.update(rho_row=rho_row, rho_matrix=rho_matrix)
    flipped = type("_Swapped", (base,), namespace)
    reader = object.__new__(flipped)
    reader.__dict__.update(q.__dict__)
    return reader


def reference_for(path: Path) -> digests.Reference:
    with digests.open_query(path) as q:
        return digests.reference_of(q)


def answers(path: Path, **kwargs: Any) -> dict[str, Any]:
    return digests.release_digests(path, **kwargs)


@cache
def plain_answers(name: str) -> dict[str, Any]:
    return answers(RELEASES / name)


@cache
def own_projected_answers(name: str) -> dict[str, Any]:
    return answers(RELEASES / name, reference=reference_for(RELEASES / name))


# ── 1. a release's projected answers are its own ─────────────────────────────


def canonical(found: Associations) -> Associations:
    return found.take(np.lexsort((found.analysis_index, found.variant_index)))


@pytest.mark.parametrize("name", COMPLETED)
def test_a_release_projected_onto_itself_answers_as_it_does_unprojected(name: str) -> None:
    """Every answer but an association answer's row order is the unprojected digest, and
    an association answer is the unprojected rows in `(variant_index, analysis_index)`
    order. `top_hits/limit` is the native *unlimited* answer in that order, then limited
    (the native `limit=3` follows the facade's own, analysis-major order: ADR 0109 §8b)."""
    path = RELEASES / name
    projected, plain = own_projected_answers(name), plain_answers(name)
    asked = 0
    with digests.open_query(path) as q:
        for question in digests.questions(q, q.variants_table(), path):
            try:
                answer = question.ask()
            except question.refusals as refused:
                answer = refused
            limited = question.name.startswith("top_hits/limit")
            if limited and not isinstance(answer, Exception):
                unlimited = q.top_hits(
                    threshold=0.2, observed_only=question.name.endswith("observed_only")
                )
                answer = canonical(unlimited).head(digests.TOP_HITS_LIMIT)
                assert len(answer["z"]) == digests.TOP_HITS_LIMIT, "the limit is exercised"
            if limited and isinstance(answer, Exception):
                assert projected["shapes"][question.name] == digests.refusal_digest(answer)
                continue
            if not isinstance(answer, Associations):
                continue
            asked += 1
            section = projected[question.section][question.name]
            expected = digests.result_digests(dict(canonical(answer)))
            del expected[digests.PVAL]
            assert section == expected, (question.section, question.name)
    assert asked > 20
    for key in plain:
        if key in (
            "shapes",
            "alid_probes",
            "misses",
            "range_probes",
            "indexed_subsets",
            "resolved",
            *digests.VALUES,
        ):
            continue  # association answers, compared above, and what is derived from them
        assert projected[key] == plain[key], key


@pytest.mark.parametrize("name", COMPLETED)
def test_a_projected_answers_values_are_the_unprojected_values(name: str) -> None:
    """`pval` and `log10_p` are kept as numbers; the same multiset of them is answered."""
    projected, plain = own_projected_answers(name), plain_answers(name)
    for kind in digests.VALUES:
        assert projected[kind].keys() == plain[kind].keys()
        for key, values in plain[kind].items():
            if "top_hits/limit" not in key:  # truncated in another order
                np.testing.assert_array_equal(
                    np.sort(projected[kind][key]), np.sort(values), err_msg=key
                )


# ── 2. a swap changes the answer without the projection, and not with it ─────────


@pytest.mark.parametrize("name", SWAPPED)
def test_two_swapped_analyses_answer_identically_under_the_projection(name: str) -> None:
    path = RELEASES / name
    reference = reference_for(path)
    ours = digests.release_digests(path, query=lambda p: swapped(p), reference=reference)
    assert digests.differences(own_projected_answers(name), ours) == []


@pytest.mark.parametrize("name", SWAPPED)
def test_two_swapped_analyses_answer_differently_without_it(name: str) -> None:
    path = RELEASES / name
    ours = digests.release_digests(path, query=lambda p: swapped(p))
    differing = digests.differences(plain_answers(name), ours)
    assert differing, "the swap must be visible without the projection"
    assert any(d.startswith("analyses_table") for d in differing)
    assert any(d.startswith("shapes/analysis/") for d in differing) or name != "ragged_completed"


# ── 3. an Analysis the release does not hold is refused ──────────────────────


def test_an_analysis_the_release_does_not_hold_is_refused_not_answered_empty() -> None:
    path = RELEASES / "ragged_completed"
    reference = reference_for(path)
    wider = replace(
        reference, ids=[*reference.ids, "dropped"], analyses={**reference.analyses, "dropped": 7}
    )
    found = answers(path, reference=wider)
    refusal = found["refusals"]["unheld/dropped"]
    with digests.open_query(path) as q, pytest.raises(UnknownAnalysisError) as raised:
        q.analysis("dropped")
    assert refusal == digests.refusal_digest(raised.value)
    assert set(refusal) == {"error"}, "a refusal digests as an error, not as arrays"
    empty = digests.result_digests(dict(Associations.empty()))
    assert refusal != empty
    assert (
        found["shapes"]["analysis/a/all"]
        == answers(path, reference=reference)["shapes"]["analysis/a/all"]
    ), "the Analyses it holds are still answered"
    assert not any(
        "dropped" in name for section in ("shapes", "alid_probes", "rho") for name in found[section]
    )


# ── 4. a limit cannot hide a displaced hit ───────────────────────────────────


def test_a_limit_does_not_hide_a_hit_another_analysis_displaced() -> None:
    """The release's `top_hits(limit=3)` lists Analysis a's hits first, so b's are all cut.
    A reference that names only b is answered with b's first three, not none."""
    path = RELEASES / "ragged_completed"
    reference = reference_for(path)
    only_b = replace(reference, ids=["b"], analyses={"b": reference.analyses["b"]})
    with digests.open_query(path) as q:
        native = q.top_hits(threshold=0.2, limit=3)
        everything = q.top_hits(threshold=0.2)
        b = reference.analyses["b"]
        assert not (native.analysis_index == b).any(), "the limit hides every hit of b"
        assert (everything.analysis_index == b).sum() > 3
        questions = {x.name: x for x in digests.questions(q, q.variants_table(), path, only_b)}
        asked = questions["top_hits/limit/all"].ask()
        assert len(asked) > 3, "the question is posed without the limit"
        projector = digests._Projector(only_b, q.variants_table(), q.analyses_table())
        limited, own = projector.apply(questions["top_hits/limit/all"], asked)
    assert len(limited["z"]) == 3 and set(limited["analysis_index"]) == {b}
    assert own["analysis_index"].tolist() == limited["analysis_index"].tolist()
    expected = canonical(everything.where(everything.analysis_index == b)).head(3)
    assert limited["z"].tolist() == expected["z"].tolist()
    assert limited["variant_index"].tolist() == expected["variant_index"].tolist()
    found = answers(path, reference=only_b)["shapes"]["top_hits/limit/all"]
    expected_digest = digests.result_digests(dict(limited))
    del expected_digest[digests.PVAL]
    assert found == expected_digest


# ── review fixes: every missing key, the stamp of a projection, decoded arrays ──


def test_every_analysis_the_release_does_not_hold_is_refused_not_only_the_sampled() -> None:
    """`Reference.ids` is a spread of eight; a dropped key outside it is probed too."""
    path = RELEASES / "ragged_completed"
    reference = reference_for(path)
    unsampled = replace(reference, analyses={**reference.analyses, "unsampled_drop": 9})
    assert "unsampled_drop" not in unsampled.ids
    refused = answers(path, reference=unsampled)["refusals"]["unheld/unsampled_drop"]
    assert set(refused) == {"error"}


def test_a_projected_record_is_stamped_with_the_reference_not_the_releases_own() -> None:
    path = RELEASES / "ragged_completed"
    reference = reference_for(path)
    assert reference.variant_space is not None
    with digests.open_query(path) as q:
        projector = digests._Projector(reference, q.variants_table(), q.analyses_table())
        native = q.analysis("a")
        projected, own = projector.associations(native, None)
    assert projected.variant_space == reference.variant_space
    assert projected.analysis_space == reference.analysis_space
    assert own.variant_space == native.variant_space, "the rows kept for `resolve` keep theirs"


def test_a_projected_record_does_not_keep_a_stamp_from_the_space_it_left() -> None:
    other = reference_for(RELEASES / "dense_completed")
    path = RELEASES / "ragged_completed"
    with digests.open_query(path) as q:
        projector = digests._Projector(
            replace(
                other, variants=reference_for(path).variants, analyses=reference_for(path).analyses
            ),
            q.variants_table(),
            q.analyses_table(),
        )
        projected, _ = projector.associations(q.analysis("a"), None)
    assert projected.variant_space == other.variant_space != q.variant_space


@pytest.mark.parametrize("name", SWAPPED)
def test_projected_arrays_of_a_swapped_release_equal_the_originals_exactly(name: str) -> None:
    path = RELEASES / name
    reference = reference_for(path)
    ours = digests.projected_arrays(path, reference)
    theirs = digests.projected_arrays(path, reference, query=lambda p: swapped(p))
    assert digests.array_differences(ours, theirs) == []
    assert any("z" in found for found in ours["shapes"].values() if isinstance(found, dict))


def test_projected_arrays_compare_eaf_within_a_tolerance_and_everything_else_exactly() -> None:
    path = RELEASES / "ragged_completed"
    reference = reference_for(path)
    ours = digests.projected_arrays(path, reference)
    name = "analysis/a/all"
    arrays = dict(ours["shapes"][name])
    assert np.isfinite(arrays["eaf"]).any()
    nudged = {**ours, "shapes": {**ours["shapes"], name: {**arrays, "eaf": arrays["eaf"] + 1e-3}}}
    assert digests.array_differences(ours, nudged, eaf_tolerance=2e-3) == []
    assert digests.array_differences(ours, nudged) == [f"shapes/{name}/eaf"]
    moved = {**ours, "shapes": {**ours["shapes"], name: {**arrays, "z": arrays["z"] + 1e-3}}}
    assert digests.array_differences(ours, moved, eaf_tolerance=1.0) == [f"shapes/{name}/z"]
