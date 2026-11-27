"""The Genotype Store's questions, and opengwasdb's answers to them (opengwas/ogquery#11).

ogquery's Genotype Store reader came from opengwasdb `964ac1b2` (ADR 0002, amended), so its
answers are held to that commit's: `QUESTIONS` is what the tests ask, and
`data/test-panel/genotype-answers/` is what `964ac1b2`'s own `opengwasdb.ld.open` answered on the
committed Genotype Store. This module wrote those files once, run in that commit's environment
(`data/SOURCE` gives the command), and `test_genotype_answers.py` imports it to ask
`ogquery.ld.open` the same questions and compare.

An answer is flattened to its fields: arrays to `answers.npz`, the rest, and every refusal (its
class, reason and text), to `answers.json`. A refusal's text names the store's path, which differs
between the run and a test's copy, so the path is written as `<store>`, and the Test Store's as
`<test-store>`. Only numpy and the standard library are imported at module level; opengwasdb only
by `main`, in its environment. `contract/support.py` picks the ALIDs asked about.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
for _directory in (HERE, HERE / "contract"):
    if str(_directory) not in sys.path:
        sys.path.append(str(_directory))

import support  # noqa: E402

TEST_PANEL = HERE / "data" / "test-panel"
GENOTYPE_STORE = TEST_PANEL / "genotype-store"
#: The LD Test Store, which a Genotype Store's reader refuses as the wrong kind.
TEST_STORE = TEST_PANEL / "store"
ANSWERS = TEST_PANEL / "genotype-answers"
#: The inputs the Genotype Store pins, beside it (`../panel.*`), as its build read them.
PINNED = ("panel.pgen", "panel.psam", "panel.pvar")
STORE = "<store>"
TEST_STORE_NAME = "<test-store>"

Ask = Callable[[Any], object]


def _names(step: int) -> list[str]:
    return support.alids(support.passing_rows()[::step])


def _matrix() -> list[str]:
    # More than 256 names, so a read at `_READ_ROWS = 128` is split across threads; reversed, with a
    # duplicate and an absent ALID, as opengwasdb's parallel-read test asks.
    names = _names(4)[::-1]
    return [*names, names[-1], "absent"]


def _questions() -> dict[str, Ask]:
    out: dict[str, Ask] = {"matrix": lambda store: store.matrix(_matrix())}
    for centre in (20_000_000, 25_000_000):
        for fill in (None, 0.0):
            out[f"region-{centre}-{fill}"] = (
                lambda store, c=centre, f=fill: store.region("chr22", c, 1_500_000, fill=f)
            )
    out["proxies"] = lambda store: store.proxies([*_names(11), "absent"], r2=0.01, kb=3000)
    clumped = _names(7)
    scores = [*np.arange(len(clumped), dtype=float).tolist(), 0.0]
    out["clump"] = lambda store: store.clump([*clumped, "absent"], scores, r2=0.01, kb=3000)
    for query in ("clump", "proxies"):
        args: tuple[list[Any], ...] = (["absent"], [1]) if query == "clump" else (["absent"],)
        for r2, kb in (
            (-1, 10),
            (float("nan"), 10),
            (0.1, -1),
            (0.1, float("inf")),
            (0.1, 10001),
            (0.001, 3000),
            (0.01, 5000),
        ):
            out[f"refused-{query}-{r2}-{kb}"] = (
                lambda store, q=query, a=args, r=r2, k=kb: getattr(store, q)(*a, r2=r, kb=k)
            )
    for radius in (-1, 5_000_001):
        out[f"refused-region-{radius}"] = lambda store, r=radius: store.region("22", 25_000_000, r)
    for label, score in (("negative", [-1]), ("inf", [np.inf]), ("nan", [np.nan]), ("two", [1, 2])):
        out[f"refused-score-{label}"] = (
            lambda store, s=score: store.clump(["absent"], s, r2=0.1, kb=3000)
        )
    out["refused-closed"] = _closed
    out["refused-lead-outside-its-window"] = _outside
    out["refused-wrong-kind-genotype-reader-on-an-ld-store"] = _genotype_reader_on_an_ld_store
    out["refused-wrong-kind-ld-reader-on-a-genotype-store"] = _ld_reader_on_a_genotype_store
    return out


def _genotype_reader_on_an_ld_store(store: Any) -> object:
    # The Genotype Store reader's own open, asked to open the LD Test Store.
    return type(store._reader).open(TEST_STORE)


def _ld_reader_on_a_genotype_store(store: Any) -> object:
    # The LD Store reader's open, asked to open this Genotype Store.
    from ogquery.ld.store import StoreReader

    return StoreReader.open(store._reader.location)


def _closed(store: Any) -> object:
    names = _names(4)[:10]
    store.matrix(names)
    store.close()
    return store.matrix(names)


def _outside(store: Any) -> object:
    # A lead the clump's hits do not hold: a store whose rows and positions disagree refuses it by
    # name rather than read a neighbour's row (opengwasdb's `_locate`).
    reader = store._reader
    hits = reader.rows_of(_names(1)[:5])
    chrom = str(reader.loci_of(hits)[0][0])
    start, stop = reader.chromosome_range(chrom)
    local = hits - start
    absent = next(row for row in range(stop - start) if row not in set(local.tolist()))
    with reader.clump_rows(chrom, local, 3000.0) as partners:
        return partners(absent)


QUESTIONS = _questions()

#: The refusals an edited copy's record gives on open: a pinned pgen moved, resized or touched.
PIN_FAULTS = ("missing", "size", "mtime_ns")


def flatten(answer: object, prefix: str = "") -> tuple[dict[str, Any], dict[str, Any]]:
    """An answer's fields as ({key: array}, {key: JSON value}), nested dataclasses by dotted key."""
    arrays: dict[str, Any] = {}
    values: dict[str, Any] = {}
    assert dataclasses.is_dataclass(answer) and not isinstance(answer, type), type(answer)
    values[prefix + "type"] = type(answer).__name__
    for field in dataclasses.fields(answer):
        value = getattr(answer, field.name)
        key = prefix + field.name
        if isinstance(value, np.ndarray):
            arrays[key] = value
        elif dataclasses.is_dataclass(value):
            inner_arrays, inner_values = flatten(value, key + ".")
            arrays.update(inner_arrays)
            values.update(inner_values)
        else:
            values[key] = list(value) if isinstance(value, tuple) else value
    kinds = getattr(answer, "kinds", None)
    # A filled matrix cannot tell a filled cell from a stored one, so it has no kinds().
    if callable(kinds) and getattr(answer, "fill", None) is None:
        arrays[prefix + "kinds()"] = kinds()
    return arrays, values


def refusal(error: BaseException, store: Path) -> dict[str, Any]:
    """A refusal as its class, its reason (a `Refusal`'s, else None) and its text, the store's
    path written as `<store>` and the Test Store's as `<test-store>`."""
    reason = getattr(error, "reason", None)
    message = str(error).replace(str(store), STORE).replace(str(TEST_STORE), TEST_STORE_NAME)
    return {
        "raised": type(error).__name__,
        "reason": None if reason is None else str(reason.value),
        "message": message,
    }


def answer(open_store: Callable[[Path], Any], store: Path, ask: Ask) -> tuple[dict, dict]:
    """`ask`'s answer on a fresh open of `store`, flattened, or the refusal it raised."""
    try:
        result = ask(open_store(store))
    except Exception as error:
        return {}, {"refusal": refusal(error, store)}
    return flatten(result)


def place(tree: Path) -> Path:
    """A copy of the Genotype Store under `tree`, beside the inputs it pins, each input's mtime
    set to the one the store records: git keeps no mtime, and an open refuses a changed one."""
    shutil.copytree(GENOTYPE_STORE, tree / GENOTYPE_STORE.name)
    for name in PINNED:
        shutil.copy2(TEST_PANEL / name, tree / name)
    store = tree / GENOTYPE_STORE.name
    document = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    record = document["attributes"]["genotype_store"]
    for fileset in record["filesets"]:
        for name in ("pgen", "psam", "pvar"):
            pin = fileset[name]
            os.utime(store / pin["path"], ns=(pin["mtime_ns"], pin["mtime_ns"]))
    return store


def edit_pin(store: Path, fault: str) -> None:
    """Edit the copy's record: the first fileset's pgen pin moved, resized or touched."""
    document = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    pin = document["attributes"]["genotype_store"]["filesets"][0]["pgen"]
    if fault == "missing":
        pin["path"] = "missing-panel.pgen"
    else:
        pin[fault] += 1
    (store / "zarr.json").write_text(json.dumps(document, indent=2), encoding="utf-8")


def ask_all(open_store: Callable[[Path], Any], tree: Path) -> tuple[dict, dict]:
    """Every question's answer and every pin fault's refusal, by name, on copies in `tree`."""
    store = place(tree / "store")
    arrays: dict[str, Any] = {}
    values: dict[str, Any] = {}
    for name, ask in QUESTIONS.items():
        found_arrays, found_values = answer(open_store, store, ask)
        arrays.update({f"{name}/{key}": value for key, value in found_arrays.items()})
        values[name] = found_values
    for fault in PIN_FAULTS:
        copy = place(tree / f"pin-{fault}")
        edit_pin(copy, fault)
        values[f"refused-pin-{fault}"] = answer(open_store, copy, lambda opened: opened)[1]
    return arrays, values


def load() -> tuple[dict[str, Any], dict[str, Any]]:
    """The committed answers, as `ask_all` returns them."""
    with np.load(ANSWERS / "answers.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    values: dict[str, Any] = json.loads((ANSWERS / "answers.json").read_text(encoding="utf-8"))
    return arrays, values


def main(tree: Path, out: Path) -> None:
    """Write `964ac1b2`'s answers: run in that commit's environment, never ogquery's."""
    import opengwasdb.ld
    from opengwasdb.ld.genotypes.reader import GenotypeReader

    probe = opengwasdb.ld.open(place(tree / "probe"))
    assert isinstance(probe._reader, GenotypeReader), type(probe._reader)
    arrays, values = ask_all(opengwasdb.ld.open, tree)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "answers.npz", **arrays)
    text = json.dumps(values, indent=1, sort_keys=True) + "\n"
    (out / "answers.json").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))  # a scratch directory, then where to write
