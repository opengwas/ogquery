"""The Store Reader's read of the Clumps, `TopHitTier.clumps` (spec §18.1 *Reading*).

Ported from opengwasdb's `tests/test_store_reader_clumps.py` (#418), on a hand-built zarr
group instead of its Clumps Test Panel Stores: each refusal is held here, a group not laid
out as §18.1 says, a group the manifest does not declare, and a declaration with no group.
What stays in opengwasdb needs its builders: the read of a release its `opengwasdb clump`
wrote, `OpenGWASDBStore.set_manifest_entry` and the Analyses column write.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from ogquery.manifest import StoreManifest
from ogquery.release import open_release
from ogquery.top_hits import (
    CLUMPS_ATTRIBUTE_KEYS,
    MalformedClumps,
    TierClumps,
    TopHitTier,
    TopHitTiers,
    declares_clumps_of,
)

OFFSETS = [0, 2, 2, 5]
N_ROWS = 5
DECLARATION = {"tier": "p_5e_08", "configuration": "default"}


def _build(path: Path) -> zarr.Group:
    root = zarr.open_group(str(path), mode="w")
    tier = root.create_group("top_hits/p_5e_08")
    tier.create_array("z", data=np.arange(N_ROWS, dtype=np.float32))
    tier.create_array("analysis_offsets", data=np.asarray(OFFSETS, dtype=np.int64))
    default = tier.create_group("clumps/default")
    default.create_array("lead", data=np.asarray([0, -1, 0, 1, -2], dtype=np.int32))
    default.create_array("r", data=np.asarray([10, 0, -5, 7, 0], dtype=np.int16))
    default.attrs["clumps"] = {key: 1 for key in sorted(CLUMPS_ATTRIBUTE_KEYS)}
    return root


def _tier(root: zarr.Group, clumps: Any = DECLARATION) -> TopHitTier:
    tier = TopHitTiers(root, clumps=clumps).tier(5e-8)
    assert tier is not None
    return tier


@pytest.fixture
def root(tmp_path: Path) -> zarr.Group:
    return _build(tmp_path / "data.zarr")


def test_a_declared_tier_reads_its_clumps_and_one_analysis_reads_its_slice(
    root: zarr.Group,
) -> None:
    tier = _tier(root)
    assert tier.has_clumps() and tier.clumps_declared
    whole = tier.clumps()
    assert isinstance(whole, TierClumps)
    assert whole.lead.dtype == np.int32 and whole.r.dtype == np.int16
    assert whole.lead.tolist() == [0, -1, 0, 1, -2] and whole.r.tolist() == [10, 0, -5, 7, 0]
    for index in range(len(OFFSETS) - 1):
        part = tier.clumps(index)
        assert part is not None
        a, b = OFFSETS[index], OFFSETS[index + 1]
        assert part.lead.tolist() == whole.lead[a:b].tolist()
        assert part.r.tolist() == whole.r[a:b].tolist()
        assert part.attribute == whole.attribute
    empty = tier.clumps(1)
    assert empty is not None and len(empty.lead) == 0


def test_a_tier_without_clumps_and_without_a_declaration_reads_none(tmp_path: Path) -> None:
    group = zarr.open_group(str(tmp_path / "data.zarr"), mode="w")
    tier = group.create_group("top_hits/p_5e_08")
    tier.create_array("z", data=np.zeros(3, dtype=np.float32))
    opened = _tier(group, None)
    assert not opened.has_clumps() and not opened.clumps_declared
    assert opened.clumps() is None


def test_the_manifest_reads_its_clumps_entry_and_never_writes_it() -> None:
    assert declares_clumps_of(DECLARATION, "p_5e_08")
    assert not declares_clumps_of(DECLARATION, "p_5e_06")
    assert not declares_clumps_of({"tier": "p_5e_08", "configuration": "other"}, "p_5e_08")
    assert not declares_clumps_of(None, "p_5e_08")
    assert not declares_clumps_of("p_5e_08", "p_5e_08")


def _undeclare(root: zarr.Group) -> Any:
    return None


def _drop_group(root: zarr.Group) -> Any:
    del root["top_hits/p_5e_08/clumps"]
    return DECLARATION


def _extra_member(root: zarr.Group) -> Any:
    root["top_hits/p_5e_08/clumps"].create_group("other")
    return DECLARATION


def _missing_member(root: zarr.Group) -> Any:
    del root["top_hits/p_5e_08/clumps/default/r"]
    return DECLARATION


def _rewritten_lead(data: Callable[[np.ndarray], np.ndarray], dtype: str) -> Callable[..., Any]:
    def corrupt(root: zarr.Group) -> Any:
        group = root["top_hits/p_5e_08/clumps/default"]
        lead = np.asarray(group["lead"][:])
        group.create_array("lead", data=data(lead).astype(dtype), overwrite=True)
        return DECLARATION
    return corrupt


def _attribute(edit: Callable[[dict[str, Any]], None]) -> Callable[..., Any]:
    def corrupt(root: zarr.Group) -> Any:
        group = root["top_hits/p_5e_08/clumps/default"]
        attrs = dict(group.attrs)
        edit(attrs)
        group.attrs.clear()
        group.attrs.update(attrs)
        return DECLARATION
    return corrupt


@pytest.mark.parametrize(("corrupt", "named"), [
    (_undeclare, r"not declared by the release's manifest"),
    (_drop_group, r"holds no clumps/ group"),
    (_extra_member, r"holds \['default', 'other'\]"),
    (_missing_member, r"default/r is missing"),
    (_rewritten_lead(lambda lead: lead, "int64"), r"default/lead is int64, not int32"),
    (_rewritten_lead(lambda lead: lead[:-1], "int32"), r"default/lead holds \d+ entries"),
    (_attribute(lambda attrs: attrs.pop("clumps")), r"has no 'clumps' attribute"),
    (_attribute(lambda attrs: attrs["clumps"].pop("kb")), r"attribute lacks \['kb'\]"),
], ids=["undeclared", "declared-no-group", "extra-member", "missing-member", "dtype", "shape",
        "missing-attribute", "missing-key"])
def test_each_malformed_or_undeclared_group_is_refused_by_name(
    root: zarr.Group, corrupt: Callable[[zarr.Group], Any], named: str
) -> None:
    declaration = corrupt(root)
    tier = _tier(root, declaration)
    with pytest.raises(MalformedClumps, match=named):
        tier.clumps()
    with pytest.raises(MalformedClumps, match=named):
        tier.clumps(0)


def test_a_manifest_reads_its_clumps_entry_and_never_writes_it() -> None:
    manifest = next(Path(__file__).parents[1].glob("contract/releases/*/manifest.json"))
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert StoreManifest.from_dict(data).clumps is None
    declared = StoreManifest.from_dict({**data, "clumps": DECLARATION})
    assert declared.clumps == DECLARATION
    assert "clumps" not in declared.to_dict()


# ── the entry points: the manifest's declaration reaches the tier ────────────

RELEASE = Path(__file__).parents[1] / "contract" / "releases" / "dense_completed"


@pytest.fixture
def declared_release(tmp_path: Path) -> Path:
    """A copy of a contract release with a `clumps/` group in its 5e-8 tier and the
    manifest's `clumps` entry declaring it."""
    path = tmp_path / "release"
    shutil.copytree(RELEASE, path)
    tier = zarr.open_group(str(path / "data.zarr"), mode="r+")["top_hits/p_5e_08"]
    n_rows = int(tier["z"].shape[0])
    default = tier.create_group("clumps/default")
    default.create_array("lead", data=np.full(n_rows, -1, dtype=np.int32))
    default.create_array("r", data=np.arange(n_rows, dtype=np.int16))
    default.attrs["clumps"] = {key: 7 for key in sorted(CLUMPS_ATTRIBUTE_KEYS)}
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    (path / "manifest.json").write_text(
        json.dumps({**manifest, "clumps": DECLARATION}), encoding="utf-8"
    )
    return path


def test_a_release_reads_the_clumps_its_manifest_declares(declared_release: Path) -> None:
    tier = open_release(declared_release).top_hit_tiers().tier(5e-8)
    assert tier is not None and tier.clumps_declared
    clumps = tier.clumps()
    assert clumps is not None
    assert clumps.attribute == {key: 7 for key in sorted(CLUMPS_ATTRIBUTE_KEYS)}
    assert clumps.r.tolist() == list(range(len(clumps.r))) and (clumps.lead == -1).all()


def test_a_release_whose_manifest_does_not_declare_its_clumps_refuses_them(
    declared_release: Path,
) -> None:
    manifest = json.loads((declared_release / "manifest.json").read_text(encoding="utf-8"))
    del manifest["clumps"]
    (declared_release / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    tier = open_release(declared_release).top_hit_tiers().tier(5e-8)
    assert tier is not None
    with pytest.raises(MalformedClumps, match="not declared by the release's manifest"):
        tier.clumps()


# ── a group, array or scalar where §18.1 says the other ──────────────────────


def _clumps_an_array(root: zarr.Group) -> Any:
    del root["top_hits/p_5e_08/clumps"]
    root["top_hits/p_5e_08"].create_array("clumps", data=np.zeros(N_ROWS, dtype=np.int32))
    return DECLARATION


def _lead_a_group(root: zarr.Group) -> Any:
    default = root["top_hits/p_5e_08/clumps/default"]
    del default["lead"]
    default.create_group("lead")
    return DECLARATION


def _lead_a_scalar(root: zarr.Group) -> Any:
    default = root["top_hits/p_5e_08/clumps/default"]
    del default["lead"]
    default.create_array("lead", data=np.int32(3))
    return DECLARATION


@pytest.mark.parametrize(("corrupt", "named"), [
    (_clumps_an_array, r"clumps is an array, not a group"),
    (_lead_a_group, r"default/lead is a group, not an array"),
    (_lead_a_scalar, r"default/lead holds a scalar and the tier 5 rows"),
], ids=["clumps-array", "lead-group", "lead-scalar"])
def test_a_member_of_the_wrong_kind_is_refused_by_name(
    root: zarr.Group, corrupt: Callable[[zarr.Group], Any], named: str
) -> None:
    tier = _tier(root, corrupt(root))
    with pytest.raises(MalformedClumps, match=named):
        tier.clumps()
