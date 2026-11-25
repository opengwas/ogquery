"""ogld#16: each LD Store records its encoding, and the reader, the writer and validation use it.

Stores of the Test Panel's reference r at k = 10 are written at × 511 (the HGDP+1kGP stores'),
× 2047 (UK Biobank EUR's) and × 32767 by the Test Store's own script, once per worker each.
Expected values are the reference r and each scale's literals: a code is computed here as ld-store
§3 writes it (`support.codes`), and each floor is ld-store §3's, never ogld's arithmetic.
"""

from __future__ import annotations

import importlib
import sys
from functools import cache
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import ogquery.ld
from ogquery.ld.format import REncoding

CONTRACT = Path(__file__).resolve().parent / "contract"
REPOSITORY = Path(__file__).resolve().parents[2]
SCALES = (511, 2047, 32767)
#: The far code floor at the Test Panel's t_far(771, 10) = 0.0274211, per scale (ld-store §3; the
#: Test Panel keeps all 771 EUR members by the owner's ruling).
FLOORS = {511: 85, 2047: 339, 32767: 5426}
SLACK = 1e-9


@cache
def contract(name: str) -> ModuleType:
    """A module of the contract tests: `support`, or `test_store` and its breakers."""
    if str(CONTRACT) not in sys.path:
        sys.path.insert(0, str(CONTRACT))
    return importlib.import_module(name)


_STORES: dict[int, Path] = {}


@cache
def every_pair(location: Path) -> tuple[ogquery.ld.LDMatrix, np.ndarray]:
    """The matrix of every variant of a store at `location`, and the reference r it holds."""
    support = contract("support")
    rows = tuple(int(row) for row in support.passing_rows())
    return ogquery.ld.open(location).matrix(support.alids(rows)), support.reference_matrix(
        10, rows
    )


# ------------------------------------------------------------------ the round trip


# ------------------------------------------------------------------ refused at open


UNIMPLEMENTED = {
    "an int8 container": {"r_dtype": "int8", "r_scale": 127},
    "a scale past int16": {"r_scale": 32768},
    "rounding half away from zero": {"rounding": "half_away_from_zero"},
}


# ------------------------------------------------------------------ the floor, the Bound, refusals


#: The Partner Floor at the Partner Thresholds the owner set per dataset (CONTEXT.md; ld-store §3),
#: and the t_far each pair of them is judged under: t_partner 0.01 on the HGDP+1kGP stores at × 511
#: and × 32767, and 0.001 = t_far on UK Biobank EUR at × 2047. The floors are literals computed
#: here.
PARTNER_FLOORS = {
    (511, 0.01, 0.0318654): 52,
    (2047, 0.001, 0.001): 65,
    (32767, 0.01, 0.0274211): 3277,
}


@pytest.mark.parametrize(("scale", "t_partner", "t_far"), list(PARTNER_FLOORS))
def test_the_partner_floor_is_the_smallest_code_reaching_the_partner_threshold(
    scale: int, t_partner: float, t_far: float
) -> None:
    floor = PARTNER_FLOORS[(scale, t_partner, t_far)]
    assert REncoding(scale).partner_code_floor(t_partner, t_far) == floor
    assert (floor / scale) ** 2 >= t_partner, "the floor's own code reaches the threshold"
    assert ((floor - 1) / scale) ** 2 < t_partner, "and no smaller code does"


def test_a_partner_threshold_at_t_far_has_the_far_code_floor() -> None:
    # UK Biobank EUR's t_partner is its t_far (CONTEXT.md): the Partner Rows are its Far Pairs too.
    encoding = REncoding(2047)
    assert encoding.partner_code_floor(0.001, 0.001) == encoding.far_code_floor(0.001) == 65


def test_a_partner_threshold_above_t_far_is_refused_naming_both() -> None:
    with pytest.raises(ValueError, match="above t_far") as refused:
        REncoding(511).partner_code_floor(0.04, 0.0318654)
    assert "0.04" in str(refused.value), refused.value
    assert "0.0318654" in str(refused.value), refused.value


# ------------------------------------------------------------------ validation at each scale


def _chr22_rows() -> np.ndarray:
    support = contract("support")
    rows = support.passing_rows()
    return rows[support.panel_data().variants().chrom[rows] == "22"]


# ------------------------------------------------------------------ one owner


