"""The Store Reader's p-value rules agree with the scipy forms they replaced (#464).

`log10_p_two_sided` was ``(log 2 + scipy.stats.norm.logsf(|z|)) / ln 10`` and
`z_critical` was ``sqrt(2) * scipy.special.erfcinv(threshold)``; the Store Reader
may not import scipy, so both are now computed with the standard library. scipy
is imported here, in the test only, as the reference.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy import special, stats

from ogquery.stats import log10_p_two_sided, p_value_from_z, z_critical
from ogquery.top_hits import TOP_HIT_THRESHOLDS


def scipy_log10_p(z: np.ndarray) -> np.ndarray:
    """The implementation `log10_p_two_sided` replaced, verbatim."""
    abs_z = np.abs(np.asarray(z, dtype="float64"))
    result: np.ndarray = (math.log(2.0) + stats.norm.logsf(abs_z)) / math.log(10.0)
    return result


_PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494459")


def _taylor_log_erfc(y: Decimal) -> Decimal:
    """``ln erfc(y)`` by erf's Taylor series: exact to the context's precision for y <= 3."""
    term, total, n = y, Decimal(0), 0
    while n == 0 or abs(term) > Decimal(10) ** -80:
        total += term / (2 * n + 1)
        n += 1
        term = -term * y * y / n
    return (1 - 2 * total / _PI.sqrt()).ln()


def _continued_fraction_log_erfc(y: Decimal, depth: int = 4000) -> Decimal:
    """``ln erfc(y)`` by Laplace's continued fraction, for y > 3, where the series
    would need hundreds of digits to cancel:
    erfc(y) = exp(-y²) / sqrt(pi) / (y + (1/2) / (y + 1 / (y + (3/2) / (y + ...))))."""
    t = y
    for n in range(depth, 0, -1):
        t = y + (Decimal(n) / 2) / t
    return -y * y - _PI.sqrt().ln() - t.ln()


def exact_log10_p(z: float, *, method: str | None = None) -> float:
    """log10 of ``erfc(|z| / sqrt(2))`` to 50 digits, for any |z| a test asks."""
    with localcontext() as context:
        context.prec = 70
        y = abs(Decimal(z)) / Decimal(2).sqrt()
        if y == 0:
            return 0.0
        series = method == "series" or (method is None and y <= 3)
        log_erfc = _taylor_log_erfc(y) if series else _continued_fraction_log_erfc(y)
        return float(log_erfc / Decimal(10).ln())


def test_the_two_references_agree_where_they_meet():
    for z in (3.0, 4.243, 5.0, 6.0):
        series = exact_log10_p(z, method="series")
        assert series == exact_log10_p(z, method="fraction"), z


#: 809 points over -40 <= z <= 40: every 0.1 in both signs, and the edges of the
#: paths `log10_p_two_sided` takes (p near 1, the series' switch at |z| = 36.8,
#: p subnormal at 37.5 and 0.0 at 38.5).
GRID = np.concatenate(
    [
        np.linspace(0.0, 40.0, 401),
        -np.linspace(0.05, 39.95, 400),
        [1e-12, -1e-9, 1e-6, -1e-4, 37.5, -38.5, 26 * math.sqrt(2), -26 * math.sqrt(2) + 1e-9],
    ]
)


#: How far `log10_p_two_sided` may be from the exact value, relative: five float64 ulps.
#:
#: It is ``log(erfc(y)) / ln 10`` (``log1p(-erf(y))`` for y < 0.5), so it is as close
#: as the platform's libm lets it be. C libraries promise `erf`/`erfc` to a few ulps, not
#: correctly rounded, and differ: glibc, macOS's libm and the Windows CRT each give
#: another last bit. opengwasdb's cross-machine analysis of this composition (its
#: pval-ulp round, mpmath over the contract's |z|) found a 4-ulp `erfc` gives up to
#: 3.8 eps in `log10_p`, and 4.7 eps with `log` also 1 ulp off; the final division by
#: ln 10 adds at most half an ulp. Five eps covers that, and is the contract's own
#: 4-ulp comparison of `log10_p` across machines plus that rounding
#: (`digests.CROSS_MACHINE_RTOL`). Measured: 3.9e-16 (1.8 eps) on glibc, 4.3e-16 on
#: Windows, 4.8e-16 (2.1 eps, z = -0.95) on macOS. A formula error is far larger:
#: `test_the_bound_catches_a_formula_error` shows a 1e-12 shift fails it.
LIBM_BOUND = 5 * float(np.finfo(np.float64).eps)


def _largest_relative_error() -> tuple[float, float]:
    exact = np.array([exact_log10_p(float(z)) for z in GRID])
    ours = log10_p_two_sided(GRID)
    zero = exact == 0.0
    assert np.all(ours[zero] == 0.0)
    relative = np.abs(ours[~zero] - exact[~zero]) / np.abs(exact[~zero])
    return float(relative.max()), float(GRID[~zero][relative.argmax()])


def test_log10_p_is_within_five_ulps_of_the_exact_value_across_minus_40_to_40():
    assert len(GRID) >= 200 and GRID.min() == -39.95 and GRID.max() == 40.0
    worst, at = _largest_relative_error()
    assert worst < LIBM_BOUND, (worst, worst / LIBM_BOUND * 5, "eps at z =", at)


def test_the_bound_catches_a_formula_error(monkeypatch: pytest.MonkeyPatch):
    """The bound is a libm's last bits, not a formula's: shifting every result by 1e-12
    relative -- ln 10 taken a little wrong -- fails it some 900 times over."""
    import ogquery.stats

    monkeypatch.setattr(ogquery.stats, "_LN10", ogquery.stats._LN10 * (1 + 1e-12))
    worst, _ = _largest_relative_error()
    assert worst > 100 * LIBM_BOUND, worst


#: Below this |z| scipy's own form loses digits: its p is ``0.5 - tiny`` doubled,
#: and the log of it is off by up to 1.3e-4 relative at |z| = 1e-12, and still by
#: 1e-12 at |z| = 2e-4, where the exact value is -7e-5. The comparison with scipy
#: starts here; the one with the exact value covers the rest.
SCIPY_ACCURATE_FROM = 1e-3


def test_log10_p_agrees_with_scipy_to_1e_12_below_37():
    z = np.concatenate(
        [
            np.linspace(SCIPY_ACCURATE_FROM, 37.0, 1_000_001, endpoint=False),
            -np.geomspace(SCIPY_ACCURATE_FROM, 37.0, 10_001, endpoint=False),
        ]
    )
    ours, theirs = log10_p_two_sided(z), scipy_log10_p(z)
    relative = np.abs(ours - theirs) / np.abs(theirs)
    assert relative.max() < 1e-12, (relative.max(), z[relative.argmax()])


@pytest.mark.parametrize("z", [0.0, 1e-12, 1e-9, 1e-6, 1e-4, 2e-4, 1e-3, 0.3, 0.7, 1.0, 47.8])
def test_log10_p_is_exact_where_scipy_is_not(z: float):
    ours = float(log10_p_two_sided(np.array([z]))[0])
    exact = exact_log10_p(z)
    assert ours == exact == 0.0 or math.isclose(ours, exact, rel_tol=1e-15), (z, ours, exact)


def test_log10_p_beyond_37_is_finite_and_within_the_old_tolerance():
    z = np.geomspace(37.0, 1000.0, 20_001)
    ours = log10_p_two_sided(z)
    assert np.isfinite(ours).all()
    # The tolerance the tests held the scipy form to (`abs_tol` 1e-3); this
    # form is far inside it, and inside 1e-12 relative too.
    np.testing.assert_allclose(ours, scipy_log10_p(z), rtol=1e-12, atol=0)


def test_the_fads_window_z_is_finite_and_far_below_minus_300():
    value = float(log10_p_two_sided(np.array([47.8]))[0])
    assert math.isfinite(value) and value < -300
    assert math.isclose(value, -497.92437972, abs_tol=1e-6)


def test_log10_p_keeps_nan_shape_sign_and_dtype():
    out = log10_p_two_sided(np.array([[np.nan, np.inf], [-np.inf, 0.0]], dtype=np.float32))
    assert out.dtype == np.float64 and out.shape == (2, 2)
    assert math.isnan(out[0, 0]) and out[0, 1] == out[1, 0] == -math.inf
    assert out[1, 1] == 0.0 and math.copysign(1.0, out[1, 1]) == 1.0
    assert log10_p_two_sided(np.float64(-3.0)).shape == ()


def test_the_scalar_p_and_the_log10_p_are_one_rule():
    for z in (-6.0, -1.0, 0.0, 0.5, 2.0, 8.0, 30.0):
        assert math.isclose(
            math.log10(p_value_from_z(z)), float(log10_p_two_sided(np.array([z]))[0]), rel_tol=1e-13
        )


def scipy_z_critical(threshold: float) -> float:
    return math.sqrt(2.0) * float(special.erfcinv(threshold))


@pytest.mark.parametrize("threshold", TOP_HIT_THRESHOLDS)
def test_z_critical_cuts_every_float32_z_where_scipy_did(threshold: float):
    ours, theirs = z_critical(threshold), scipy_z_critical(threshold)
    # The two differ by at most a few ulps (5e-8: 2 ulps) ...
    assert abs(ours - theirs) <= 4 * math.ulp(theirs)
    # ... and no float32 lies between them, so every |z| a Store holds -- float32
    # in a tier, a plane or a CSR -- is cut exactly where the scipy value cut it.
    low, high = sorted((ours, theirs))
    above = np.nextafter(np.float32(low), np.float32(np.inf))
    nearest = np.float32(low) if float(np.float32(low)) >= low else above
    assert not (low <= float(nearest) <= high), (threshold, low, float(nearest), high)
    for value in (np.float32(low), np.float32(high), nearest):
        assert (float(value) >= ours) == (float(value) >= theirs)


@pytest.mark.parametrize("threshold", [1e-300, 1e-100, 1e-20, 1e-10, 1e-3, 0.01, 0.05, 0.2, 0.9])
def test_z_critical_agrees_with_scipy_for_any_threshold(threshold: float):
    assert math.isclose(z_critical(threshold), scipy_z_critical(threshold), rel_tol=1e-15)
