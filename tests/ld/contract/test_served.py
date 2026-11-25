"""A store served over HTTP answers every query exactly as it does from disk (ld-store §1;
ogquery#2): the four queries, and the reads of what it holds.

The Test Store's directory is served from this machine by a static Range-serving server
(`served_store`), as a static host would serve it, and read through the `http` extra.
"""

from __future__ import annotations

import numpy as np
import pytest
import support
from support import alids, passing_rows

import ogquery.ld

pytestmark = pytest.mark.http


def test_every_query_over_http_is_the_local_answer(served_store: str) -> None:
    local = ogquery.ld.open(support.TEST_STORE)
    remote = ogquery.ld.open(served_store)
    names = alids(passing_rows()[::3])
    some = names[:60]
    z = np.abs(support.panel_data().gwas().z[passing_rows()[::3]])

    def answers(store: ogquery.ld.LDStore) -> dict[str, object]:
        return {
            "matrix": store.matrix(some).r,
            "kinds": store.matrix(some).kinds(),
            "region": store.region("21", 30_000_000, 2_000_000).r,
            "proxies": store.proxies(some, r2=0.2, kb=3000).columns(),
            "clump": store.clump(names, z, r2=0.1, kb=3000).columns(),
            "variants_in": store.variants_in("22", 16_000_000, 30_000_000).columns(),
            "count_in": store.count_in("21", 1, 3_000_000_000),
            "has": store.has([*some, "22:1:A:C"]),
        }

    here, there = answers(local), answers(remote)
    assert len(here["clump"]["alid"]) > 300, "the clump must be one worth comparing"
    for name, value in here.items():
        if isinstance(value, dict):
            assert value.keys() == there[name].keys(), name
            for column in value:
                np.testing.assert_array_equal(value[column], there[name][column], err_msg=name)
        else:
            np.testing.assert_array_equal(value, there[name], err_msg=name)


def test_a_scheme_in_capitals_opens_the_served_store(served_store: str) -> None:
    # A URL's scheme is case-insensitive (RFC 3986 §3.1); fsspec knows `http`, not `HTTP`.
    assert served_store.startswith("http://")
    shouting = "HTTP" + served_store[len("http") :]
    remote, local = ogquery.ld.open(shouting), ogquery.ld.open(support.TEST_STORE)
    some = alids(passing_rows()[:30])
    np.testing.assert_array_equal(remote.matrix(some).r, local.matrix(some).r)
    assert remote._reader.location == shouting, "a refusal names the location as it was given"
