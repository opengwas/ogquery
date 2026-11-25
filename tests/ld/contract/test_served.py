"""A store served over HTTP answers every query exactly as it does from disk (ld-store §1;
ogquery#2).

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
