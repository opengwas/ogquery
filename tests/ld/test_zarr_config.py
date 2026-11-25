"""The LD read path leaves zarr's process-wide configuration as it finds it (ogquery#2, ADR 0002).

The GWAS read path sets zarr's codec pipeline to the fused one with one worker, writes empty
chunks and turns Blosc's threads on, when `ogquery.arrays` is imported (ADR 0056). ogld's readers
ran under zarr's defaults, and opengwasdb measured `proxies` 17-29 % slower cold under the fused
pipeline (its #460). So `import ogquery.ld` and an LD query, in a fresh interpreter, must leave
every one of those settings at zarr's default; the GWAS path must still set them.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TEST_STORE = Path(__file__).resolve().parent / "data" / "test-panel" / "store"

SETTINGS = """
import json, numcodecs, zarr
import zarr.codecs.blosc
print(json.dumps({
    "codec_pipeline.path": zarr.config.get("codec_pipeline.path"),
    "codec_pipeline.max_workers": zarr.config.get("codec_pipeline.max_workers"),
    "array.write_empty_chunks": zarr.config.get("array.write_empty_chunks"),
    "blosc.use_threads": numcodecs.blosc.use_threads,
}))
"""

LD_QUERIES = f"""
import numpy as np
import ogquery.ld
store = ogquery.ld.open({str(TEST_STORE)!r})
rows = store.variants_in("22", 1, 3_000_000_000).alid.tolist()
store.matrix(rows[:50])
store.region("22", 30_000_000, 1_000_000)
store.proxies(rows[:20], r2=0.5, kb=3000)
store.clump(rows[:200], np.arange(200.0), r2=0.1, kb=3000)
store.has(rows[:5])
"""


def _settings_after(statements: str) -> dict[str, object]:
    done = subprocess.run(
        [sys.executable, "-c", statements + SETTINGS], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    found: dict[str, object] = json.loads(done.stdout.strip().splitlines()[-1])
    return found


def test_the_ld_read_path_leaves_zarr_s_configuration_at_its_defaults() -> None:
    assert _settings_after(LD_QUERIES) == _settings_after("")


def test_the_gwas_read_path_still_sets_it() -> None:
    # The probe sees a change when one is made: the GWAS reader's configuration (ADR 0056).
    defaults, gwas = _settings_after(""), _settings_after("import ogquery.facade\n")
    assert gwas["codec_pipeline.path"] == "zarr.core.codec_pipeline.FusedCodecPipeline"
    assert gwas["codec_pipeline.max_workers"] == 1
    assert gwas["blosc.use_threads"] is True
    assert {key for key in gwas if gwas[key] != defaults[key]} >= {
        "codec_pipeline.path",
        "blosc.use_threads",
    }
