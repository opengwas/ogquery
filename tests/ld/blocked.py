"""A fresh interpreter in which some modules cannot be imported: how the extras' tests
(`test_http_extra.py`, `test_genotype_extra.py`) run without an extra, installed or not.

`sys.modules[name] = None` makes `import name` raise and `importlib.util.find_spec(name)` return
None, which is what `ogquery.opening`'s checks look for.
"""

from __future__ import annotations

import json
import subprocess
import sys


def without(blocked: tuple[str, ...], statements: str) -> dict[str, object]:
    """Run `statements` in a fresh interpreter in which `blocked` cannot be imported, and return
    the JSON it prints."""
    code = (
        "import json, sys\n"
        f"for name in {blocked!r}:\n"
        "    sys.modules[name] = None\n"
        f"{statements}\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    found: dict[str, object] = json.loads(done.stdout)
    return found
