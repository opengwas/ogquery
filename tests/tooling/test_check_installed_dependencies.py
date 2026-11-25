"""`scripts/check_installed_dependencies.py`, on the environment the tests run in.

CI runs the script itself in a clean environment the wheel was installed into; here the
environment also holds the test tools, so these check its rules, not that environment.
"""

from __future__ import annotations

from script_modules import load_script

check = load_script("check_installed_dependencies")


def test_ogquery_requires_exactly_its_three_dependencies():
    requirements = check._requirements(check.distribution("ogquery"), frozenset())
    direct = {check.canonical(r.name) for r in requirements}
    assert direct == check.DIRECT == {"numpy", "zarr", "numcodecs"}


def test_the_closure_follows_requirements_but_not_ogquerys_extras():
    reached = set(check.closure("ogquery"))
    assert check.DIRECT | {"ogquery"} <= reached
    assert "zarr" in reached and "packaging" in reached  # zarr's own
    # The `test` extra's tools are installed here, and are not ogquery's dependencies.
    assert not {"pytest", "scipy", "ruff", "mypy"} & reached


def test_a_source_build_in_the_report_is_refused():
    report = {
        "install": [
            {"download_info": {"url": "https://x/numpy-2.4.6-cp312-cp312-x.whl"},
             "metadata": {"name": "numpy"}},
            {"download_info": {"url": "https://x/numcodecs-0.17.1.tar.gz"},
             "metadata": {"name": "numcodecs"}},
        ]
    }
    found = [p for p in check.problems(report) if "wheel" in p]
    assert found == ["numcodecs was not installed from a wheel: https://x/numcodecs-0.17.1.tar.gz"]


def test_names_are_compared_normalised():
    assert check.canonical("Typing_Extensions") == check.canonical("typing.extensions")
