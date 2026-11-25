"""The public names of a Store Reader checkout, as the committed list in
tests/unit/test_store_reader_public_names.py (ogquery#4).

A name is public when a module's `__all__` lists it or a top-level definition or
assignment (tuple unpacking included) does not start with an underscore. Run it on an
unpacked `git archive` of the opengwasdb commit, never on a live checkout:

    git -C <opengwasdb> archive c8ebe4ca2e7697f3b723fd89ebb645bb1d221fd4 \\
        opengwasdb/store_reader opengwasdb/store/query_digests.py | tar -x -C <dir>
    python scripts/public_names.py <dir>

It prints the `PUBLIC_NAMES` literal. Standard library only.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

DIGESTS = "opengwasdb/store/query_digests.py"
READER = "opengwasdb/store_reader"


def _targets(node: ast.expr) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [name for element in node.elts for name in _targets(element)]
    return []


def public_names(source: str) -> list[str]:
    """The public names `source` defines or exports, sorted."""
    listed: set[str] = set()
    defined: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [name for target in targets for name in _targets(target)]
            if "__all__" in names and node.value is not None:
                elements = getattr(node.value, "elts", [])
                constants = [e.value for e in elements if isinstance(e, ast.Constant)]
                listed.update(c for c in constants if isinstance(c, str))
            defined.update(names)
    return sorted(listed | {name for name in defined if not name.startswith("_")})


def module_name(root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    if relative == DIGESTS:
        return "ogquery.digests"
    dotted = relative.removeprefix(f"{READER}/").removesuffix(".py").replace("/", ".")
    dotted = dotted.removesuffix(".__init__")
    return "ogquery" if dotted == "__init__" else f"ogquery.{dotted}"


def collect(root: Path) -> dict[str, list[str]]:
    paths = sorted((root / READER).rglob("*.py")) + [root / DIGESTS]
    return {module_name(root, p): public_names(p.read_text(encoding="utf-8")) for p in paths}


def render(names: dict[str, list[str]]) -> str:
    lines = ["PUBLIC_NAMES: dict[str, tuple[str, ...]] = {"]
    for module, public in names.items():
        lines.append(f'    "{module}": (')
        lines.extend(f'        "{name}",' for name in public)
        lines.append("    ),")
    lines.append("}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    names = collect(Path(argv[0]))
    print(f"{len(names)} modules, {sum(map(len, names.values()))} names", file=sys.stderr)
    print(render(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
