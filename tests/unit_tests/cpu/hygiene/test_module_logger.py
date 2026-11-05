# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every module that calls `logger.*` must actually have a `logger`.

Guards a real defect: a `logger = logging.getLogger(__name__)` statement landed
inside a function's parameter list in two modules, becoming a keyword parameter
and leaving the module with no module-level `logger`. It parses, imports and
lints clean; every `logger.*` call then raises NameError when its branch runs.
A grep for `logging.getLogger` does not catch it -- the string is there -- so
the check has to be semantic.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[4] / "torchtitan"


def _modules() -> list[pathlib.Path]:
    return sorted(_ROOT.rglob("*.py"))


def _uses_logger_as_global(tree: ast.AST, bound: set[str]) -> bool:
    """True if `logger.x` is read somewhere it is not a local binding."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "logger"
            and "logger" not in bound
        ):
            return True
    return False


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.name))
def test_logger_is_not_swallowed_into_a_signature(path: pathlib.Path):
    """No function may take a parameter literally named `logger`... almost.

    `observability.logging.warn_once(logger, msg)` legitimately receives one,
    so the rule is scoped to functions that ALSO do not use it as a parameter
    deliberately -- expressed here as an explicit allowlist, because the failure
    mode being guarded is an accident, not a design.
    """
    allowed = {("logging.py", "warn_once")}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
        if "logger" in names and (path.name, node.name) not in allowed:
            pytest.fail(
                f"{path}:{node.lineno} `def {node.name}` takes a parameter named "
                "`logger`. If a `logger = logging.getLogger(__name__)` statement "
                "was meant to be at module scope, it has been absorbed into the "
                "parameter list -- valid Python, but the module then has no "
                "module-level `logger` and every `logger.*` call raises NameError."
            )


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.name))
def test_modules_calling_logger_define_one(path: pathlib.Path):
    """A module that reads `logger.*` as a global must bind one at module scope.

    `logger` counts as bound by a plain assign, an ANNOTATED assign
    (`logger: logging.Logger = ...`, which several rl modules use), or an
    import. A module that binds it only inside a function, or that receives it
    as a parameter, uses it locally and is not our concern.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bound: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            bound |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound |= {a.asname or a.name.split(".")[0] for a in node.names}
    if "logger" in bound:
        return

    # Otherwise: is every `logger.*` read covered by a local binding -- a
    # parameter, or an assignment inside the same function?
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        a = func.args
        local = {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
        for n in ast.walk(func):
            if isinstance(n, ast.Assign):
                local |= {t.id for t in n.targets if isinstance(t, ast.Name)}
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                local.add(n.target.id)
        if "logger" in local:
            return

    if _uses_logger_as_global(tree, bound):
        pytest.fail(
            f"{path} reads `logger.*` but binds no `logger` at module scope and "
            "none locally. Every one of those calls raises NameError when its "
            "branch runs."
        )
