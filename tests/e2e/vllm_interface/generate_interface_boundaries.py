# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate exact vLLM interface relations from the current source pair.

The collector is intentionally consumer-first: it records patch and inheritance
intent from vllm-ascend before resolving the target in vLLM. A missing upstream
target is therefore kept as an explicit risk finding instead of silently
disappearing.

The first implementation covers:

* explicit monkey patches (assignment and literal-name ``setattr``);
* direct inheritance from an imported vLLM class;
* verified overrides whose effective owner is found in the combined MRO.

It targets vLLM main: an exact ``vllm_version_is("<tag>")`` branch is treated
as release-only, and the opposite branch is scanned. An incomplete MRO is
reported instead of being guessed.

It does not import either package and does not require an NPU.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 4
GENERATOR_VERSION = "0.19.0"
SUPPORTED_RELATIONS = frozenset({"inheritance", "monkey_patch", "override"})
FINDING_STATUSES = frozenset({"expected", "excluded", "review", "risk", "verified"})
STDLIB_STRUCTURAL_BASES: dict[str, tuple[str, ...]] = {
    "abc.ABC": (),
    "typing.Generic": (),
    "typing.Protocol": ("typing.Generic",),
}


def _jsonable_signature(node: ast.AST | None) -> list[object] | None:
    if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef, ast.Lambda)):
        return None

    arguments = node.args
    positional = [*arguments.posonlyargs, *arguments.args]
    required_count = len(positional) - len(arguments.defaults)
    return [
        "async" if isinstance(node, ast.AsyncFunctionDef) else "sync",
        [[argument.arg, index < required_count] for index, argument in enumerate(arguments.posonlyargs)],
        [
            [
                argument.arg,
                index + len(arguments.posonlyargs) < required_count,
            ]
            for index, argument in enumerate(arguments.args)
        ],
        arguments.vararg.arg if arguments.vararg else None,
        [
            [argument.arg, default is None]
            for argument, default in zip(
                arguments.kwonlyargs,
                arguments.kw_defaults,
            )
        ],
        arguments.kwarg.arg if arguments.kwarg else None,
    ]


def _expression_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expression_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Subscript):
        return _expression_name(node.value)
    return None


def _module_name(package_name: str, package_root: Path, path: Path) -> tuple[str, bool]:
    relative = path.relative_to(package_root)
    parts = list(relative.with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    suffix = ".".join(parts)
    return (f"{package_name}.{suffix}" if suffix else package_name), is_package


def _relative_import_module(
    current_module: str,
    is_package: bool,
    level: int,
    imported_module: str | None,
) -> str:
    if level == 0:
        return imported_module or ""

    package_parts = current_module.split(".") if is_package else current_module.split(".")[:-1]
    keep = len(package_parts) - (level - 1)
    if keep < 0:
        return imported_module or ""
    result = package_parts[:keep]
    if imported_module:
        result.extend(imported_module.split("."))
    return ".".join(result)


def _method_nodes(node: ast.ClassDef) -> dict[str, ast.AST]:
    return {child.name: child for child in node.body if isinstance(child, (ast.AsyncFunctionDef, ast.FunctionDef))}


@dataclass(frozen=True, order=True)
class _ScopeBinding:
    """One possible final runtime binding for a module/class namespace name."""

    kind: str
    line: int
    column: int
    end_line: int
    end_column: int
    node: ast.AST | None = field(default=None, compare=False, hash=False, repr=False)


_UNBOUND_SCOPE_BINDING = _ScopeBinding("unbound", -1, -1, -1, -1)


def _scope_binding(kind: str, node: ast.AST) -> _ScopeBinding:
    return _ScopeBinding(
        kind=kind,
        line=getattr(node, "lineno", 0),
        column=getattr(node, "col_offset", 0),
        end_line=getattr(node, "end_lineno", getattr(node, "lineno", 0)),
        end_column=getattr(node, "end_col_offset", getattr(node, "col_offset", 0)),
        node=node,
    )


def _merge_scope_binding_states(
    states: Sequence[dict[str, tuple[_ScopeBinding, ...]]],
) -> dict[str, tuple[_ScopeBinding, ...]] | None:
    live_states = [state for state in states if state is not None]
    if not live_states:
        return None
    names = {name for state in live_states for name in state}
    merged: dict[str, tuple[_ScopeBinding, ...]] = {}
    for name in names:
        alternatives = {
            alternative for state in live_states for alternative in state.get(name, (_UNBOUND_SCOPE_BINDING,))
        }
        merged[name] = tuple(sorted(alternatives))
    return merged


def _bind_scope_names(
    state: dict[str, tuple[_ScopeBinding, ...]],
    names: Iterable[str],
    binding: _ScopeBinding,
) -> None:
    for name in names:
        state[name] = (binding,)


def _scope_final_binding_state(
    statements: Sequence[ast.stmt],
    tag_guard_names: set[str],
    incoming: dict[str, tuple[_ScopeBinding, ...]] | None = None,
    *,
    loop_body: bool = False,
) -> dict[str, tuple[_ScopeBinding, ...]] | None:
    """Interpret namespace writes and retain only normally completing paths."""

    state = {name: tuple(values) for name, values in (incoming or {}).items()}
    for node in statements:
        if isinstance(node, ast.If):
            condition = _main_condition_value(node.test, tag_guard_names)
            if condition is True:
                selected = _scope_final_binding_state(
                    node.body,
                    tag_guard_names,
                    state,
                    loop_body=loop_body,
                )
            elif condition is False:
                selected = _scope_final_binding_state(
                    node.orelse,
                    tag_guard_names,
                    state,
                    loop_body=loop_body,
                )
            else:
                selected = _merge_scope_binding_states(
                    [
                        _scope_final_binding_state(
                            node.body,
                            tag_guard_names,
                            state,
                            loop_body=loop_body,
                        ),
                        _scope_final_binding_state(
                            node.orelse,
                            tag_guard_names,
                            state,
                            loop_body=loop_body,
                        ),
                    ]
                )
            if selected is None:
                return None
            state = selected
            continue

        if isinstance(node, (ast.Try, ast.TryStar)):
            normal = _scope_final_binding_state(
                node.body,
                tag_guard_names,
                state,
                loop_body=loop_body,
            )
            if normal is not None:
                normal = _scope_final_binding_state(
                    node.orelse,
                    tag_guard_names,
                    normal,
                    loop_body=loop_body,
                )
            paths: list[dict[str, tuple[_ScopeBinding, ...]] | None] = [normal]
            for handler in node.handlers:
                handler_state = {name: tuple(values) for name, values in state.items()}
                if handler.name:
                    handler_state[handler.name] = (_scope_binding("value", handler),)
                handled = _scope_final_binding_state(
                    handler.body,
                    tag_guard_names,
                    handler_state,
                    loop_body=loop_body,
                )
                if handled is not None and handler.name:
                    handled[handler.name] = (_UNBOUND_SCOPE_BINDING,)
                paths.append(handled)
            live_paths = [path for path in paths if path is not None]
            if node.finalbody:
                live_paths = [
                    final
                    for path in live_paths
                    if (
                        final := _scope_final_binding_state(
                            node.finalbody,
                            tag_guard_names,
                            path,
                            loop_body=loop_body,
                        )
                    )
                    is not None
                ]
            selected = _merge_scope_binding_states(live_paths)
            if selected is None:
                return None
            state = selected
            continue

        if isinstance(node, (ast.With, ast.AsyncWith)):
            child = {name: tuple(values) for name, values in state.items()}
            for item in node.items:
                if item.optional_vars is not None:
                    _bind_scope_names(
                        child,
                        _bound_target_names(item.optional_vars),
                        _scope_binding("value", item.optional_vars),
                    )
            selected = _scope_final_binding_state(
                node.body,
                tag_guard_names,
                child,
                loop_body=loop_body,
            )
            if selected is None:
                return None
            state = selected
            continue

        if isinstance(node, (ast.AsyncFor, ast.For, ast.While)):
            body_state = {name: tuple(values) for name, values in state.items()}
            if isinstance(node, (ast.AsyncFor, ast.For)):
                _bind_scope_names(
                    body_state,
                    _bound_target_names(node.target),
                    _scope_binding("value", node.target),
                )
            body_result = _scope_final_binding_state(
                node.body,
                tag_guard_names,
                body_state,
                loop_body=True,
            )
            loop_result = _merge_scope_binding_states([state, body_result])
            if loop_result is None:
                return None
            if node.orelse:
                else_result = _scope_final_binding_state(
                    node.orelse,
                    tag_guard_names,
                    loop_result,
                    loop_body=loop_body,
                )
                loop_result = _merge_scope_binding_states([loop_result, else_result])
                if loop_result is None:
                    return None
            state = loop_result
            continue

        if isinstance(node, ast.Match):
            paths = [state]
            paths.extend(
                _scope_final_binding_state(
                    case.body,
                    tag_guard_names,
                    state,
                    loop_body=loop_body,
                )
                for case in node.cases
            )
            selected = _merge_scope_binding_states(paths)
            if selected is None:
                return None
            state = selected
            continue

        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            state[node.name] = (_scope_binding("function", node),)
            continue
        if isinstance(node, ast.ClassDef):
            state[node.name] = (_scope_binding("class", node),)
            continue
        if isinstance(node, ast.Import):
            _bind_scope_names(
                state,
                (alias.asname or alias.name.split(".", 1)[0] for alias in node.names),
                _scope_binding("value", node),
            )
            continue
        if isinstance(node, ast.ImportFrom):
            _bind_scope_names(
                state,
                (alias.asname or alias.name for alias in node.names if alias.name != "*"),
                _scope_binding("value", node),
            )
            continue
        if isinstance(node, ast.Assign):
            source_binding = state.get(node.value.id) if isinstance(node.value, ast.Name) else None
            for target in node.targets:
                names = _bound_target_names(target)
                if len(names) == 1 and isinstance(target, ast.Name) and source_binding is not None:
                    state[target.id] = tuple(source_binding)
                else:
                    _bind_scope_names(
                        state,
                        names,
                        _scope_binding("value", node),
                    )
            continue
        if isinstance(node, ast.AnnAssign):
            if node.value is not None:
                _bind_scope_names(
                    state,
                    _bound_target_names(node.target),
                    _scope_binding("value", node),
                )
            continue
        if isinstance(node, (ast.AugAssign, ast.NamedExpr)):
            target = node.target
            _bind_scope_names(
                state,
                _bound_target_names(target),
                _scope_binding("value", node),
            )
            continue
        if isinstance(node, ast.Delete):
            _bind_scope_names(
                state,
                (
                    child.id
                    for target in node.targets
                    for child in ast.walk(target)
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Del)
                ),
                _UNBOUND_SCOPE_BINDING,
            )
            continue
        if isinstance(node, (ast.Break, ast.Continue)):
            return state if loop_body else None
        if isinstance(node, (ast.Raise, ast.Return)):
            return None
    return state


def _scope_final_bindings(
    statements: Sequence[ast.stmt],
    tag_guard_names: set[str],
) -> dict[str, tuple[_ScopeBinding, ...]]:
    return _scope_final_binding_state(statements, tag_guard_names) or {}


def _possible_method_nodes(
    node: ast.ClassDef,
    tag_guard_names: set[str],
) -> dict[str, ast.AST]:
    """Index methods that can exist on the active main-version path."""
    return {name: candidates[0] for name, candidates in _possible_method_variants(node, tag_guard_names).items()}


def _possible_method_variants(
    node: ast.ClassDef,
    tag_guard_names: set[str],
) -> dict[str, tuple[ast.AST, ...]]:
    bindings = _scope_final_bindings(node.body, tag_guard_names)
    return {
        name: tuple(
            candidate.node for candidate in candidates if candidate.kind == "function" and candidate.node is not None
        )
        for name, candidates in bindings.items()
        if any(candidate.kind == "function" for candidate in candidates)
    }


def _function_scope_nodes(
    node: ast.AsyncFunctionDef | ast.FunctionDef,
) -> Iterable[ast.AST]:
    """Walk one function scope without entering nested scopes."""
    stack: list[ast.AST] = list(reversed(node.body))
    while stack:
        current = stack.pop()
        yield current
        if isinstance(
            current,
            (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef, ast.Lambda),
        ):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(current))))


def _function_local_names(
    node: ast.AsyncFunctionDef | ast.FunctionDef,
) -> set[str]:
    """Return names compiled as locals in exactly one function scope."""

    class LocalCollector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.names: set[str] = set()
            self.globals: set[str] = set()
            self.nonlocals: set[str] = set()

        def visit_Name(self, child: ast.Name) -> None:  # noqa: N802
            if isinstance(child.ctx, (ast.Del, ast.Store)):
                self.names.add(child.id)

        def visit_Global(self, child: ast.Global) -> None:  # noqa: N802
            self.globals.update(child.names)

        def visit_Nonlocal(self, child: ast.Nonlocal) -> None:  # noqa: N802
            self.nonlocals.update(child.names)

        def visit_Import(self, child: ast.Import) -> None:  # noqa: N802
            self.names.update(alias.asname or alias.name.split(".", 1)[0] for alias in child.names)

        def visit_ImportFrom(self, child: ast.ImportFrom) -> None:  # noqa: N802
            self.names.update(alias.asname or alias.name for alias in child.names if alias.name != "*")

        def visit_FunctionDef(self, child: ast.FunctionDef) -> None:  # noqa: N802
            self.names.add(child.name)

        def visit_AsyncFunctionDef(self, child: ast.AsyncFunctionDef) -> None:  # noqa: N802
            self.names.add(child.name)

        def visit_ClassDef(self, child: ast.ClassDef) -> None:  # noqa: N802
            self.names.add(child.name)

        def visit_Lambda(self, child: ast.Lambda) -> None:  # noqa: N802
            return

        def visit_ExceptHandler(self, child: ast.ExceptHandler) -> None:  # noqa: N802
            if child.type is not None:
                self.visit(child.type)
            if child.name:
                self.names.add(child.name)
            for statement in child.body:
                self.visit(statement)

        def _visit_comprehension_scope(
            self,
            generators: Sequence[ast.comprehension],
            values: Sequence[ast.AST],
        ) -> None:
            # Comprehension iteration targets belong to the implicit nested
            # scope.  Their iterable/filter expressions and assignment
            # expressions still execute in the surrounding function.
            for generator in generators:
                self.visit(generator.iter)
                for condition in generator.ifs:
                    self.visit(condition)
            for value in values:
                self.visit(value)

        def visit_ListComp(self, child: ast.ListComp) -> None:  # noqa: N802
            self._visit_comprehension_scope(child.generators, (child.elt,))

        def visit_SetComp(self, child: ast.SetComp) -> None:  # noqa: N802
            self._visit_comprehension_scope(child.generators, (child.elt,))

        def visit_GeneratorExp(self, child: ast.GeneratorExp) -> None:  # noqa: N802
            self._visit_comprehension_scope(child.generators, (child.elt,))

        def visit_DictComp(self, child: ast.DictComp) -> None:  # noqa: N802
            self._visit_comprehension_scope(child.generators, (child.key, child.value))

        def visit_MatchAs(self, child: ast.MatchAs) -> None:  # noqa: N802
            if child.name:
                self.names.add(child.name)
            if child.pattern is not None:
                self.visit(child.pattern)

        def visit_MatchStar(self, child: ast.MatchStar) -> None:  # noqa: N802
            if child.name:
                self.names.add(child.name)

        def visit_MatchMapping(self, child: ast.MatchMapping) -> None:  # noqa: N802
            if child.rest:
                self.names.add(child.rest)
            self.generic_visit(child)

    collector = LocalCollector()
    collector.names.update(
        argument.arg
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
    )
    if node.args.vararg is not None:
        collector.names.add(node.args.vararg.arg)
    if node.args.kwarg is not None:
        collector.names.add(node.args.kwarg.arg)
    for statement in node.body:
        collector.visit(statement)
    return collector.names - collector.globals - collector.nonlocals


def _statements_must_terminate(statements: Sequence[ast.stmt]) -> bool:
    return any(_statement_must_terminate(statement) for statement in statements)


def _statement_must_terminate(node: ast.stmt) -> bool:
    if isinstance(node, (ast.Break, ast.Continue, ast.Raise, ast.Return)):
        return True
    if isinstance(node, ast.If):
        return bool(node.orelse) and _statements_must_terminate(node.body) and _statements_must_terminate(node.orelse)
    if isinstance(node, ast.Try):
        if _statements_must_terminate(node.finalbody):
            return True
        success = (*node.body, *node.orelse)
        return (
            bool(node.handlers)
            and _statements_must_terminate(success)
            and all(_statements_must_terminate(handler.body) for handler in node.handlers)
        )
    return False


def _none_comparison(
    node: ast.AST,
) -> tuple[ast.AST, bool] | None:
    """Return the compared expression and whether the test means non-None."""

    if not (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], (ast.Is, ast.IsNot))
        and len(node.comparators) == 1
    ):
        return None
    left = node.left
    right = node.comparators[0]
    if isinstance(left, ast.Constant) and left.value is None:
        subject = right
    elif isinstance(right, ast.Constant) and right.value is None:
        subject = left
    else:
        return None
    return subject, isinstance(node.ops[0], ast.IsNot)


def _canonical_guard(
    node: ast.AST,
    *,
    truth: bool = True,
) -> tuple[str, bool, str]:
    """Normalize one predicate without relying on its rendered spelling."""

    while isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        truth = not truth
        node = node.operand

    none_check = _none_comparison(node)
    if none_check is not None:
        subject, test_means_non_none = none_check
        means_non_none = test_means_non_none if truth else not test_means_non_none
        subject_text = " ".join(ast.unparse(subject).split())
        key = f"none:{ast.dump(subject, include_attributes=False)}"
        text = f"{subject_text} is not None" if means_non_none else f"{subject_text} is None"
        return key, means_non_none, text

    expression = " ".join(ast.unparse(node).split())
    key = f"expr:{ast.dump(node, include_attributes=False)}"
    text = expression if truth else f"not ({expression})"
    return key, truth, text


def _canonical_guard_text(text: str) -> tuple[str, bool, str]:
    """Canonicalize a stored guard; keep synthetic flow labels opaque."""

    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        return f"opaque:{text}", True, text
    return _canonical_guard(node)


def _call_guard_polarity(text: str, call_name: str) -> bool | None:
    """Return whether a stored guard requires a direct call to be truthy."""

    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        return None
    truth = True
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        truth = not truth
        node = node.operand
    if isinstance(node, ast.Call) and _expression_name(node.func) == call_name:
        return truth
    return None


def _main_expression_calls(
    node: ast.AST | None,
    tag_guard_names: set[str],
) -> Iterable[ast.Call]:
    """Walk calls that may be evaluated on the main-version path.

    Function and lambda bodies are deferred scopes. Boolean operands and
    conditional-expression arms may be skipped at the current call site, so
    apply an exact release-tag result before attributing a helper call to main.
    """
    if node is None:
        return

    def walk(current: ast.AST) -> Iterable[ast.Call]:
        if isinstance(
            current,
            (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef, ast.Lambda),
        ):
            return
        if isinstance(current, ast.BoolOp):
            for value in current.values:
                yield from walk(value)
                condition = _main_condition_value(value, tag_guard_names)
                if isinstance(current.op, ast.And) and condition is False:
                    break
                if isinstance(current.op, ast.Or) and condition is True:
                    break
            return
        if isinstance(current, ast.IfExp):
            yield from walk(current.test)
            condition = _main_condition_value(current.test, tag_guard_names)
            if condition is True:
                yield from walk(current.body)
            elif condition is False:
                yield from walk(current.orelse)
            else:
                yield from walk(current.body)
                yield from walk(current.orelse)
            return
        if isinstance(current, ast.Call):
            yield current
        for child in ast.iter_child_nodes(current):
            yield from walk(child)

    yield from walk(node)


def _lazy_getattr_names(node: ast.AST) -> set[str]:
    if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
        return set()
    parameters = [*node.args.posonlyargs, *node.args.args]
    if not parameters:
        return set()
    parameter = parameters[0].arg
    names: set[str] = set()
    for child in _function_scope_nodes(node):
        if not isinstance(child, ast.If):
            continue
        test = child.test
        if not (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
        ):
            continue
        left, right = test.left, test.comparators[0]
        candidates = ((left, right), (right, left))
        for name_node, value_node in candidates:
            if (
                isinstance(name_node, ast.Name)
                and name_node.id == parameter
                and isinstance(value_node, ast.Constant)
                and isinstance(value_node.value, str)
                and any(isinstance(item, ast.Return) for item in child.body)
            ):
                names.add(value_node.value)
    return names


def _is_exact_tag_check(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and _expression_name(node.func) == "vllm_version_is"
        and bool(node.args)
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    )


def _tag_guard_names(statements: Sequence[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for node in statements:
        if isinstance(node, ast.Assign) and _is_exact_tag_check(node.value):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and _is_exact_tag_check(node.value):
            names.add(node.target.id)
    return names


def _main_condition_value(
    node: ast.AST,
    tag_guard_names: set[str],
) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    if _is_exact_tag_check(node):
        return False
    if isinstance(node, ast.Name) and node.id in tag_guard_names:
        return False
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _main_condition_value(node.operand, tag_guard_names)
        return None if value is None else not value
    if isinstance(node, ast.BoolOp):
        values = [_main_condition_value(value, tag_guard_names) for value in node.values]
        if isinstance(node.op, ast.And):
            if False in values:
                return False
            return True if all(value is True for value in values) else None
        if isinstance(node.op, ast.Or):
            if True in values:
                return True
            return False if all(value is False for value in values) else None
    return None


def _main_module_statements(
    statements: Sequence[ast.stmt],
    tag_guard_names: set[str],
) -> Iterable[ast.stmt]:
    for node in statements:
        if isinstance(node, ast.If):
            condition = _main_condition_value(
                node.test,
                tag_guard_names,
            )
            if condition is True:
                selected = node.body
                yield from _main_module_statements(
                    selected,
                    tag_guard_names,
                )
            elif condition is False:
                selected = node.orelse
                yield from _main_module_statements(
                    selected,
                    tag_guard_names,
                )
            else:
                selected = None
                yield from _main_module_statements(
                    node.body,
                    tag_guard_names,
                )
                yield from _main_module_statements(
                    node.orelse,
                    tag_guard_names,
                )
            if (selected is not None and _statements_must_terminate(selected)) or (
                selected is None and _statement_must_terminate(node)
            ):
                return
            continue
        if isinstance(node, ast.Try):
            yield from _main_module_statements(
                node.body,
                tag_guard_names,
            )
            for handler in node.handlers:
                yield from _main_module_statements(
                    handler.body,
                    tag_guard_names,
                )
            yield from _main_module_statements(
                node.orelse,
                tag_guard_names,
            )
            yield from _main_module_statements(
                node.finalbody,
                tag_guard_names,
            )
            continue
        yield node
        if isinstance(node, (ast.Break, ast.Continue, ast.Raise, ast.Return)):
            return


def _bound_target_names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)}


def _direct_bound_names(node: ast.stmt) -> set[str]:
    """Names bound in the current scope by one non-compound statement."""
    if isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {name for target in node.targets for name in _bound_target_names(target)}
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        return _bound_target_names(node.target)
    if isinstance(node, ast.Import):
        return {alias.asname or alias.name.split(".", 1)[0] for alias in node.names}
    if isinstance(node, ast.ImportFrom):
        return {alias.asname or alias.name for alias in node.names if alias.name != "*"}
    return set()


def _scope_must_bound_state(
    statements: Sequence[ast.stmt],
    tag_guard_names: set[str],
    incoming: set[str] | None = None,
) -> set[str] | None:
    """Return MUST names, or ``None`` when no path completes normally."""
    state = set(incoming or ())
    for node in statements:
        if isinstance(node, ast.If):
            condition = _main_condition_value(node.test, tag_guard_names)
            if condition is True:
                selected = _scope_must_bound_state(node.body, tag_guard_names, state)
            elif condition is False:
                selected = _scope_must_bound_state(node.orelse, tag_guard_names, state)
            else:
                alternatives = [
                    branch
                    for branch in (
                        _scope_must_bound_state(node.body, tag_guard_names, state),
                        _scope_must_bound_state(node.orelse, tag_guard_names, state),
                    )
                    if branch is not None
                ]
                selected = set.intersection(*alternatives) if alternatives else None
            if selected is None:
                return None
            state = selected
            continue
        if isinstance(node, ast.Try):
            normal = _scope_must_bound_state(node.body, tag_guard_names, state)
            if normal is not None:
                normal = _scope_must_bound_state(node.orelse, tag_guard_names, normal)
            paths = [normal] if normal is not None else []
            paths.extend(
                handler_state
                for handler in node.handlers
                if (
                    handler_state := _scope_must_bound_state(
                        handler.body,
                        tag_guard_names,
                        state,
                    )
                )
                is not None
            )
            if node.finalbody:
                paths = [
                    final_state
                    for path in paths
                    if (
                        final_state := _scope_must_bound_state(
                            node.finalbody,
                            tag_guard_names,
                            path,
                        )
                    )
                    is not None
                ]
            if not paths:
                return None
            state = set.intersection(*paths)
            continue
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    state.update(_bound_target_names(item.optional_vars))
            selected = _scope_must_bound_state(node.body, tag_guard_names, state)
            if selected is None:
                return None
            state = selected
            continue
        if isinstance(node, ast.Delete):
            state.difference_update(
                child.id for target in node.targets for child in ast.walk(target) if isinstance(child, ast.Name)
            )
            continue
        # Loop bodies may execute zero times, so they cannot add a MUST name.
        if isinstance(node, (ast.AsyncFor, ast.For, ast.While)):
            continue
        if isinstance(node, (ast.Break, ast.Continue, ast.Raise, ast.Return)):
            return None
        state.update(_direct_bound_names(node))
    return state


def _scope_must_bound_names(
    statements: Sequence[ast.stmt],
    tag_guard_names: set[str],
    incoming: set[str] | None = None,
) -> set[str]:
    """Return names present after every normally completing active-main path."""
    return _scope_must_bound_state(statements, tag_guard_names, incoming) or set()


def _main_module_statement_records(
    statements: Sequence[ast.stmt],
    tag_guard_names: set[str],
    *,
    unconditional: bool = True,
) -> Iterable[tuple[ast.stmt, bool]]:
    """Yield active-main statements together with runtime availability.

    Unknown branches remain indexed because they may contain a real interface,
    but a definition in such a branch must not prove ``hasattr`` true.  This is
    intentionally more conservative than ``_main_module_statements``, whose
    flattened output is still used by the general interface collector.
    """

    for node in statements:
        if isinstance(node, ast.If):
            condition = _main_condition_value(node.test, tag_guard_names)
            if condition is True:
                yield from _main_module_statement_records(
                    node.body,
                    tag_guard_names,
                    unconditional=unconditional,
                )
            elif condition is False:
                yield from _main_module_statement_records(
                    node.orelse,
                    tag_guard_names,
                    unconditional=unconditional,
                )
            else:
                yield from _main_module_statement_records(
                    node.body,
                    tag_guard_names,
                    unconditional=False,
                )
                yield from _main_module_statement_records(
                    node.orelse,
                    tag_guard_names,
                    unconditional=False,
                )
            continue
        if isinstance(node, ast.Try):
            # Imports and definitions in a try/except arm are path-dependent.
            yield from _main_module_statement_records(
                node.body,
                tag_guard_names,
                unconditional=False,
            )
            for handler in node.handlers:
                yield from _main_module_statement_records(
                    handler.body,
                    tag_guard_names,
                    unconditional=False,
                )
            yield from _main_module_statement_records(
                node.orelse,
                tag_guard_names,
                unconditional=False,
            )
            yield from _main_module_statement_records(
                node.finalbody,
                tag_guard_names,
                unconditional=False,
            )
            continue
        yield node, unconditional


def _main_ast_walk(tree: ast.AST) -> Iterable[ast.AST]:
    statements = tree.body if isinstance(tree, ast.Module) else ()
    tag_guard_names = _tag_guard_names(statements)

    def walk(node: ast.AST) -> Iterable[ast.AST]:
        yield node
        if isinstance(node, ast.If):
            condition = _main_condition_value(
                node.test,
                tag_guard_names,
            )
            branches: Sequence[ast.stmt]
            if condition is True:
                branches = node.body
            elif condition is False:
                branches = node.orelse
            else:
                branches = (*node.body, *node.orelse)
            for child in branches:
                yield from walk(child)
            return
        for child in ast.iter_child_nodes(node):
            yield from walk(child)

    yield from walk(tree)


def _string_assignment(
    node: ast.AST,
) -> tuple[str, str] | None:
    target: ast.AST | None = None
    value: ast.AST | None = None
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        value = node.value
    elif isinstance(node, ast.AnnAssign):
        target = node.target
        value = node.value
    if isinstance(target, ast.Name) and isinstance(value, ast.Constant) and isinstance(value.value, str):
        return target.id, value.value
    return None


def _resolve_bound_reference(
    module: str,
    expression: str,
    imports: dict[str, str],
    local_names: set[str],
) -> str:
    parts = expression.split(".")
    if parts[0] in imports:
        return ".".join([imports[parts[0]], *parts[1:]])
    if parts[0] in local_names:
        return f"{module}.{expression}"
    if expression.startswith(("vllm.", "vllm_ascend.")):
        return expression
    return f"{module}.{expression}"


@dataclass(frozen=True)
class ClassInfo:
    qualified_name: str
    module: str
    file: str
    name: str
    bases: tuple[str, ...]
    resolved_bases: tuple[str, ...]
    methods: dict[str, ast.AST] = field(compare=False, hash=False, repr=False)
    method_variants: dict[str, tuple[ast.AST, ...]] = field(
        default_factory=dict,
        compare=False,
        hash=False,
        repr=False,
    )


@dataclass(frozen=True)
class CallableInfo:
    qualified_name: str
    module: str
    file: str
    owner: str | None
    name: str
    node: ast.AST | None = field(compare=False, hash=False, repr=False)
    binding_line: int | None = None
    origin_kind: str = "definition"
    signature_override: list[object] | None = field(
        default=None,
        compare=False,
        hash=False,
        repr=False,
    )

    @property
    def signature(self) -> list[object] | None:
        if self.signature_override is not None:
            return self.signature_override
        return _jsonable_signature(self.node)


@dataclass(frozen=True)
class ValueInfo:
    qualified_name: str
    module: str
    file: str
    owner: str | None
    name: str
    node: ast.AST | None = field(compare=False, hash=False, repr=False)


@dataclass
class ModuleInfo:
    name: str
    file: str
    is_package: bool
    tree: ast.Module
    imports: dict[str, str]
    classes: dict[str, ClassInfo]
    functions: dict[str, CallableInfo]
    loose_functions: dict[str, list[CallableInfo]]
    string_constants: dict[str, tuple[str, ...]]
    star_imports: tuple[str, ...]
    typed_lazy_exports: dict[str, str]


@dataclass(frozen=True)
class MroResult:
    owners: tuple[str, ...]
    complete: bool
    reason: str | None = None


@dataclass(frozen=True)
class RelationEvidence:
    file: str
    line: int
    scope: str | None = None
    guards: tuple[str, ...] = ()
    patch_kind: str | None = None
    definition_line: int | None = None
    binding_line: int | None = None
    target_expression: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "file": self.file,
            "line": self.line,
        }
        if self.scope:
            payload["scope"] = self.scope
        if self.guards:
            payload["guards"] = list(self.guards)
        if self.patch_kind:
            payload["patch_kind"] = self.patch_kind
        if self.definition_line is not None:
            payload["definition_line"] = self.definition_line
        if self.binding_line is not None:
            payload["binding_line"] = self.binding_line
        if self.target_expression is not None:
            payload["target_expression"] = self.target_expression
        return payload


@dataclass(frozen=True)
class Relation:
    relation: str
    upstream_file: str
    upstream_owner: str | None
    upstream_name: str
    upstream_signature: list[object] | None = field(compare=False, hash=False)
    downstream_file: str
    downstream_owner: str | None
    downstream_name: str
    downstream_signature: list[object] | None = field(compare=False, hash=False)
    evidence_file: str
    evidence_line: int
    evidence: tuple[RelationEvidence, ...] = field(
        default=(),
        compare=False,
        hash=False,
    )
    upstream_package: str = "vllm"

    def upstream_key(self) -> tuple[str, str, str, str]:
        return (
            self.upstream_package,
            self.upstream_file,
            self.upstream_owner or "",
            self.upstream_name,
        )

    def downstream_key(self) -> tuple[str, str, str, str]:
        return (
            self.relation,
            self.downstream_file,
            self.downstream_owner or "",
            self.downstream_name,
        )

    def exact_key(self) -> tuple[str, ...]:
        return (*self.downstream_key(), *self.upstream_key())

    def comparison_downstream_keys(
        self,
    ) -> tuple[tuple[str, str, str, str], ...]:
        keys = {self.downstream_key()}
        if self.relation == "monkey_patch":
            keys.update(
                (
                    self.relation,
                    evidence.file,
                    self.downstream_owner or "",
                    self.downstream_name,
                )
                for evidence in self.evidence
            )
        return tuple(sorted(keys))

    def comparison_exact_keys(self) -> tuple[tuple[str, ...], ...]:
        return tuple((*downstream_key, *self.upstream_key()) for downstream_key in self.comparison_downstream_keys())


@dataclass(frozen=True)
class CandidateFinding:
    relation: str
    downstream_file: str
    downstream_owner: str | None
    downstream_name: str
    target_expression: str
    evidence_line: int
    reason: str
    status: str = "review"
    reason_code: str = "analysis_gap"
    generator_issue: bool = True
    evidence_scope: str | None = None
    evidence_guards: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        if self.status not in FINDING_STATUSES:
            raise ValueError(f"unsupported finding status: {self.status}")
        return {
            "relation": self.relation,
            "downstream": {
                "file": self.downstream_file,
                "owner": self.downstream_owner,
                "name": self.downstream_name,
            },
            "target_expression": self.target_expression,
            "evidence": {
                "file": self.downstream_file,
                "line": self.evidence_line,
                **({"scope": self.evidence_scope} if self.evidence_scope else {}),
                **({"guards": list(self.evidence_guards)} if self.evidence_guards else {}),
            },
            "status": self.status,
            "reason_code": self.reason_code,
            "generator_issue": self.generator_issue,
            "reason": self.reason,
        }


# Kept as a source-compatible alias for callers of the v0.3 POC.
UnresolvedRelation = CandidateFinding


@dataclass(frozen=True, order=True)
class GuardFact:
    """One normalized predicate in one lexical scope activation."""

    scope: str
    activation: str
    key: str
    polarity: bool
    text: str = field(compare=False)
    hasattr_target: tuple[str, str] | None = field(default=None, compare=False)


@dataclass
class PatchScanContext:
    bindings: dict[str, set[str]] = field(default_factory=dict)
    binding_alternatives: dict[str, set[str | None]] = field(default_factory=dict)
    unknown_bindings: set[str] = field(default_factory=set)
    upstream_binding_provenance: dict[str, set[str]] = field(default_factory=dict)
    upstream_binding_history: set[str] = field(default_factory=set)
    strings: dict[str, set[str]] = field(default_factory=dict)
    local_callables: dict[str, list[CallableInfo]] = field(default_factory=dict)
    runtime_modules: dict[str, set[str]] = field(default_factory=dict)
    parameter_names: set[str] = field(default_factory=set)
    scope: tuple[str, ...] = ()
    guard_scope: str = "<module>"
    activation: str = "<module>"
    guards: tuple[GuardFact, ...] = ()

    @property
    def guard_texts(self) -> tuple[str, ...]:
        return tuple(sorted({guard.text for guard in self.guards}))

    def clone(
        self,
        *,
        scope: tuple[str, ...] | None = None,
        guard_scope: str | None = None,
        activation: str | None = None,
        guards: tuple[GuardFact, ...] | None = None,
    ) -> PatchScanContext:
        return PatchScanContext(
            bindings={name: set(values) for name, values in self.bindings.items()},
            binding_alternatives={name: set(values) for name, values in self.binding_alternatives.items()},
            unknown_bindings=set(self.unknown_bindings),
            upstream_binding_provenance={
                name: set(values) for name, values in self.upstream_binding_provenance.items()
            },
            upstream_binding_history=set(self.upstream_binding_history),
            strings={name: set(values) for name, values in self.strings.items()},
            local_callables={name: list(values) for name, values in self.local_callables.items()},
            runtime_modules={name: set(values) for name, values in self.runtime_modules.items()},
            parameter_names=set(self.parameter_names),
            scope=self.scope if scope is None else scope,
            guard_scope=self.guard_scope if guard_scope is None else guard_scope,
            activation=self.activation if activation is None else activation,
            guards=self.guards if guards is None else guards,
        )

    def bind_exact(self, name: str, references: Iterable[str]) -> None:
        """Replace one name with an exact value and its latest provenance."""
        exact = set(references)
        self.bindings[name] = exact
        self.binding_alternatives[name] = set(exact)
        self.unknown_bindings.discard(name)
        upstream = {reference for reference in exact if reference == "vllm" or reference.startswith("vllm.")}
        if upstream:
            self.upstream_binding_provenance[name] = upstream
            self.upstream_binding_history.add(name)
        else:
            self.upstream_binding_provenance.pop(name, None)

    def bind_none(self, name: str) -> None:
        """Record a proven ``None`` binding and clear stale upstream origin."""
        self.bindings[name] = set()
        self.binding_alternatives[name] = {None}
        self.unknown_bindings.discard(name)
        self.upstream_binding_provenance.pop(name, None)

    def bind_unknown(self, name: str) -> None:
        """Tombstone an exact binding while retaining its last known origin."""
        self.bindings.pop(name, None)
        self.binding_alternatives.pop(name, None)
        self.unknown_bindings.add(name)

    def merge(self, contexts: Sequence[PatchScanContext]) -> None:
        if not contexts:
            return
        self.bindings = _merge_candidate_maps(context.bindings for context in contexts)
        self.binding_alternatives = _merge_binding_alternative_maps(
            context.binding_alternatives for context in contexts
        )
        all_binding_names = {
            name
            for context in contexts
            for name in (
                *context.bindings,
                *context.binding_alternatives,
                *context.unknown_bindings,
                *context.upstream_binding_provenance,
            )
        }
        self.unknown_bindings = {
            name
            for name in all_binding_names
            if any(
                name in context.unknown_bindings
                or (
                    name not in context.bindings
                    and name not in context.binding_alternatives
                    and any(
                        name in other.bindings or name in other.binding_alternatives or name in other.unknown_bindings
                        for other in contexts
                    )
                )
                for context in contexts
            )
        }
        merged_provenance: dict[str, set[str]] = defaultdict(set)
        for branch in contexts:
            for name, references in branch.upstream_binding_provenance.items():
                merged_provenance[name].update(references)
        self.upstream_binding_provenance = dict(merged_provenance)
        self.upstream_binding_history = {name for branch in contexts for name in branch.upstream_binding_history}
        self.strings = _merge_candidate_maps(context.strings for context in contexts)
        self.runtime_modules = _merge_candidate_maps(context.runtime_modules for context in contexts)
        callable_names = {name for context in contexts for name in context.local_callables}
        merged_callables: dict[str, list[CallableInfo]] = {}
        for name in callable_names:
            if any(name not in context.local_callables for context in contexts):
                continue
            candidates: dict[tuple[str, str | None, str, int], CallableInfo] = {}
            for context in contexts:
                for candidate in context.local_callables.get(name, []):
                    key = (
                        candidate.file,
                        candidate.owner,
                        candidate.name,
                        getattr(candidate.node, "lineno", 0),
                    )
                    candidates[key] = candidate
            merged_callables[name] = list(candidates.values())
        self.local_callables = merged_callables


@dataclass
class PatchFlowExit:
    kind: str
    context: PatchScanContext
    exception_name: str | None = None


@dataclass
class PatchFlowResult:
    live: bool = True
    exits: list[PatchFlowExit] = field(default_factory=list)


@dataclass(frozen=True)
class PrivateHelperInvocation:
    """One statically exact private-helper call on the active main path."""

    bindings: tuple[tuple[str, str], ...]
    guards: tuple[GuardFact, ...] = ()
    activation: str = ""


@dataclass
class PrivateHelperDefinition:
    identity: str
    info: CallableInfo
    module_info: ModuleInfo
    tag_guard_names: frozenset[str]
    entry_context: PatchScanContext | None = None


@dataclass(frozen=True)
class StaticValueAlternative:
    target: str | None
    truth: bool
    guards: tuple[GuardFact, ...] = ()


@dataclass(frozen=True)
class PatchReplacement:
    info: CallableInfo | None
    kind: str
    reason: str | None = None
    is_restore: bool = False
    is_save: bool = False
    lifecycle_source: str | None = None


def _merge_candidate_maps(
    mappings: Iterable[dict[str, set[str]]],
) -> dict[str, set[str]]:
    materialized = list(mappings)
    names = {name for mapping in materialized for name in mapping}
    merged: dict[str, set[str]] = {}
    for name in names:
        branch_values = [mapping.get(name) for mapping in materialized]
        if any(not values for values in branch_values):
            merged[name] = set()
        else:
            merged[name] = {value for values in branch_values if values is not None for value in values}
    return merged


def _merge_binding_alternative_maps(
    mappings: Iterable[dict[str, set[str | None]]],
) -> dict[str, set[str | None]]:
    """Keep alternatives only when every incoming path is fully described."""

    materialized = list(mappings)
    names = {name for mapping in materialized for name in mapping}
    return {
        name: {value for mapping in materialized for value in mapping[name]}
        for name in names
        if all(name in mapping and mapping[name] for mapping in materialized)
    }


class RepositoryIndex:
    """AST-only symbol and import index for one Python package."""

    def __init__(self, repo_root: Path, package_name: str):
        self.repo_root = repo_root.resolve()
        self.package_name = package_name
        self.package_root = self.repo_root / package_name
        if not self.package_root.is_dir():
            raise ValueError(f"package directory not found: {self.package_root}")

        self.modules: dict[str, ModuleInfo] = {}
        self.classes: dict[str, ClassInfo] = {}
        self.callables: dict[str, CallableInfo] = {}
        self.callable_variants: dict[str, tuple[CallableInfo, ...]] = {}
        self.values: dict[str, ValueInfo] = {}
        self.aliases: dict[str, str] = {}
        self.unconditional_exports: set[str] = set()
        self.unconditional_symbols: set[str] = set()
        self._unconditional_star_imports: set[tuple[str, str]] = set()
        self._pending_method_aliases: list[tuple[str, str, str, str, int]] = []
        self.parse_errors: list[dict[str, str]] = []
        self._parse()

    def _parse(self) -> None:
        for path in sorted(self.package_root.rglob("*.py")):
            relative_file = path.relative_to(self.repo_root).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (SyntaxError, UnicodeDecodeError) as error:
                self.parse_errors.append(
                    {
                        "file": relative_file,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue

            module, is_package = _module_name(self.package_name, self.package_root, path)
            imports: dict[str, str] = {}
            classes: dict[str, ClassInfo] = {}
            functions: dict[str, CallableInfo] = {}
            loose_functions: dict[str, list[CallableInfo]] = defaultdict(list)
            string_constants: dict[str, set[str]] = defaultdict(set)
            star_imports: list[str] = []
            annotated_exports: list[tuple[str, str]] = []
            tag_guard_names = _tag_guard_names(tree.body)
            module_final_bindings = _scope_final_bindings(
                tree.body,
                tag_guard_names,
            )
            module_must_names = _scope_must_bound_names(
                tree.body,
                tag_guard_names,
            )
            module_statements = list(
                _main_module_statements(
                    tree.body,
                    tag_guard_names,
                )
            )
            statement_availability = {
                id(node): unconditional
                for node, unconditional in _main_module_statement_records(
                    tree.body,
                    tag_guard_names,
                )
            }

            for node in module_statements:
                unconditional = statement_availability.get(id(node), False)
                assignment_targets: Sequence[ast.AST] = ()
                assignment_value: ast.AST | None = None
                if isinstance(node, ast.Assign):
                    assignment_targets = node.targets
                    assignment_value = node.value
                elif isinstance(node, ast.AnnAssign):
                    assignment_targets = (node.target,)
                    assignment_value = node.value
                for target in assignment_targets:
                    if not isinstance(target, ast.Name):
                        continue
                    qualified_value = f"{module}.{target.id}"
                    self.values[qualified_value] = ValueInfo(
                        qualified_name=qualified_value,
                        module=module,
                        file=relative_file,
                        owner=None,
                        name=target.id,
                        node=assignment_value,
                    )
                    if unconditional or target.id in module_must_names:
                        self.unconditional_exports.add(qualified_value)
                        self.unconditional_symbols.add(qualified_value)
                string_assignment = _string_assignment(node)
                if string_assignment:
                    name, value = string_assignment
                    string_constants[name].add(value)
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        local_name = alias.asname or alias.name.split(".", 1)[0]
                        imports[local_name] = alias.name if alias.asname else local_name
                        if unconditional or local_name in module_must_names:
                            self.unconditional_exports.add(f"{module}.{local_name}")
                elif isinstance(node, ast.ImportFrom):
                    source_module = _relative_import_module(
                        module,
                        is_package,
                        node.level,
                        node.module,
                    )
                    for alias in node.names:
                        if alias.name == "*":
                            star_imports.append(source_module)
                            if unconditional:
                                self._unconditional_star_imports.add((module, source_module))
                            continue
                        local_name = alias.asname or alias.name
                        imports[local_name] = f"{source_module}.{alias.name}" if source_module else alias.name
                        if unconditional or local_name in module_must_names:
                            self.unconditional_exports.add(f"{module}.{local_name}")
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    annotation = _expression_name(node.annotation)
                    if annotation:
                        annotated_exports.append((node.target.id, annotation))
                elif isinstance(node, ast.ClassDef):
                    bases = tuple(name for name in (_expression_name(base) for base in node.bases) if name)
                    resolved_bases = tuple(
                        _resolve_bound_reference(
                            module,
                            base,
                            imports,
                            {*classes, *functions},
                        )
                        for base in bases
                    )
                    imports.pop(node.name, None)
                    qualified_name = f"{module}.{node.name}"
                    class_is_unconditional = unconditional or node.name in module_must_names
                    class_final_bindings = _scope_final_bindings(
                        node.body,
                        tag_guard_names,
                    )
                    class_must_callable_names = {
                        name
                        for name, candidates in class_final_bindings.items()
                        if candidates and all(candidate.kind == "function" for candidate in candidates)
                    }
                    method_variants = _possible_method_variants(
                        node,
                        tag_guard_names,
                    )
                    info = ClassInfo(
                        qualified_name=qualified_name,
                        module=module,
                        file=relative_file,
                        name=node.name,
                        bases=bases,
                        resolved_bases=resolved_bases,
                        methods={name: candidates[0] for name, candidates in method_variants.items()},
                        method_variants=method_variants,
                    )
                    classes[node.name] = info
                    self.classes[qualified_name] = info
                    if class_is_unconditional:
                        self.unconditional_exports.add(qualified_name)
                        self.unconditional_symbols.add(qualified_name)
                    self.callables[qualified_name] = CallableInfo(
                        qualified_name=qualified_name,
                        module=module,
                        file=relative_file,
                        owner=None,
                        name=node.name,
                        node=node,
                    )
                    for class_statement in node.body:
                        class_targets: Sequence[ast.AST] = ()
                        class_value: ast.AST | None = None
                        if isinstance(class_statement, ast.Assign):
                            class_targets = class_statement.targets
                            class_value = class_statement.value
                        elif isinstance(class_statement, ast.AnnAssign):
                            class_targets = (class_statement.target,)
                            class_value = class_statement.value
                        for target in class_targets:
                            if not isinstance(target, ast.Name):
                                continue
                            qualified_value = f"{qualified_name}.{target.id}"
                            self.values[qualified_value] = ValueInfo(
                                qualified_name=qualified_value,
                                module=module,
                                file=relative_file,
                                owner=node.name,
                                name=target.id,
                                node=class_value,
                            )
                    for method_name, method_node in info.methods.items():
                        method_qualified_name = f"{qualified_name}.{method_name}"
                        variants = tuple(
                            CallableInfo(
                                qualified_name=method_qualified_name,
                                module=module,
                                file=relative_file,
                                owner=node.name,
                                name=method_name,
                                node=candidate,
                            )
                            for candidate in info.method_variants.get(method_name, (method_node,))
                        )
                        self.callable_variants[method_qualified_name] = variants
                        self.callables[method_qualified_name] = variants[0]
                        if class_is_unconditional and method_name in class_must_callable_names:
                            self.unconditional_symbols.add(method_qualified_name)
                    self._collect_class_callable_aliases(
                        node,
                        module,
                        qualified_name,
                        imports,
                        {*classes, *functions},
                    )
                elif isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                    imports.pop(node.name, None)
                    qualified_name = f"{module}.{node.name}"
                    info = CallableInfo(
                        qualified_name=qualified_name,
                        module=module,
                        file=relative_file,
                        owner=None,
                        name=node.name,
                        node=node,
                    )
                    functions[node.name] = info
                    self.callables[qualified_name] = info
                    if unconditional or node.name in module_must_names:
                        self.unconditional_exports.add(qualified_name)
                        self.unconditional_symbols.add(qualified_name)

            module_function_names = {
                *functions,
                *(
                    name
                    for name, candidates in module_final_bindings.items()
                    if any(candidate.kind == "function" for candidate in candidates)
                ),
            }
            for function_name in module_function_names:
                qualified_name = f"{module}.{function_name}"
                candidates = tuple(
                    candidate.node
                    for candidate in module_final_bindings.get(function_name, ())
                    if candidate.kind == "function" and candidate.node is not None
                )
                self.unconditional_exports.discard(qualified_name)
                self.unconditional_symbols.discard(qualified_name)
                if not candidates:
                    functions.pop(function_name, None)
                    self.callables.pop(qualified_name, None)
                    self.callable_variants.pop(qualified_name, None)
                    continue
                variants = tuple(
                    CallableInfo(
                        qualified_name=qualified_name,
                        module=module,
                        file=relative_file,
                        owner=None,
                        name=function_name,
                        node=candidate,
                    )
                    for candidate in candidates
                )
                functions[function_name] = variants[0]
                self.callables[qualified_name] = variants[0]
                self.callable_variants[qualified_name] = variants
                final_alternatives = module_final_bindings[function_name]
                if final_alternatives and all(candidate.kind == "function" for candidate in final_alternatives):
                    self.unconditional_exports.add(qualified_name)
                    self.unconditional_symbols.add(qualified_name)

            for node in _main_ast_walk(tree):
                if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                    continue
                qualified_name = f"{module}.{node.name}"
                loose_functions[node.name].append(
                    CallableInfo(
                        qualified_name=qualified_name,
                        module=module,
                        file=relative_file,
                        owner=None,
                        name=node.name,
                        node=node,
                    )
                )

            lazy_names = {
                name
                for candidate in module_statements
                if isinstance(candidate, (ast.AsyncFunctionDef, ast.FunctionDef)) and candidate.name == "__getattr__"
                for name in _lazy_getattr_names(candidate)
            }
            typed_lazy_exports = {
                name: _resolve_bound_reference(
                    module,
                    annotation,
                    imports,
                    {*classes, *functions},
                )
                for name, annotation in annotated_exports
                if name in lazy_names
            }
            module_info = ModuleInfo(
                name=module,
                file=relative_file,
                is_package=is_package,
                tree=tree,
                imports=imports,
                classes=classes,
                functions=functions,
                loose_functions=dict(loose_functions),
                string_constants={name: tuple(sorted(values)) for name, values in string_constants.items()},
                star_imports=tuple(star_imports),
                typed_lazy_exports=typed_lazy_exports,
            )
            self.modules[module] = module_info
            for local_name, target in imports.items():
                self.aliases[f"{module}.{local_name}"] = target
            for export_name, target in typed_lazy_exports.items():
                self.aliases[f"{module}.{export_name}"] = target

        self._materialize_star_import_aliases()
        self._materialize_dataclass_initializers()
        self._materialize_class_callable_aliases()

    def _materialize_star_import_aliases(self) -> None:
        """Resolve public top-level callables imported with ``import *``."""
        changed = True
        while changed:
            changed = False
            for module_info in self.modules.values():
                desired: dict[str, str] = {}
                for source_module in module_info.star_imports:
                    source = self.modules.get(source_module)
                    if source is None:
                        continue
                    exported_names = {
                        *source.classes,
                        *source.functions,
                        *(
                            alias.rsplit(".", 1)[-1]
                            for alias in self.aliases
                            if alias.startswith(f"{source_module}.") and "." not in alias[len(source_module) + 1 :]
                        ),
                    }
                    for name in sorted(exported_names):
                        if name.startswith("_"):
                            continue
                        alias = f"{module_info.name}.{name}"
                        target = f"{source_module}.{name}"
                        desired[alias] = target
                for alias, target in desired.items():
                    if self.aliases.get(alias) == target:
                        continue
                    self.aliases[alias] = target
                    source_module = target.rsplit(".", 1)[0]
                    if (
                        module_info.name,
                        source_module,
                    ) in self._unconditional_star_imports and target in self.unconditional_exports:
                        self.unconditional_exports.add(alias)
                    changed = True

    def _materialize_dataclass_initializers(self) -> None:
        field_cache: dict[
            str,
            list[tuple[str, bool, bool]],
        ] = {}
        for class_info in self.classes.values():
            if "__init__" in class_info.methods:
                continue
            class_node = self.callables[class_info.qualified_name].node
            config = self._dataclass_config(class_info.module, class_node)
            if config is None or not config[0]:
                continue
            fields = self._dataclass_fields(class_info, field_cache, frozenset())
            if fields is None:
                continue
            self_name = "__dataclass_self__" if any(name == "self" for name, _, _ in fields) else "self"
            positional = [[self_name, True]]
            positional.extend([name, required] for name, required, kw_only in fields if not kw_only)
            keyword_only = [[name, required] for name, required, kw_only in fields if kw_only]
            signature: list[object] = [
                "sync",
                [],
                positional,
                None,
                keyword_only,
                None,
            ]
            class_info.methods["__init__"] = class_node or ast.Pass()
            qualified_name = f"{class_info.qualified_name}.__init__"
            generated = CallableInfo(
                qualified_name=f"{class_info.qualified_name}.__init__",
                module=class_info.module,
                file=class_info.file,
                owner=class_info.name,
                name="__init__",
                node=None,
                binding_line=getattr(class_node, "lineno", 0),
                origin_kind="generated_dataclass_method",
                signature_override=signature,
            )
            class_info.method_variants["__init__"] = (class_node or ast.Pass(),)
            self.callables[qualified_name] = generated
            self.callable_variants[qualified_name] = (generated,)
            if class_info.qualified_name in self.unconditional_symbols:
                self.unconditional_symbols.add(f"{class_info.qualified_name}.__init__")

    def _dataclass_fields(
        self,
        class_info: ClassInfo,
        cache: dict[str, list[tuple[str, bool, bool]]],
        visiting: frozenset[str],
    ) -> list[tuple[str, bool, bool]] | None:
        if class_info.qualified_name in cache:
            return list(cache[class_info.qualified_name])
        if class_info.qualified_name in visiting:
            return None
        class_node = self.callables[class_info.qualified_name].node
        if not isinstance(class_node, ast.ClassDef):
            return None
        config = self._dataclass_config(class_info.module, class_node)
        if config is None:
            return None
        _, default_kw_only = config

        fields: list[tuple[str, bool, bool]] = []
        positions: dict[str, int] = {}
        next_visiting = frozenset((*visiting, class_info.qualified_name))
        for base_name in class_info.resolved_bases:
            if base_name in {"builtins.object", "object"}:
                continue
            base = self.find_class(base_name)
            if base is None:
                return None
            base_config = self._dataclass_config(
                base.module,
                self.callables[base.qualified_name].node,
            )
            if base_config is None:
                continue
            base_fields = self._dataclass_fields(
                base,
                cache,
                next_visiting,
            )
            if base_fields is None:
                return None
            for field_info in base_fields:
                positions[field_info[0]] = len(fields)
                fields.append(field_info)

        kw_only = default_kw_only
        for statement in class_node.body:
            if not isinstance(statement, ast.AnnAssign):
                continue
            if not isinstance(statement.target, ast.Name):
                continue
            annotation = "".join(ast.unparse(statement.annotation).split())
            if annotation.rsplit(".", 1)[-1] == "KW_ONLY":
                kw_only = True
                continue
            if "ClassVar" in annotation:
                continue
            field_config = self._dataclass_field_config(
                statement.value,
                kw_only,
            )
            if field_config is None:
                return None
            include, required, field_kw_only = field_config
            if not include:
                continue
            field_info = (
                statement.target.id,
                required,
                field_kw_only,
            )
            if statement.target.id in positions:
                fields[positions[statement.target.id]] = field_info
            else:
                positions[statement.target.id] = len(fields)
                fields.append(field_info)

        cache[class_info.qualified_name] = list(fields)
        return fields

    def _dataclass_config(
        self,
        module: str,
        node: ast.AST | None,
    ) -> tuple[bool, bool] | None:
        if not isinstance(node, ast.ClassDef):
            return None
        for decorator in node.decorator_list:
            call = decorator if isinstance(decorator, ast.Call) else None
            expression = _expression_name(call.func if call else decorator)
            if expression is None:
                continue
            reference = self.canonical_name(self.resolve_reference(module, expression))
            if reference != "dataclasses.dataclass":
                continue
            init = True
            kw_only = False
            if call:
                for keyword in call.keywords:
                    if keyword.arg not in {"init", "kw_only"}:
                        continue
                    if not isinstance(keyword.value, ast.Constant) or not isinstance(
                        keyword.value.value,
                        bool,
                    ):
                        return None
                    if keyword.arg == "init":
                        init = keyword.value.value
                    else:
                        kw_only = keyword.value.value
            return init, kw_only
        return None

    def _dataclass_field_config(
        self,
        value: ast.AST | None,
        default_kw_only: bool,
    ) -> tuple[bool, bool, bool] | None:
        if not isinstance(value, ast.Call):
            return True, value is None, default_kw_only
        function_name = _expression_name(value.func)
        if not function_name or function_name.rsplit(".", 1)[-1] != "field":
            return True, False, default_kw_only

        include = True
        kw_only = default_kw_only
        has_default = bool(value.args)
        for keyword in value.keywords:
            if keyword.arg in {"default", "default_factory"}:
                has_default = True
            elif keyword.arg in {"init", "kw_only"}:
                if not isinstance(keyword.value, ast.Constant) or not isinstance(
                    keyword.value.value,
                    bool,
                ):
                    return None
                if keyword.arg == "init":
                    include = keyword.value.value
                else:
                    kw_only = keyword.value.value
        return include, not has_default, kw_only

    def _collect_class_callable_aliases(
        self,
        node: ast.ClassDef,
        module: str,
        class_name: str,
        imports: dict[str, str],
        local_names: set[str],
    ) -> None:
        explicit_methods = _method_nodes(node)
        for statement in node.body:
            value: ast.AST | None = None
            targets: Sequence[ast.AST] = ()
            if isinstance(statement, ast.Assign):
                value = statement.value
                targets = statement.targets
            elif isinstance(statement, ast.AnnAssign):
                value = statement.value
                targets = (statement.target,)
            else:
                continue

            kind = "callable_alias"
            if isinstance(value, ast.Call):
                wrapper = _expression_name(value.func)
                if wrapper not in {"classmethod", "property", "staticmethod"}:
                    continue
                if len(value.args) != 1:
                    continue
                kind = wrapper
                value = value.args[0]
            expression = _expression_name(value)
            if expression is None:
                continue
            resolved = _resolve_bound_reference(
                module,
                expression,
                imports,
                local_names,
            )
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id in explicit_methods:
                    continue
                self._pending_method_aliases.append(
                    (
                        class_name,
                        target.id,
                        resolved,
                        kind,
                        getattr(statement, "lineno", 0),
                    )
                )

    def _materialize_class_callable_aliases(self) -> None:
        for class_name, member_name, target, kind, line in self._pending_method_aliases:
            source = self.find_callable(target)
            if source is None or not isinstance(
                source.node,
                (ast.AsyncFunctionDef, ast.FunctionDef, ast.Lambda),
            ):
                continue
            class_info = self.classes[class_name]
            if member_name in class_info.methods:
                continue
            qualified_name = f"{class_name}.{member_name}"
            alias = CallableInfo(
                qualified_name=qualified_name,
                module=class_info.module,
                file=class_info.file,
                owner=class_info.name,
                name=member_name,
                node=source.node,
                binding_line=line,
                origin_kind=kind,
            )
            class_info.methods[member_name] = source.node
            class_info.method_variants[member_name] = (source.node,)
            self.callables[qualified_name] = alias
            self.callable_variants[qualified_name] = (alias,)
            if class_name in self.unconditional_symbols and target in self.unconditional_symbols:
                self.unconditional_symbols.add(qualified_name)

    def resolve_reference(self, module: str, expression: str) -> str:
        parts = expression.split(".")
        module_info = self.modules[module]
        if parts[0] in module_info.imports:
            target = module_info.imports[parts[0]]
            return ".".join([target, *parts[1:]])
        if parts[0] in module_info.classes or parts[0] in module_info.functions:
            return f"{module}.{expression}"
        if expression.startswith((f"{self.package_name}.", "vllm.", "vllm_ascend.")):
            return expression
        return f"{module}.{expression}"

    def canonical_name(self, qualified_name: str) -> str:
        result = qualified_name
        visited: set[str] = set()
        while result not in visited:
            visited.add(result)
            replacement = None
            for alias in sorted(self.aliases, key=len, reverse=True):
                if result == alias or result.startswith(f"{alias}."):
                    replacement = f"{self.aliases[alias]}{result[len(alias) :]}"
                    break
            if replacement is None or replacement == result:
                break
            result = replacement
        return result

    def find_class(self, qualified_name: str) -> ClassInfo | None:
        canonical = self.canonical_name(qualified_name)
        return self.classes.get(canonical)

    def find_callable(self, qualified_name: str) -> CallableInfo | None:
        canonical = self.canonical_name(qualified_name)
        return self.callables.get(canonical)

    def find_callable_variants(
        self,
        qualified_name: str,
    ) -> tuple[CallableInfo, ...]:
        canonical = self.canonical_name(qualified_name)
        direct = self.callable_variants.get(canonical)
        if direct is not None:
            return direct
        callable_info = self.callables.get(canonical)
        return (callable_info,) if callable_info is not None else ()

    def find_loose_function(self, module: str, name: str) -> CallableInfo | None:
        candidates = self.modules[module].loose_functions.get(name, [])
        return candidates[0] if len(candidates) == 1 else None

    def find_value(self, qualified_name: str) -> ValueInfo | None:
        direct = self.values.get(qualified_name)
        if direct is not None:
            return direct
        return self.values.get(self.canonical_name(qualified_name))


class InterfaceBoundaryGenerator:
    def __init__(
        self,
        vllm_root: Path,
        ascend_root: Path,
        external_roots: dict[str, Path] | None = None,
    ):
        self.upstream = RepositoryIndex(vllm_root, "vllm")
        self.downstream = RepositoryIndex(ascend_root, "vllm_ascend")
        self.externals = {
            package: RepositoryIndex(root, package) for package, root in sorted((external_roots or {}).items())
        }
        parse_errors = (
            [("vLLM", error) for error in self.upstream.parse_errors]
            + [("vllm-ascend", error) for error in self.downstream.parse_errors]
            + [(package, error) for package, index in self.externals.items() for error in index.parse_errors]
        )
        if parse_errors:
            details = "; ".join(f"{repository}:{error['file']}: {error['error']}" for repository, error in parse_errors)
            raise ValueError(f"Python source parsing failed: {details}")
        self.relations: list[Relation] = []
        self.findings: list[CandidateFinding] = []
        self._mro_cache: dict[str, MroResult] = {}
        self._private_helper_invocations: dict[str, tuple[PrivateHelperInvocation, ...]] = {}
        self._private_helper_definitions: dict[str, PrivateHelperDefinition] = {}
        self._private_helper_exports: dict[str, str] = {}
        self._private_helper_node_identities: dict[int, str] = {}

    def generate(self) -> tuple[list[Relation], list[CandidateFinding]]:
        self._collect_inheritance()
        self._collect_verified_overrides()
        self._collect_monkey_patches()
        self._reclassify_missing_patch_members()
        grouped: dict[tuple[str, ...], list[Relation]] = defaultdict(list)
        for relation in self.relations:
            grouped[relation.exact_key()].append(relation)
        deduplicated = {}
        for key, occurrences in grouped.items():
            first = min(
                occurrences,
                key=lambda item: (
                    item.evidence_file,
                    item.evidence_line,
                ),
            )
            evidence = {
                item
                for relation in occurrences
                for item in (
                    relation.evidence
                    or (
                        RelationEvidence(
                            file=relation.evidence_file,
                            line=relation.evidence_line,
                        ),
                    )
                )
            }
            deduplicated[key] = replace(
                first,
                evidence=tuple(
                    sorted(
                        evidence,
                        key=lambda item: (
                            item.file,
                            item.line,
                            item.scope or "",
                            item.guards,
                            item.patch_kind or "",
                            item.target_expression or "",
                        ),
                    )
                ),
            )
        self.relations = sorted(
            deduplicated.values(),
            key=lambda relation: (
                relation.upstream_key(),
                relation.downstream_key(),
            ),
        )
        self.findings = sorted(
            set(self.findings),
            key=lambda relation: (
                relation.status,
                relation.reason_code,
                relation.relation,
                relation.downstream_file,
                relation.downstream_owner or "",
                relation.downstream_name,
                relation.target_expression,
                relation.evidence_line,
                relation.evidence_scope or "",
                relation.evidence_guards,
                relation.reason,
            ),
        )
        return self.relations, self.findings

    def _resolve_downstream_reference(self, module: str, expression: str) -> str:
        qualified = self.downstream.resolve_reference(module, expression)
        return self._canonical_reference(qualified)

    def _canonical_reference(self, qualified_name: str) -> str:
        if qualified_name.startswith("vllm."):
            return self.upstream.canonical_name(qualified_name)
        if qualified_name.startswith("vllm_ascend."):
            return self.downstream.canonical_name(qualified_name)
        for package, index in self.externals.items():
            if qualified_name == package or qualified_name.startswith(f"{package}."):
                return index.canonical_name(qualified_name)
        return qualified_name

    def _class_info(self, qualified_name: str) -> ClassInfo | None:
        if qualified_name.startswith("vllm_ascend."):
            return self.downstream.find_class(qualified_name)
        if qualified_name.startswith("vllm."):
            return self.upstream.find_class(qualified_name)
        for package, index in self.externals.items():
            if qualified_name == package or qualified_name.startswith(f"{package}."):
                return index.find_class(qualified_name)
        return None

    def _callable_info(self, qualified_name: str) -> CallableInfo | None:
        if qualified_name.startswith("vllm_ascend."):
            return self.downstream.find_callable(qualified_name)
        if qualified_name.startswith("vllm."):
            return self.upstream.find_callable(qualified_name)
        for package, index in self.externals.items():
            if qualified_name == package or qualified_name.startswith(f"{package}."):
                return index.find_callable(qualified_name)
        return None

    def _callable_variants(
        self,
        qualified_name: str,
    ) -> tuple[CallableInfo, ...]:
        if qualified_name.startswith("vllm_ascend."):
            return self.downstream.find_callable_variants(qualified_name)
        if qualified_name.startswith("vllm."):
            return self.upstream.find_callable_variants(qualified_name)
        for package, index in self.externals.items():
            if qualified_name == package or qualified_name.startswith(f"{package}."):
                return index.find_callable_variants(qualified_name)
        return ()

    def _member_is_unconditional(
        self,
        owner: str,
        member: str,
    ) -> bool:
        qualified_name = self._canonical_reference(f"{owner}.{member}")
        if qualified_name.startswith("vllm_ascend."):
            return qualified_name in self.downstream.unconditional_symbols
        if qualified_name.startswith("vllm."):
            return qualified_name in self.upstream.unconditional_symbols
        for package, index in self.externals.items():
            if qualified_name == package or qualified_name.startswith(f"{package}."):
                return qualified_name in index.unconditional_symbols
        return False

    def _source_package(self, qualified_name: str) -> str:
        if qualified_name == "vllm" or qualified_name.startswith("vllm."):
            return "vllm"
        for package in self.externals:
            if qualified_name == package or qualified_name.startswith(f"{package}."):
                return package
        raise ValueError(f"interface source package was not indexed: {qualified_name}")

    def _class_defines_method(
        self,
        qualified_name: str,
        method_name: str,
    ) -> bool:
        class_info = self._class_info(qualified_name)
        return class_info is not None and method_name in class_info.methods

    def _class_bases(
        self,
        qualified_name: str,
    ) -> tuple[list[str], list[str]]:
        if qualified_name in STDLIB_STRUCTURAL_BASES:
            return list(STDLIB_STRUCTURAL_BASES[qualified_name]), []
        info = self._class_info(qualified_name)
        if info is None:
            return [], [qualified_name]
        bases: list[str] = []
        missing: list[str] = []
        normalized_bases: list[str] = []
        for candidate in info.resolved_bases:
            normalized_bases.append(self._canonical_reference(candidate))

        for candidate in normalized_bases:
            if self._class_info(candidate) or candidate in STDLIB_STRUCTURAL_BASES:
                bases.append(candidate)
            elif candidate not in {"builtins.object", "object"}:
                missing.append(f"opaque or unresolved base: {candidate}")
                break
        return bases, missing

    def _linearized_mro(
        self,
        qualified_name: str,
        stack: tuple[str, ...] = (),
    ) -> MroResult:
        if qualified_name in self._mro_cache:
            return self._mro_cache[qualified_name]
        if qualified_name in stack:
            return MroResult(
                owners=(qualified_name,),
                complete=False,
                reason=f"inheritance cycle at {qualified_name}",
            )

        bases, missing = self._class_bases(qualified_name)
        if not bases:
            if missing:
                result = MroResult(
                    owners=(qualified_name,),
                    complete=False,
                    reason=(f"unresolved base(s): {', '.join(sorted(missing))}"),
                )
                self._mro_cache[qualified_name] = result
                return result
            result = MroResult(
                owners=(qualified_name,),
                complete=True,
            )
            self._mro_cache[qualified_name] = result
            return result

        base_results = [self._linearized_mro(base, (*stack, qualified_name)) for base in bases]
        incomplete = next(
            (result for result in base_results if not result.complete),
            None,
        )
        if missing or incomplete is not None:
            prefix: tuple[str, ...] = (qualified_name,)
            if len(base_results) == 1:
                prefix = (*prefix, *base_results[0].owners)
            reason_parts = []
            if missing:
                reason_parts.append(f"unresolved base(s): {', '.join(sorted(missing))}")
            if incomplete is not None and incomplete.reason:
                reason_parts.append(incomplete.reason)
            result = MroResult(
                owners=prefix,
                complete=False,
                reason="; ".join(reason_parts),
            )
            self._mro_cache[qualified_name] = result
            return result

        sequences = [list(result.owners) for result in base_results]
        sequences.append(bases.copy())
        result = [qualified_name]
        while any(sequences):
            sequences = [sequence for sequence in sequences if sequence]
            candidate = next(
                (sequence[0] for sequence in sequences if not any(sequence[0] in other[1:] for other in sequences)),
                None,
            )
            if candidate is None:
                incomplete_result = MroResult(
                    owners=tuple(result),
                    complete=False,
                    reason=f"invalid or ambiguous MRO at {qualified_name}",
                )
                self._mro_cache[qualified_name] = incomplete_result
                return incomplete_result
            result.append(candidate)
            for sequence in sequences:
                if sequence and sequence[0] == candidate:
                    sequence.pop(0)

        complete_result = MroResult(
            owners=tuple(result),
            complete=True,
        )
        self._mro_cache[qualified_name] = complete_result
        return complete_result

    def _collect_inheritance(self) -> None:
        for class_info in self.downstream.classes.values():
            for base_expression, resolved in zip(
                class_info.bases,
                class_info.resolved_bases,
            ):
                resolved = self._canonical_reference(resolved)
                if not resolved.startswith("vllm."):
                    continue
                upstream_class = self.upstream.find_class(resolved)
                if upstream_class is None:
                    self.findings.append(
                        CandidateFinding(
                            relation="inheritance",
                            downstream_file=class_info.file,
                            downstream_owner=class_info.name,
                            downstream_name=class_info.name,
                            target_expression=resolved,
                            evidence_line=self._class_line(class_info),
                            reason="upstream base class was not found",
                            status="risk",
                            reason_code="missing_upstream_base",
                            generator_issue=False,
                        )
                    )
                    continue
                self.relations.append(
                    Relation(
                        relation="inheritance",
                        upstream_file=upstream_class.file,
                        upstream_owner=None,
                        upstream_name=upstream_class.name,
                        upstream_signature=None,
                        downstream_file=class_info.file,
                        downstream_owner=class_info.name,
                        downstream_name=base_expression.rsplit(".", 1)[-1],
                        downstream_signature=None,
                        evidence_file=class_info.file,
                        evidence_line=self._class_line(class_info),
                    )
                )

    def _collect_verified_overrides(self) -> None:
        for class_info in self.downstream.classes.values():
            mro_result = self._linearized_mro(class_info.qualified_name)
            mro = mro_result.owners
            if mro_result.complete and not any(owner.startswith("vllm.") for owner in mro[1:]):
                continue
            for method_name, method_node in class_info.methods.items():
                effective_owners = self._effective_method_owners(mro[1:], method_name)
                if not effective_owners:
                    candidates = (
                        self._candidate_upstream_method_owners(
                            class_info.qualified_name,
                            method_name,
                        )
                        if not mro_result.complete
                        else ()
                    )
                    if candidates:
                        self.findings.append(
                            CandidateFinding(
                                relation="override",
                                downstream_file=class_info.file,
                                downstream_owner=class_info.name,
                                downstream_name=method_name,
                                target_expression=", ".join(candidates),
                                evidence_line=getattr(
                                    method_node,
                                    "lineno",
                                    0,
                                ),
                                reason=(
                                    f"incomplete MRO ({mro_result.reason}); candidate upstream owner was not selected"
                                ),
                                status="review",
                                reason_code="ambiguous_mro",
                                generator_issue=False,
                            )
                        )
                    continue
                for effective_owner in effective_owners:
                    self._record_verified_override_owner(
                        class_info,
                        method_name,
                        method_node,
                        effective_owner,
                        mro,
                    )

    def _record_verified_override_owner(
        self,
        class_info: ClassInfo,
        method_name: str,
        method_node: ast.AST,
        effective_owner: str,
        mro: Sequence[str],
    ) -> None:
        is_external = self._is_external_owner(effective_owner)
        if not effective_owner.startswith("vllm.") and not is_external:
            return
        if is_external:
            shadowed = next(
                (
                    owner
                    for owner in mro[1:]
                    if owner.startswith("vllm.")
                    and self._class_defines_method(
                        owner,
                        method_name,
                    )
                ),
                None,
            )
            target_expression = f"{effective_owner}.{method_name}"
            reason = f"the effective overridden method is owned by external package class {effective_owner}, not vLLM"
            reason_code = "external_only_override"
            if shadowed is not None:
                target_expression = f"{shadowed}.{method_name}"
                reason = f"external owner {effective_owner} defines the effective method before this vLLM candidate"
                reason_code = "external_override_owner"
            self.findings.append(
                CandidateFinding(
                    relation="override",
                    downstream_file=class_info.file,
                    downstream_owner=class_info.name,
                    downstream_name=method_name,
                    target_expression=target_expression,
                    evidence_line=getattr(method_node, "lineno", 0),
                    reason=reason,
                    status="excluded",
                    reason_code=reason_code,
                    generator_issue=False,
                )
            )
            return
        upstream_name = f"{effective_owner}.{method_name}"
        downstream_name = f"{class_info.qualified_name}.{method_name}"
        upstream_variants = self._callable_variants(upstream_name)
        downstream_variants = self._callable_variants(downstream_name)
        upstream_signatures = {
            json.dumps(
                candidate.signature,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for candidate in upstream_variants
        }
        downstream_signatures = {
            json.dumps(
                candidate.signature,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for candidate in downstream_variants
        }
        if len(upstream_signatures) > 1 or len(downstream_signatures) > 1:
            self.findings.append(
                CandidateFinding(
                    relation="override",
                    downstream_file=class_info.file,
                    downstream_owner=class_info.name,
                    downstream_name=method_name,
                    target_expression=upstream_name,
                    evidence_line=getattr(method_node, "lineno", 0),
                    reason=("conditional upstream or downstream callable has incompatible signature variants"),
                    status="review",
                    reason_code="conditional_callable_variants",
                    generator_issue=False,
                )
            )
            return

        upstream_callable = upstream_variants[0] if upstream_variants else self._callable_info(upstream_name)
        if upstream_callable is None:
            return
        downstream_callable = (
            downstream_variants[0] if downstream_variants else self.downstream.find_callable(downstream_name)
        )
        evidence_line = (
            downstream_callable.binding_line
            if downstream_callable and downstream_callable.binding_line is not None
            else getattr(method_node, "lineno", 0)
        )
        self.relations.append(
            Relation(
                relation="override",
                upstream_file=upstream_callable.file,
                upstream_owner=upstream_callable.owner,
                upstream_name=upstream_callable.name,
                upstream_signature=upstream_callable.signature,
                downstream_file=class_info.file,
                downstream_owner=class_info.name,
                downstream_name=method_name,
                downstream_signature=(
                    downstream_callable.signature if downstream_callable else _jsonable_signature(method_node)
                ),
                evidence_file=class_info.file,
                evidence_line=evidence_line,
                upstream_package=self._source_package(upstream_callable.qualified_name),
            )
        )

    def _effective_method_owners(
        self,
        mro: Sequence[str],
        method_name: str,
    ) -> tuple[str, ...]:
        owners: list[str] = []
        for owner in mro:
            class_info = self._class_info(owner)
            if class_info is None or method_name not in class_info.methods:
                continue
            owners.append(owner)
            if self._member_is_unconditional(owner, method_name):
                break
        return tuple(owners)

    def _effective_method_owner(
        self,
        mro: Sequence[str],
        method_name: str,
    ) -> str | None:
        owners = self._effective_method_owners(mro, method_name)
        return owners[0] if owners else None

    def _is_external_owner(self, qualified_name: str) -> bool:
        return any(qualified_name == package or qualified_name.startswith(f"{package}.") for package in self.externals)

    def _candidate_upstream_method_owners(
        self,
        qualified_name: str,
        method_name: str,
        seen: frozenset[str] = frozenset(),
    ) -> tuple[str, ...]:
        if qualified_name in seen:
            return ()
        class_info = self._class_info(qualified_name)
        if class_info is None:
            return ()

        candidates: set[str] = set()
        next_seen = (*seen, qualified_name)
        for base in class_info.resolved_bases:
            base = self._canonical_reference(base)
            base_info = self._class_info(base)
            if base_info is None:
                continue
            if base.startswith("vllm.") and method_name in base_info.methods:
                candidates.add(base)
            candidates.update(
                self._candidate_upstream_method_owners(
                    base,
                    method_name,
                    frozenset(next_seen),
                )
            )
        return tuple(sorted(candidates))

    def _index_private_helper_definitions(self) -> None:
        definitions: dict[str, PrivateHelperDefinition] = {}
        node_identities: dict[int, str] = {}
        for module_info in self.downstream.modules.values():
            tag_guard_names = _tag_guard_names(module_info.tree.body)
            for node in _main_module_statements(
                module_info.tree.body,
                tag_guard_names,
            ):
                if not isinstance(
                    node,
                    (ast.AsyncFunctionDef, ast.FunctionDef),
                ) or not (node.name.startswith("_") and not node.name.startswith("__")):
                    continue
                qualified_name = f"{module_info.name}.{node.name}"
                identity = (
                    f"<private-helper>:{qualified_name}:{getattr(node, 'lineno', 0)}:{getattr(node, 'col_offset', 0)}"
                )
                info = CallableInfo(
                    qualified_name=qualified_name,
                    module=module_info.name,
                    file=module_info.file,
                    owner=None,
                    name=node.name,
                    node=node,
                )
                definitions[identity] = PrivateHelperDefinition(
                    identity=identity,
                    info=info,
                    module_info=module_info,
                    tag_guard_names=frozenset(tag_guard_names),
                )
                node_identities[id(node)] = identity

        exports = {}
        for module_info in self.downstream.modules.values():
            for name, info in module_info.functions.items():
                if not (name.startswith("_") and not name.startswith("__")):
                    continue
                identity = node_identities.get(id(info.node))
                if identity is not None:
                    exports[info.qualified_name] = identity

        self._private_helper_definitions = definitions
        self._private_helper_exports = exports
        self._private_helper_node_identities = node_identities

    def _prepare_private_helper_parameter_bindings(self) -> None:
        self._index_private_helper_definitions()
        calls: dict[
            str,
            list[tuple[dict[str, set[str] | None], tuple[GuardFact, ...]]],
        ] = defaultdict(list)
        for module_info in self.downstream.modules.values():
            self._scan_private_helper_calls(
                module_info,
                module_info.tree.body,
                PatchScanContext(
                    guard_scope=module_info.name,
                    activation=module_info.name,
                ),
                _tag_guard_names(module_info.tree.body),
                calls,
            )

        known = self._exact_private_helper_invocations(calls)
        processed: set[tuple[str, PrivateHelperInvocation]] = set()
        while True:
            frontier = sorted(
                (
                    (helper_name, invocation)
                    for helper_name, invocations in known.items()
                    for invocation in invocations
                    if (helper_name, invocation) not in processed
                ),
                key=lambda item: (item[0], item[1].bindings, item[1].guards),
            )
            if not frontier:
                break
            for helper_name, invocation in frontier:
                processed.add((helper_name, invocation))
                definition = self._private_helper_definitions[helper_name]
                if definition.entry_context is None or not isinstance(
                    definition.info.node,
                    (ast.AsyncFunctionDef, ast.FunctionDef),
                ):
                    continue
                guards = self._merge_guard_paths(
                    definition.entry_context.guards,
                    invocation.guards,
                )
                if guards is None:
                    continue
                context = definition.entry_context.clone(
                    guards=guards,
                    activation=invocation.activation,
                )
                for parameter, target in invocation.bindings:
                    context.bindings[parameter] = {target}
                    context.binding_alternatives[parameter] = {target}
                forwarded_calls: dict[
                    str,
                    list[
                        tuple[
                            dict[str, set[str] | None],
                            tuple[GuardFact, ...],
                        ]
                    ],
                ] = defaultdict(list)
                self._scan_private_helper_calls(
                    definition.module_info,
                    definition.info.node.body,
                    context,
                    set(definition.tag_guard_names),
                    forwarded_calls,
                )
                for forwarded_name, forwarded_invocations in self._exact_private_helper_invocations(
                    forwarded_calls
                ).items():
                    known[forwarded_name].update(forwarded_invocations)

        self._private_helper_invocations = {
            helper_name: tuple(
                sorted(
                    invocations,
                    key=lambda invocation: (
                        invocation.bindings,
                        invocation.guards,
                    ),
                )
            )
            for helper_name, invocations in known.items()
            if invocations
        }

    def _exact_private_helper_invocations(
        self,
        calls: dict[
            str,
            list[tuple[dict[str, set[str] | None], tuple[GuardFact, ...]]],
        ],
    ) -> defaultdict[str, set[PrivateHelperInvocation]]:
        invocations: defaultdict[str, set[PrivateHelperInvocation]] = defaultdict(set)
        for helper_name, helper_calls in calls.items():
            helper = self._private_helper_definitions[helper_name].info
            if not isinstance(
                helper.node,
                (ast.AsyncFunctionDef, ast.FunctionDef),
            ):
                continue
            parameters = self._callable_parameter_names(helper.node)
            reassigned = {
                parameter for parameter in parameters if self._parameter_is_reassigned(helper.node, parameter)
            }
            exact_calls = set()
            for arguments, guards in helper_calls:
                exact_bindings = []
                for parameter in parameters:
                    values = arguments.get(parameter)
                    if parameter not in reassigned and values is not None and len(values) == 1:
                        exact_bindings.append((parameter, next(iter(values))))
                if exact_bindings:
                    normalized_bindings = tuple(sorted(exact_bindings))
                    normalized_guards = tuple(sorted(set(guards)))
                    activation_payload = repr(
                        (
                            helper_name,
                            normalized_bindings,
                            tuple(
                                (guard.scope, guard.activation, guard.key, guard.polarity)
                                for guard in normalized_guards
                            ),
                        )
                    ).encode("utf-8")
                    exact_calls.add(
                        PrivateHelperInvocation(
                            bindings=normalized_bindings,
                            guards=normalized_guards,
                            activation=(f"{helper_name}:{hashlib.sha256(activation_payload).hexdigest()[:16]}"),
                        )
                    )
            invocations[helper_name].update(exact_calls)
        return invocations

    def _scan_flow_if(
        self,
        module_info: ModuleInfo,
        node: ast.If,
        context: PatchScanContext,
        tag_guard_names: set[str],
        scan_branch: Any,
    ) -> PatchFlowResult:
        """Shared path-sensitive ``if`` handling for helper and patch scans."""

        live_branches: list[PatchScanContext] = []
        exits: list[PatchFlowExit] = []
        for branch_statements, truth in (
            (node.body, True),
            (node.orelse, False),
        ):
            for branch in self._condition_contexts(
                module_info,
                node.test,
                context,
                tag_guard_names,
                truth=truth,
            ):
                result = scan_branch(branch_statements, branch)
                exits.extend(result.exits)
                if result.live:
                    live_branches.append(branch)
        if live_branches:
            context.merge(live_branches)
        return PatchFlowResult(live=bool(live_branches), exits=exits)

    def _scan_private_helper_calls(
        self,
        module_info: ModuleInfo,
        statements: Sequence[ast.stmt],
        context: PatchScanContext,
        tag_guard_names: set[str],
        calls: dict[
            str,
            list[tuple[dict[str, set[str] | None], tuple[GuardFact, ...]]],
        ],
    ) -> PatchFlowResult:
        exits: list[PatchFlowExit] = []
        for node in statements:
            if self._statement_may_raise(module_info, node, context):
                exits.append(PatchFlowExit("raise", context.clone()))
            for expression in self._statement_expressions(node):
                for call in _main_expression_calls(expression, tag_guard_names):
                    self._record_private_helper_call(
                        module_info,
                        call,
                        context,
                        calls,
                        tag_guard_names,
                    )

            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._update_import_bindings(module_info, node, context)
                continue
            if isinstance(node, ast.If):
                result = self._scan_flow_if(
                    module_info,
                    node,
                    context,
                    tag_guard_names,
                    lambda branch_statements, branch: self._scan_private_helper_calls(
                        module_info,
                        branch_statements,
                        branch,
                        tag_guard_names,
                        calls,
                    ),
                )
                exits.extend(result.exits)
                if not result.live:
                    return PatchFlowResult(live=False, exits=exits)
                continue
            if isinstance(node, ast.Try):
                result = self._scan_private_helper_try(
                    module_info,
                    node,
                    context,
                    tag_guard_names,
                    calls,
                )
                exits.extend(result.exits)
                if not result.live:
                    return PatchFlowResult(live=False, exits=exits)
                continue
            if isinstance(node, (ast.Break, ast.Continue, ast.Raise, ast.Return)):
                return PatchFlowResult(
                    live=False,
                    exits=[
                        *exits,
                        PatchFlowExit(
                            kind=type(node).__name__.lower(),
                            context=context.clone(),
                            exception_name=(
                                self._raised_exception_name(module_info, node, context)
                                if isinstance(node, ast.Raise)
                                else None
                            ),
                        ),
                    ],
                )
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified_name = f"{module_info.name}.{'.'.join((*context.scope, node.name))}"
                identity = self._private_helper_node_identities.get(id(node))
                bound_name = identity or qualified_name
                context.bindings[node.name] = {bound_name}
                context.binding_alternatives[node.name] = {bound_name}
                scope_identity = (
                    f"{module_info.name}:{getattr(node, 'lineno', 0)}:{getattr(node, 'col_offset', 0)}:{node.name}"
                )
                child = context.clone(
                    scope=(*context.scope, node.name),
                    guard_scope=scope_identity,
                    activation=scope_identity,
                )
                self._clear_function_parameter_bindings(node, child)
                if identity is not None:
                    definition = self._private_helper_definitions[identity]
                    if definition.entry_context is None:
                        definition.entry_context = child.clone()
                    else:
                        merged = definition.entry_context.clone()
                        merged.merge([definition.entry_context, child])
                        definition.entry_context = merged
                self._scan_private_helper_calls(
                    module_info,
                    node.body,
                    child,
                    tag_guard_names,
                    calls,
                )
                continue
            if isinstance(node, ast.ClassDef):
                qualified_name = f"{module_info.name}.{'.'.join((*context.scope, node.name))}"
                context.bindings[node.name] = {qualified_name}
                context.binding_alternatives[node.name] = {qualified_name}
                scope_identity = (
                    f"{module_info.name}:{getattr(node, 'lineno', 0)}:{getattr(node, 'col_offset', 0)}:{node.name}"
                )
                child = context.clone(
                    scope=(*context.scope, node.name),
                    guard_scope=scope_identity,
                    activation=scope_identity,
                )
                self._scan_private_helper_calls(
                    module_info,
                    node.body,
                    child,
                    tag_guard_names,
                    calls,
                )
                continue
            if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                branches = [context.clone()]
                branch = context.clone()
                self._scan_private_helper_calls(
                    module_info,
                    node.body,
                    branch,
                    tag_guard_names,
                    calls,
                )
                branches.append(branch)
                context.merge(branches)
                self._scan_private_helper_calls(
                    module_info,
                    node.orelse,
                    context,
                    tag_guard_names,
                    calls,
                )
                continue
            if isinstance(node, (ast.With, ast.AsyncWith)):
                suppressed_exception_names = self._suppress_exception_names(
                    module_info,
                    node,
                    context,
                )
                branch = context.clone()
                self._update_with_bindings(module_info, node, branch)
                result = self._scan_private_helper_calls(
                    module_info,
                    node.body,
                    branch,
                    tag_guard_names,
                    calls,
                )
                with_result = self._finish_with_flow(
                    context,
                    branch,
                    result,
                    suppressed_exception_names,
                )
                exits.extend(with_result.exits)
                if not with_result.live:
                    return PatchFlowResult(live=False, exits=exits)
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                self._update_owner_call_bindings(
                    module_info,
                    targets,
                    node.value,
                    context,
                )

        return PatchFlowResult(live=True, exits=exits)

    def _handler_exception_names(
        self,
        module_info: ModuleInfo,
        handler: ast.ExceptHandler,
        context: PatchScanContext,
    ) -> tuple[tuple[str, ...], bool] | None:
        """Return known caught exceptions and whether any member is unknown."""

        if handler.type is None:
            return None
        handler_nodes = handler.type.elts if isinstance(handler.type, ast.Tuple) else (handler.type,)
        resolved = tuple(
            self._canonical_exception_node(
                module_info,
                candidate,
                context,
            )
            for candidate in handler_nodes
        )
        return tuple(name for name in resolved if name is not None), any(name is None for name in resolved)

    def _canonical_exception_node(
        self,
        module_info: ModuleInfo,
        node: ast.AST | None,
        context: PatchScanContext,
    ) -> str | None:
        expression = _expression_name(node)
        if expression is None:
            return None
        root = expression.split(".", 1)[0]
        builtin_type = getattr(builtins, expression, None) if "." not in expression else None
        if (
            isinstance(builtin_type, type)
            and issubclass(builtin_type, BaseException)
            and root not in context.bindings
            and root not in context.unknown_bindings
            and root not in context.local_callables
            and root not in context.parameter_names
        ):
            return f"builtins.{expression}"
        references = self._resolve_patch_references(
            module_info,
            expression,
            context,
        )
        if len(references) != 1:
            return None
        return self._canonical_reference(next(iter(references)))

    def _raised_exception_name(
        self,
        module_info: ModuleInfo,
        node: ast.Raise,
        context: PatchScanContext,
    ) -> str | None:
        exception_node = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        return self._canonical_exception_node(
            module_info,
            exception_node,
            context,
        )

    def _exception_name_is_subclass(
        self,
        child_name: str,
        parent_name: str,
    ) -> bool:
        if child_name == parent_name:
            return True
        child_type = (
            getattr(builtins, child_name.removeprefix("builtins."), None)
            if child_name.startswith("builtins.")
            else None
        )
        parent_type = (
            getattr(builtins, parent_name.removeprefix("builtins."), None)
            if parent_name.startswith("builtins.")
            else None
        )
        if (
            isinstance(child_type, type)
            and isinstance(parent_type, type)
            and issubclass(child_type, BaseException)
            and issubclass(parent_type, BaseException)
        ):
            return issubclass(child_type, parent_type)

        pending = [child_name]
        seen: set[str] = set()
        while pending:
            candidate = pending.pop()
            if candidate in seen:
                continue
            seen.add(candidate)
            class_info = self._class_info(candidate)
            if class_info is None:
                continue
            for expression, resolved in zip(
                class_info.bases,
                class_info.resolved_bases,
            ):
                builtin_base = getattr(builtins, expression, None) if "." not in expression else None
                base = (
                    f"builtins.{expression}"
                    if isinstance(builtin_base, type) and issubclass(builtin_base, BaseException)
                    else self._canonical_reference(resolved)
                )
                if base == parent_name:
                    return True
                pending.append(base)
        return False

    def _handler_matches_raise(
        self,
        module_info: ModuleInfo,
        handler: ast.ExceptHandler,
        raised: PatchFlowExit,
    ) -> bool:
        handler_resolution = self._handler_exception_names(
            module_info,
            handler,
            raised.context,
        )
        if handler_resolution is None or raised.exception_name is None:
            return True
        handler_names, has_unknown = handler_resolution
        return has_unknown or any(
            self._exception_name_is_subclass(
                raised.exception_name,
                handler_name,
            )
            for handler_name in handler_names
        )

    def _suppress_exception_names(
        self,
        module_info: ModuleInfo,
        node: ast.With | ast.AsyncWith,
        context: PatchScanContext,
    ) -> tuple[str, ...] | None:
        """Return suppressed exceptions when every manager is statically known."""

        if not isinstance(node, ast.With):
            return None
        exception_names: list[str] = []
        for item in node.items:
            manager = item.context_expr
            if not isinstance(manager, ast.Call):
                return None
            function = _expression_name(manager.func)
            functions = (
                self._resolve_patch_references(
                    module_info,
                    function,
                    context,
                )
                if function
                else set()
            )
            if functions == {"contextlib.nullcontext"}:
                continue
            if functions != {"contextlib.suppress"} or manager.keywords:
                return None
            for argument in manager.args:
                exception_name = self._canonical_exception_node(
                    module_info,
                    argument,
                    context,
                )
                if exception_name is None:
                    return None
                exception_names.append(exception_name)
        return tuple(exception_names)

    def _finish_with_flow(
        self,
        context: PatchScanContext,
        body_context: PatchScanContext,
        result: PatchFlowResult,
        suppressed_exception_names: tuple[str, ...] | None,
    ) -> PatchFlowResult:
        """Merge normal and exactly suppressed exits from one ``with`` body."""

        live_contexts = [body_context] if result.live else []
        remaining_exits: list[PatchFlowExit] = []
        for flow_exit in result.exits:
            is_suppressed = (
                suppressed_exception_names is not None
                and flow_exit.kind == "raise"
                and flow_exit.exception_name is not None
                and any(
                    self._exception_name_is_subclass(
                        flow_exit.exception_name,
                        suppressed_name,
                    )
                    for suppressed_name in suppressed_exception_names
                )
            )
            if is_suppressed:
                live_contexts.append(flow_exit.context)
            else:
                remaining_exits.append(flow_exit)

        if live_contexts:
            context.merge(live_contexts)
        return PatchFlowResult(
            live=bool(live_contexts),
            exits=remaining_exits,
        )

    def _handler_covers_handler(
        self,
        module_info: ModuleInfo,
        earlier: ast.ExceptHandler,
        later: ast.ExceptHandler,
        context: PatchScanContext,
    ) -> bool:
        """Whether every exception caught by ``later`` was caught earlier."""

        earlier_resolution = self._handler_exception_names(module_info, earlier, context)
        if earlier_resolution is None:
            return True
        earlier_names, _ = earlier_resolution
        later_resolution = self._handler_exception_names(module_info, later, context)
        if later_resolution is None:
            return any(
                self._exception_name_is_subclass(
                    "builtins.BaseException",
                    earlier_name,
                )
                for earlier_name in earlier_names
            )
        later_names, later_has_unknown = later_resolution
        if later_has_unknown:
            return False
        if not earlier_names or not later_names:
            return False
        return all(
            any(
                self._exception_name_is_subclass(
                    later_name,
                    earlier_name,
                )
                for earlier_name in earlier_names
            )
            for later_name in later_names
        )

    def _handler_catches_all_implicit_exceptions(
        self,
        module_info: ModuleInfo,
        handler: ast.ExceptHandler,
        context: PatchScanContext,
    ) -> bool:
        handler_resolution = self._handler_exception_names(module_info, handler, context)
        if handler_resolution is None:
            return True
        handler_names, _ = handler_resolution
        return any(
            self._exception_name_is_subclass(
                "builtins.Exception",
                handler_name,
            )
            for handler_name in handler_names
        )

    def _route_try_handlers(
        self,
        module_info: ModuleInfo,
        context: PatchScanContext,
        handlers: Sequence[ast.ExceptHandler],
        raised_exits: Sequence[PatchFlowExit],
        scan_handler: Any,
    ) -> tuple[list[PatchFlowExit], list[PatchFlowExit]]:
        """Route exact and implicit raises through ordered ``except`` arms."""

        outcomes: list[PatchFlowExit] = []
        remaining_exact = [raised for raised in raised_exits if raised.exception_name is not None]
        implicit = [raised for raised in raised_exits if raised.exception_name is None]
        previous_handlers: list[ast.ExceptHandler] = []
        implicit_consumed = False

        for handler in handlers:
            exact_sources = [
                raised for raised in remaining_exact if self._handler_matches_raise(module_info, handler, raised)
            ]
            shadowed = any(
                self._handler_covers_handler(
                    module_info,
                    previous,
                    handler,
                    context,
                )
                for previous in previous_handlers
            )
            implicit_sources = [] if implicit_consumed or shadowed else implicit
            for source in (*exact_sources, *implicit_sources):
                exception_name = (
                    " ".join(ast.unparse(handler.type).split()) if handler.type is not None else "Exception"
                )
                source_guards = tuple(guard for guard in source.context.guards if guard.text != "try-success")
                handler_guards = self._merge_guard_paths(
                    source_guards,
                    (f"except {exception_name}",),
                )
                if handler_guards is None:
                    continue
                branch = source.context.clone(guards=handler_guards)
                handler_result = scan_handler(handler, branch)
                outcomes.extend(handler_result.exits)
                if handler_result.live:
                    outcomes.append(PatchFlowExit("live", branch))

            remaining_exact = [raised for raised in remaining_exact if raised not in exact_sources]
            previous_handlers.append(handler)
            if self._handler_catches_all_implicit_exceptions(
                module_info,
                handler,
                context,
            ):
                implicit_consumed = True

        unhandled = list(remaining_exact)
        if not implicit_consumed:
            unhandled.extend(implicit)
        return outcomes, unhandled

    def _scan_private_helper_try(
        self,
        module_info: ModuleInfo,
        node: ast.Try,
        context: PatchScanContext,
        tag_guard_names: set[str],
        calls: dict[
            str,
            list[tuple[dict[str, set[str] | None], tuple[GuardFact, ...]]],
        ],
    ) -> PatchFlowResult:
        outcomes: list[PatchFlowExit] = []
        raised_exits: list[PatchFlowExit] = []
        success_guards = self._merge_guard_paths(
            context.guards,
            ("try-success",),
        )
        if success_guards is not None:
            success = context.clone(guards=success_guards)
            body_result = self._scan_private_helper_calls(
                module_info,
                node.body,
                success,
                tag_guard_names,
                calls,
            )
            raised_exits = [outcome for outcome in body_result.exits if outcome.kind == "raise"]
            outcomes.extend(outcome for outcome in body_result.exits if outcome.kind != "raise")
            if body_result.live:
                else_result = self._scan_private_helper_calls(
                    module_info,
                    node.orelse,
                    success,
                    tag_guard_names,
                    calls,
                )
                outcomes.extend(else_result.exits)
                if else_result.live:
                    outcomes.append(PatchFlowExit("live", success))

        handler_outcomes, remaining_raises = self._route_try_handlers(
            module_info,
            context,
            node.handlers,
            raised_exits,
            lambda handler, branch: self._scan_private_helper_calls(
                module_info,
                handler.body,
                branch,
                tag_guard_names,
                calls,
            ),
        )
        outcomes.extend(handler_outcomes)
        outcomes.extend(remaining_raises)

        live_contexts: list[PatchScanContext] = []
        exits: list[PatchFlowExit] = []
        for outcome in outcomes:
            # ``finally`` is unconditional. Keep each branch's value state,
            # but do not leak synthetic try/except labels into its evidence.
            final_context = outcome.context.clone(guards=context.guards)
            final_result = self._scan_private_helper_calls(
                module_info,
                node.finalbody,
                final_context,
                tag_guard_names,
                calls,
            )
            exits.extend(final_result.exits)
            if not final_result.live:
                continue
            if outcome.kind == "live":
                live_contexts.append(final_context)
            else:
                exits.append(
                    PatchFlowExit(
                        outcome.kind,
                        final_context,
                        exception_name=outcome.exception_name,
                    )
                )

        if live_contexts:
            context.merge(live_contexts)
        return PatchFlowResult(live=bool(live_contexts), exits=exits)

    def _record_private_helper_call(
        self,
        module_info: ModuleInfo,
        call: ast.Call,
        context: PatchScanContext,
        calls: dict[
            str,
            list[tuple[dict[str, set[str] | None], tuple[GuardFact, ...]]],
        ],
        tag_guard_names: set[str],
    ) -> None:
        expression = _expression_name(call.func)
        if expression is None:
            return
        local_name = expression.rsplit(".", 1)[-1]
        root_name = expression.split(".", 1)[0]
        imported_private_helper = any(
            candidate in self._private_helper_definitions
            or self.downstream.canonical_name(candidate) in self._private_helper_exports
            for candidate in context.bindings.get(root_name, ())
        )
        if not local_name.startswith("_") and not imported_private_helper:
            return
        references = self._resolve_patch_references(
            module_info,
            expression,
            context,
        )
        candidates = sorted(
            {
                identity
                for reference in references
                for identity in (
                    (
                        reference
                        if reference in self._private_helper_definitions
                        else self._private_helper_exports.get(self.downstream.canonical_name(reference))
                    ),
                )
                if identity is not None
            }
        )
        if not candidates:
            return
        for helper_name in candidates:
            helper = self._private_helper_definitions[helper_name].info
            if len(references) != 1 or not isinstance(
                helper.node,
                (ast.AsyncFunctionDef, ast.FunctionDef),
            ):
                calls[helper_name].append(({}, context.guards))
                continue
            calls[helper_name].extend(
                self._bound_owner_arguments(
                    module_info,
                    helper.node,
                    call,
                    context,
                    tag_guard_names,
                )
            )

    def _update_owner_call_bindings(
        self,
        module_info: ModuleInfo,
        targets: Sequence[ast.AST],
        value: ast.AST | None,
        context: PatchScanContext,
    ) -> None:
        expression = _expression_name(value)
        references = (
            self._resolve_patch_references(
                module_info,
                expression,
                context,
            )
            if expression
            else set()
        )
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if references:
                context.bind_exact(target.id, references)
            elif isinstance(value, ast.Constant) and value.value is None:
                context.bind_none(target.id)
            else:
                context.bind_unknown(target.id)

    def _bound_owner_arguments(
        self,
        module_info: ModuleInfo,
        function: ast.AsyncFunctionDef | ast.FunctionDef,
        call: ast.Call,
        context: PatchScanContext,
        tag_guard_names: set[str],
    ) -> list[tuple[dict[str, set[str] | None], tuple[GuardFact, ...]]]:
        positional = [*function.args.posonlyargs, *function.args.args]
        explicit_keywords = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg is not None}
        has_starred = any(isinstance(argument, ast.Starred) for argument in call.args)
        has_kwargs = any(keyword.arg is None for keyword in call.keywords)
        actuals: list[tuple[str, ast.AST | None, bool]] = []
        for index, parameter in enumerate(positional):
            actual = explicit_keywords.get(parameter.arg)
            if actual is None and index < len(call.args) and not has_starred:
                actual = call.args[index]
            actuals.append(
                (
                    parameter.arg,
                    actual,
                    actual is None and (has_starred or has_kwargs),
                )
            )
        for parameter in function.args.kwonlyargs:
            actual = explicit_keywords.get(parameter.arg)
            actuals.append((parameter.arg, actual, actual is None and has_kwargs))

        contexts: list[tuple[dict[str, set[str] | None], tuple[GuardFact, ...]]] = [({}, context.guards)]
        for parameter, actual, forced_unknown in actuals:
            alternatives = (
                None
                if forced_unknown
                else self._owner_argument_alternatives(
                    module_info,
                    actual,
                    context,
                    tag_guard_names,
                )
            )
            if alternatives is None:
                for arguments, _guards in contexts:
                    arguments[parameter] = None
                continue

            expanded = []
            for arguments, guards in contexts:
                for target, alternative_guards in alternatives:
                    merged_guards = self._merge_guard_paths(
                        guards,
                        alternative_guards,
                    )
                    if merged_guards is None:
                        continue
                    expanded.append(
                        (
                            {**arguments, parameter: {target}},
                            merged_guards,
                        )
                    )
            contexts = expanded
        return contexts

    def _owner_argument_alternatives(
        self,
        module_info: ModuleInfo,
        node: ast.AST | None,
        context: PatchScanContext,
        tag_guard_names: set[str],
    ) -> tuple[tuple[str, tuple[GuardFact, ...]], ...] | None:
        alternatives = self._static_value_alternatives(
            module_info,
            node,
            context,
            tag_guard_names,
        )
        if alternatives is None or any(alternative.target is None for alternative in alternatives):
            return None
        return tuple(
            sorted(
                {
                    (alternative.target, alternative.guards)
                    for alternative in alternatives
                    if alternative.target is not None
                }
            )
        )

    def _patch_condition_value(
        self,
        module_info: ModuleInfo,
        node: ast.AST,
        context: PatchScanContext,
        tag_guard_names: set[str],
    ) -> bool | None:
        main_value = _main_condition_value(node, tag_guard_names)
        if main_value is not None:
            return main_value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = self._patch_condition_value(
                module_info,
                node.operand,
                context,
                tag_guard_names,
            )
            return None if value is None else not value
        if isinstance(node, ast.BoolOp):
            values = [
                self._patch_condition_value(
                    module_info,
                    value,
                    context,
                    tag_guard_names,
                )
                for value in node.values
            ]
            if isinstance(node.op, ast.And):
                if False in values:
                    return False
                return True if all(value is True for value in values) else None
            if isinstance(node.op, ast.Or):
                if True in values:
                    return True
                return False if all(value is False for value in values) else None
        if not (
            isinstance(node, ast.Call)
            and _expression_name(node.func) == "hasattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            return None
        owner_expression = _expression_name(node.args[0])
        if owner_expression is None:
            return None
        owners = {
            self._canonical_reference(owner)
            for owner in self._resolve_patch_references(
                module_info,
                owner_expression,
                context,
            )
            if owner == "vllm" or owner.startswith("vllm.")
        }
        if len(owners) != 1:
            return None
        owner = next(iter(owners))
        member = node.args[1].value
        return True if self._upstream_member_is_proven(owner, member) else None

    def _condition_contexts(
        self,
        module_info: ModuleInfo,
        node: ast.AST,
        context: PatchScanContext,
        tag_guard_names: set[str],
        *,
        truth: bool,
    ) -> list[PatchScanContext]:
        """Return the feasible, narrowed contexts for one condition outcome."""

        exact = self._patch_condition_value(
            module_info,
            node,
            context,
            tag_guard_names,
        )
        if exact is not None:
            return [context.clone()] if exact is truth else []

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self._condition_contexts(
                module_info,
                node.operand,
                context,
                tag_guard_names,
                truth=not truth,
            )

        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                if truth:
                    contexts = [context.clone()]
                    for value in node.values:
                        contexts = [
                            refined
                            for candidate in contexts
                            for refined in self._condition_contexts(
                                module_info,
                                value,
                                candidate,
                                tag_guard_names,
                                truth=True,
                            )
                        ]
                    return contexts
                results: list[PatchScanContext] = []
                prefixes = [context.clone()]
                for value in node.values:
                    results.extend(
                        refined
                        for candidate in prefixes
                        for refined in self._condition_contexts(
                            module_info,
                            value,
                            candidate,
                            tag_guard_names,
                            truth=False,
                        )
                    )
                    prefixes = [
                        refined
                        for candidate in prefixes
                        for refined in self._condition_contexts(
                            module_info,
                            value,
                            candidate,
                            tag_guard_names,
                            truth=True,
                        )
                    ]
                return results

            if truth:
                results = []
                prefixes = [context.clone()]
                for value in node.values:
                    results.extend(
                        refined
                        for candidate in prefixes
                        for refined in self._condition_contexts(
                            module_info,
                            value,
                            candidate,
                            tag_guard_names,
                            truth=True,
                        )
                    )
                    prefixes = [
                        refined
                        for candidate in prefixes
                        for refined in self._condition_contexts(
                            module_info,
                            value,
                            candidate,
                            tag_guard_names,
                            truth=False,
                        )
                    ]
                return results
            contexts = [context.clone()]
            for value in node.values:
                contexts = [
                    refined
                    for candidate in contexts
                    for refined in self._condition_contexts(
                        module_info,
                        value,
                        candidate,
                        tag_guard_names,
                        truth=False,
                    )
                ]
            return contexts

        branch = context.clone()
        if not self._refine_none_guard(branch, node, truth=truth):
            return []
        guard = self._guard_fact(
            branch,
            node,
            truth=truth,
            module_info=module_info,
        )
        guards = self._merge_guard_paths(branch.guards, (guard,))
        if guards is None:
            return []
        branch.guards = guards
        return [branch]

    def _refine_none_guard(
        self,
        context: PatchScanContext,
        node: ast.AST,
        *,
        truth: bool,
    ) -> bool:
        """Narrow an exact-value/None join; return False for an impossible path."""

        while isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            truth = not truth
            node = node.operand
        none_check = _none_comparison(node)
        if none_check is None:
            return True
        subject, test_means_non_none = none_check
        if not isinstance(subject, ast.Name):
            return True
        alternatives = context.binding_alternatives.get(subject.id)
        if alternatives is None:
            return True
        means_non_none = test_means_non_none if truth else not test_means_non_none
        selected = {target for target in alternatives if (target is not None) == means_non_none}
        if not selected:
            return False
        exact = {target for target in selected if target is not None}
        if exact:
            context.bind_exact(subject.id, exact)
        else:
            context.bind_none(subject.id)
        return True

    def _upstream_member_is_proven(
        self,
        owner: str,
        member: str,
    ) -> bool:
        owner = self._canonical_reference(owner)
        if owner in self.upstream.modules:
            # A child module existing on disk does not make it an attribute of
            # the package object.  Require a direct, unconditional export.
            return f"{owner}.{member}" in self.upstream.unconditional_exports
        target = self._canonical_reference(f"{owner}.{member}")
        if target in self.upstream.unconditional_symbols:
            return True
        owner_info = self.upstream.find_class(owner)
        if owner_info is None:
            return False
        mro_result = self._linearized_mro(owner_info.qualified_name)
        if not mro_result.complete:
            return False
        return any(
            self._canonical_reference(f"{candidate}.{member}") in self.upstream.unconditional_symbols
            for candidate in mro_result.owners[1:]
        )

    def _static_value_alternatives(
        self,
        module_info: ModuleInfo,
        node: ast.AST | None,
        context: PatchScanContext,
        tag_guard_names: set[str],
    ) -> tuple[StaticValueAlternative, ...] | None:
        if node is None:
            return None

        if isinstance(node, ast.IfExp):
            condition = self._patch_condition_value(
                module_info,
                node.test,
                context,
                tag_guard_names,
            )
            if condition is not None:
                selected = node.body if condition else node.orelse
                return self._static_value_alternatives(
                    module_info,
                    selected,
                    context,
                    tag_guard_names,
                )
            body = self._static_value_alternatives(
                module_info,
                node.body,
                context,
                tag_guard_names,
            )
            otherwise = self._static_value_alternatives(
                module_info,
                node.orelse,
                context,
                tag_guard_names,
            )
            if body is None or otherwise is None:
                return None
            guard = self._guard_fact(
                context,
                node.test,
                module_info=module_info,
            )
            opposite = self._guard_fact(
                context,
                node.test,
                truth=False,
                module_info=module_info,
            )
            return self._guarded_value_alternatives(
                body,
                guard,
            ) + self._guarded_value_alternatives(
                otherwise,
                opposite,
            )

        if isinstance(node, ast.BoolOp):
            alternatives = self._static_value_alternatives(
                module_info,
                node.values[0],
                context,
                tag_guard_names,
            )
            if alternatives is None:
                return None
            for value in node.values[1:]:
                next_alternatives = self._static_value_alternatives(
                    module_info,
                    value,
                    context,
                    tag_guard_names,
                )
                if next_alternatives is None:
                    return None
                combined = []
                for alternative in alternatives:
                    short_circuits = (isinstance(node.op, ast.And) and not alternative.truth) or (
                        isinstance(node.op, ast.Or) and alternative.truth
                    )
                    if short_circuits:
                        combined.append(alternative)
                        continue
                    for next_alternative in next_alternatives:
                        guards = self._merge_guard_paths(
                            alternative.guards,
                            next_alternative.guards,
                        )
                        if guards is None:
                            continue
                        combined.append(
                            StaticValueAlternative(
                                target=next_alternative.target,
                                truth=next_alternative.truth,
                                guards=guards,
                            )
                        )
                alternatives = tuple(set(combined))
            return alternatives

        expression = _expression_name(node)
        if expression is not None:
            references = self._resolve_patch_references(
                module_info,
                expression,
                context,
            )
            candidates = {
                reference for reference in references if (reference == "vllm" or reference.startswith("vllm."))
            }
            if len(candidates) == 1:
                return (
                    StaticValueAlternative(
                        target=next(iter(candidates)),
                        truth=True,
                    ),
                )
            if len(candidates) > 1:
                return None

        condition = self._patch_condition_value(
            module_info,
            node,
            context,
            tag_guard_names,
        )
        if condition is not None:
            return (StaticValueAlternative(target=None, truth=condition),)
        guard = self._guard_fact(
            context,
            node,
            module_info=module_info,
        )
        opposite = self._guard_fact(
            context,
            node,
            truth=False,
            module_info=module_info,
        )
        return (
            StaticValueAlternative(
                target=None,
                truth=True,
                guards=(guard,),
            ),
            StaticValueAlternative(
                target=None,
                truth=False,
                guards=(opposite,),
            ),
        )

    def _guarded_value_alternatives(
        self,
        alternatives: Sequence[StaticValueAlternative],
        guard: GuardFact,
    ) -> tuple[StaticValueAlternative, ...]:
        guarded = []
        for alternative in alternatives:
            guards = self._merge_guard_paths(alternative.guards, (guard,))
            if guards is None:
                continue
            guarded.append(replace(alternative, guards=guards))
        return tuple(guarded)

    def _guard_fact(
        self,
        context: PatchScanContext,
        node: ast.AST,
        *,
        truth: bool = True,
        module_info: ModuleInfo | None = None,
    ) -> GuardFact:
        key, polarity, text = _canonical_guard(node, truth=truth)
        hasattr_target = self._hasattr_guard_target(module_info, node, context) if module_info is not None else None
        return GuardFact(
            scope=context.guard_scope,
            activation=context.activation,
            key=key,
            polarity=polarity,
            text=text,
            hasattr_target=hasattr_target,
        )

    def _opaque_guard(
        self,
        context: PatchScanContext,
        text: str,
    ) -> GuardFact:
        return GuardFact(
            scope=context.guard_scope,
            activation=context.activation,
            key=f"opaque:{text}",
            polarity=True,
            text=text,
        )

    def _hasattr_guard_target(
        self,
        module_info: ModuleInfo,
        node: ast.AST,
        context: PatchScanContext,
    ) -> tuple[str, str] | None:
        while isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            node = node.operand
        if not (
            isinstance(node, ast.Call)
            and _expression_name(node.func) == "hasattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            return None
        owner_expression = _expression_name(node.args[0])
        if owner_expression is None:
            return None
        owners = {
            self._canonical_reference(owner)
            for owner in self._resolve_patch_references(
                module_info,
                owner_expression,
                context,
            )
            if owner == "vllm" or owner.startswith("vllm.")
        }
        if len(owners) != 1:
            return None
        return next(iter(owners)), node.args[1].value

    def _merge_guard_paths(
        self,
        *paths: Sequence[GuardFact],
    ) -> tuple[GuardFact, ...] | None:
        predicates: dict[tuple[str, str, str], GuardFact] = {}
        for guard in (guard for path in paths for guard in path):
            if isinstance(guard, str):
                key, polarity, text = _canonical_guard_text(guard)
                guard = GuardFact(
                    scope="<flow>",
                    activation="<flow>",
                    key=key,
                    polarity=polarity,
                    text=text,
                )
            identity = (guard.scope, guard.activation, guard.key)
            previous = predicates.get(identity)
            if previous is not None and previous.polarity != guard.polarity:
                return None
            predicates[identity] = guard
        return tuple(sorted(predicates.values()))

    def _statement_expressions(
        self,
        node: ast.stmt,
    ) -> tuple[ast.AST | None, ...]:
        if isinstance(node, ast.Expr):
            return (node.value,)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            return (node.value,)
        if isinstance(node, (ast.If, ast.While)):
            return (node.test,)
        if isinstance(node, (ast.For, ast.AsyncFor)):
            return (node.iter,)
        if isinstance(node, ast.Return):
            return (node.value,)
        if isinstance(node, ast.Raise):
            return (node.exc,)
        if isinstance(node, ast.Assert):
            return (node.test, node.msg)
        if isinstance(node, (ast.With, ast.AsyncWith)):
            return tuple(item.context_expr for item in node.items)
        return ()

    def _proven_safe_literal_call(
        self,
        module_info: ModuleInfo,
        node: ast.Call,
        context: PatchScanContext,
    ) -> bool:
        """Recognize a small builtin call whose literal input cannot raise."""
        if not (
            isinstance(node.func, ast.Name) and node.func.id == "len" and len(node.args) == 1 and not node.keywords
        ):
            return False
        name = node.func.id
        if (
            name in context.bindings
            or name in context.unknown_bindings
            or name in context.local_callables
            or name in context.parameter_names
        ):
            return False
        try:
            value = ast.literal_eval(node.args[0])
        except (TypeError, ValueError):
            return False
        return isinstance(value, (bytes, dict, list, set, str, tuple))

    def _statement_may_raise(
        self,
        module_info: ModuleInfo,
        node: ast.stmt,
        context: PatchScanContext,
    ) -> bool:
        """Whether evaluating one statement can implicitly raise before it commits."""
        if isinstance(node, ast.Raise):
            return False
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return True
        for expression in self._statement_expressions(node):
            if expression is None:
                continue
            for candidate in ast.walk(expression):
                if isinstance(candidate, (ast.Await, ast.Subscript, ast.YieldFrom)):
                    return True
                if isinstance(candidate, ast.Call) and not self._proven_safe_literal_call(
                    module_info,
                    candidate,
                    context,
                ):
                    return True
        return False

    def _update_with_bindings(
        self,
        module_info: ModuleInfo,
        node: ast.With | ast.AsyncWith,
        context: PatchScanContext,
    ) -> None:
        """Bind ``with ... as name`` without reusing a stale imported owner."""
        for item in node.items:
            target = item.optional_vars
            if target is None:
                continue
            names = [candidate.id for candidate in ast.walk(target) if isinstance(candidate, ast.Name)]
            manager = item.context_expr
            references: set[str] = set()
            known_none = False
            if isinstance(manager, ast.Call):
                function = _expression_name(manager.func)
                function_targets = self._resolve_patch_references(module_info, function, context) if function else set()
                if function_targets == {"contextlib.nullcontext"}:
                    enter_value = (
                        manager.args[0]
                        if manager.args
                        else next(
                            (keyword.value for keyword in manager.keywords if keyword.arg == "enter_result"),
                            None,
                        )
                    )
                    if enter_value is None or (isinstance(enter_value, ast.Constant) and enter_value.value is None):
                        known_none = True
                    elif expression := _expression_name(enter_value):
                        references = self._resolve_patch_references(
                            module_info,
                            expression,
                            context,
                        )
            for name in names:
                if len(names) == 1 and references:
                    context.bind_exact(name, references)
                elif len(names) == 1 and known_none:
                    context.bind_none(name)
                else:
                    context.bind_unknown(name)

    def _callable_parameter_names(
        self,
        node: ast.AsyncFunctionDef | ast.FunctionDef,
    ) -> tuple[str, ...]:
        parameters = [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]
        if node.args.vararg is not None:
            parameters.append(node.args.vararg)
        if node.args.kwarg is not None:
            parameters.append(node.args.kwarg)
        return tuple(argument.arg for argument in parameters)

    def _clear_function_parameter_bindings(
        self,
        node: ast.AsyncFunctionDef | ast.FunctionDef,
        context: PatchScanContext,
    ) -> None:
        """Remove outer values shadowed by any lexical function local."""
        for parameter in _function_local_names(node):
            # An empty lexical binding is a tombstone. Without it, resolution
            # falls back to the module import index and can reuse a shadowed
            # ``import ... as <parameter>`` value.
            context.bindings[parameter] = set()
            context.binding_alternatives.pop(parameter, None)
            context.strings.pop(parameter, None)
            context.local_callables.pop(parameter, None)
            context.runtime_modules.pop(parameter, None)
            context.unknown_bindings.discard(parameter)
            context.upstream_binding_provenance.pop(parameter, None)
            context.upstream_binding_history.discard(parameter)
            context.parameter_names.add(parameter)

    def _parameter_is_reassigned(
        self,
        node: ast.AsyncFunctionDef | ast.FunctionDef,
        parameter: str,
    ) -> bool:
        return any(
            isinstance(child, ast.Name) and child.id == parameter and isinstance(child.ctx, (ast.Del, ast.Store))
            for child in _function_scope_nodes(node)
        )

    def _collect_monkey_patches(self) -> None:
        self._prepare_private_helper_parameter_bindings()
        for module_info in self.downstream.modules.values():
            context = PatchScanContext(
                guard_scope=module_info.name,
                activation=module_info.name,
            )
            self._scan_patch_statements(
                module_info,
                module_info.tree.body,
                context,
                _tag_guard_names(module_info.tree.body),
            )

    def _scan_patch_statements(
        self,
        module_info: ModuleInfo,
        statements: Sequence[ast.stmt],
        context: PatchScanContext,
        tag_guard_names: set[str],
    ) -> PatchFlowResult:
        exits: list[PatchFlowExit] = []
        for node in statements:
            if self._statement_may_raise(module_info, node, context):
                exits.append(PatchFlowExit("raise", context.clone()))
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._update_import_bindings(module_info, node, context)
                continue

            if isinstance(node, ast.If):
                result = self._scan_patch_if(
                    module_info,
                    node,
                    context,
                    tag_guard_names,
                )
                exits.extend(result.exits)
                if not result.live:
                    return PatchFlowResult(live=False, exits=exits)
                continue

            if isinstance(node, ast.Try):
                result = self._scan_patch_try(
                    module_info,
                    node,
                    context,
                    tag_guard_names,
                )
                exits.extend(result.exits)
                if not result.live:
                    return PatchFlowResult(live=False, exits=exits)
                continue

            if isinstance(node, (ast.Break, ast.Continue, ast.Raise, ast.Return)):
                return PatchFlowResult(
                    live=False,
                    exits=[
                        *exits,
                        PatchFlowExit(
                            kind=type(node).__name__.lower(),
                            context=context.clone(),
                            exception_name=(
                                self._raised_exception_name(module_info, node, context)
                                if isinstance(node, ast.Raise)
                                else None
                            ),
                        ),
                    ],
                )

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                callable_info = CallableInfo(
                    qualified_name=(f"{module_info.name}.{'.'.join((*context.scope, node.name))}"),
                    module=module_info.name,
                    file=module_info.file,
                    owner=None,
                    name=node.name,
                    node=node,
                )
                context.bindings.pop(node.name, None)
                context.binding_alternatives.pop(node.name, None)
                context.local_callables[node.name] = [callable_info]
                scope_identity = (
                    f"{module_info.name}:{getattr(node, 'lineno', 0)}:{getattr(node, 'col_offset', 0)}:{node.name}"
                )
                base_child = context.clone(
                    scope=(*context.scope, node.name),
                    guard_scope=scope_identity,
                    activation=scope_identity,
                )
                self._clear_function_parameter_bindings(node, base_child)
                helper_name = self._private_helper_node_identities.get(id(node))
                invocations = self._private_helper_invocations.get(helper_name)
                if invocations:
                    for invocation in invocations:
                        guards = self._merge_guard_paths(
                            base_child.guards,
                            invocation.guards,
                        )
                        if guards is None:
                            continue
                        child = base_child.clone(
                            guards=guards,
                            activation=invocation.activation,
                        )
                        for parameter, target in invocation.bindings:
                            child.bindings[parameter] = {target}
                            child.binding_alternatives[parameter] = {target}
                        self._scan_patch_statements(
                            module_info,
                            node.body,
                            child,
                            tag_guard_names,
                        )
                else:
                    self._scan_patch_statements(
                        module_info,
                        node.body,
                        base_child,
                        tag_guard_names,
                    )
                continue

            if isinstance(node, ast.ClassDef):
                qualified_name = f"{module_info.name}.{node.name}"
                context.bindings[node.name] = {qualified_name}
                context.binding_alternatives[node.name] = {qualified_name}
                context.local_callables[node.name] = [
                    CallableInfo(
                        qualified_name=qualified_name,
                        module=module_info.name,
                        file=module_info.file,
                        owner=None,
                        name=node.name,
                        node=node,
                    )
                ]
                scope_identity = (
                    f"{module_info.name}:{getattr(node, 'lineno', 0)}:{getattr(node, 'col_offset', 0)}:{node.name}"
                )
                child = context.clone(
                    scope=(*context.scope, node.name),
                    guard_scope=scope_identity,
                    activation=scope_identity,
                )
                self._scan_patch_statements(
                    module_info,
                    node.body,
                    child,
                    tag_guard_names,
                )
                continue

            if isinstance(node, (ast.For, ast.AsyncFor)):
                self._scan_patch_for(
                    module_info,
                    node,
                    context,
                    tag_guard_names,
                )
                continue

            if isinstance(node, ast.While):
                guard = self._guard_fact(
                    context,
                    node.test,
                    module_info=module_info,
                )
                body = context.clone(
                    guards=self._merge_guard_paths(context.guards, (guard,)) or context.guards,
                )
                self._scan_patch_statements(
                    module_info,
                    node.body,
                    body,
                    tag_guard_names,
                )
                empty = context.clone()
                context.merge([body, empty])
                if node.orelse:
                    self._scan_patch_statements(
                        module_info,
                        node.orelse,
                        context,
                        tag_guard_names,
                    )
                continue

            if isinstance(node, (ast.With, ast.AsyncWith)):
                suppressed_exception_names = self._suppress_exception_names(
                    module_info,
                    node,
                    context,
                )
                if suppressed_exception_names is None:
                    with_guard = self._opaque_guard(context, "with-context")
                    child = context.clone(
                        guards=self._merge_guard_paths(
                            context.guards,
                            (with_guard,),
                        )
                        or context.guards,
                    )
                else:
                    child = context.clone()
                self._update_with_bindings(module_info, node, child)
                result = self._scan_patch_statements(
                    module_info,
                    node.body,
                    child,
                    tag_guard_names,
                )
                with_result = self._finish_with_flow(
                    context,
                    child,
                    result,
                    suppressed_exception_names,
                )
                exits.extend(with_result.exits)
                if not with_result.live:
                    return PatchFlowResult(live=False, exits=exits)
                continue

            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Attribute):
                        self._record_patch_node(
                            module_info,
                            target,
                            value,
                            context,
                            getattr(node, "lineno", 0),
                        )
                self._update_assignment_bindings(
                    module_info,
                    targets,
                    value,
                    context,
                )
                continue

            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and _expression_name(node.value.func) == "setattr"
                and len(node.value.args) >= 3
            ):
                self._record_setattr_patch(
                    module_info,
                    node.value,
                    context,
                    getattr(node, "lineno", 0),
                )

        return PatchFlowResult(live=True, exits=exits)

    def _scan_patch_if(
        self,
        module_info: ModuleInfo,
        node: ast.If,
        context: PatchScanContext,
        tag_guard_names: set[str],
    ) -> PatchFlowResult:
        return self._scan_flow_if(
            module_info,
            node,
            context,
            tag_guard_names,
            lambda branch_statements, branch: self._scan_patch_statements(
                module_info,
                branch_statements,
                branch,
                tag_guard_names,
            ),
        )

    def _scan_patch_try(
        self,
        module_info: ModuleInfo,
        node: ast.Try,
        context: PatchScanContext,
        tag_guard_names: set[str],
    ) -> PatchFlowResult:
        outcomes: list[PatchFlowExit] = []
        raised_exits: list[PatchFlowExit] = []
        success_guards = self._merge_guard_paths(
            context.guards,
            ("try-success",),
        )
        if success_guards is not None:
            success = context.clone(guards=success_guards)
            body_result = self._scan_patch_statements(
                module_info,
                node.body,
                success,
                tag_guard_names,
            )
            raised_exits = [outcome for outcome in body_result.exits if outcome.kind == "raise"]
            outcomes.extend(outcome for outcome in body_result.exits if outcome.kind != "raise")
            if body_result.live:
                else_result = self._scan_patch_statements(
                    module_info,
                    node.orelse,
                    success,
                    tag_guard_names,
                )
                outcomes.extend(else_result.exits)
                if else_result.live:
                    outcomes.append(PatchFlowExit("live", success))

        handler_outcomes, remaining_raises = self._route_try_handlers(
            module_info,
            context,
            node.handlers,
            raised_exits,
            lambda handler, branch: self._scan_patch_statements(
                module_info,
                handler.body,
                branch,
                tag_guard_names,
            ),
        )
        outcomes.extend(handler_outcomes)
        outcomes.extend(remaining_raises)

        live_contexts: list[PatchScanContext] = []
        exits: list[PatchFlowExit] = []
        for outcome in outcomes:
            # ``finally`` is unconditional. Keep each branch's value state,
            # but do not leak synthetic try/except labels into its evidence.
            final_context = outcome.context.clone(guards=context.guards)
            final_result = self._scan_patch_statements(
                module_info,
                node.finalbody,
                final_context,
                tag_guard_names,
            )
            exits.extend(final_result.exits)
            if not final_result.live:
                continue
            if outcome.kind == "live":
                live_contexts.append(final_context)
            else:
                exits.append(
                    PatchFlowExit(
                        outcome.kind,
                        final_context,
                        exception_name=outcome.exception_name,
                    )
                )

        if live_contexts:
            context.merge(live_contexts)
        return PatchFlowResult(live=bool(live_contexts), exits=exits)

    def _scan_patch_for(
        self,
        module_info: ModuleInfo,
        node: ast.For | ast.AsyncFor,
        context: PatchScanContext,
        tag_guard_names: set[str],
    ) -> None:
        values = self._string_values(node.iter, context)
        branches = [context.clone()]
        if isinstance(node.target, ast.Name) and values:
            for value in sorted(values):
                branch = context.clone(
                    guards=self._merge_guard_paths(
                        context.guards,
                        (
                            self._opaque_guard(
                                context,
                                f"for {node.target.id}={value!r}",
                            ),
                        ),
                    )
                    or context.guards,
                )
                branch.strings[node.target.id] = {value}
                self._scan_patch_statements(
                    module_info,
                    node.body,
                    branch,
                    tag_guard_names,
                )
                branches.append(branch)
        else:
            branch = context.clone(
                guards=self._merge_guard_paths(
                    context.guards,
                    (self._opaque_guard(context, "for-loop"),),
                )
                or context.guards,
            )
            self._scan_patch_statements(
                module_info,
                node.body,
                branch,
                tag_guard_names,
            )
            branches.append(branch)
        context.merge(branches)
        self._scan_patch_statements(
            module_info,
            node.orelse,
            context,
            tag_guard_names,
        )

    def _update_import_bindings(
        self,
        module_info: ModuleInfo,
        node: ast.Import | ast.ImportFrom,
        context: PatchScanContext,
    ) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".", 1)[0]
                target = alias.name if alias.asname else local_name
                context.bind_exact(local_name, {target})
                context.runtime_modules.pop(local_name, None)
            return

        source_module = _relative_import_module(
            module_info.name,
            module_info.is_package,
            node.level,
            node.module,
        )
        for alias in node.names:
            if alias.name == "*":
                continue
            local_name = alias.asname or alias.name
            target = f"{source_module}.{alias.name}" if source_module else alias.name
            context.bind_exact(local_name, {target})
            context.runtime_modules.pop(local_name, None)

    def _update_assignment_bindings(
        self,
        module_info: ModuleInfo,
        targets: Sequence[ast.AST],
        value: ast.AST | None,
        context: PatchScanContext,
    ) -> None:
        produced = (
            self._resolve_wrapper_factory_call(
                module_info,
                value,
                context,
                line=getattr(value, "lineno", 0),
            )
            if isinstance(value, ast.Call)
            else None
        )
        string_values = self._string_values(value, context)
        expression = _expression_name(value)
        runtime_modules = self._runtime_module_references(
            module_info,
            value,
            context,
        )
        getattr_references = self._getattr_references(
            module_info,
            value,
            context,
        )
        references = (
            runtime_modules
            or getattr_references
            or (
                self._resolve_patch_references(
                    module_info,
                    expression,
                    context,
                )
                if expression
                else set()
            )
        )
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if produced is not None and produced.info is not None:
                context.local_callables[target.id] = [produced.info]
            elif target.id in context.local_callables:
                context.local_callables.pop(target.id)
            if string_values:
                context.strings[target.id] = set(string_values)
            else:
                context.strings.pop(target.id, None)
            if references:
                context.bind_exact(target.id, references)
            elif isinstance(value, ast.Constant) and value.value is None:
                context.bind_none(target.id)
            else:
                context.bind_unknown(target.id)
            if runtime_modules:
                context.runtime_modules[target.id] = set(runtime_modules)
            else:
                context.runtime_modules.pop(target.id, None)

    def _runtime_module_references(
        self,
        module_info: ModuleInfo,
        node: ast.AST | None,
        context: PatchScanContext,
    ) -> set[str]:
        attributes: list[str] = []
        while isinstance(node, ast.Attribute):
            attributes.append(node.attr)
            node = node.value
        module_name: str | None = None
        owner_node: ast.AST | None = None
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            owner_node = node.func.value
            module_name = node.args[0].value
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            owner_node = node.value
            module_name = node.slice.value
        if owner_node is None or not (module_name == "vllm" or module_name.startswith("vllm.")):
            return set()
        owner = _expression_name(owner_node)
        if owner is None:
            return set()
        references = self._resolve_patch_references(
            module_info,
            owner,
            context,
        )
        if references != {"sys.modules"}:
            return set()
        return {".".join((module_name, *reversed(attributes))) if attributes else module_name}

    def _getattr_references(
        self,
        module_info: ModuleInfo,
        node: ast.AST | None,
        context: PatchScanContext,
    ) -> set[str]:
        if not (isinstance(node, ast.Call) and _expression_name(node.func) == "getattr" and len(node.args) >= 2):
            return set()
        owner = _expression_name(node.args[0])
        attributes = self._string_values(node.args[1], context)
        if owner is None or len(attributes) != 1:
            return set()
        attribute = next(iter(attributes))
        return {
            f"{candidate}.{attribute}"
            for candidate in self._resolve_patch_references(
                module_info,
                owner,
                context,
            )
        }

    def _resolve_patch_references(
        self,
        module_info: ModuleInfo,
        expression: str,
        context: PatchScanContext,
    ) -> set[str]:
        parts = expression.split(".")
        if parts[0] in context.unknown_bindings:
            candidates = set()
        elif parts[0] in context.bindings:
            candidates = {".".join([candidate, *parts[1:]]) for candidate in context.bindings[parts[0]]}
        elif expression.startswith(("vllm.", "vllm_ascend.")):
            candidates = {expression}
        else:
            candidates = {f"{module_info.name}.{expression}"}

        resolved = set()
        for candidate in candidates:
            if candidate.startswith("vllm."):
                candidate = self.upstream.canonical_name(candidate)
            elif candidate.startswith("vllm_ascend."):
                candidate = self.downstream.canonical_name(candidate)
            resolved.add(candidate)
        return resolved

    def _string_values(
        self,
        node: ast.AST | None,
        context: PatchScanContext,
    ) -> set[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.Name):
            return set(context.strings.get(node.id, ()))
        if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
            return {value for element in node.elts for value in self._string_values(element, context)}
        if isinstance(node, ast.IfExp):
            return {
                *self._string_values(node.body, context),
                *self._string_values(node.orelse, context),
            }
        return set()

    def _record_setattr_patch(
        self,
        module_info: ModuleInfo,
        call: ast.Call,
        context: PatchScanContext,
        line: int,
    ) -> None:
        owner = _expression_name(call.args[0])
        attributes = self._string_values(call.args[1], context)
        if not owner:
            return
        owner_targets = self._resolve_patch_references(
            module_info,
            owner,
            context,
        )
        upstream_owners = sorted(target for target in owner_targets if target.startswith("vllm."))
        if not upstream_owners:
            for attribute in sorted(attributes):
                synthetic_target = ast.copy_location(
                    ast.Attribute(
                        value=call.args[0],
                        attr=attribute,
                        ctx=ast.Store(),
                    ),
                    call.args[0],
                )
                self._record_unresolved_patch_owner(
                    module_info,
                    synthetic_target,
                    call.args[2],
                    context,
                    line,
                )
            return
        if not attributes:
            self._append_unresolved_patch(
                module_info,
                context,
                ", ".join(upstream_owners),
                call.args[2],
                line,
                "dynamic setattr attribute name",
            )
            return
        target_expressions = {
            f"{owner_target}.{attribute}" for owner_target in upstream_owners for attribute in attributes
        }
        live_targets = {target for target in target_expressions if self._find_upstream_patch_target(target) is not None}
        selected = live_targets or target_expressions
        if len(selected) != 1:
            self._append_unresolved_patch(
                module_info,
                context,
                ", ".join(sorted(selected)),
                call.args[2],
                line,
                "ambiguous setattr patch target",
            )
            return
        self._record_resolved_patch(
            module_info,
            next(iter(selected)),
            call.args[2],
            context,
            line,
            evidence_target=(f"{owner}.{next(iter(selected)).rsplit('.', 1)[-1]}"),
        )

    def _record_patch_node(
        self,
        module_info: ModuleInfo,
        target_node: ast.Attribute,
        replacement_node: ast.AST | None,
        context: PatchScanContext,
        line: int,
    ) -> None:
        expression = _expression_name(target_node)
        direct_runtime_modules = self._runtime_module_references(
            module_info,
            target_node.value,
            context,
        )
        if direct_runtime_modules:
            target_expressions = {f"{module}.{target_node.attr}" for module in direct_runtime_modules}
            targets = sorted(
                target
                for target_expression in target_expressions
                for target in self._resolve_patch_references(
                    module_info,
                    target_expression,
                    context,
                )
                if target.startswith("vllm.")
            )
            evidence_target = next(iter(target_expressions)) if len(target_expressions) == 1 else None
        else:
            if not expression:
                return
            targets = sorted(
                target
                for target in self._resolve_patch_references(
                    module_info,
                    expression,
                    context,
                )
                if target.startswith("vllm.")
            )
            evidence_target = expression
        if not targets:
            self._record_unresolved_patch_owner(
                module_info,
                target_node,
                replacement_node,
                context,
                line,
            )
            return
        if len(targets) != 1:
            self._append_unresolved_patch(
                module_info,
                context,
                ", ".join(targets),
                replacement_node,
                line,
                "ambiguous patch target alias",
            )
            return
        if expression:
            parts = expression.split(".")
            runtime_modules = context.runtime_modules.get(parts[0], set())
            if len(runtime_modules) == 1:
                evidence_target = ".".join([next(iter(runtime_modules)), *parts[1:]])
        self._record_resolved_patch(
            module_info,
            targets[0],
            replacement_node,
            context,
            line,
            evidence_target=evidence_target,
        )

    def _record_unresolved_patch_owner(
        self,
        module_info: ModuleInfo,
        target_node: ast.Attribute,
        replacement_node: ast.AST | None,
        context: PatchScanContext,
        line: int,
    ) -> None:
        expression = _expression_name(target_node)
        if expression is None:
            return
        parts = expression.split(".")
        root = parts[0]
        if root in context.unknown_bindings:
            owners = {
                owner
                for owner in context.upstream_binding_provenance.get(root, ())
                if owner == "vllm" or owner.startswith("vllm.")
            }
            if not owners:
                if root not in context.upstream_binding_history:
                    return
                self._append_unresolved_patch(
                    module_info,
                    context,
                    expression,
                    replacement_node,
                    line,
                    "upstream-derived patch owner now has only a dynamic runtime value",
                    status="review",
                    reason_code="dynamic_patch_owner",
                    generator_issue=False,
                )
                return
            targets = {self._canonical_reference(".".join((owner, *parts[1:]))) for owner in owners}
            self._append_unresolved_patch(
                module_info,
                context,
                ", ".join(sorted(targets)),
                replacement_node,
                line,
                "upstream-derived patch owner was overwritten by a dynamic value",
                status="review",
                reason_code="dynamic_patch_owner",
                generator_issue=False,
            )
            return
        if root in context.parameter_names or root not in context.bindings or context.bindings[root]:
            return
        alternatives = context.binding_alternatives.get(root)
        if not alternatives:
            return
        owners = {
            owner for owner in alternatives if owner is not None and (owner == "vllm" or owner.startswith("vllm."))
        }
        if not owners:
            return
        none_key = f"none:{ast.dump(ast.Name(id=root, ctx=ast.Load()), include_attributes=False)}"
        if not any(guard.key == none_key and guard.polarity for guard in context.guards):
            return
        targets = {self._canonical_reference(".".join((owner, *parts[1:]))) for owner in owners}
        self._append_unresolved_patch(
            module_info,
            context,
            ", ".join(sorted(targets)),
            replacement_node,
            line,
            ("upstream patch owner is path-dependent after branch merge; the active non-None path was not resolved"),
            status="review",
            reason_code="unresolved_patch_owner",
            generator_issue=True,
        )

    def _record_resolved_patch(
        self,
        module_info: ModuleInfo,
        target: str,
        replacement_node: ast.AST | None,
        context: PatchScanContext,
        line: int,
        *,
        evidence_target: str | None = None,
    ) -> None:
        field_finding = self._field_patch_finding(
            module_info,
            target,
            replacement_node,
            context,
            line,
            evidence_target=evidence_target,
        )
        if field_finding is not None:
            self.findings.append(field_finding)
            return
        replacement = self._resolve_patch_replacement(
            module_info,
            replacement_node,
            context,
            target,
            line,
        )
        if replacement.is_restore:
            self._append_unresolved_patch(
                module_info,
                context,
                target,
                replacement_node,
                line,
                "assignment restores the original upstream callable",
                status="excluded",
                reason_code="restore_original",
                generator_issue=False,
            )
            return
        if replacement.is_save:
            self._append_unresolved_patch(
                module_info,
                context,
                target,
                replacement_node,
                line,
                (f"assignment saves the original upstream callable from {replacement.lifecycle_source}"),
                status="excluded",
                reason_code="save_original",
                generator_issue=False,
            )
            return
        if replacement.info is None:
            self._append_unresolved_patch(
                module_info,
                context,
                target,
                replacement_node,
                line,
                replacement.reason or "replacement callable was not resolved",
            )
            return

        upstream_variants = self._callable_variants(target)
        variant_signatures = {
            json.dumps(
                candidate.signature,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for candidate in upstream_variants
        }
        if len(variant_signatures) > 1:
            self.findings.append(
                CandidateFinding(
                    relation="monkey_patch",
                    downstream_file=replacement.info.file,
                    downstream_owner=replacement.info.owner,
                    downstream_name=replacement.info.name,
                    target_expression=target,
                    evidence_line=line,
                    reason="conditional upstream callable has incompatible signature variants",
                    status="review",
                    reason_code="conditional_callable_variants",
                    generator_issue=False,
                    evidence_scope=self._scope_name(context),
                    evidence_guards=context.guard_texts,
                )
            )
            return

        upstream_callable = self._find_upstream_patch_target(target)
        if upstream_callable is None:
            status, reason_code, generator_issue = self._missing_patch_target_classification(
                target,
                context,
            )
            self.findings.append(
                CandidateFinding(
                    relation="monkey_patch",
                    downstream_file=module_info.file,
                    downstream_owner=replacement.info.owner,
                    downstream_name=replacement.info.name,
                    target_expression=target,
                    evidence_line=line,
                    reason="upstream patch target was not found",
                    status=status,
                    reason_code=reason_code,
                    generator_issue=generator_issue,
                    evidence_scope=self._scope_name(context),
                    evidence_guards=context.guard_texts,
                )
            )
            return

        definition_line = getattr(replacement.info.node, "lineno", None)
        evidence = RelationEvidence(
            file=module_info.file,
            line=line,
            scope=self._scope_name(context),
            guards=context.guard_texts,
            patch_kind=replacement.kind,
            definition_line=definition_line,
            binding_line=replacement.info.binding_line,
            target_expression=evidence_target or target,
        )
        self.relations.append(
            Relation(
                relation="monkey_patch",
                upstream_file=upstream_callable.file,
                upstream_owner=upstream_callable.owner,
                upstream_name=upstream_callable.name,
                upstream_signature=upstream_callable.signature,
                downstream_file=replacement.info.file,
                downstream_owner=replacement.info.owner,
                downstream_name=replacement.info.name,
                downstream_signature=replacement.info.signature,
                evidence_file=module_info.file,
                evidence_line=line,
                evidence=(evidence,),
                upstream_package=self._source_package(upstream_callable.qualified_name),
            )
        )

    def _field_patch_finding(
        self,
        module_info: ModuleInfo,
        target: str,
        replacement_node: ast.AST | None,
        context: PatchScanContext,
        line: int,
        *,
        evidence_target: str | None,
    ) -> CandidateFinding | None:
        if self._find_upstream_patch_target(target) is not None:
            return None
        upstream_value = self.upstream.find_value(target)
        if upstream_value is not None:
            return CandidateFinding(
                relation="monkey_patch",
                downstream_file=module_info.file,
                downstream_owner=None,
                downstream_name=target.rsplit(".", 1)[-1],
                target_expression=evidence_target or target,
                evidence_line=line,
                reason=(f"assignment mutates an existing upstream field declared in {upstream_value.file}"),
                status="verified",
                reason_code="field_mutation",
                generator_issue=False,
                evidence_scope=self._scope_name(context),
                evidence_guards=context.guard_texts,
            )
        if not self._definitely_non_callable(replacement_node):
            return None

        owner_name = target.rsplit(".", 1)[0]
        owner_exists = (
            self.upstream.find_class(owner_name) is not None
            or owner_name in self.upstream.modules
            or self.upstream.find_value(owner_name) is not None
        )
        if not owner_exists:
            return None

        guards = " ".join(context.guard_texts)
        hasattr_polarities = self._matching_hasattr_polarities(
            target,
            context,
        )
        if False in hasattr_polarities or " not in " in guards:
            status = "expected"
            reason_code = "inject_missing_field"
            reason = "assignment injects a missing upstream field under a negative guard"
        elif True in hasattr_polarities:
            status = "excluded"
            reason_code = "inactive_guard"
            reason = "field assignment is inactive because its positive guard is false"
        else:
            status = "risk"
            reason_code = "missing_upstream_field"
            reason = "assignment injects an unguarded field missing from the upstream owner"
        return CandidateFinding(
            relation="monkey_patch",
            downstream_file=module_info.file,
            downstream_owner=None,
            downstream_name=target.rsplit(".", 1)[-1],
            target_expression=evidence_target or target,
            evidence_line=line,
            reason=reason,
            status=status,
            reason_code=reason_code,
            generator_issue=False,
            evidence_scope=self._scope_name(context),
            evidence_guards=context.guard_texts,
        )

    def _definitely_non_callable(self, node: ast.AST | None) -> bool:
        return isinstance(
            node,
            (
                ast.Constant,
                ast.Dict,
                ast.DictComp,
                ast.JoinedStr,
                ast.List,
                ast.ListComp,
                ast.Set,
                ast.SetComp,
                ast.Tuple,
            ),
        )

    def _resolve_patch_replacement(
        self,
        module_info: ModuleInfo,
        node: ast.AST | None,
        context: PatchScanContext,
        target: str,
        line: int,
    ) -> PatchReplacement:
        kind = "replacement"
        if isinstance(node, ast.Call):
            wrapper = _expression_name(node.func)
            if wrapper in {"classmethod", "staticmethod"} and len(node.args) == 1:
                kind = wrapper
                node = node.args[0]
            elif wrapper == "property" and node.args:
                kind = "property"
                node = node.args[0]
            else:
                produced = self._resolve_wrapper_factory_call(
                    module_info,
                    node,
                    context,
                    target=target,
                    line=line,
                )
                if produced is not None:
                    return produced
                return PatchReplacement(
                    info=None,
                    kind="wrapper",
                    reason="patch replacement is produced by an unresolved call",
                )

        if isinstance(node, ast.Lambda):
            definition_line = getattr(node, "lineno", line)
            return PatchReplacement(
                info=CallableInfo(
                    qualified_name=(f"{module_info.name}.<lambda>@{definition_line}"),
                    module=module_info.name,
                    file=module_info.file,
                    owner=None,
                    name=f"<lambda>@{definition_line}",
                    node=node,
                ),
                kind="lambda",
            )

        expression = _expression_name(node)
        if not expression:
            return PatchReplacement(
                info=None,
                kind=kind,
                reason="unsupported patch replacement expression",
            )

        if "." not in expression:
            local_candidates = context.local_callables.get(expression, [])
            if len(local_candidates) == 1:
                candidate = local_candidates[0]
                return PatchReplacement(
                    info=candidate,
                    kind=(candidate.origin_kind if candidate.origin_kind != "definition" else kind),
                )
            if len(local_candidates) > 1:
                return PatchReplacement(
                    info=None,
                    kind=kind,
                    reason="ambiguous local replacement callable",
                )

        references = self._resolve_patch_references(
            module_info,
            expression,
            context,
        )
        if references == {target}:
            return PatchReplacement(
                info=None,
                kind="restore_original",
                is_restore=True,
                lifecycle_source=target,
            )
        upstream_references = {reference for reference in references if reference.startswith("vllm.")}
        if len(upstream_references) == 1:
            source = next(iter(upstream_references))
            target_owner, target_name = target.rsplit(".", 1)
            source_owner = source.rsplit(".", 1)[0]
            if (
                target_owner == source_owner
                and "original" in target_name.lower()
                and self._find_upstream_patch_target(target) is None
            ):
                return PatchReplacement(
                    info=None,
                    kind="save_original",
                    is_save=True,
                    lifecycle_source=source,
                )
        if upstream_references:
            return PatchReplacement(
                info=None,
                kind="alias_rebind",
                reason="replacement is another upstream callable",
            )

        candidates: dict[tuple[str, str | None, str], CallableInfo] = {}
        for reference in references:
            candidate = self._find_downstream_patch_replacement(reference)
            if candidate is None and reference.startswith(f"{module_info.name}."):
                candidate = self.downstream.find_loose_function(
                    module_info.name,
                    reference.rsplit(".", 1)[-1],
                )
            if candidate:
                candidates[(candidate.file, candidate.owner, candidate.name)] = candidate
        if len(candidates) == 1:
            return PatchReplacement(
                info=next(iter(candidates.values())),
                kind=kind,
            )
        return PatchReplacement(
            info=None,
            kind=kind,
            reason=("ambiguous replacement callable" if candidates else "replacement callable was not found"),
        )

    def _resolve_wrapper_factory_call(
        self,
        module_info: ModuleInfo,
        node: ast.Call,
        context: PatchScanContext,
        *,
        target: str | None = None,
        line: int,
    ) -> PatchReplacement | None:
        expression = _expression_name(node.func)
        if expression is None:
            return None

        root_name = expression.split(".", 1)[0]
        local_factory = expression in context.local_callables or expression in module_info.functions
        downstream_binding = any(
            candidate.startswith("vllm_ascend.") for candidate in context.bindings.get(root_name, ())
        )
        if not (local_factory or downstream_binding or expression.startswith("vllm_ascend.")):
            return None

        factories: dict[tuple[str, str | None, str], CallableInfo] = {}
        if "." not in expression:
            for candidate in context.local_callables.get(expression, []):
                factories[(candidate.file, candidate.owner, candidate.name)] = candidate
        references = self._resolve_patch_references(
            module_info,
            expression,
            context,
        )
        for reference in references:
            if not reference.startswith("vllm_ascend."):
                continue
            candidate = self._find_downstream_patch_replacement(reference)
            if candidate is not None:
                factories[(candidate.file, candidate.owner, candidate.name)] = candidate
        if len(factories) != 1:
            return None

        factory = next(iter(factories.values()))
        if not isinstance(
            factory.node,
            (ast.AsyncFunctionDef, ast.FunctionDef),
        ):
            return None
        scope_nodes = list(_function_scope_nodes(factory.node))
        nested = {
            child.name: child for child in scope_nodes if isinstance(child, (ast.AsyncFunctionDef, ast.FunctionDef))
        }
        returns = [child for child in scope_nodes if isinstance(child, ast.Return)]
        if not returns:
            return None

        parameters = {
            argument.arg
            for argument in (
                *factory.node.args.posonlyargs,
                *factory.node.args.args,
                *factory.node.args.kwonlyargs,
            )
        }
        if factory.node.args.vararg:
            parameters.add(factory.node.args.vararg.arg)
        if factory.node.args.kwarg:
            parameters.add(factory.node.args.kwarg.arg)

        returned_nodes: dict[str, ast.AST] = {}
        identity_return = False
        for return_node in returns:
            value = return_node.value
            if isinstance(value, ast.Name) and value.id in nested:
                returned_nodes[value.id] = nested[value.id]
                continue
            if isinstance(value, ast.Lambda):
                returned_nodes[f"<lambda>@{getattr(value, 'lineno', line)}"] = value
                continue
            if isinstance(value, ast.Name) and value.id in parameters:
                identity_return = True
                continue
            return PatchReplacement(
                info=None,
                kind="wrapper_factory",
                reason="wrapper factory has unsupported return values",
            )

        if len(returned_nodes) != 1:
            return PatchReplacement(
                info=None,
                kind="wrapper_factory",
                reason=(
                    "wrapper factory has ambiguous callable returns"
                    if returned_nodes
                    else "wrapper factory only returns an input callable"
                ),
                is_restore=bool(target and identity_return),
            )

        returned_name, returned_node = next(iter(returned_nodes.items()))
        kind = "wrapper_or_identity" if identity_return else "wrapper_factory"
        return PatchReplacement(
            info=CallableInfo(
                qualified_name=f"{factory.qualified_name}.<return>.{returned_name}",
                module=factory.module,
                file=factory.file,
                owner=None,
                name=returned_name,
                node=returned_node,
                binding_line=line,
                origin_kind=kind,
            ),
            kind=kind,
        )

    def _find_downstream_patch_replacement(
        self,
        qualified_name: str,
    ) -> CallableInfo | None:
        direct = self.downstream.find_callable(qualified_name)
        if direct is not None:
            return direct
        if "." not in qualified_name:
            return None

        owner_name, method_name = qualified_name.rsplit(".", 1)
        owner = self.downstream.find_class(owner_name)
        if owner is None:
            return None
        mro_result = self._linearized_mro(owner.qualified_name)
        effective_owner = self._effective_method_owner(
            mro_result.owners[1:],
            method_name,
        )
        if effective_owner is None:
            return None
        if effective_owner.startswith("vllm_ascend."):
            return self.downstream.find_callable(f"{effective_owner}.{method_name}")
        return None

    def _append_unresolved_patch(
        self,
        module_info: ModuleInfo,
        context: PatchScanContext,
        target_expression: str,
        replacement_node: ast.AST | None,
        line: int,
        reason: str,
        *,
        status: str = "review",
        reason_code: str | None = None,
        generator_issue: bool = True,
    ) -> None:
        replacement_name = _expression_name(replacement_node)
        if replacement_name is None and isinstance(replacement_node, ast.Lambda):
            replacement_name = f"<lambda>@{line}"
        codes = {
            "ambiguous local replacement callable": "ambiguous_replacement_callable",
            "ambiguous patch target alias": "ambiguous_patch_target",
            "ambiguous replacement callable": "ambiguous_replacement_callable",
            "ambiguous setattr patch target": "ambiguous_patch_target",
            "dynamic setattr attribute name": "dynamic_setattr_name",
            "patch replacement is produced by an unresolved call": "wrapper_factory",
            "replacement callable was not found": "missing_replacement_callable",
            "replacement is another upstream callable": "upstream_alias_rebind",
            "unsupported patch replacement expression": "unsupported_replacement_expression",
            "wrapper factory has ambiguous callable returns": "ambiguous_wrapper_factory",
            "wrapper factory has unsupported return values": "unsupported_wrapper_factory",
            "wrapper factory only returns an input callable": "identity_wrapper_factory",
        }
        self.findings.append(
            CandidateFinding(
                relation="monkey_patch",
                downstream_file=module_info.file,
                downstream_owner=None,
                downstream_name=replacement_name or "<unknown>",
                target_expression=target_expression,
                evidence_line=line,
                reason=reason,
                status=status,
                reason_code=reason_code or codes.get(reason, "analysis_gap"),
                generator_issue=generator_issue,
                evidence_scope=self._scope_name(context),
                evidence_guards=context.guard_texts,
            )
        )

    def _matching_hasattr_polarities(
        self,
        target: str,
        context: PatchScanContext,
    ) -> set[bool]:
        owner, member = self._canonical_reference(target).rsplit(".", 1)
        return {
            guard.polarity
            for guard in context.guards
            if guard.hasattr_target == (self._canonical_reference(owner), member)
        }

    def _missing_patch_target_classification(
        self,
        target: str,
        context: PatchScanContext,
    ) -> tuple[str, str, bool]:
        hasattr_polarities = self._matching_hasattr_polarities(
            target,
            context,
        )
        if False in hasattr_polarities:
            return "expected", "inject_missing_member", False
        if True in hasattr_polarities:
            return "excluded", "inactive_guard", False

        owner_name = target.rsplit(".", 1)[0]
        owner_class = self.upstream.find_class(owner_name)
        owner_exists = (
            owner_class is not None
            or owner_name in self.upstream.modules
            or self.upstream.find_value(owner_name) is not None
        )
        if owner_exists:
            return "risk", "possible_stale_patch", False
        return "risk", "possible_stale_patch", False

    def _reclassify_missing_patch_members(self) -> None:
        candidate_indexes = [
            index
            for index, finding in enumerate(self.findings)
            if finding.reason_code == "possible_stale_patch" and finding.relation == "monkey_patch"
        ]
        grouped: dict[str, list[int]] = defaultdict(list)
        for index in candidate_indexes:
            target = self.findings[index].target_expression
            if "." not in target:
                continue
            grouped[target.rsplit(".", 1)[0]].append(index)

        for owner_name, indexes in grouped.items():
            owner = self.upstream.find_class(owner_name)
            if owner is None:
                continue
            bindings: dict[str, list[int]] = defaultdict(list)
            binding_dependencies: dict[str, set[str]] = defaultdict(set)
            for index in indexes:
                finding = self.findings[index]
                member_name = finding.target_expression.rsplit(".", 1)[-1]
                bindings[member_name].append(index)
                replacement = self._finding_downstream_callable(finding)
                if replacement is not None:
                    binding_dependencies[member_name].update(self._self_member_references(replacement))

            reachable = {
                member
                for relation in self.relations
                if relation.relation == "monkey_patch"
                and relation.upstream_file == owner.file
                and relation.upstream_owner == owner.name
                for replacement in [self._relation_downstream_callable(relation)]
                if replacement is not None
                for member in self._self_member_references(replacement)
            }
            queue = list(reachable)
            promoted: set[str] = set()
            while queue:
                member = queue.pop()
                if member not in bindings or member in promoted:
                    continue
                promoted.add(member)
                for dependency in binding_dependencies.get(member, ()):
                    if dependency not in reachable:
                        reachable.add(dependency)
                        queue.append(dependency)

            for member in promoted:
                for index in bindings[member]:
                    self.findings[index] = replace(
                        self.findings[index],
                        status="expected",
                        reason_code="inject_missing_member",
                        reason=("missing member is injected and is reachable from a verified patch replacement"),
                    )

            has_external_base = any(not base.startswith(("vllm.", "vllm_ascend.")) for base in owner.resolved_bases)
            if has_external_base:
                for index in indexes:
                    if self.findings[index].reason_code != "possible_stale_patch":
                        continue
                    self.findings[index] = replace(
                        self.findings[index],
                        status="review",
                        reason_code="external_inherited_method",
                        reason=(
                            "member may be inherited from an external base; "
                            "the pinned source pair cannot prove its owner"
                        ),
                    )

    def _finding_downstream_callable(
        self,
        finding: CandidateFinding,
    ) -> CallableInfo | None:
        return next(
            (
                candidate
                for candidate in self.downstream.callables.values()
                if candidate.file == finding.downstream_file
                and candidate.owner == finding.downstream_owner
                and candidate.name == finding.downstream_name
            ),
            None,
        )

    def _relation_downstream_callable(
        self,
        relation: Relation,
    ) -> CallableInfo | None:
        return next(
            (
                candidate
                for candidate in self.downstream.callables.values()
                if candidate.file == relation.downstream_file
                and candidate.owner == relation.downstream_owner
                and candidate.name == relation.downstream_name
            ),
            None,
        )

    def _self_member_references(
        self,
        callable_info: CallableInfo,
    ) -> set[str]:
        if not isinstance(
            callable_info.node,
            (ast.AsyncFunctionDef, ast.FunctionDef),
        ):
            return set()
        return {
            node.attr
            for node in _function_scope_nodes(callable_info.node)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in {"cls", "self"}
        }

    def _scope_name(self, context: PatchScanContext) -> str | None:
        return ".".join(context.scope) if context.scope else None

    def _guard_text(self, node: ast.AST, *, truth: bool = True) -> str:
        return _canonical_guard(node, truth=truth)[2]

    def _find_upstream_patch_target(
        self,
        qualified_name: str,
    ) -> CallableInfo | None:
        qualified_name = self._canonical_reference(qualified_name)
        direct = self._callable_info(qualified_name)
        if direct is not None:
            return direct
        if "." not in qualified_name:
            return None

        owner_name, method_name = qualified_name.rsplit(".", 1)
        owner = self._class_info(owner_name)
        if owner is None:
            return None
        mro_result = self._linearized_mro(owner.qualified_name)
        effective_owner = self._effective_method_owner(
            mro_result.owners[1:],
            method_name,
        )
        if effective_owner is None:
            return None
        return self._callable_info(f"{effective_owner}.{method_name}")

    def _class_line(self, class_info: ClassInfo) -> int:
        node = self.downstream.find_callable(class_info.qualified_name)
        return getattr(node.node, "lineno", 0) if node else 0


def _relation_payloads(
    relations: Iterable[Relation],
    *,
    vllm_sha: str,
    ascend_sha: str,
    findings: Iterable[CandidateFinding] = (),
    external_sources: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str | None, str, str],
        list[Relation],
    ] = defaultdict(list)
    for relation in relations:
        signature_key = json.dumps(
            relation.upstream_signature,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        grouped[
            (
                relation.upstream_package,
                relation.upstream_file,
                relation.upstream_owner,
                relation.upstream_name,
                signature_key,
            )
        ].append(relation)

    payloads: list[dict[str, Any]] = []
    relation_count = 0
    for key in sorted(
        grouped,
        key=lambda item: (
            item[0],
            item[1],
            item[2] or "",
            item[3],
            item[4],
        ),
    ):
        source_package, upstream_file, owner, name, signature_key = key
        consumers = []
        evidence_records = []
        for relation in sorted(
            grouped[key],
            key=lambda item: (
                item.relation,
                item.downstream_file,
                item.downstream_owner or "",
                item.downstream_name,
            ),
        ):
            consumers.append(
                [
                    relation.relation,
                    relation.downstream_file,
                    relation.downstream_owner,
                    relation.downstream_name,
                    relation.downstream_signature,
                ]
            )
            evidence_records.append(
                {
                    "consumer": [
                        relation.relation,
                        relation.downstream_file,
                        relation.downstream_owner,
                        relation.downstream_name,
                    ],
                    "occurrences": [evidence.as_dict() for evidence in relation.evidence],
                }
            )
            relation_count += 1
        payloads.append(
            {
                "p": source_package,
                "u": [
                    upstream_file,
                    owner,
                    name,
                    json.loads(signature_key),
                ],
                "c": consumers,
                "e": evidence_records,
            }
        )

    finding_payloads = [
        {"f": finding.as_dict()}
        for finding in sorted(
            findings,
            key=lambda item: (
                item.status,
                item.reason_code,
                item.relation,
                item.downstream_file,
                item.evidence_line,
                item.target_expression,
            ),
        )
    ]
    finding_statuses = Counter(payload["f"]["status"] for payload in finding_payloads)
    meta = {
        "_meta": {
            "schema": SCHEMA_VERSION,
            "generator": GENERATOR_VERSION,
            "vllm": vllm_sha,
            "vllm_ascend": ascend_sha,
            "external_sources": dict(sorted((external_sources or {}).items())),
            "contracts": len(payloads),
            "relations": relation_count,
            "findings": len(finding_payloads),
            "findings_by_status": dict(sorted(finding_statuses.items())),
            "scope": sorted(SUPPORTED_RELATIONS),
        }
    }
    return [meta, *payloads, *finding_payloads]


def _write_jsonl(path: Path, payloads: Iterable[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) for payload in payloads)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{text}\n", encoding="utf-8")


def _load_compact_relations(path: Path) -> list[Relation]:
    relations = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if "_meta" in payload or "f" in payload:
            continue
        upstream_file, upstream_owner, upstream_name, upstream_signature = payload["u"]
        for consumer in payload["c"]:
            relation, downstream_file, downstream_owner, downstream_name, downstream_signature = consumer
            if relation not in SUPPORTED_RELATIONS:
                continue
            relations.append(
                Relation(
                    relation=relation,
                    upstream_file=upstream_file,
                    upstream_owner=upstream_owner,
                    upstream_name=upstream_name,
                    upstream_signature=upstream_signature,
                    downstream_file=downstream_file,
                    downstream_owner=downstream_owner,
                    downstream_name=downstream_name,
                    downstream_signature=downstream_signature,
                    evidence_file=downstream_file,
                    evidence_line=0,
                    upstream_package=payload.get("p", "vllm"),
                )
            )
    return relations


def _relation_label(relation: Relation) -> dict[str, Any]:
    return {
        "relation": relation.relation,
        "upstream": {
            "package": relation.upstream_package,
            "file": relation.upstream_file,
            "owner": relation.upstream_owner,
            "name": relation.upstream_name,
        },
        "downstream": {
            "file": relation.downstream_file,
            "owner": relation.downstream_owner,
            "name": relation.downstream_name,
        },
    }


def _downstream_label(
    key: tuple[str, str, str, str],
) -> dict[str, Any]:
    return {
        "relation": key[0],
        "file": key[1],
        "owner": key[2] or None,
        "name": key[3],
    }


def compare_relations(
    generated: Sequence[Relation],
    baseline: Sequence[Relation],
    findings: Sequence[CandidateFinding],
) -> dict[str, Any]:
    finding_statuses = Counter(finding.status for finding in findings)
    generated_exact = {relation.exact_key(): relation for relation in generated}
    baseline_exact = {relation.exact_key(): relation for relation in baseline}
    generated_exact_aliases = {key: relation for relation in generated for key in relation.comparison_exact_keys()}
    baseline_exact_aliases = {key: relation for relation in baseline for key in relation.comparison_exact_keys()}
    generated_downstream: dict[
        tuple[str, str, str, str],
        list[Relation],
    ] = defaultdict(list)
    baseline_downstream: dict[
        tuple[str, str, str, str],
        list[Relation],
    ] = defaultdict(list)
    for relation in generated:
        for key in relation.comparison_downstream_keys():
            generated_downstream[key].append(relation)
    for relation in baseline:
        for key in relation.comparison_downstream_keys():
            baseline_downstream[key].append(relation)

    exact_matches = {
        key
        for key in baseline_exact
        if any(alias in generated_exact_aliases for alias in baseline_exact[key].comparison_exact_keys())
    }
    different_upstream = []
    baseline_downstream_keys = {relation.downstream_key() for relation in baseline}
    for key in sorted(baseline_downstream_keys & set(generated_downstream)):
        generated_targets = sorted(relation.upstream_key() for relation in generated_downstream[key])
        baseline_targets = sorted(relation.upstream_key() for relation in baseline_downstream[key])
        if generated_targets != baseline_targets:
            different_upstream.append(
                {
                    "downstream": {
                        "relation": key[0],
                        "file": key[1],
                        "owner": key[2] or None,
                        "name": key[3],
                    },
                    "baseline_upstream": baseline_targets,
                    "generated_upstream": generated_targets,
                }
            )

    old_only_keys = set(baseline_exact) - exact_matches
    new_only_keys = {
        key
        for key, relation in generated_exact.items()
        if not any(alias in baseline_exact_aliases for alias in relation.comparison_exact_keys())
    }
    generated_downstream_keys = {relation.downstream_key() for relation in generated}
    covered_downstream_keys = {key for key in baseline_downstream_keys if key in generated_downstream}
    missing_downstream_keys = baseline_downstream_keys - covered_downstream_keys
    new_downstream_keys = {
        key
        for key in generated_downstream_keys
        if not any(
            alias in baseline_downstream
            for relation in generated
            if relation.downstream_key() == key
            for alias in relation.comparison_downstream_keys()
        )
    }
    downstream_coverage = (
        len(covered_downstream_keys) / len(baseline_downstream_keys) * 100 if baseline_downstream_keys else 100.0
    )
    report = {
        "summary": {
            "generated_relations": len(generated),
            "baseline_relations": len(baseline),
            "exact_matches": len(exact_matches),
            "same_downstream_different_upstream": len(different_upstream),
            "old_only": len(old_only_keys),
            "new_only": len(new_only_keys),
            "findings": len(findings),
            "unresolved": finding_statuses["review"],
            "upstream_risks": finding_statuses["risk"],
            "expected": finding_statuses["expected"],
            "excluded": finding_statuses["excluded"],
            "verified_findings": finding_statuses["verified"],
            "generator_issues": sum(finding.generator_issue for finding in findings),
            "generated_downstream_endpoints": len(generated_downstream_keys),
            "baseline_downstream_endpoints": len(baseline_downstream_keys),
            "covered_downstream_endpoints": len(covered_downstream_keys),
            "missing_downstream_endpoints": len(missing_downstream_keys),
            "new_downstream_endpoints": len(new_downstream_keys),
            "downstream_coverage_percent": round(
                downstream_coverage,
                2,
            ),
            "generated_by_relation": dict(sorted(Counter(relation.relation for relation in generated).items())),
            "baseline_by_relation": dict(sorted(Counter(relation.relation for relation in baseline).items())),
        },
        "same_downstream_different_upstream": different_upstream,
        "old_only": [_relation_label(baseline_exact[key]) for key in sorted(old_only_keys)],
        "new_only": [_relation_label(generated_exact[key]) for key in sorted(new_only_keys)],
        "missing_downstream": [_downstream_label(key) for key in sorted(missing_downstream_keys)],
        "new_downstream": [_downstream_label(key) for key in sorted(new_downstream_keys)],
        "findings": [finding.as_dict() for finding in findings],
    }
    return report


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _verify_sha(label: str, actual: str, expected: str | None) -> None:
    if expected and actual != expected:
        raise SystemExit(f"{label} SHA mismatch: expected {expected}, found {actual}")


def _canonical_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _named_values(values: Sequence[str], option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{option} must use PACKAGE=VALUE: {value}")
        package, item = value.split("=", 1)
        if not package or not item:
            raise SystemExit(f"{option} must use PACKAGE=VALUE: {value}")
        if not package.isidentifier():
            raise SystemExit(f"invalid package name for {option}: {package}")
        if package in result:
            raise SystemExit(f"duplicate {option} package: {package}")
        result[package] = item
    return result


def _snapshot_source_sha(root: Path, package: str) -> str:
    manifest_path = root / ".interface-source.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"external source is neither a Git checkout nor a valid snapshot: {package}={root}") from error

    if manifest.get("schema") != 1 or manifest.get("package") != package:
        raise SystemExit(f"invalid external source manifest identity: {manifest_path}")
    commit = manifest.get("commit")
    expected_files = manifest.get("files")
    if not isinstance(commit, str) or not isinstance(expected_files, dict):
        raise SystemExit(f"invalid external source manifest: {manifest_path}")

    package_root = root / package
    actual_files = {path.relative_to(root).as_posix() for path in package_root.rglob("*.py")}
    if actual_files != set(expected_files):
        missing = sorted(set(expected_files) - actual_files)
        extra = sorted(actual_files - set(expected_files))
        raise SystemExit(f"external source snapshot file set changed for {package}: missing={missing}, extra={extra}")

    for relative_path, expected_digest in sorted(expected_files.items()):
        if not isinstance(relative_path, str) or not isinstance(
            expected_digest,
            str,
        ):
            raise SystemExit(f"invalid external source file record: {manifest_path}")
        path = root / relative_path
        actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            raise SystemExit(
                f"external source snapshot digest mismatch: {relative_path}; "
                f"expected {expected_digest}, found {actual_digest}"
            )
    return commit


def _verified_external_sources(
    roots: dict[str, Path],
    expected_shas: dict[str, str],
) -> dict[str, str]:
    if set(roots) != set(expected_shas):
        raise SystemExit("--external-root and --expect-external-sha must name the same packages")
    actual_shas: dict[str, str] = {}
    for package, root in sorted(roots.items()):
        try:
            actual = _git_head(root)
        except subprocess.CalledProcessError:
            actual = _snapshot_source_sha(root, package)
        _verify_sha(
            f"external package {package}",
            actual,
            expected_shas[package],
        )
        actual_shas[package] = actual
    return actual_shas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--ascend-root", type=Path, required=True)
    parser.add_argument("--expect-vllm-sha")
    parser.add_argument("--expect-ascend-sha")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unresolved-output", type=Path)
    parser.add_argument("--compare-with", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--external-root",
        action="append",
        default=[],
        metavar="PACKAGE=PATH",
    )
    parser.add_argument(
        "--expect-external-sha",
        action="append",
        default=[],
        metavar="PACKAGE=SHA",
    )
    args = parser.parse_args()

    vllm_sha = _git_head(args.vllm_root)
    ascend_sha = _git_head(args.ascend_root)
    _verify_sha("vLLM", vllm_sha, args.expect_vllm_sha)
    _verify_sha("vllm-ascend", ascend_sha, args.expect_ascend_sha)

    external_root_values = _named_values(
        args.external_root,
        "--external-root",
    )
    expected_external_shas = _named_values(
        args.expect_external_sha,
        "--expect-external-sha",
    )
    external_roots = {package: Path(path) for package, path in external_root_values.items()}
    external_sources = _verified_external_sources(
        external_roots,
        expected_external_shas,
    )

    generator = InterfaceBoundaryGenerator(
        args.vllm_root,
        args.ascend_root,
        external_roots,
    )
    relations, findings = generator.generate()
    _write_jsonl(
        args.output,
        _relation_payloads(
            relations,
            vllm_sha=vllm_sha,
            ascend_sha=ascend_sha,
            findings=findings,
            external_sources=external_sources,
        ),
    )

    if args.unresolved_output:
        _write_jsonl(
            args.unresolved_output,
            (finding.as_dict() for finding in findings),
        )

    finding_statuses = Counter(finding.status for finding in findings)
    report: dict[str, Any] = {
        "inputs": {
            "vllm_sha": vllm_sha,
            "vllm_ascend_sha": ascend_sha,
            "generator_version": GENERATOR_VERSION,
            "external_sources": dict(sorted(external_sources.items())),
        },
        "generated": {
            "relations": len(relations),
            "findings": len(findings),
            "unresolved": finding_statuses["review"],
            "upstream_risks": finding_statuses["risk"],
            "expected": finding_statuses["expected"],
            "excluded": finding_statuses["excluded"],
            "verified_findings": finding_statuses["verified"],
            "generator_issues": sum(finding.generator_issue for finding in findings),
            "findings_by_status": dict(sorted(finding_statuses.items())),
            "by_relation": dict(sorted(Counter(relation.relation for relation in relations).items())),
            "sha256": _canonical_digest(args.output),
        },
    }
    if args.compare_with:
        baseline = _load_compact_relations(args.compare_with)
        report["comparison"] = compare_relations(relations, baseline, findings)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            f"{json.dumps(report, ensure_ascii=False, indent=2)}\n",
            encoding="utf-8",
        )
    console_report = {
        "inputs": report["inputs"],
        "generated": report["generated"],
    }
    if "comparison" in report:
        console_report["comparison"] = report["comparison"]["summary"]
    print(json.dumps(console_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
