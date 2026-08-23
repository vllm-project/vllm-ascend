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

"""Generate vLLM override relations required by the upstream interface CI.

Inheritance is collected only to resolve the complete MRO behind a downstream
override. Monkey-patch collection and main2main-only review findings are outside
this package's scope.

It does not import either package and does not require an NPU.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import hashlib
import inspect
import json
import os
import pickle
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .analysis_plans import VLLM_INTERFACE_PLAN, AnalysisPlan

GENERATOR_VERSION = "0.42.0"
REPOSITORY_INDEX_CACHE_SCHEMA_VERSION = 1
REPOSITORY_FILE_FRAGMENT_CACHE_SCHEMA_VERSION = 1
FINDING_STATUSES = frozenset({"expected", "excluded", "review", "risk", "verified"})
DESCRIPTOR_KINDS = frozenset(
    {
        "ordinary",
        "property",
        "classmethod",
        "staticmethod",
        "unknown",
    }
)
_BUILTIN_DESCRIPTOR_DECORATORS = {
    "builtins.classmethod": "classmethod",
    "builtins.property": "property",
    "builtins.staticmethod": "staticmethod",
}
_TRANSPARENT_DESCRIPTOR_DECORATORS = frozenset(
    {
        "abc.abstractmethod",
        "functools.wraps",
        "typing.final",
        "typing.override",
        "typing_extensions.final",
        "typing_extensions.override",
    }
)
_KNOWN_ORDINARY_DESCRIPTOR_DECORATORS = frozenset(
    {
        "torch.inference_mode",
        "vllm.tracing.instrument",
    }
)
_KNOWN_TRANSPARENT_SIGNATURE_DECORATORS = frozenset(
    {
        "torch.inference_mode",
        "vllm.tracing.instrument",
    }
)
_KNOWN_WRAPS_SIGNATURE_DECORATORS = frozenset({"torch.compiler.disable"})
_STDLIB_WRAPS_SIGNATURE_DECORATORS = frozenset({"contextlib.contextmanager"})
_TRITON_JIT_DECORATOR = "vllm.triton_utils.triton.jit"
_TRITON_HEURISTICS_DECORATOR = "vllm.triton_utils.triton.heuristics"
_TRITON_KERNEL_PROTOCOL = "triton_kernel_launch"
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


@dataclass(frozen=True)
class _ScopeFlowExit:
    """One non-local exit from module/class namespace execution."""

    kind: str
    state: dict[str, tuple[_ScopeBinding, ...]] = field(
        compare=False,
        hash=False,
        repr=False,
    )
    exception_name: str | None = None


@dataclass
class _ScopeFlowResult:
    """Normally completing namespace states and their abrupt exits."""

    normal: list[dict[str, tuple[_ScopeBinding, ...]]] = field(
        default_factory=list,
    )
    exits: list[_ScopeFlowExit] = field(default_factory=list)


_HANDLER_NEVER = "never"
_HANDLER_MAYBE = "maybe"
_HANDLER_ALWAYS = "always"


def _clone_scope_binding_state(
    state: dict[str, tuple[_ScopeBinding, ...]],
) -> dict[str, tuple[_ScopeBinding, ...]]:
    return {name: tuple(values) for name, values in state.items()}


def _scope_state_key(
    state: dict[str, tuple[_ScopeBinding, ...]],
) -> tuple[tuple[str, tuple[_ScopeBinding, ...]], ...]:
    return tuple(sorted(state.items()))


def _compact_scope_states(
    states: Iterable[dict[str, tuple[_ScopeBinding, ...]]],
) -> list[dict[str, tuple[_ScopeBinding, ...]]]:
    """Merge path states without losing any per-name binding alternative."""

    unique = {_scope_state_key(state): state for state in states}
    if not unique:
        return []
    merged = _merge_scope_binding_states(list(unique.values()))
    return [merged] if merged is not None else []


def _compact_scope_exits(exits: Iterable[_ScopeFlowExit]) -> list[_ScopeFlowExit]:
    grouped: dict[tuple[str, str | None], list[dict[str, tuple[_ScopeBinding, ...]]]] = defaultdict(list)
    for flow_exit in exits:
        grouped[(flow_exit.kind, flow_exit.exception_name)].append(flow_exit.state)
    compacted: list[_ScopeFlowExit] = []
    for (kind, exception_name), states in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1] or ""),
    ):
        merged = _merge_scope_binding_states(states)
        if merged is not None:
            compacted.append(
                _ScopeFlowExit(
                    kind=kind,
                    state=merged,
                    exception_name=exception_name,
                )
            )
    return compacted


def _compact_scope_flow(result: _ScopeFlowResult) -> _ScopeFlowResult:
    return _ScopeFlowResult(
        normal=_compact_scope_states(result.normal),
        exits=_compact_scope_exits(result.exits),
    )


def _scope_exception_name(
    node: ast.AST | None,
    state: dict[str, tuple[_ScopeBinding, ...]],
) -> str | None:
    """Resolve a statically named exception without guessing dynamic values."""

    expression = _expression_name(node.func if isinstance(node, ast.Call) else node)
    if expression is None:
        return None
    if "." not in expression:
        builtin_type = getattr(builtins, expression, None)
        root_bindings = state.get(expression, (_UNBOUND_SCOPE_BINDING,))
        if (
            isinstance(builtin_type, type)
            and issubclass(builtin_type, BaseException)
            and all(binding.kind == "unbound" for binding in root_bindings)
        ):
            return f"builtins.{expression}"
    return expression


def _scope_exception_is_subclass(child_name: str, parent_name: str) -> bool:
    if child_name == parent_name:
        return True
    child_type = (
        getattr(builtins, child_name.removeprefix("builtins."), None) if child_name.startswith("builtins.") else None
    )
    parent_type = (
        getattr(builtins, parent_name.removeprefix("builtins."), None) if parent_name.startswith("builtins.") else None
    )
    return bool(
        isinstance(child_type, type)
        and isinstance(parent_type, type)
        and issubclass(child_type, BaseException)
        and issubclass(parent_type, BaseException)
        and issubclass(child_type, parent_type)
    )


def _scope_handler_names(
    handler: ast.ExceptHandler,
    state: dict[str, tuple[_ScopeBinding, ...]],
) -> tuple[tuple[str, ...], bool] | None:
    if handler.type is None:
        return None
    nodes = handler.type.elts if isinstance(handler.type, ast.Tuple) else (handler.type,)
    resolved = tuple(_scope_exception_name(node, state) for node in nodes)
    return (
        tuple(name for name in resolved if name is not None),
        any(name is None for name in resolved),
    )


def _scope_handler_match(
    flow_exit: _ScopeFlowExit,
    handler: ast.ExceptHandler,
) -> str:
    resolution = _scope_handler_names(handler, flow_exit.state)
    if resolution is None:
        return _HANDLER_ALWAYS
    handler_names, has_unknown = resolution
    if flow_exit.exception_name is None:
        if any(name == "builtins.BaseException" for name in handler_names):
            return _HANDLER_ALWAYS
        return _HANDLER_MAYBE
    if any(_scope_exception_is_subclass(flow_exit.exception_name, handler_name) for handler_name in handler_names):
        return _HANDLER_ALWAYS
    return _HANDLER_MAYBE if has_unknown else _HANDLER_NEVER


def _scope_expression_may_raise(node: ast.AST | None) -> bool:
    if node is None:
        return False
    return any(
        isinstance(candidate, (ast.Await, ast.Call, ast.Subscript, ast.YieldFrom)) for candidate in ast.walk(node)
    )


def _scope_function_header_may_raise(
    node: ast.AsyncFunctionDef | ast.FunctionDef,
) -> bool:
    expressions: list[ast.AST | None] = [
        *node.decorator_list,
        *node.args.defaults,
        *node.args.kw_defaults,
        *(argument.annotation for argument in node.args.posonlyargs),
        *(argument.annotation for argument in node.args.args),
        *(argument.annotation for argument in node.args.kwonlyargs),
        node.args.vararg.annotation if node.args.vararg else None,
        node.args.kwarg.annotation if node.args.kwarg else None,
        node.returns,
    ]
    expressions.extend(getattr(node, "type_params", ()))
    return any(_scope_expression_may_raise(expression) for expression in expressions)


def _scope_class_header_may_raise(node: ast.ClassDef) -> bool:
    expressions: list[ast.AST] = [
        *node.decorator_list,
        *node.bases,
        *(keyword.value for keyword in node.keywords),
        *getattr(node, "type_params", ()),
    ]
    return any(_scope_expression_may_raise(expression) for expression in expressions)


def _scope_simple_statement_may_raise(node: ast.stmt) -> bool:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return True
    if isinstance(node, ast.Expr):
        return _scope_expression_may_raise(node.value)
    if isinstance(node, ast.Assign):
        return _scope_expression_may_raise(node.value)
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        return _scope_expression_may_raise(node.value)
    if isinstance(node, ast.Assert):
        return _scope_expression_may_raise(node.test) or _scope_expression_may_raise(node.msg)
    if isinstance(node, ast.Delete):
        return any(not isinstance(target, ast.Name) for target in node.targets)
    return False


def _unbind_handler_name(
    states: Iterable[dict[str, tuple[_ScopeBinding, ...]]],
    name: str | None,
) -> None:
    if not name:
        return
    for state in states:
        state[name] = (_UNBOUND_SCOPE_BINDING,)


def _unbind_handler_name_from_exits(
    exits: Iterable[_ScopeFlowExit],
    name: str | None,
) -> None:
    if not name:
        return
    for flow_exit in exits:
        flow_exit.state[name] = (_UNBOUND_SCOPE_BINDING,)


def _scope_flow_statement(
    node: ast.stmt,
    tag_guard_names: set[str],
    incoming: dict[str, tuple[_ScopeBinding, ...]],
    *,
    loop_body: bool,
    active_exception: str | None,
) -> _ScopeFlowResult:
    """Interpret one namespace statement and preserve its exception snapshot."""

    state = _clone_scope_binding_state(incoming)

    if isinstance(node, ast.If):
        exits: list[_ScopeFlowExit] = []
        if _scope_expression_may_raise(node.test):
            exits.append(_ScopeFlowExit("raise", _clone_scope_binding_state(state)))
        condition = _main_condition_value(node.test, tag_guard_names)
        branches: list[Sequence[ast.stmt]]
        if condition is True:
            branches = [node.body]
        elif condition is False:
            branches = [node.orelse]
        else:
            branches = [node.body, node.orelse]
        normal: list[dict[str, tuple[_ScopeBinding, ...]]] = []
        for statements in branches:
            branch = _scope_binding_flow(
                statements,
                tag_guard_names,
                state,
                loop_body=loop_body,
                active_exception=active_exception,
            )
            normal.extend(branch.normal)
            exits.extend(branch.exits)
        return _compact_scope_flow(_ScopeFlowResult(normal=normal, exits=exits))

    if isinstance(node, ast.TryStar):
        # ExceptionGroup routing differs from ordinary try/except.  Preserve
        # every explicit path conservatively until a dedicated model exists.
        paths = [
            _scope_binding_flow(
                node.body,
                tag_guard_names,
                state,
                loop_body=loop_body,
                active_exception=active_exception,
            ),
            *(
                _scope_binding_flow(
                    handler.body,
                    tag_guard_names,
                    state,
                    loop_body=loop_body,
                    active_exception=None,
                )
                for handler in node.handlers
            ),
        ]
        normal = [candidate for path in paths for candidate in path.normal]
        exits = [candidate for path in paths for candidate in path.exits]
        if node.orelse:
            else_flow = _scope_binding_flow(
                node.orelse,
                tag_guard_names,
                _merge_scope_binding_states(normal) or state,
                loop_body=loop_body,
                active_exception=active_exception,
            )
            normal.extend(else_flow.normal)
            exits.extend(else_flow.exits)
        result = _compact_scope_flow(_ScopeFlowResult(normal=normal, exits=exits))
        if node.finalbody:
            return _apply_scope_finally(
                result,
                node.finalbody,
                tag_guard_names,
                loop_body=loop_body,
            )
        return result

    if isinstance(node, ast.Try):
        body = _scope_binding_flow(
            node.body,
            tag_guard_names,
            state,
            loop_body=loop_body,
            active_exception=active_exception,
        )
        normal: list[dict[str, tuple[_ScopeBinding, ...]]] = []
        exits: list[_ScopeFlowExit] = [candidate for candidate in body.exits if candidate.kind != "raise"]

        for body_state in body.normal:
            else_flow = _scope_binding_flow(
                node.orelse,
                tag_guard_names,
                body_state,
                loop_body=loop_body,
                active_exception=active_exception,
            )
            normal.extend(else_flow.normal)
            exits.extend(else_flow.exits)

        pending = [candidate for candidate in body.exits if candidate.kind == "raise"]
        for handler in node.handlers:
            next_pending: list[_ScopeFlowExit] = []
            for raised in pending:
                match = _scope_handler_match(raised, handler)
                if match in {_HANDLER_MAYBE, _HANDLER_ALWAYS}:
                    handler_state = _clone_scope_binding_state(raised.state)
                    if handler.name:
                        handler_state[handler.name] = (_scope_binding("value", handler),)
                    handler_flow = _scope_binding_flow(
                        handler.body,
                        tag_guard_names,
                        handler_state,
                        loop_body=loop_body,
                        active_exception=raised.exception_name,
                    )
                    _unbind_handler_name(handler_flow.normal, handler.name)
                    _unbind_handler_name_from_exits(handler_flow.exits, handler.name)
                    normal.extend(handler_flow.normal)
                    exits.extend(handler_flow.exits)
                if match in {_HANDLER_NEVER, _HANDLER_MAYBE}:
                    next_pending.append(raised)
            pending = next_pending
        exits.extend(pending)
        result = _compact_scope_flow(_ScopeFlowResult(normal=normal, exits=exits))
        if node.finalbody:
            result = _apply_scope_finally(
                result,
                node.finalbody,
                tag_guard_names,
                loop_body=loop_body,
            )
        return result

    if isinstance(node, (ast.With, ast.AsyncWith)):
        exits: list[_ScopeFlowExit] = []
        if any(_scope_expression_may_raise(item.context_expr) for item in node.items):
            exits.append(_ScopeFlowExit("raise", _clone_scope_binding_state(state)))
        for item in node.items:
            if item.optional_vars is not None:
                _bind_scope_names(
                    state,
                    _bound_target_names(item.optional_vars),
                    _scope_binding("value", item.optional_vars),
                )
        body = _scope_binding_flow(
            node.body,
            tag_guard_names,
            state,
            loop_body=loop_body,
            active_exception=active_exception,
        )
        return _compact_scope_flow(_ScopeFlowResult(normal=body.normal, exits=[*exits, *body.exits]))

    if isinstance(node, (ast.AsyncFor, ast.For, ast.While)):
        exits: list[_ScopeFlowExit] = []
        test = node.iter if isinstance(node, (ast.AsyncFor, ast.For)) else node.test
        if _scope_expression_may_raise(test):
            exits.append(_ScopeFlowExit("raise", _clone_scope_binding_state(state)))
        body_state = _clone_scope_binding_state(state)
        if isinstance(node, (ast.AsyncFor, ast.For)):
            _bind_scope_names(
                body_state,
                _bound_target_names(node.target),
                _scope_binding("value", node.target),
            )
        body = _scope_binding_flow(
            node.body,
            tag_guard_names,
            body_state,
            loop_body=True,
            active_exception=active_exception,
        )
        normal = [state, *body.normal]
        exits.extend(candidate for candidate in body.exits if candidate.kind not in {"break", "continue"})
        normal.extend(candidate.state for candidate in body.exits if candidate.kind in {"break", "continue"})
        if node.orelse:
            merged = _merge_scope_binding_states(normal)
            if merged is not None:
                else_flow = _scope_binding_flow(
                    node.orelse,
                    tag_guard_names,
                    merged,
                    loop_body=loop_body,
                    active_exception=active_exception,
                )
                normal.extend(else_flow.normal)
                exits.extend(else_flow.exits)
        return _compact_scope_flow(_ScopeFlowResult(normal=normal, exits=exits))

    if isinstance(node, ast.Match):
        exits: list[_ScopeFlowExit] = []
        if _scope_expression_may_raise(node.subject):
            exits.append(_ScopeFlowExit("raise", _clone_scope_binding_state(state)))
        normal = [state]
        for case in node.cases:
            branch = _scope_binding_flow(
                case.body,
                tag_guard_names,
                state,
                loop_body=loop_body,
                active_exception=active_exception,
            )
            normal.extend(branch.normal)
            exits.extend(branch.exits)
        return _compact_scope_flow(_ScopeFlowResult(normal=normal, exits=exits))

    implicit_raise = _scope_simple_statement_may_raise(node)
    exits = [_ScopeFlowExit("raise", _clone_scope_binding_state(state))] if implicit_raise else []

    if isinstance(node, ast.Raise):
        exception_name = active_exception if node.exc is None else _scope_exception_name(node.exc, state)
        return _ScopeFlowResult(exits=[_ScopeFlowExit("raise", state, exception_name)])
    if isinstance(node, ast.Return):
        return _ScopeFlowResult(exits=[_ScopeFlowExit("return", state)])
    if isinstance(node, ast.Break):
        return _ScopeFlowResult(exits=[_ScopeFlowExit("break", state)])
    if isinstance(node, ast.Continue):
        return _ScopeFlowResult(exits=[_ScopeFlowExit("continue", state)])

    if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
        if _scope_function_header_may_raise(node):
            exits.append(_ScopeFlowExit("raise", _clone_scope_binding_state(state)))
        state[node.name] = (_scope_binding("function", node),)
    elif isinstance(node, ast.ClassDef):
        if _scope_class_header_may_raise(node):
            exits.append(_ScopeFlowExit("raise", _clone_scope_binding_state(state)))
        class_flow = _scope_binding_flow(
            node.body,
            tag_guard_names,
            {},
            loop_body=False,
            active_exception=None,
        )
        exits.extend(
            _ScopeFlowExit(
                kind=candidate.kind,
                state=_clone_scope_binding_state(state),
                exception_name=candidate.exception_name,
            )
            for candidate in class_flow.exits
            if candidate.kind == "raise"
        )
        if not class_flow.normal:
            return _compact_scope_flow(_ScopeFlowResult(exits=exits))
        state[node.name] = (_scope_binding("class", node),)
    elif isinstance(node, ast.Import):
        _bind_scope_names(
            state,
            (alias.asname or alias.name.split(".", 1)[0] for alias in node.names),
            _scope_binding("value", node),
        )
    elif isinstance(node, ast.ImportFrom):
        _bind_scope_names(
            state,
            (alias.asname or alias.name for alias in node.names if alias.name != "*"),
            _scope_binding("value", node),
        )
    elif isinstance(node, ast.Assign):
        source_binding = state.get(node.value.id) if isinstance(node.value, ast.Name) else None
        for target in node.targets:
            names = _bound_target_names(target)
            if len(names) == 1 and isinstance(target, ast.Name) and isinstance(node.value, ast.Name):
                if source_binding is not None and all(
                    binding.kind in {"class", "function"} for binding in source_binding
                ):
                    state[target.id] = tuple(source_binding)
                else:
                    state[target.id] = (_scope_binding("alias", node),)
            else:
                _bind_scope_names(state, names, _scope_binding("value", node))
    elif isinstance(node, ast.AnnAssign):
        if node.value is not None:
            source_binding = state.get(node.value.id) if isinstance(node.value, ast.Name) else None
            if isinstance(node.target, ast.Name) and isinstance(node.value, ast.Name):
                if source_binding is not None and all(
                    binding.kind in {"class", "function"} for binding in source_binding
                ):
                    state[node.target.id] = tuple(source_binding)
                else:
                    state[node.target.id] = (_scope_binding("alias", node),)
            else:
                _bind_scope_names(
                    state,
                    _bound_target_names(node.target),
                    _scope_binding("value", node),
                )
    elif isinstance(node, ast.AugAssign):
        _bind_scope_names(
            state,
            _bound_target_names(node.target),
            _scope_binding("value", node),
        )
    elif isinstance(node, ast.Delete):
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

    return _compact_scope_flow(_ScopeFlowResult(normal=[state], exits=exits))


def _scope_binding_flow(
    statements: Sequence[ast.stmt],
    tag_guard_names: set[str],
    incoming: dict[str, tuple[_ScopeBinding, ...]] | None = None,
    *,
    loop_body: bool = False,
    active_exception: str | None = None,
) -> _ScopeFlowResult:
    normal = [_clone_scope_binding_state(incoming or {})]
    exits: list[_ScopeFlowExit] = []
    for node in statements:
        next_normal: list[dict[str, tuple[_ScopeBinding, ...]]] = []
        for state in normal:
            result = _scope_flow_statement(
                node,
                tag_guard_names,
                state,
                loop_body=loop_body,
                active_exception=active_exception,
            )
            next_normal.extend(result.normal)
            exits.extend(result.exits)
        normal = _compact_scope_states(next_normal)
        exits = _compact_scope_exits(exits)
        if not normal:
            break
    return _ScopeFlowResult(normal=normal, exits=exits)


def _apply_scope_finally(
    incoming: _ScopeFlowResult,
    statements: Sequence[ast.stmt],
    tag_guard_names: set[str],
    *,
    loop_body: bool,
) -> _ScopeFlowResult:
    normal: list[dict[str, tuple[_ScopeBinding, ...]]] = []
    exits: list[_ScopeFlowExit] = []
    sources = [
        *(_ScopeFlowExit("normal", state) for state in incoming.normal),
        *incoming.exits,
    ]
    for source in sources:
        final_flow = _scope_binding_flow(
            statements,
            tag_guard_names,
            source.state,
            loop_body=loop_body,
            active_exception=(source.exception_name if source.kind == "raise" else None),
        )
        exits.extend(final_flow.exits)
        if source.kind == "normal":
            normal.extend(final_flow.normal)
        else:
            exits.extend(
                _ScopeFlowExit(
                    kind=source.kind,
                    state=state,
                    exception_name=source.exception_name,
                )
                for state in final_flow.normal
            )
    return _compact_scope_flow(_ScopeFlowResult(normal=normal, exits=exits))


def _scope_final_binding_state(
    statements: Sequence[ast.stmt],
    tag_guard_names: set[str],
    incoming: dict[str, tuple[_ScopeBinding, ...]] | None = None,
    *,
    loop_body: bool = False,
) -> dict[str, tuple[_ScopeBinding, ...]] | None:
    """Interpret namespace writes and retain only normally completing paths."""

    flow = _scope_binding_flow(
        statements,
        tag_guard_names,
        incoming,
        loop_body=loop_body,
    )
    return _merge_scope_binding_states(flow.normal)


def _scope_final_bindings(
    statements: Sequence[ast.stmt],
    tag_guard_names: set[str],
) -> dict[str, tuple[_ScopeBinding, ...]]:
    return _scope_final_binding_state(statements, tag_guard_names) or {}


def _scope_state_before(
    statements: Sequence[ast.stmt],
    line: int,
    tag_guard_names: set[str],
) -> dict[str, tuple[_ScopeBinding, ...]]:
    """Return bindings after statements that finish before ``line``.

    Descriptor decorators are evaluated at definition time.  Looking at the
    final import table, or at every lexical assignment before a line, is not
    sufficient: an inactive branch must not shadow a builtin and ``del`` can
    restore fallback lookup.  The normal-path scope interpreter already owns
    those rules, so descriptor resolution reuses its state.
    """

    prefix = [
        statement
        for statement in statements
        if getattr(statement, "end_lineno", getattr(statement, "lineno", 0)) < line
    ]
    return _scope_final_binding_state(prefix, tag_guard_names) or {}


def _import_binding_reference(
    node: ast.Import | ast.ImportFrom,
    local_name: str,
    *,
    module: str,
    is_package: bool,
) -> str | None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            bound_name = alias.asname or alias.name.split(".", 1)[0]
            if bound_name == local_name:
                return alias.name if alias.asname else alias.name.split(".", 1)[0]
        return None

    source_module = _relative_import_module(
        module,
        is_package,
        node.level,
        node.module,
    )
    for alias in node.names:
        if alias.name == "*":
            continue
        if (alias.asname or alias.name) == local_name:
            return f"{source_module}.{alias.name}" if source_module else alias.name
    return None


def _scope_reference_variants(
    expression_node: ast.AST,
    *,
    statements: Sequence[ast.stmt],
    line: int,
    tag_guard_names: set[str],
    module: str,
    is_package: bool,
    fallback: Callable[[ast.AST], set[str | None]] | None = None,
    seen: frozenset[tuple[str, int]] = frozenset(),
) -> set[str | None]:
    """Resolve one expression on every normal path reaching ``line``.

    ``None`` is an explicit unresolved alternative.  Returning all variants
    lets callers distinguish a conditional classmethod/staticmethod choice
    from a genuinely dynamic decorator instead of silently selecting the last
    import seen in the file.
    """

    candidate = expression_node.func if isinstance(expression_node, ast.Call) else expression_node
    expression = _expression_name(candidate)
    if expression is None:
        return {None}
    root, separator, remainder = expression.partition(".")
    state = _scope_state_before(statements, line, tag_guard_names)
    bindings = state.get(root, ())

    def fallback_references() -> set[str | None]:
        if fallback is not None:
            return fallback(candidate)
        if not separator and root in _BUILTIN_DESCRIPTOR_DECORATORS.values():
            return {f"builtins.{root}"}
        return {None}

    if not bindings or all(binding.kind == "unbound" for binding in bindings):
        return fallback_references()

    references: set[str | None] = set()
    for binding in bindings:
        if binding.kind == "unbound":
            references.update(fallback_references())
            continue
        binding_node = binding.node
        reference: str | None = None
        if isinstance(binding_node, (ast.Import, ast.ImportFrom)):
            reference = _import_binding_reference(
                binding_node,
                root,
                module=module,
                is_package=is_package,
            )
        elif isinstance(binding_node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)):
            reference = f"{module}.{binding_node.name}"
        elif isinstance(binding_node, (ast.Assign, ast.AnnAssign)):
            value = binding_node.value
            recursion_key = (root, binding.line)
            if value is not None and recursion_key not in seen:
                nested = _scope_reference_variants(
                    value,
                    statements=statements,
                    line=binding.line,
                    tag_guard_names=tag_guard_names,
                    module=module,
                    is_package=is_package,
                    fallback=fallback,
                    seen=frozenset((*seen, recursion_key)),
                )
                references.update(
                    (f"{item}.{remainder}" if item is not None and separator else item) for item in nested
                )
                continue
        if reference is None:
            references.add(None)
        else:
            references.add(f"{reference}.{remainder}" if separator else reference)
    return references or {None}


def _decorator_reference_tuple(
    node: ast.AST | None,
    reference_resolver: Callable[[ast.AST], set[str | None]],
) -> tuple[str | None, ...]:
    """Keep one exact reference per decorator, or ``None`` when ambiguous."""

    if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
        return ()
    return tuple(
        next(iter(references)) if len(references) == 1 else None
        for decorator in node.decorator_list
        for references in (reference_resolver(decorator),)
    )


def _scope_decorator_reference_tuple(
    node: ast.AST | None,
    *,
    statements: Sequence[ast.stmt],
    tag_guard_names: set[str],
    module: str,
    is_package: bool,
) -> tuple[str | None, ...]:
    """Resolve function decorators against their enclosing module scope."""

    if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
        return ()
    line = getattr(node, "lineno", 0)
    return tuple(
        next(iter(references)) if len(references) == 1 else None
        for decorator in node.decorator_list
        for references in (
            _scope_reference_variants(
                decorator,
                statements=statements,
                line=line,
                tag_guard_names=tag_guard_names,
                module=module,
                is_package=is_package,
            ),
        )
    )


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
    if truth:
        text = expression
    elif isinstance(node, ast.Call) and _expression_name(node.func) == "hasattr":
        text = f"not {expression}"
    else:
        text = f"not ({expression})"
    return key, truth, text


def _canonical_guard_text(text: str) -> tuple[str, bool, str]:
    """Canonicalize a stored guard; keep synthetic flow labels opaque."""

    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        return f"opaque:{text}", True, text
    return _canonical_guard(node)


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
    if (
        isinstance(node, ast.Call)
        and not node.args
        and not node.keywords
        and _expression_name(node.func) == "current_platform.is_cpu"
    ):
        # This mapping is generated for the vllm-ascend/NPU consumer.  A CPU
        # implementation alias is not a runtime alternative for that target.
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


def _scope_bound_names_before(
    statements: Sequence[ast.stmt],
    line: int,
) -> set[str]:
    """Return conservative current-scope bindings created before ``line``.

    The helper deliberately does not enter nested function or class scopes.
    A binding seen on only one control-flow path is still returned: that is
    enough to prove that a bare builtin decorator is not unconditionally the
    builtin and must therefore be reported as ``unknown``.
    """

    names: set[str] = set()

    def visit_statement(node: ast.stmt) -> None:
        node_line = getattr(node, "lineno", 0)
        if node_line >= line:
            return
        names.update(_direct_bound_names(node))
        if isinstance(node, (ast.AsyncFor, ast.For)):
            names.update(_bound_target_names(node.target))
        elif isinstance(node, (ast.AsyncWith, ast.With)):
            for item in node.items:
                if item.optional_vars is not None:
                    names.update(_bound_target_names(item.optional_vars))
        elif isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            names.add("*")

        if isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)):
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ExceptHandler):
                if child.name and getattr(child, "lineno", 0) < line:
                    names.add(child.name)
                for statement in child.body:
                    visit_statement(statement)
            elif isinstance(child, ast.stmt):
                visit_statement(child)

    for statement in statements:
        visit_statement(statement)
    return names


def _resolved_decorator_reference(
    node: ast.AST,
    imports: dict[str, str],
    shadowed_names: set[str],
) -> str | None:
    """Resolve a decorator name only when its lexical root is provable."""

    expression_node = node.func if isinstance(node, ast.Call) else node
    expression = _expression_name(expression_node)
    if expression is None:
        return None
    root, separator, remainder = expression.partition(".")
    if root in imports:
        imported = imports[root]
        return f"{imported}.{remainder}" if separator else imported
    if root in {"classmethod", "property", "staticmethod"}:
        if root in shadowed_names or "*" in shadowed_names:
            return None
        return f"builtins.{root}"
    if root in shadowed_names:
        return None
    return expression


def _definition_descriptor_kinds(
    node: ast.AST | None,
    *,
    imports: dict[str, str] | None = None,
    shadowed_names: set[str] | None = None,
    known_properties: set[str] | None = None,
    ordinary_decorators: set[str] | frozenset[str] | None = None,
    reference_resolver: Callable[[ast.AST], set[str | None]] | None = None,
) -> tuple[str | None, ...]:
    """Classify the object produced by a function definition.

    Decorators are applied from bottom to top.  A known outer descriptor
    wrapper therefore determines the installed kind even when an inner
    decorator is dynamic.  An unknown outer decorator is never guessed.
    """

    if isinstance(node, ast.Lambda):
        return ("ordinary",)
    if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
        return (None,)

    imports = imports or {}
    shadowed_names = shadowed_names or set()
    known_properties = known_properties or set()
    ordinary_decorators = ordinary_decorators or set()
    kinds: set[str | None] = {"ordinary"}
    for decorator in reversed(node.decorator_list):
        expression = _expression_name(decorator)
        if (
            expression is not None
            and expression.rsplit(".", 1)[-1] in {"deleter", "getter", "setter"}
            and expression.rsplit(".", 1)[0] in known_properties
        ):
            kinds = {"property"}
            continue
        references = (
            reference_resolver(decorator)
            if reference_resolver is not None
            else {
                _resolved_decorator_reference(
                    decorator,
                    imports,
                    shadowed_names,
                )
            }
        )
        next_kinds: set[str | None] = set()
        for kind in kinds:
            for reference in references:
                descriptor_kind = _BUILTIN_DESCRIPTOR_DECORATORS.get(reference or "")
                if descriptor_kind is not None and not isinstance(decorator, ast.Call):
                    next_kinds.add(descriptor_kind)
                elif reference in _TRANSPARENT_DESCRIPTOR_DECORATORS:
                    next_kinds.add(kind)
                elif reference in ordinary_decorators:
                    next_kinds.add("ordinary" if kind == "ordinary" else "unknown")
                else:
                    next_kinds.add("unknown")
        kinds = next_kinds or {"unknown"}
    return tuple(sorted(kinds, key=lambda item: item or ""))


def _definition_descriptor_kind(
    node: ast.AST | None,
    *,
    imports: dict[str, str] | None = None,
    shadowed_names: set[str] | None = None,
    known_properties: set[str] | None = None,
    ordinary_decorators: set[str] | frozenset[str] | None = None,
    reference_resolver: Callable[[ast.AST], set[str | None]] | None = None,
) -> str | None:
    kinds = _definition_descriptor_kinds(
        node,
        imports=imports,
        shadowed_names=shadowed_names,
        known_properties=known_properties,
        ordinary_decorators=ordinary_decorators,
        reference_resolver=reference_resolver,
    )
    return kinds[0] if len(kinds) == 1 else "unknown"


def _scope_must_bound_names(
    statements: Sequence[ast.stmt],
    tag_guard_names: set[str],
    incoming: set[str] | None = None,
) -> set[str]:
    """Return names present after every normally completing active-main path."""

    initial = {name: (_scope_binding("value", ast.Pass()),) for name in incoming or ()}
    final = _scope_final_binding_state(
        statements,
        tag_guard_names,
        initial,
    )
    if final is None:
        return set()
    return {
        name
        for name, alternatives in final.items()
        if alternatives and all(alternative.kind != "unbound" for alternative in alternatives)
    }


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
class SignatureContract:
    """Static views of one callable after decorators and descriptor binding."""

    definition_signature: list[object] | None
    runtime_entry_signature: list[object] | None
    reported_signature: list[object] | None
    bound_call_signature: list[object] | None
    forwarded_targets: tuple[str, ...] = ()
    protocol: str = "python_call"
    status: str = "exact"
    provenance: tuple[str, ...] = ("ast_definition",)


@dataclass(frozen=True)
class StaticDecoratorTransform:
    wrapper_signature: list[object]
    preserves_reported_signature: bool
    wrapper_name: str


def _one_json_value(values: Iterable[object]) -> tuple[object, bool]:
    keyed = {json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")): value for value in values}
    if len(keyed) == 1:
        return next(iter(keyed.values())), True
    return None, False


def _merge_signature_contracts(
    contracts: Sequence[SignatureContract | None],
) -> tuple[SignatureContract | None, bool]:
    if not contracts or all(contract is None for contract in contracts):
        return None, False
    if any(contract is None for contract in contracts):
        present = [contract for contract in contracts if contract is not None]
        definition, _ = _one_json_value(contract.definition_signature for contract in present)
        return (
            SignatureContract(
                definition_signature=definition,
                runtime_entry_signature=None,
                reported_signature=None,
                bound_call_signature=None,
                forwarded_targets=tuple(
                    sorted({target for contract in present for target in contract.forwarded_targets})
                ),
                protocol="unknown",
                status="unknown",
                provenance=tuple(
                    dict.fromkeys(
                        [
                            *(item for contract in present for item in contract.provenance),
                            "conditional_signature_variants",
                        ]
                    )
                ),
            ),
            True,
        )

    present = [contract for contract in contracts if contract is not None]
    semantic_payloads = [
        [
            contract.definition_signature,
            contract.runtime_entry_signature,
            contract.reported_signature,
            contract.bound_call_signature,
            list(contract.forwarded_targets),
            contract.protocol,
            contract.status,
        ]
        for contract in present
    ]
    _, one_semantic_contract = _one_json_value(semantic_payloads)
    provenance = tuple(dict.fromkeys(item for contract in present for item in contract.provenance))
    if one_semantic_contract:
        first = present[0]
        return replace(first, provenance=provenance), False

    definition, _ = _one_json_value(contract.definition_signature for contract in present)
    runtime_entry, _ = _one_json_value(contract.runtime_entry_signature for contract in present)
    reported, _ = _one_json_value(contract.reported_signature for contract in present)
    forwarded_targets = tuple(sorted({target for contract in present for target in contract.forwarded_targets}))
    protocols = {contract.protocol for contract in present}
    return (
        SignatureContract(
            definition_signature=definition,
            runtime_entry_signature=runtime_entry,
            reported_signature=reported,
            bound_call_signature=None,
            forwarded_targets=forwarded_targets,
            protocol=next(iter(protocols)) if len(protocols) == 1 else "unknown",
            status="unknown",
            provenance=(*provenance, "conditional_signature_variants"),
        ),
        True,
    )


def _inspect_signature(
    signature: list[object],
) -> inspect.Signature | None:
    if len(signature) != 6:
        return None
    positional_only, positional_or_keyword = signature[1], signature[2]
    vararg, keyword_only, kwarg = signature[3], signature[4], signature[5]
    if not all(isinstance(items, list) for items in (positional_only, positional_or_keyword, keyword_only)):
        return None

    parameters: list[inspect.Parameter] = []

    def add_named(items: list[object], kind: inspect._ParameterKind) -> bool:
        for item in items:
            if not (
                isinstance(item, list) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], bool)
            ):
                return False
            parameters.append(
                inspect.Parameter(
                    item[0],
                    kind,
                    default=(inspect.Parameter.empty if item[1] else None),
                )
            )
        return True

    if not add_named(positional_only, inspect.Parameter.POSITIONAL_ONLY):
        return None
    if not add_named(positional_or_keyword, inspect.Parameter.POSITIONAL_OR_KEYWORD):
        return None
    if vararg is not None:
        if not isinstance(vararg, str):
            return None
        parameters.append(inspect.Parameter(vararg, inspect.Parameter.VAR_POSITIONAL))
    if not add_named(keyword_only, inspect.Parameter.KEYWORD_ONLY):
        return None
    if kwarg is not None:
        if not isinstance(kwarg, str):
            return None
        parameters.append(inspect.Parameter(kwarg, inspect.Parameter.VAR_KEYWORD))
    try:
        return inspect.Signature(parameters)
    except ValueError:
        return None


def _signature_call_witnesses(
    signature: list[object],
) -> list[tuple[list[object], dict[str, object]]]:
    positional_only = signature[1]
    positional_or_keyword = signature[2]
    keyword_only = signature[4]
    marker = {item[0]: object() for items in (positional_only, positional_or_keyword, keyword_only) for item in items}

    witnesses: list[tuple[list[object], dict[str, object]]] = []
    minimal_args = [marker[name] for name, required in positional_only if required]
    minimal_kwargs = {name: marker[name] for name, required in [*positional_or_keyword, *keyword_only] if required}
    witnesses.append((minimal_args, minimal_kwargs))

    all_positional_args = [marker[name] for name, _ in [*positional_only, *positional_or_keyword]]
    all_positional_kwargs = {name: marker[name] for name, _ in keyword_only}
    witnesses.append((all_positional_args, all_positional_kwargs))

    all_keyword_args = [marker[name] for name, _ in positional_only]
    all_keyword_kwargs = {name: marker[name] for name, _ in [*positional_or_keyword, *keyword_only]}
    witnesses.append((all_keyword_args, all_keyword_kwargs))

    for split in range(len(positional_or_keyword) + 1):
        args = [marker[name] for name, _ in positional_only]
        args.extend(marker[name] for name, _ in positional_or_keyword[:split])
        kwargs = {name: marker[name] for name, _ in [*positional_or_keyword[split:], *keyword_only]}
        witnesses.append((args, kwargs))

    if signature[3] is not None:
        witnesses.append(([*all_positional_args, object(), object()], dict(all_positional_kwargs)))
    if signature[5] is not None:
        witnesses.append((list(all_keyword_args), {**all_keyword_kwargs, "__interface_extra_keyword__": object()}))

    unique: dict[str, tuple[list[object], dict[str, object]]] = {}
    for args, kwargs in witnesses:
        shape = json.dumps(
            [len(args), sorted(kwargs)],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        unique[shape] = (args, kwargs)
    return list(unique.values())


def _accepts_signature_contract(
    upstream_signature: list[object],
    installed_signature: list[object],
) -> bool:
    if upstream_signature[0] != installed_signature[0]:
        return False
    candidate = _inspect_signature(installed_signature)
    if candidate is None:
        return False
    for args, kwargs in _signature_call_witnesses(upstream_signature):
        try:
            candidate.bind(*args, **kwargs)
        except TypeError:
            return False

    upstream_positional = [*upstream_signature[1], *upstream_signature[2]]
    installed_positional = [*installed_signature[1], *installed_signature[2]]
    for index, upstream_parameter in enumerate(upstream_positional):
        if index >= len(installed_positional):
            break
        installed_parameter = installed_positional[index]
        upstream_is_positional_or_keyword = index >= len(upstream_signature[1])
        installed_is_positional_only = index < len(installed_signature[1])
        if upstream_is_positional_or_keyword and (
            installed_is_positional_only or upstream_parameter[0] != installed_parameter[0]
        ):
            return False
    return True


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
    descriptor_kind: str | None = "ordinary"
    descriptor_variants: tuple[str | None, ...] = ()
    decorator_references: tuple[str | None, ...] = ()
    decorator_forwarded_targets: tuple[tuple[str, ...] | None, ...] | None = None
    property_accessor_nodes: tuple[ast.AST | None, ast.AST | None, ast.AST | None] | None = field(
        default=None,
        compare=False,
        hash=False,
        repr=False,
    )
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

    @property
    def property_accessors(
        self,
    ) -> tuple[list[object] | None, list[object] | None, list[object] | None] | None:
        if self.property_accessor_nodes is None:
            return None
        return tuple(_jsonable_signature(node) for node in self.property_accessor_nodes)


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
    star_imports: tuple[str, ...]


@dataclass(frozen=True)
class MroResult:
    owners: tuple[str, ...]
    complete: bool
    reason: str | None = None


@dataclass(frozen=True)
class EffectiveMethodResolution:
    """All outcomes of Python attribute lookup for one method name."""

    callable_owners: tuple[str, ...]
    may_be_missing: bool = False
    may_be_non_callable: bool = False
    has_unresolved_value: bool = False
    blocking_owners: tuple[str, ...] = ()

    @property
    def is_total_callable(self) -> bool:
        return bool(self.callable_owners) and not (
            self.may_be_missing or self.may_be_non_callable or self.has_unresolved_value
        )


@dataclass(frozen=True)
class RelationEvidence:
    file: str
    line: int
    scope: str | None = None
    guards: tuple[str, ...] = ()
    definition_line: int | None = None
    binding_line: int | None = None
    target_expression: str | None = None
    installed_descriptor_kind: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "file": self.file,
            "line": self.line,
        }
        if self.scope:
            payload["scope"] = self.scope
        if self.guards:
            payload["guards"] = list(self.guards)
        if self.definition_line is not None:
            payload["definition_line"] = self.definition_line
        if self.binding_line is not None:
            payload["binding_line"] = self.binding_line
        if self.target_expression is not None:
            payload["target_expression"] = self.target_expression
        if self.installed_descriptor_kind is not None:
            payload["installed_descriptor_kind"] = self.installed_descriptor_kind
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
    upstream_descriptor_kind: str | None = None
    downstream_descriptor_kind: str | None = None
    installed_descriptor_kind: str | None = None
    upstream_property_accessors: (
        tuple[
            list[object] | None,
            list[object] | None,
            list[object] | None,
        ]
        | None
    ) = field(default=None, compare=False, hash=False)
    downstream_property_accessors: (
        tuple[
            list[object] | None,
            list[object] | None,
            list[object] | None,
        ]
        | None
    ) = field(default=None, compare=False, hash=False)
    installed_property_accessors: (
        tuple[
            list[object] | None,
            list[object] | None,
            list[object] | None,
        ]
        | None
    ) = field(default=None, compare=False, hash=False)
    upstream_signature_contract: SignatureContract | None = field(
        default=None,
        compare=False,
        hash=False,
    )
    downstream_signature_contract: SignatureContract | None = field(
        default=None,
        compare=False,
        hash=False,
    )
    installed_signature_contract: SignatureContract | None = field(
        default=None,
        compare=False,
        hash=False,
    )
    override_paths: tuple[tuple[str, ...], ...] = field(
        default=(),
        compare=False,
        hash=False,
    )

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
        return (self.downstream_key(),)

    def comparison_exact_keys(self) -> tuple[tuple[str, ...], ...]:
        return tuple((*downstream_key, *self.upstream_key()) for downstream_key in self.comparison_downstream_keys())


@dataclass(frozen=True)
class HistoricalOverrideCandidate:
    """A downstream method whose new MRO has no upstream implementation.

    This is not yet a verified override relation.  The range layer must prove
    that the same lookup root resolved to a callable at ``old`` before it can
    promote the candidate into the exact relation graph.
    """

    lookup_root: str
    downstream_file: str
    downstream_owner: str
    downstream_qualified_owner: str
    downstream_name: str
    evidence_line: int


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
    supplemental: bool = False
    upstream_descriptor_kind: str | None = None
    downstream_descriptor_kind: str | None = None
    installed_descriptor_kind: str | None = None

    def as_dict(self) -> dict[str, Any]:
        if self.status not in FINDING_STATUSES:
            raise ValueError(f"unsupported finding status: {self.status}")
        payload = {
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
        if self.supplemental:
            payload["supplemental"] = True
        if self.upstream_descriptor_kind is not None:
            payload["upstream_descriptor_kind"] = self.upstream_descriptor_kind
        if self.downstream_descriptor_kind is not None:
            payload["downstream_descriptor_kind"] = self.downstream_descriptor_kind
        if self.installed_descriptor_kind is not None:
            payload["installed_descriptor_kind"] = self.installed_descriptor_kind
        return payload


class RepositoryIndex:
    """AST-only symbol and import index for one Python package."""

    def __init__(
        self,
        repo_root: Path,
        package_name: str,
        *,
        ordinary_descriptor_decorators: set[str] | frozenset[str] = frozenset(),
        _source_paths: Sequence[Path] | None = None,
        _finalize: bool = True,
    ):
        self.repo_root = repo_root.resolve()
        self.package_name = package_name
        self.ordinary_descriptor_decorators = frozenset(ordinary_descriptor_decorators)
        self.package_root = self.repo_root / package_name
        if not self.package_root.is_dir():
            raise ValueError(f"package directory not found: {self.package_root}")

        self.modules: dict[str, ModuleInfo] = {}
        self.classes: dict[str, ClassInfo] = {}
        self.callables: dict[str, CallableInfo] = {}
        self.callable_variants: dict[str, tuple[CallableInfo, ...]] = {}
        self.class_variants: dict[str, list[ClassInfo]] = defaultdict(list)
        self._class_variant_bindings: dict[
            str,
            list[dict[str, tuple[_ScopeBinding, ...]]],
        ] = defaultdict(list)
        self.class_base_conflicts: set[str] = set()
        self.final_bindings: dict[str, tuple[_ScopeBinding, ...]] = {}
        self.values: dict[str, ValueInfo] = {}
        self.aliases: dict[str, str] = {}
        self.typed_instance_aliases: set[str] = set()
        self.unconditional_exports: set[str] = set()
        self.unconditional_symbols: set[str] = set()
        self._unconditional_star_imports: set[tuple[str, str]] = set()
        self._pending_method_aliases: list[tuple[str, str, str, str, int]] = []
        self._descriptor_kinds_by_node: dict[int, str | None] = {}
        self._descriptor_variants_by_node: dict[int, tuple[str | None, ...]] = {}
        self._decorator_references_by_node: dict[
            int,
            tuple[str | None, ...],
        ] = {}
        self._class_alias_descriptor_kinds: dict[tuple[str, int], str | None] = {}
        self.parse_errors: list[dict[str, str]] = []
        self._source_paths = tuple(_source_paths) if _source_paths is not None else None
        self._finalize_after_parse = _finalize
        self._parse()
        del self._source_paths
        del self._finalize_after_parse

    def __getstate__(self) -> dict[str, object]:
        """Preserve AST identity-based maps when the index is serialized.

        Several resolver maps use ``id(ast_node)`` for fast lookup.  Numeric
        identities are process-local, so a plain pickle would silently retain
        stale keys after loading.  Store the AST node objects beside their
        values and rebuild the numeric keys in ``__setstate__`` instead.
        """

        state = dict(self.__dict__)
        nodes_by_id = {id(node): node for module in self.modules.values() for node in ast.walk(module.tree)}
        for name in (
            "_descriptor_kinds_by_node",
            "_descriptor_variants_by_node",
            "_decorator_references_by_node",
        ):
            mapping = state.pop(name)
            serialized: list[tuple[ast.AST, object]] = []
            for node_id, value in mapping.items():
                node = nodes_by_id.get(node_id)
                if node is None:
                    raise ValueError(f"repository index contains an unreachable AST identity in {name}")
                serialized.append((node, value))
            state[f"__serialized{name}"] = serialized
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        for name in (
            "_descriptor_kinds_by_node",
            "_descriptor_variants_by_node",
            "_decorator_references_by_node",
        ):
            serialized = state.pop(f"__serialized{name}")
            state[name] = {id(node): value for node, value in serialized}
        self.__dict__.update(state)

    @classmethod
    def _from_serial_file_fragments(
        cls,
        repo_root: Path,
        package_name: str,
        *,
        ordinary_descriptor_decorators: set[str] | frozenset[str] = frozenset(),
    ) -> RepositoryIndex:
        """Build an index through isolated file fragments for parity tests."""

        package_root = repo_root.resolve() / package_name
        paths = sorted(package_root.rglob("*.py"))
        combined = cls(
            repo_root,
            package_name,
            ordinary_descriptor_decorators=ordinary_descriptor_decorators,
            _source_paths=(),
            _finalize=False,
        )
        for path in paths:
            fragment = cls(
                repo_root,
                package_name,
                ordinary_descriptor_decorators=ordinary_descriptor_decorators,
                _source_paths=(path,),
                _finalize=False,
            )
            combined._merge_pre_final_fragment(fragment)
        combined._finalize_index()
        return combined

    def _parse(self) -> None:
        """Parse repository modules and build the static symbol indexes."""
        paths = sorted(self.package_root.rglob("*.py")) if self._source_paths is None else sorted(self._source_paths)
        for path in paths:
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
            star_imports: list[str] = []
            annotated_exports: list[tuple[str, str]] = []
            tag_guard_names = _tag_guard_names(tree.body)
            module_final_bindings = _scope_final_bindings(
                tree.body,
                tag_guard_names,
            )
            self.final_bindings.update(
                {f"{module}.{name}": alternatives for name, alternatives in module_final_bindings.items()}
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
                    if not any(
                        binding.kind == "class" and binding.node is node
                        for binding in module_final_bindings.get(node.name, ())
                    ):
                        continue
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
                    module_shadowed_names = _scope_bound_names_before(
                        tree.body,
                        getattr(node, "lineno", 0),
                    )
                    descriptor_kinds: dict[int, str | None] = {}
                    descriptor_variants: dict[int, tuple[str | None, ...]] = {}
                    property_accessors: dict[
                        int,
                        tuple[ast.AST | None, ast.AST | None, ast.AST | None],
                    ] = {}

                    def module_reference_resolver(
                        expression: ast.AST,
                        module_tree: ast.Module = tree,
                        class_line: int = getattr(node, "lineno", 0),
                        active_tag_guards: set[str] = tag_guard_names,
                        current_module: str = module,
                        current_is_package: bool = is_package,
                    ) -> set[str | None]:
                        return _scope_reference_variants(
                            expression,
                            statements=module_tree.body,
                            line=class_line,
                            tag_guard_names=active_tag_guards,
                            module=current_module,
                            is_package=current_is_package,
                        )

                    def class_reference_resolver(
                        expression: ast.AST,
                        line: int,
                        class_node: ast.ClassDef = node,
                        active_tag_guards: set[str] = tag_guard_names,
                        current_class: str = qualified_name,
                        module_fallback: Callable[[ast.AST], set[str | None]] = module_reference_resolver,
                    ) -> set[str | None]:
                        return _scope_reference_variants(
                            expression,
                            statements=class_node.body,
                            line=line,
                            tag_guard_names=active_tag_guards,
                            module=current_class,
                            is_package=False,
                            fallback=module_fallback,
                        )

                    class_functions = sorted(
                        (
                            statement
                            for statement in _main_module_statements(
                                node.body,
                                tag_guard_names,
                            )
                            if isinstance(
                                statement,
                                (ast.AsyncFunctionDef, ast.FunctionDef),
                            )
                        ),
                        key=lambda statement: (
                            getattr(statement, "lineno", 0),
                            getattr(statement, "col_offset", 0),
                        ),
                    )

                    def callable_node_for_expression(
                        expression_node: ast.AST,
                        line: int,
                        class_node: ast.ClassDef = node,
                        active_tag_guards: set[str] = tag_guard_names,
                        current_module: str = module,
                        current_imports: dict[str, str] = imports,
                        local_classes: dict[str, ClassInfo] = classes,
                        local_functions: dict[str, CallableInfo] = functions,
                    ) -> ast.AST | None:
                        expression = _expression_name(expression_node)
                        if expression is None:
                            return None
                        if "." not in expression:
                            local_state = _scope_state_before(
                                class_node.body,
                                line,
                                active_tag_guards,
                            )
                            local_nodes = {
                                alternative.node
                                for alternative in local_state.get(
                                    expression,
                                    (),
                                )
                                if alternative.kind == "function" and alternative.node is not None
                            }
                            if len(local_nodes) == 1:
                                return next(iter(local_nodes))
                        resolved = _resolve_bound_reference(
                            current_module,
                            expression,
                            current_imports,
                            {*local_classes, *local_functions},
                        )
                        callable_info = self.find_callable(resolved)
                        return callable_info.node if callable_info is not None else None

                    def register_property_assignment(
                        binding: _ScopeBinding,
                        accessors_by_node: dict[
                            int,
                            tuple[
                                ast.AST | None,
                                ast.AST | None,
                                ast.AST | None,
                            ],
                        ] = property_accessors,
                    ) -> None:
                        binding_node = binding.node
                        if binding_node is None or id(binding_node) in accessors_by_node:
                            return
                        if isinstance(binding_node, (ast.AnnAssign, ast.Assign)):
                            value = binding_node.value
                        else:
                            return
                        if not (
                            isinstance(value, ast.Call)
                            and len(value.args) <= 3
                            and not value.keywords
                            and class_reference_resolver(
                                value.func,
                                binding.line,
                            )
                            == {"builtins.property"}
                        ):
                            return
                        nodes = [callable_node_for_expression(argument, binding.line) for argument in value.args[:3]]
                        nodes.extend([None] * (3 - len(nodes)))
                        if value.args and nodes[0] is None:
                            return
                        accessors = (nodes[0], nodes[1], nodes[2])
                        accessors_by_node[id(binding_node)] = accessors

                    for function_node in class_functions:
                        function_line = getattr(function_node, "lineno", 0)
                        class_state = _scope_state_before(
                            node.body,
                            function_line,
                            tag_guard_names,
                        )
                        for alternatives in class_state.values():
                            for alternative in alternatives:
                                register_property_assignment(alternative)
                        known_properties = {
                            name
                            for name, alternatives in class_state.items()
                            if alternatives
                            and all(
                                alternative.node is not None and id(alternative.node) in property_accessors
                                for alternative in alternatives
                            )
                        }
                        class_shadowed_names = {
                            *module_shadowed_names,
                            *_scope_bound_names_before(
                                node.body,
                                function_line,
                            ),
                        }
                        variants_for_node = _definition_descriptor_kinds(
                            function_node,
                            imports=imports,
                            shadowed_names=class_shadowed_names,
                            known_properties=known_properties,
                            ordinary_decorators=self.ordinary_descriptor_decorators,
                            reference_resolver=lambda expression, line=function_line: class_reference_resolver(
                                expression,
                                line,
                            ),
                        )
                        descriptor_kind = variants_for_node[0] if len(variants_for_node) == 1 else "unknown"
                        descriptor_kinds[id(function_node)] = descriptor_kind
                        descriptor_variants[id(function_node)] = variants_for_node
                        self._descriptor_kinds_by_node[id(function_node)] = descriptor_kind
                        self._descriptor_variants_by_node[id(function_node)] = variants_for_node
                        self._decorator_references_by_node[id(function_node)] = _decorator_reference_tuple(
                            function_node,
                            lambda expression, line=function_line: class_reference_resolver(
                                expression,
                                line,
                            ),
                        )

                        accessor_kind: str | None = None
                        accessor_name: str | None = None
                        for decorator in reversed(function_node.decorator_list):
                            expression = _expression_name(decorator)
                            if expression is None or "." not in expression:
                                continue
                            candidate_name, candidate_kind = expression.rsplit(".", 1)
                            if candidate_kind in {"deleter", "getter", "setter"}:
                                accessor_name = candidate_name
                                accessor_kind = candidate_kind
                                break

                        resolved_accessors: (
                            tuple[
                                ast.AST | None,
                                ast.AST | None,
                                ast.AST | None,
                            ]
                            | None
                        ) = None
                        if accessor_name in known_properties and accessor_kind is not None:
                            accessor_bases = {
                                property_accessors[id(alternative.node)]
                                for alternative in class_state.get(accessor_name, ())
                                if alternative.node is not None and id(alternative.node) in property_accessors
                            }
                            if len(accessor_bases) == 1:
                                getter, setter, deleter = next(iter(accessor_bases))
                                if accessor_kind == "getter":
                                    getter = function_node
                                elif accessor_kind == "setter":
                                    setter = function_node
                                else:
                                    deleter = function_node
                                resolved_accessors = (getter, setter, deleter)
                        elif descriptor_kind == "property" and any(
                            _BUILTIN_DESCRIPTOR_DECORATORS.get(reference or "") == "property"
                            for decorator in function_node.decorator_list
                            for reference in class_reference_resolver(
                                decorator,
                                function_line,
                            )
                        ):
                            resolved_accessors = (function_node, None, None)

                        if resolved_accessors is not None:
                            property_accessors[id(function_node)] = resolved_accessors
                    self.final_bindings.update(
                        {
                            f"{qualified_name}.{name}": alternatives
                            for name, alternatives in class_final_bindings.items()
                        }
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
                    self.class_variants[qualified_name].append(info)
                    self._class_variant_bindings[qualified_name].append(class_final_bindings)
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
                        descriptor_kind=None,
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
                                descriptor_kind=descriptor_kinds.get(
                                    id(candidate),
                                    "unknown",
                                ),
                                descriptor_variants=descriptor_variants.get(
                                    id(candidate),
                                    (),
                                ),
                                decorator_references=self._decorator_references_by_node.get(
                                    id(candidate),
                                    (),
                                ),
                                property_accessor_nodes=property_accessors.get(
                                    id(candidate),
                                ),
                                signature_override=(
                                    _jsonable_signature(property_accessors[id(candidate)][0])
                                    if id(candidate) in property_accessors
                                    and property_accessors[id(candidate)][0] is not None
                                    else None
                                ),
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
                        class_reference_resolver,
                        tag_guard_names,
                    )
                elif isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                    imports.pop(node.name, None)
                    qualified_name = f"{module}.{node.name}"
                    decorator_references = _scope_decorator_reference_tuple(
                        node,
                        statements=tree.body,
                        tag_guard_names=tag_guard_names,
                        module=module,
                        is_package=is_package,
                    )
                    self._decorator_references_by_node[id(node)] = decorator_references
                    info = CallableInfo(
                        qualified_name=qualified_name,
                        module=module,
                        file=relative_file,
                        owner=None,
                        name=node.name,
                        node=node,
                        descriptor_kind=_definition_descriptor_kind(
                            node,
                            imports=imports,
                            shadowed_names=_scope_bound_names_before(
                                tree.body,
                                getattr(node, "lineno", 0),
                            ),
                            ordinary_decorators=self.ordinary_descriptor_decorators,
                        ),
                        decorator_references=decorator_references,
                    )
                    functions[node.name] = info
                    self._descriptor_kinds_by_node[id(node)] = info.descriptor_kind
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
                variants_list: list[CallableInfo] = []
                for candidate in candidates:
                    decorator_references = _scope_decorator_reference_tuple(
                        candidate,
                        statements=tree.body,
                        tag_guard_names=tag_guard_names,
                        module=module,
                        is_package=is_package,
                    )
                    self._decorator_references_by_node[id(candidate)] = decorator_references
                    variants_list.append(
                        CallableInfo(
                            qualified_name=qualified_name,
                            module=module,
                            file=relative_file,
                            owner=None,
                            name=function_name,
                            node=candidate,
                            descriptor_kind=_definition_descriptor_kind(
                                candidate,
                                imports=imports,
                                shadowed_names=_scope_bound_names_before(
                                    tree.body,
                                    getattr(candidate, "lineno", 0),
                                ),
                                ordinary_decorators=self.ordinary_descriptor_decorators,
                            ),
                            decorator_references=decorator_references,
                        )
                    )
                variants = tuple(variants_list)
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
                decorator_references = _scope_decorator_reference_tuple(
                    node,
                    statements=tree.body,
                    tag_guard_names=tag_guard_names,
                    module=module,
                    is_package=is_package,
                )
                self._decorator_references_by_node.setdefault(
                    id(node),
                    decorator_references,
                )
                loose_functions[node.name].append(
                    CallableInfo(
                        qualified_name=qualified_name,
                        module=module,
                        file=relative_file,
                        owner=None,
                        name=node.name,
                        node=node,
                        descriptor_kind=_definition_descriptor_kind(
                            node,
                            imports=imports,
                            shadowed_names=_scope_bound_names_before(
                                tree.body,
                                getattr(node, "lineno", 0),
                            ),
                            ordinary_decorators=self.ordinary_descriptor_decorators,
                        ),
                        decorator_references=self._decorator_references_by_node[id(node)],
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
                star_imports=tuple(star_imports),
            )
            self.modules[module] = module_info
            for local_name, target in imports.items():
                self.aliases[f"{module}.{local_name}"] = target
            for export_name, target in typed_lazy_exports.items():
                self.aliases[f"{module}.{export_name}"] = target
                self.typed_instance_aliases.add(f"{module}.{export_name}")

        if self._finalize_after_parse:
            self._finalize_index()

    def _finalize_index(self) -> None:
        self._aggregate_class_variants()
        self._materialize_star_import_aliases()
        self._materialize_dataclass_initializers()
        self._materialize_class_callable_aliases()
        self._validate_index_consistency()

    def _merge_pre_final_fragment(self, fragment: RepositoryIndex) -> None:
        """Merge one source-ordered file fragment before global finalization."""

        for name in (
            "modules",
            "classes",
            "callables",
            "callable_variants",
            "final_bindings",
            "values",
            "aliases",
            "_descriptor_kinds_by_node",
            "_descriptor_variants_by_node",
            "_decorator_references_by_node",
            "_class_alias_descriptor_kinds",
        ):
            getattr(self, name).update(getattr(fragment, name))
        for name in ("class_variants", "_class_variant_bindings"):
            destination = getattr(self, name)
            for key, values in getattr(fragment, name).items():
                destination[key].extend(values)
        for name in (
            "class_base_conflicts",
            "typed_instance_aliases",
            "unconditional_exports",
            "unconditional_symbols",
            "_unconditional_star_imports",
        ):
            getattr(self, name).update(getattr(fragment, name))
        self._pending_method_aliases.extend(fragment._pending_method_aliases)
        self.parse_errors.extend(fragment.parse_errors)

    def _validate_index_consistency(self) -> None:
        """Fail closed when representative and variant indexes drift apart."""
        for qualified_name, variants in self.callable_variants.items():
            if not variants:
                raise RuntimeError(f"empty callable variant index: {qualified_name}")
            if self.callables.get(qualified_name) != variants[0]:
                raise RuntimeError(f"callable representative does not match first variant: {qualified_name}")
        missing_classes = self.class_variants.keys() - self.classes.keys()
        if missing_classes:
            names = ", ".join(sorted(missing_classes))
            raise RuntimeError(f"class variants have no representative: {names}")

    def _aggregate_class_variants(self) -> None:
        """Merge same-name class definitions that can be final at runtime."""

        for qualified_name, variants in self.class_variants.items():
            if len(variants) < 2:
                continue
            binding_states = self._class_variant_bindings[qualified_name]
            merged_bindings = _merge_scope_binding_states(binding_states) or {}
            member_names = {name for state in binding_states for name in state}
            for member_name in member_names:
                member_qualified_name = f"{qualified_name}.{member_name}"
                alternatives = merged_bindings[member_name]
                self.final_bindings[member_qualified_name] = alternatives
                function_nodes = tuple(
                    dict.fromkeys(
                        alternative.node
                        for alternative in alternatives
                        if alternative.kind == "function" and alternative.node is not None
                    )
                )
                self.unconditional_symbols.discard(member_qualified_name)
                if not function_nodes:
                    self.callables.pop(member_qualified_name, None)
                    self.callable_variants.pop(member_qualified_name, None)
                    continue
                representative = variants[0]
                callable_variants = tuple(
                    CallableInfo(
                        qualified_name=member_qualified_name,
                        module=representative.module,
                        file=representative.file,
                        owner=representative.name,
                        name=member_name,
                        node=node,
                        descriptor_kind=self._descriptor_kinds_by_node.get(
                            id(node),
                            "unknown",
                        ),
                        decorator_references=self._decorator_references_by_node.get(
                            id(node),
                            (),
                        ),
                    )
                    for node in function_nodes
                )
                self.callables[member_qualified_name] = callable_variants[0]
                self.callable_variants[member_qualified_name] = callable_variants
                if (
                    qualified_name in self.unconditional_symbols
                    and alternatives
                    and all(alternative.kind == "function" for alternative in alternatives)
                ):
                    self.unconditional_symbols.add(member_qualified_name)

            base_shapes = {(variant.bases, variant.resolved_bases) for variant in variants}
            if len(base_shapes) != 1:
                self.class_base_conflicts.add(qualified_name)
            representative = variants[0]
            method_variants = {
                name: tuple(
                    candidate.node
                    for candidate in self.callable_variants.get(
                        f"{qualified_name}.{name}",
                        (),
                    )
                    if candidate.node is not None
                )
                for name in member_names
                if self.callable_variants.get(f"{qualified_name}.{name}")
            }
            aggregate = ClassInfo(
                qualified_name=qualified_name,
                module=representative.module,
                file=representative.file,
                name=representative.name,
                bases=representative.bases,
                resolved_bases=representative.resolved_bases,
                methods={name: candidates[0] for name, candidates in method_variants.items()},
                method_variants=method_variants,
            )
            self.classes[qualified_name] = aggregate
            self.modules[aggregate.module].classes[aggregate.name] = aggregate
            class_nodes = tuple(
                candidate.node
                for candidate in self.final_bindings.get(qualified_name, ())
                if candidate.kind == "class" and candidate.node is not None
            )
            if class_nodes:
                self.callables[qualified_name] = CallableInfo(
                    qualified_name=qualified_name,
                    module=aggregate.module,
                    file=aggregate.file,
                    owner=None,
                    name=aggregate.name,
                    node=class_nodes[0],
                    descriptor_kind=None,
                )

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
                descriptor_kind="ordinary",
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
        reference_resolver: Callable[[ast.AST, int], set[str | None]],
        tag_guard_names: set[str],
    ) -> None:
        explicit_methods = _method_nodes(node)
        for statement in _main_module_statements(node.body, tag_guard_names):
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
                wrapper_kinds = {
                    _BUILTIN_DESCRIPTOR_DECORATORS[reference]
                    for reference in reference_resolver(
                        value.func,
                        getattr(statement, "lineno", 0),
                    )
                    if reference in _BUILTIN_DESCRIPTOR_DECORATORS
                }
                if len(wrapper_kinds) != 1:
                    continue
                if len(value.args) != 1 or value.keywords:
                    continue
                kind = next(iter(wrapper_kinds))
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
                self._class_alias_descriptor_kinds[(f"{class_name}.{target.id}", getattr(statement, "lineno", 0))] = (
                    kind
                )

    def _materialize_class_callable_aliases(self) -> None:
        grouped: dict[
            tuple[str, str],
            list[tuple[str, str, int]],
        ] = defaultdict(list)
        for class_name, member_name, target, kind, line in self._pending_method_aliases:
            grouped[(class_name, member_name)].append((target, kind, line))

        for (class_name, member_name), pending in grouped.items():
            class_info = self.classes[class_name]
            qualified_name = f"{class_name}.{member_name}"
            variants = list(self.callable_variants.get(qualified_name, ()))
            exact_alias_targets: list[str] = []
            for target, kind, line in pending:
                source = self.find_callable(target)
                if source is None or not isinstance(
                    source.node,
                    (ast.AsyncFunctionDef, ast.FunctionDef, ast.Lambda),
                ):
                    continue

                source_variants = source.descriptor_variants or (source.descriptor_kind,)
                if kind in {"classmethod", "property", "staticmethod"}:
                    installed_variants = (kind,)
                elif source.owner is None:
                    installed_variants = source_variants
                else:
                    installed_variants = tuple(
                        sorted(
                            {
                                (
                                    "ordinary"
                                    if candidate in {"ordinary", "staticmethod"}
                                    else ("property" if candidate == "property" else "unknown")
                                )
                                for candidate in source_variants
                            }
                        )
                    )
                descriptor_kind = installed_variants[0] if len(installed_variants) == 1 else "unknown"
                property_nodes = None
                if descriptor_kind == "property":
                    property_nodes = (source.node, None, None) if kind == "property" else source.property_accessor_nodes
                variants.append(
                    CallableInfo(
                        qualified_name=qualified_name,
                        module=class_info.module,
                        file=class_info.file,
                        owner=class_info.name,
                        name=member_name,
                        node=source.node,
                        binding_line=line,
                        origin_kind=kind,
                        descriptor_kind=descriptor_kind,
                        descriptor_variants=installed_variants,
                        decorator_references=source.decorator_references,
                        property_accessor_nodes=property_nodes,
                        signature_override=source.signature,
                    )
                )
                exact_alias_targets.append(target)

            if not variants:
                continue
            unique = {
                (
                    candidate.binding_line or -1,
                    getattr(candidate.node, "lineno", 0),
                    candidate.descriptor_kind or "",
                    json.dumps(
                        candidate.signature,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ): candidate
                for candidate in variants
            }
            variants = [unique[key] for key in sorted(unique)]
            class_info.methods.setdefault(member_name, variants[0].node)
            class_info.method_variants[member_name] = tuple(candidate.node for candidate in variants)
            self.callables[qualified_name] = variants[0]
            self.callable_variants[qualified_name] = tuple(variants)
            if (
                class_name in self.unconditional_symbols
                and exact_alias_targets
                and all(target in self.unconditional_symbols for target in exact_alias_targets)
            ):
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
        visited_aliases: set[str] = set()
        while result not in visited:
            visited.add(result)
            replacement = None
            for alias in sorted(self.aliases, key=len, reverse=True):
                if result == alias or result.startswith(f"{alias}."):
                    if alias in visited_aliases:
                        # An alias can only match again when another alias maps
                        # back to it or when it expands into its own namespace.
                        # Neither chain has one statically provable canonical
                        # target, so fail closed instead of growing forever.
                        return qualified_name
                    visited_aliases.add(alias)
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

    def find_final_bindings(
        self,
        qualified_name: str,
    ) -> tuple[_ScopeBinding, ...]:
        canonical = self.canonical_name(qualified_name)
        refined = {
            candidate
            for binding in self.final_bindings.get(canonical, ())
            for candidate in self._refine_final_binding_variants(
                canonical,
                binding,
                frozenset(),
            )
        }
        return tuple(sorted(refined))

    def _final_alias_target(
        self,
        qualified_name: str,
        binding: _ScopeBinding,
    ) -> str | None:
        node = binding.node
        if isinstance(node, (ast.AnnAssign, ast.Assign)):
            value = node.value
        else:
            return None
        alias_kind = self._class_alias_descriptor_kinds.get((qualified_name, binding.line))
        if isinstance(value, ast.Call):
            wrapper = alias_kind
            if wrapper not in {"classmethod", "property", "staticmethod"} or len(value.args) != 1 or value.keywords:
                return None
            owner_name = qualified_name.rsplit(".", 1)[0]
            if owner_name not in self.classes:
                # classmethod/staticmethod objects are descriptors only when
                # installed in a class namespace; at module scope they are
                # ordinary non-callable values.
                return None
            value = value.args[0]
        elif binding.kind != "alias" and alias_kind != "callable_alias":
            return None
        expression = _expression_name(value)
        if expression is None:
            return None

        owner_name = qualified_name.rsplit(".", 1)[0]
        owner = self.classes.get(owner_name)
        if owner is not None:
            same_class = f"{owner_name}.{expression}"
            if "." not in expression and self.find_callable(same_class) is not None:
                return self.canonical_name(same_class)
            module = owner.module
        else:
            modules = [name for name in self.modules if qualified_name.startswith(f"{name}.")]
            if not modules:
                return None
            module = max(modules, key=len)
        return self.canonical_name(
            self.resolve_reference(
                module,
                expression,
            )
        )

    def _refine_final_binding_variants(
        self,
        qualified_name: str,
        binding: _ScopeBinding,
        seen: frozenset[str],
    ) -> tuple[_ScopeBinding, ...]:
        """Propagate every final kind through a provable callable alias."""

        if qualified_name in seen:
            return (binding,)
        target = self._final_alias_target(qualified_name, binding)
        if target is None:
            return (binding,)
        source_bindings = self.final_bindings.get(target, ())
        if not source_bindings:
            return (self._refine_final_binding(qualified_name, binding),)
        refined_sources = (
            candidate
            for source_binding in source_bindings
            for candidate in self._refine_final_binding_variants(
                target,
                source_binding,
                frozenset((*seen, qualified_name)),
            )
        )
        return tuple(
            replace(
                binding,
                kind=source.kind,
                node=source.node,
            )
            for source in refined_sources
        )

    def _refine_final_binding(
        self,
        qualified_name: str,
        binding: _ScopeBinding,
    ) -> _ScopeBinding:
        target = self._final_alias_target(qualified_name, binding)
        if target is None:
            return binding
        source = self.find_callable(target)
        if source is not None:
            return replace(
                binding,
                kind="function",
                node=source.node,
            )
        source_class = self.find_class(target)
        if source_class is not None:
            return replace(
                binding,
                kind="class",
                node=self.find_callable(source_class.qualified_name).node,
            )
        return binding

    def find_final_callable_variants(
        self,
        qualified_name: str,
        seen: frozenset[str] = frozenset(),
    ) -> tuple[CallableInfo, ...]:
        canonical = self.canonical_name(qualified_name)
        if canonical in seen:
            return ()
        raw = self.final_bindings.get(canonical, ())
        if not raw:
            return self.find_callable_variants(canonical)

        endpoint = self.find_callable(canonical)
        direct = self.find_callable_variants(canonical)
        variants: list[CallableInfo] = []
        for binding in raw:
            if binding.kind == "function" and binding.node is not None:
                matching = [candidate for candidate in direct if candidate.node is binding.node]
                if matching:
                    variants.extend(matching)
                elif endpoint is not None:
                    variants.append(replace(endpoint, node=binding.node))
                continue
            target = self._final_alias_target(canonical, binding)
            if target is None:
                continue
            for source in self.find_final_callable_variants(
                target,
                frozenset((*seen, canonical)),
            ):
                alias_template = next(
                    (candidate for candidate in direct if candidate.binding_line == binding.line),
                    endpoint,
                )
                if alias_template is None:
                    owner_name, member_name = canonical.rsplit(".", 1)
                    owner = self.find_class(owner_name)
                    if owner is None:
                        variants.append(source)
                    else:
                        variants.append(
                            replace(
                                source,
                                qualified_name=canonical,
                                module=owner.module,
                                file=owner.file,
                                owner=owner.name,
                                name=member_name,
                                binding_line=binding.line,
                                origin_kind="callable_alias",
                            )
                        )
                else:
                    variants.append(
                        replace(
                            alias_template,
                            node=source.node,
                            binding_line=binding.line,
                            origin_kind="callable_alias",
                            signature_override=(alias_template.signature_override or source.signature),
                        )
                    )

        unique: dict[tuple[str, str | None, str, int, int, str, str], CallableInfo] = {}
        for candidate in variants:
            key = (
                candidate.file,
                candidate.owner,
                candidate.name,
                getattr(candidate.node, "lineno", 0),
                candidate.binding_line if candidate.binding_line is not None else -1,
                json.dumps(candidate.signature, ensure_ascii=False, separators=(",", ":")),
                candidate.descriptor_kind or "",
            )
            unique[key] = candidate
        return tuple(unique[key] for key in sorted(unique))

    def find_loose_function(self, module: str, name: str) -> CallableInfo | None:
        candidates = self.modules[module].loose_functions.get(name, [])
        return candidates[0] if len(candidates) == 1 else None

    def find_value(self, qualified_name: str) -> ValueInfo | None:
        direct = self.values.get(qualified_name)
        if direct is not None:
            return direct
        return self.values.get(self.canonical_name(qualified_name))


def _repository_fragment_batch(
    args: tuple[str, str, tuple[str, ...], tuple[str, ...]],
) -> list[tuple[str, RepositoryIndex]]:
    repo_root_value, package_name, relative_files, ordinary_decorators = args
    repo_root = Path(repo_root_value)
    results: list[tuple[str, RepositoryIndex]] = []
    for relative_file in relative_files:
        path = repo_root.joinpath(*relative_file.split("/"))
        results.append(
            (
                relative_file,
                RepositoryIndex(
                    repo_root,
                    package_name,
                    ordinary_descriptor_decorators=frozenset(ordinary_decorators),
                    _source_paths=(path,),
                    _finalize=False,
                ),
            )
        )
    return results


def _repository_file_cache_identities(
    repo_root: Path,
    package_name: str,
    source_version: str | None,
    relative_files: Sequence[str],
    ordinary_descriptor_decorators: frozenset[str],
) -> tuple[dict[str, tuple[str, str]] | None, str | None]:
    if not source_version:
        return None, "source version is unavailable"
    try:
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                package_name,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if dirty:
            return None, f"{package_name} contains uncommitted source changes"
        tree = subprocess.run(
            ["git", "-C", str(repo_root), "ls-tree", "-r", source_version, "--", package_name],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        return None, f"Git file-cache identity failed: {error}"

    blob_ids: dict[str, str] = {}
    for line in tree.splitlines():
        metadata, separator, relative_file = line.partition("\t")
        if not separator or not relative_file.endswith(".py"):
            continue
        parts = metadata.split()
        if len(parts) == 3 and parts[1] == "blob":
            blob_ids[relative_file] = parts[2]
    missing = sorted(set(relative_files) - blob_ids.keys())
    if missing:
        return None, f"Git file-cache identity is missing {len(missing)} Python files"

    identities: dict[str, tuple[str, str]] = {}
    for relative_file in relative_files:
        identity = {
            "cache_schema_version": REPOSITORY_FILE_FRAGMENT_CACHE_SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "python_cache_tag": sys.implementation.cache_tag,
            "package_name": package_name,
            "relative_file": relative_file,
            "blob_sha": blob_ids[relative_file],
            "ordinary_descriptor_decorators": sorted(ordinary_descriptor_decorators),
        }
        serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        identities[relative_file] = (
            hashlib.sha256(serialized.encode()).hexdigest(),
            serialized,
        )
    return identities, None


def _sqlite_rows(
    connection: sqlite3.Connection,
    keys: Sequence[str],
) -> Iterable[tuple[str, str, bytes]]:
    for start in range(0, len(keys), 500):
        batch = keys[start : start + 500]
        placeholders = ",".join("?" for _ in batch)
        yield from connection.execute(
            f"SELECT cache_key, identity, payload FROM fragments WHERE cache_key IN ({placeholders})",  # noqa: S608
            batch,
        )


def _repository_index_from_file_fragments(
    repo_root: Path,
    package_name: str,
    *,
    ordinary_descriptor_decorators: frozenset[str],
    source_version: str | None,
    cache_dir: Path | None,
    index_workers: int,
) -> tuple[RepositoryIndex, dict[str, object]]:
    if index_workers < 1:
        raise ValueError("index_workers must be at least 1")
    repo_root = repo_root.resolve()
    package_root = repo_root / package_name
    paths = sorted(package_root.rglob("*.py"))
    relative_files = tuple(path.relative_to(repo_root).as_posix() for path in paths)
    status: dict[str, object] = {
        "enabled": cache_dir is not None,
        "status": "disabled",
        "database": None,
        "files_total": len(relative_files),
        "cache_hits": 0,
        "cache_misses": len(relative_files),
        "invalid_entries": 0,
        "workers_requested": index_workers,
        "workers_used": 1,
        "load_seconds": 0.0,
        "build_seconds": 0.0,
        "write_seconds": 0.0,
        "merge_finalize_seconds": 0.0,
        "database_bytes": None,
        "hit_ratio": 0.0,
        "reason": None,
    }
    identities: dict[str, tuple[str, str]] | None = None
    connection: sqlite3.Connection | None = None
    fragments: dict[str, RepositoryIndex] = {}
    load_started = time.perf_counter()
    if cache_dir is not None:
        identities, reason = _repository_file_cache_identities(
            repo_root,
            package_name,
            source_version,
            relative_files,
            ordinary_descriptor_decorators,
        )
        if identities is None:
            status.update(status="bypassed", reason=reason)
        else:
            database = cache_dir.resolve() / (
                f"{package_name}-file-fragments-v{REPOSITORY_FILE_FRAGMENT_CACHE_SCHEMA_VERSION}.sqlite3"
            )
            status["database"] = str(database)
            try:
                database.parent.mkdir(parents=True, exist_ok=True)
                connection = sqlite3.connect(database, timeout=30)
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS fragments ("
                    "cache_key TEXT PRIMARY KEY, identity TEXT NOT NULL, payload BLOB NOT NULL)"
                )
                keys = [identities[name][0] for name in relative_files]
                key_to_file = {identities[name][0]: name for name in relative_files}
                for cache_key, identity, payload in _sqlite_rows(connection, keys):
                    relative_file = key_to_file.get(cache_key)
                    if relative_file is None or identity != identities[relative_file][1]:
                        status["invalid_entries"] = int(status["invalid_entries"]) + 1
                        continue
                    try:
                        fragment = pickle.loads(payload)  # noqa: S301 - trusted CI cache, documented in README.
                    except Exception:
                        status["invalid_entries"] = int(status["invalid_entries"]) + 1
                        continue
                    if not isinstance(fragment, RepositoryIndex):
                        status["invalid_entries"] = int(status["invalid_entries"]) + 1
                        continue
                    fragment.repo_root = repo_root
                    fragment.package_root = package_root
                    fragments[relative_file] = fragment
            except (OSError, sqlite3.DatabaseError) as error:
                if connection is not None:
                    connection.close()
                    connection = None
                identities = None
                status.update(status="unavailable", reason=f"{type(error).__name__}: {error}")
    status["load_seconds"] = round(time.perf_counter() - load_started, 6)

    missing_files = tuple(name for name in relative_files if name not in fragments)
    status["cache_hits"] = len(fragments)
    status["cache_misses"] = len(missing_files)
    status["hit_ratio"] = round(len(fragments) / len(relative_files), 6) if relative_files else 1.0
    build_started = time.perf_counter()
    effective_workers = min(index_workers, len(missing_files)) if missing_files else 0
    status["workers_used"] = effective_workers
    if missing_files:
        task_count = max(1, effective_workers * 4)
        batch_size = max(1, min(64, (len(missing_files) + task_count - 1) // task_count))
        tasks = [
            (
                str(repo_root),
                package_name,
                missing_files[start : start + batch_size],
                tuple(sorted(ordinary_descriptor_decorators)),
            )
            for start in range(0, len(missing_files), batch_size)
        ]
        batches: Iterable[list[tuple[str, RepositoryIndex]]]
        if effective_workers > 1:
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                batches = executor.map(_repository_fragment_batch, tasks)
                for batch in batches:
                    fragments.update(batch)
        else:
            for task in tasks:
                fragments.update(_repository_fragment_batch(task))
    status["build_seconds"] = round(time.perf_counter() - build_started, 6)

    write_started = time.perf_counter()
    if connection is not None and identities is not None and missing_files:
        try:
            with connection:
                connection.executemany(
                    "INSERT OR REPLACE INTO fragments(cache_key, identity, payload) VALUES (?, ?, ?)",
                    [
                        (
                            identities[name][0],
                            identities[name][1],
                            pickle.dumps(fragments[name], protocol=pickle.HIGHEST_PROTOCOL),
                        )
                        for name in missing_files
                    ],
                )
        except Exception as error:  # Cache serialization must not invalidate source analysis.
            status.update(status="write_error", reason=f"{type(error).__name__}: {error}")
    status["write_seconds"] = round(time.perf_counter() - write_started, 6)
    if connection is not None:
        connection.close()
    if status["database"] is not None:
        with contextlib.suppress(OSError):
            status["database_bytes"] = Path(str(status["database"])).stat().st_size

    merge_started = time.perf_counter()
    combined = RepositoryIndex(
        repo_root,
        package_name,
        ordinary_descriptor_decorators=ordinary_descriptor_decorators,
        _source_paths=(),
        _finalize=False,
    )
    for relative_file in relative_files:
        combined._merge_pre_final_fragment(fragments[relative_file])
    combined._finalize_index()
    status["merge_finalize_seconds"] = round(time.perf_counter() - merge_started, 6)
    if status["status"] not in {"bypassed", "unavailable", "write_error"}:
        hits = int(status["cache_hits"])
        status["status"] = "hit" if hits == len(relative_files) else "partial_hit" if hits else "miss"
    return combined, status


def _repository_index_cache_identity(
    repo_root: Path,
    package_name: str,
    source_version: str | None,
    ordinary_descriptor_decorators: frozenset[str],
) -> tuple[dict[str, object] | None, str | None]:
    if not source_version:
        return None, "source version is unavailable"
    try:
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                package_name,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if dirty:
            return None, f"{package_name} contains uncommitted source changes"
        tree_sha = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", f"{source_version}:{package_name}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return None, f"Git cache identity failed: {error}"

    return (
        {
            "cache_schema_version": REPOSITORY_INDEX_CACHE_SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "python_cache_tag": sys.implementation.cache_tag,
            "package_name": package_name,
            "source_version": source_version,
            "tree_sha": tree_sha,
            "ordinary_descriptor_decorators": sorted(ordinary_descriptor_decorators),
        },
        None,
    )


def _repository_index_with_cache(
    repo_root: Path,
    package_name: str,
    *,
    ordinary_descriptor_decorators: frozenset[str],
    source_version: str | None,
    cache_dir: Path | None,
) -> tuple[RepositoryIndex, dict[str, object]]:
    status: dict[str, object] = {
        "enabled": cache_dir is not None,
        "status": "disabled",
        "key": None,
        "path": None,
        "reason": None,
    }
    if cache_dir is None:
        return (
            RepositoryIndex(
                repo_root,
                package_name,
                ordinary_descriptor_decorators=ordinary_descriptor_decorators,
            ),
            status,
        )

    identity, reason = _repository_index_cache_identity(
        repo_root,
        package_name,
        source_version,
        ordinary_descriptor_decorators,
    )
    if identity is None:
        status.update(status="bypassed", reason=reason)
        return (
            RepositoryIndex(
                repo_root,
                package_name,
                ordinary_descriptor_decorators=ordinary_descriptor_decorators,
            ),
            status,
        )

    serialized_identity = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    cache_key = hashlib.sha256(serialized_identity.encode()).hexdigest()
    cache_path = cache_dir.resolve() / f"{package_name}-{cache_key}.pickle"
    status.update(key=cache_key, path=str(cache_path))
    invalid_cache = False
    try:
        if cache_path.is_file():
            with cache_path.open("rb") as stream:
                payload = pickle.load(stream)  # noqa: S301 - the configured cache directory is trusted CI state.
            if not isinstance(payload, dict) or payload.get("identity") != identity:
                raise ValueError("repository index cache identity does not match")
            index = payload.get("index")
            if not isinstance(index, RepositoryIndex):
                raise ValueError("repository index cache payload has an invalid index")
            index.repo_root = repo_root.resolve()
            index.package_root = index.repo_root / package_name
            status["status"] = "hit"
            return index, status
    except Exception as error:  # Cache corruption must not invalidate source analysis.
        invalid_cache = True
        status.update(status="invalid", reason=f"{type(error).__name__}: {error}")

    index = RepositoryIndex(
        repo_root,
        package_name,
        ordinary_descriptor_decorators=ordinary_descriptor_decorators,
    )
    temporary_path: Path | None = None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=cache_dir,
            prefix=f".{package_name}-{cache_key}-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            pickle.dump(
                {"identity": identity, "index": index},
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, cache_path)
        status["status"] = "invalid_rebuilt" if invalid_cache else "miss"
    except Exception as error:  # A cache write failure must fall back to the fresh index.
        status.update(status="write_error", reason=f"{type(error).__name__}: {error}")
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return index, status


class InterfaceBoundaryGenerator:
    def __init__(
        self,
        vllm_root: Path,
        ascend_root: Path,
        external_roots: dict[str, Path] | None = None,
        *,
        source_versions: dict[str, str] | None = None,
        downstream_index_cache_dir: Path | None = None,
        upstream_file_index_cache_dir: Path | None = None,
        index_workers: int = 1,
    ):
        source_versions = source_versions or {}
        ordinary_descriptor_decorators = _KNOWN_ORDINARY_DESCRIPTOR_DECORATORS
        self.repository_index_timings: dict[str, float] = {}
        index_started = time.perf_counter()
        if upstream_file_index_cache_dir is not None or index_workers > 1:
            self.upstream, upstream_file_cache = _repository_index_from_file_fragments(
                vllm_root,
                "vllm",
                ordinary_descriptor_decorators=frozenset(ordinary_descriptor_decorators),
                source_version=source_versions.get("vllm"),
                cache_dir=upstream_file_index_cache_dir,
                index_workers=index_workers,
            )
        else:
            self.upstream = RepositoryIndex(
                vllm_root,
                "vllm",
                ordinary_descriptor_decorators=ordinary_descriptor_decorators,
            )
            upstream_file_cache = {
                "enabled": False,
                "status": "disabled",
                "workers_requested": index_workers,
                "workers_used": 1,
            }
        self.repository_index_timings["upstream"] = round(time.perf_counter() - index_started, 6)
        index_started = time.perf_counter()
        self.downstream, downstream_cache = _repository_index_with_cache(
            ascend_root,
            "vllm_ascend",
            ordinary_descriptor_decorators=ordinary_descriptor_decorators,
            source_version=source_versions.get("vllm_ascend"),
            cache_dir=downstream_index_cache_dir,
        )
        self.repository_index_timings["downstream"] = round(time.perf_counter() - index_started, 6)
        self.repository_index_cache = {
            "upstream_file_fragments": upstream_file_cache,
            "downstream": downstream_cache,
        }
        index_started = time.perf_counter()
        self.externals = {
            package: RepositoryIndex(
                root,
                package,
                ordinary_descriptor_decorators=ordinary_descriptor_decorators,
            )
            for package, root in sorted((external_roots or {}).items())
        }
        self.repository_index_timings["external"] = round(time.perf_counter() - index_started, 6)
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
        self.historical_override_candidates: list[HistoricalOverrideCandidate] = []
        self._mro_cache: dict[str, MroResult] = {}
        self._override_root_path_cache: dict[
            tuple[str, str],
            tuple[tuple[str, tuple[str, ...]], ...],
        ] = {}
        self._mro_resolution_seconds = 0.0
        self.phase_timings: dict[str, float | None] = {}

    def generate(
        self,
        plan: AnalysisPlan = VLLM_INTERFACE_PLAN,
    ) -> tuple[list[Relation], list[CandidateFinding]]:
        """Generate verified overrides using lazily resolved inheritance MROs."""

        if plan != VLLM_INTERFACE_PLAN:
            raise ValueError("only the vllm-interface analysis plan is supported")
        self.relations = []
        self.findings = []
        self.historical_override_candidates = []
        self._override_root_path_cache = {}
        self._mro_resolution_seconds = 0.0
        self.phase_timings = {
            "inheritance_mro": None,
            "override": None,
            "monkey_patch": None,
        }
        phase_started = time.perf_counter()
        self._collect_verified_overrides()
        override_elapsed = time.perf_counter() - phase_started
        self.phase_timings["inheritance_mro"] = round(
            self._mro_resolution_seconds,
            6,
        )
        self.phase_timings["override"] = round(
            max(0.0, override_elapsed - self._mro_resolution_seconds),
            6,
        )
        phase_started = time.perf_counter()
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
            descriptor_sets = {
                field_name: {getattr(relation, field_name) for relation in occurrences}
                for field_name in (
                    "upstream_descriptor_kind",
                    "downstream_descriptor_kind",
                    "installed_descriptor_kind",
                )
            }
            merged_descriptor_kinds = {
                field_name: (next(iter(kinds)) if len(kinds) == 1 else "unknown")
                for field_name, kinds in descriptor_sets.items()
            }
            merged_signature_contracts: dict[str, SignatureContract | None] = {}
            conditional_signature_contract = False
            for field_name in (
                "upstream_signature_contract",
                "downstream_signature_contract",
                "installed_signature_contract",
            ):
                merged_contract, conditional = _merge_signature_contracts(
                    [getattr(relation, field_name) for relation in occurrences]
                )
                merged_signature_contracts[field_name] = merged_contract
                conditional_signature_contract = conditional_signature_contract or conditional
            override_paths = tuple(sorted({path for relation in occurrences for path in relation.override_paths}))
            merged_relation = replace(
                first,
                evidence=tuple(
                    sorted(
                        evidence,
                        key=lambda item: (
                            item.file,
                            item.line,
                            item.scope or "",
                            item.guards,
                            item.installed_descriptor_kind or "",
                            item.target_expression or "",
                        ),
                    )
                ),
                **merged_descriptor_kinds,
                **merged_signature_contracts,
                override_paths=override_paths,
            )
            deduplicated[key] = merged_relation
            if any(len(kinds) > 1 for kinds in descriptor_sets.values()):
                first_evidence = merged_relation.evidence[0] if merged_relation.evidence else None
                self._append_descriptor_finding(
                    merged_relation,
                    target_expression=(
                        first_evidence.target_expression
                        if first_evidence and first_evidence.target_expression
                        else f"{merged_relation.upstream_owner or ''}.{merged_relation.upstream_name}".lstrip(".")
                    ),
                    evidence_line=merged_relation.evidence_line,
                    conditional=True,
                    evidence_scope=(first_evidence.scope if first_evidence else None),
                    evidence_guards=(first_evidence.guards if first_evidence else ()),
                )
            if conditional_signature_contract:
                first_evidence = merged_relation.evidence[0] if merged_relation.evidence else None
                self.findings.append(
                    CandidateFinding(
                        relation=merged_relation.relation,
                        downstream_file=merged_relation.downstream_file,
                        downstream_owner=merged_relation.downstream_owner,
                        downstream_name=merged_relation.downstream_name,
                        target_expression=(
                            first_evidence.target_expression
                            if first_evidence and first_evidence.target_expression
                            else f"{merged_relation.upstream_owner or ''}.{merged_relation.upstream_name}".lstrip(".")
                        ),
                        evidence_line=merged_relation.evidence_line,
                        reason=(
                            "different reachable branches install different runtime "
                            "signature contracts for the same dependency edge"
                        ),
                        status="review",
                        reason_code="conditional_signature_contract",
                        generator_issue=False,
                        evidence_scope=(first_evidence.scope if first_evidence else None),
                        evidence_guards=(first_evidence.guards if first_evidence else ()),
                        supplemental=True,
                    )
                )
        self.relations = sorted(
            deduplicated.values(),
            key=lambda relation: (
                relation.upstream_key(),
                relation.downstream_key(),
            ),
        )
        unique_findings: dict[object, CandidateFinding] = {}
        for finding in set(self.findings):
            key: object = finding
            if finding.supplemental:
                key = (
                    finding.relation,
                    finding.downstream_file,
                    finding.downstream_owner,
                    finding.downstream_name,
                    finding.target_expression,
                    finding.reason,
                    finding.status,
                    finding.reason_code,
                    finding.generator_issue,
                    finding.supplemental,
                    finding.upstream_descriptor_kind,
                    finding.downstream_descriptor_kind,
                    finding.installed_descriptor_kind,
                )
            previous = unique_findings.get(key)
            if previous is None or (
                finding.evidence_line,
                finding.evidence_scope or "",
                finding.evidence_guards,
            ) < (
                previous.evidence_line,
                previous.evidence_scope or "",
                previous.evidence_guards,
            ):
                unique_findings[key] = finding
        self.findings = sorted(
            unique_findings.values(),
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
        self.phase_timings["relation_finalization"] = round(
            time.perf_counter() - phase_started,
            6,
        )
        return self.relations, self.findings

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
            return self.downstream.find_final_callable_variants(qualified_name)
        if qualified_name.startswith("vllm."):
            return self.upstream.find_final_callable_variants(qualified_name)
        for package, index in self.externals.items():
            if qualified_name == package or qualified_name.startswith(f"{package}."):
                return index.find_final_callable_variants(qualified_name)
        return ()

    def _aggregate_descriptor_kinds(
        self,
        candidates: Sequence[CallableInfo],
    ) -> tuple[str | None, bool]:
        kinds = {
            None if candidate_kind is None else (candidate_kind if candidate_kind in DESCRIPTOR_KINDS else "unknown")
            for candidate in candidates
            for candidate_kind in (candidate.descriptor_variants or (candidate.descriptor_kind,))
        }
        if not kinds:
            return None, False
        if len(kinds) == 1:
            return next(iter(kinds)), False
        return "unknown", True

    def _repository_for_callable(
        self,
        callable_info: CallableInfo,
    ) -> RepositoryIndex | None:
        if callable_info.qualified_name.startswith("vllm_ascend."):
            return self.downstream
        if callable_info.qualified_name.startswith("vllm."):
            return self.upstream
        for package, index in self.externals.items():
            if callable_info.qualified_name == package or callable_info.qualified_name.startswith(f"{package}."):
                return index
        return None

    def _bound_call_signature(
        self,
        signature: list[object] | None,
        *,
        descriptor_kind: str | None,
        binds_receiver: bool,
    ) -> tuple[list[object] | None, str]:
        if signature is None:
            return None, "unknown"
        result = json.loads(json.dumps(signature))
        if not binds_receiver:
            return result, "exact"
        if descriptor_kind == "staticmethod":
            return result, "exact"
        if descriptor_kind not in {"classmethod", "ordinary", "property"}:
            return None, "unknown"
        positional_only = result[1]
        positional_or_keyword = result[2]
        if positional_only:
            positional_only.pop(0)
        elif positional_or_keyword:
            positional_or_keyword.pop(0)
        elif result[3] is not None:
            return result, "exact"
        else:
            return None, "invalid"
        return result, "exact"

    def _signature_contract(
        self,
        callable_info: CallableInfo,
        *,
        descriptor_kind: str | None = None,
        binds_receiver: bool | None = None,
    ) -> SignatureContract:
        """Derive the callable contract after statically known wrappers."""
        definition_signature = callable_info.signature
        runtime_entry_signature = definition_signature
        reported_signature = definition_signature
        status = "exact"
        provenance = ["ast_definition"]
        forwarded_targets: list[str] = []
        protocol = (
            "property_access" if (descriptor_kind or callable_info.descriptor_kind) == "property" else "python_call"
        )
        node = callable_info.node
        decorators = tuple(node.decorator_list) if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) else ()
        references = callable_info.decorator_references
        if len(references) != len(decorators):
            references = tuple(None for _ in decorators)
        captured_targets = callable_info.decorator_forwarded_targets
        if captured_targets is None or len(captured_targets) != len(decorators):
            forwarded_target_variants: tuple[tuple[str, ...] | None, ...] = tuple(None for _ in decorators)
        else:
            forwarded_target_variants = captured_targets

        repository = self._repository_for_callable(callable_info)
        for decorator, reference, captured in reversed(tuple(zip(decorators, references, forwarded_target_variants))):
            expression = _expression_name(decorator.func if isinstance(decorator, ast.Call) else decorator)
            label = reference or expression or "<dynamic-decorator>"
            if reference == _TRITON_JIT_DECORATOR:
                runtime_entry_signature = definition_signature
                reported_signature = None
                protocol = _TRITON_KERNEL_PROTOCOL
                provenance.append(f"{label}:kernel_launch")
                continue
            if reference == _TRITON_HEURISTICS_DECORATOR:
                generated_names = self._triton_heuristic_names(decorator)
                transformed_signature = (
                    self._triton_heuristics_signature(
                        runtime_entry_signature,
                        generated_names,
                    )
                    if protocol == _TRITON_KERNEL_PROTOCOL and generated_names is not None
                    else None
                )
                if transformed_signature is None:
                    runtime_entry_signature = None
                    reported_signature = None
                    status = "unknown"
                    provenance.append(f"{label}:unresolved_kernel_heuristics")
                else:
                    runtime_entry_signature = transformed_signature
                    provenance.append(f"{label}:generated={','.join(generated_names)}")
                continue
            if reference in _STDLIB_WRAPS_SIGNATURE_DECORATORS and not isinstance(decorator, ast.Call):
                runtime_entry_signature = ["sync", [], [], "args", [], "kwargs"]
                reported_signature = definition_signature
                forwarded_targets.append(callable_info.qualified_name)
                provenance.append(f"{label}:stdlib_wrapped")
                continue
            if reference == "functools.wraps" and isinstance(decorator, ast.Call) and decorator.args:
                target_expression = _expression_name(decorator.args[0])
                target_names = captured
                if target_names is None:
                    resolved_name = None
                    if repository is not None and target_expression is not None:
                        resolved_name = self._canonical_reference(
                            repository.resolve_reference(
                                callable_info.module,
                                target_expression,
                            )
                        )
                    target_names = (resolved_name,) if resolved_name is not None else ()
                target_name = target_names[0] if len(target_names) == 1 else None
                target_callable = self._callable_info(target_name) if target_name is not None else None
                target_label = target_name
                if target_label is None and len(target_names) > 1:
                    target_label = f"<ambiguous:{'|'.join(target_names)}>"
                provenance.append(f"functools.wraps:{target_label or target_expression or '<unknown>'}")
                if target_name is not None:
                    forwarded_targets.append(target_name)
                if target_callable is None:
                    reported_signature = None
                    status = "unknown"
                else:
                    reported_signature = target_callable.signature
                continue
            if reference in _BUILTIN_DESCRIPTOR_DECORATORS or reference in (
                _TRANSPARENT_DESCRIPTOR_DECORATORS - {"functools.wraps"}
            ):
                provenance.append(label)
                continue

            if reference in _KNOWN_TRANSPARENT_SIGNATURE_DECORATORS:
                provenance.append(label)
                continue

            if reference in _KNOWN_WRAPS_SIGNATURE_DECORATORS and not isinstance(decorator, ast.Call):
                runtime_entry_signature = ["sync", [], [], "args", [], "kwargs"]
                forwarded_targets.append(callable_info.qualified_name)
                provenance.append(f"{label}:wrapped")
                continue

            if expression is not None and expression.rsplit(".", 1)[-1] in {
                "deleter",
                "getter",
                "setter",
            }:
                provenance.append(label)
                continue

            static_transform = (
                self._static_decorator_transform(reference)
                if reference is not None and not isinstance(decorator, ast.Call)
                else None
            )
            if static_transform is not None:
                runtime_entry_signature = static_transform.wrapper_signature
                if static_transform.preserves_reported_signature:
                    forwarded_targets.append(callable_info.qualified_name)
                else:
                    reported_signature = static_transform.wrapper_signature
                provenance.append(f"{label}:static_wrapper:{static_transform.wrapper_name}")
                continue

            runtime_entry_signature = None
            reported_signature = None
            status = "unknown"
            provenance.append(label)

        effective_kind = callable_info.descriptor_kind if descriptor_kind is None else descriptor_kind
        receiver_binding = callable_info.owner is not None if binds_receiver is None else binds_receiver
        bound_call_signature, binding_status = self._bound_call_signature(
            runtime_entry_signature,
            descriptor_kind=effective_kind,
            binds_receiver=receiver_binding,
        )
        if status == "exact" and binding_status != "exact":
            status = binding_status
            provenance.append(
                "invalid_receiver_binding" if binding_status == "invalid" else "unknown_descriptor_binding"
            )
        return SignatureContract(
            definition_signature=definition_signature,
            runtime_entry_signature=runtime_entry_signature,
            reported_signature=reported_signature,
            bound_call_signature=bound_call_signature,
            forwarded_targets=tuple(dict.fromkeys(forwarded_targets)),
            protocol=protocol,
            status=status,
            provenance=tuple(provenance),
        )

    @staticmethod
    def _triton_heuristic_names(
        decorator: ast.AST,
    ) -> tuple[str, ...] | None:
        """Return the literal names injected by a pinned Triton heuristic."""

        if not isinstance(decorator, ast.Call):
            return None
        values: ast.AST | None = decorator.args[0] if len(decorator.args) == 1 else None
        for keyword in decorator.keywords:
            if keyword.arg == "values":
                if values is not None:
                    return None
                values = keyword.value
            else:
                return None
        if not isinstance(values, ast.Dict) or len(values.keys) != len(values.values):
            return None
        names: list[str] = []
        for key in values.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                return None
            names.append(key.value)
        return tuple(dict.fromkeys(names))

    @staticmethod
    def _triton_heuristics_signature(
        signature: list[object] | None,
        generated_names: tuple[str, ...],
    ) -> list[object] | None:
        """Model the public ``kernel[grid](...)`` shape after heuristics."""

        if signature is None:
            return None
        result = json.loads(json.dumps(signature))
        positional_only = result[1]
        positional_or_keyword = result[2]
        keyword_only = result[4]
        if not all(isinstance(items, list) for items in (positional_only, positional_or_keyword, keyword_only)):
            return None
        generated = set(generated_names)
        known_names = {
            item[0]
            for items in (positional_only, positional_or_keyword, keyword_only)
            for item in items
            if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str)
        }
        if not generated.issubset(known_names):
            return None
        if any(item[0] in generated for item in positional_only):
            return None

        first_generated = next(
            (index for index, item in enumerate(positional_or_keyword) if item[0] in generated),
            None,
        )
        if first_generated is not None:
            trailing = positional_or_keyword[first_generated:]
            result[2] = positional_or_keyword[:first_generated]
            result[4] = [[name, False if name in generated else required] for name, required in trailing] + keyword_only
        result[4] = [[name, False if name in generated else required] for name, required in result[4]]
        return result

    def _static_decorator_transform(
        self,
        reference: str,
    ) -> StaticDecoratorTransform | None:
        """Resolve a direct decorator that returns one local wrapper."""

        decorator = self._callable_info(reference)
        node = decorator.node if decorator is not None else None
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            return None
        if node.decorator_list or node.args.vararg is not None or node.args.kwarg is not None:
            return None
        positional = [*node.args.posonlyargs, *node.args.args]
        if len(positional) != 1 or node.args.kwonlyargs:
            return None
        parameter = positional[0].arg
        if self._parameter_is_reassigned(node, parameter):
            return None

        scope_nodes = list(_function_scope_nodes(node))
        if any(isinstance(child, (ast.Yield, ast.YieldFrom)) for child in scope_nodes):
            return None
        nested = {
            child.name: child for child in scope_nodes if isinstance(child, (ast.AsyncFunctionDef, ast.FunctionDef))
        }
        returns = [child for child in scope_nodes if isinstance(child, ast.Return)]
        if not returns or not isinstance(node.body[-1], ast.Return):
            return None
        returned_names = {
            child.value.id for child in returns if isinstance(child.value, ast.Name) and child.value.id in nested
        }
        if len(returned_names) != 1 or len(returned_names) != len(returns):
            return None
        wrapper_name = next(iter(returned_names))
        final_return = node.body[-1]
        if not isinstance(final_return.value, ast.Name) or final_return.value.id != wrapper_name:
            return None
        wrapper = nested[wrapper_name]
        wrapper_signature = _jsonable_signature(wrapper)
        if wrapper_signature is None:
            return None

        preserves_reported_signature = False
        if wrapper.decorator_list:
            repository = self._repository_for_callable(decorator)
            wrapper_references = (
                repository._decorator_references_by_node.get(id(wrapper), ()) if repository is not None else ()
            )
            if len(wrapper_references) != 1 or len(wrapper.decorator_list) != 1:
                return None
            wrapper_decorator = wrapper.decorator_list[0]
            if not (
                wrapper_references[0] == "functools.wraps"
                and isinstance(wrapper_decorator, ast.Call)
                and len(wrapper_decorator.args) == 1
                and not wrapper_decorator.keywords
                and isinstance(wrapper_decorator.args[0], ast.Name)
                and wrapper_decorator.args[0].id == parameter
            ):
                return None
            preserves_reported_signature = True

        return StaticDecoratorTransform(
            wrapper_signature=wrapper_signature,
            preserves_reported_signature=preserves_reported_signature,
            wrapper_name=wrapper_name,
        )

    def _parameter_is_reassigned(
        self,
        node: ast.AsyncFunctionDef | ast.FunctionDef,
        parameter: str,
    ) -> bool:
        return any(
            isinstance(child, ast.Name) and child.id == parameter and isinstance(child.ctx, (ast.Del, ast.Store))
            for child in _function_scope_nodes(node)
        )

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

    def _append_signature_finding(
        self,
        relation: Relation,
        *,
        target_expression: str,
        evidence_line: int,
        evidence_scope: str | None = None,
        evidence_guards: tuple[str, ...] = (),
    ) -> None:
        contracts = (
            relation.upstream_signature_contract,
            relation.downstream_signature_contract,
            relation.installed_signature_contract,
        )
        if not any(contract is not None and contract.status != "exact" for contract in contracts):
            return
        invalid_binding = any(contract is not None and contract.status == "invalid" for contract in contracts)
        unknown_binding = any(
            contract is not None and "unknown_descriptor_binding" in contract.provenance for contract in contracts
        )
        unknown_transform = any(
            contract is not None
            and contract.status == "unknown"
            and "unknown_descriptor_binding" not in contract.provenance
            for contract in contracts
        )
        known_descriptor_mismatch = (
            relation.upstream_descriptor_kind is not None
            and relation.installed_descriptor_kind is not None
            and relation.upstream_descriptor_kind != relation.installed_descriptor_kind
        )
        if invalid_binding and known_descriptor_mismatch:
            # The invalid binding is derived from the already reported change
            # in descriptor protocol. Keep an independent decorator unknown,
            # but do not report the same binding break twice.
            if not unknown_transform:
                return
            invalid_binding = False
        if unknown_binding and not invalid_binding and not unknown_transform:
            # The descriptor finding already owns this uncertainty. Emitting a
            # second signature finding for the same unknown binding would add
            # noise without providing independent evidence.
            return
        self.findings.append(
            CandidateFinding(
                relation=relation.relation,
                downstream_file=relation.downstream_file,
                downstream_owner=relation.downstream_owner,
                downstream_name=relation.downstream_name,
                target_expression=target_expression,
                evidence_line=evidence_line,
                reason=(
                    "descriptor binding requires a receiver but the callable has no positional slot"
                    if invalid_binding
                    else (
                        (
                            "a decorator changes the runtime callable contract and "
                            "its exact signature effect is not proven for this source version"
                        )
                        if unknown_transform
                        else (
                            "the descriptor kind is conditional or unknown, so the "
                            "bound runtime signature cannot be proven"
                        )
                    )
                ),
                status="risk" if invalid_binding else "review",
                reason_code=(
                    "invalid_receiver_binding"
                    if invalid_binding
                    else ("unknown_signature_transform" if unknown_transform else "unknown_signature_binding")
                ),
                generator_issue=False,
                evidence_scope=evidence_scope,
                evidence_guards=evidence_guards,
                supplemental=True,
            )
        )

    def _append_signature_compatibility_finding(
        self,
        relation: Relation,
        *,
        target_expression: str,
        evidence_line: int,
        evidence_scope: str | None = None,
        evidence_guards: tuple[str, ...] = (),
    ) -> None:
        upstream = relation.upstream_signature_contract
        installed = relation.installed_signature_contract
        if (
            upstream is None
            or installed is None
            or upstream.status != "exact"
            or installed.status != "exact"
            or upstream.bound_call_signature is None
            or installed.bound_call_signature is None
        ):
            return
        if (
            relation.upstream_descriptor_kind is not None
            and relation.installed_descriptor_kind is not None
            and relation.upstream_descriptor_kind != relation.installed_descriptor_kind
        ):
            # A known descriptor mismatch already reports the access-protocol
            # break. Do not duplicate it as a derived signature failure.
            return

        incompatible_views: list[str] = []
        compared_signatures: set[str] = set()

        def compare_view(
            label: str,
            upstream_signature: list[object] | None,
            installed_signature: list[object] | None,
        ) -> None:
            if upstream_signature is None or installed_signature is None:
                return
            comparison_key = json.dumps(
                [upstream_signature, installed_signature],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if comparison_key in compared_signatures:
                return
            compared_signatures.add(comparison_key)
            if not _accepts_signature_contract(
                upstream_signature,
                installed_signature,
            ):
                incompatible_views.append(label)

        compare_view(
            "runtime entry",
            upstream.bound_call_signature,
            installed.bound_call_signature,
        )
        binds_receiver = relation.upstream_owner is not None
        extra_views = ()
        if _TRITON_KERNEL_PROTOCOL not in {upstream.protocol, installed.protocol}:
            extra_views = (
                ("reported", "reported_signature"),
                ("definition", "definition_signature"),
            )
        for label, attribute in extra_views:
            upstream_signature, upstream_status = self._bound_call_signature(
                getattr(upstream, attribute),
                descriptor_kind=relation.upstream_descriptor_kind,
                binds_receiver=binds_receiver,
            )
            installed_signature, installed_status = self._bound_call_signature(
                getattr(installed, attribute),
                descriptor_kind=relation.installed_descriptor_kind,
                binds_receiver=binds_receiver,
            )
            if upstream_status == "exact" and installed_status == "exact":
                compare_view(
                    label,
                    upstream_signature,
                    installed_signature,
                )

        if upstream.protocol != installed.protocol:
            incompatible_views.insert(0, "access protocol")
        if not incompatible_views:
            return
        self.findings.append(
            CandidateFinding(
                relation=relation.relation,
                downstream_file=relation.downstream_file,
                downstream_owner=relation.downstream_owner,
                downstream_name=relation.downstream_name,
                target_expression=target_expression,
                evidence_line=evidence_line,
                reason=(
                    "the installed downstream callable does not accept every "
                    "call shape allowed by the upstream contract; incompatible "
                    f"views: {', '.join(incompatible_views)}"
                ),
                status="risk",
                reason_code="signature_incompatible",
                generator_issue=False,
                evidence_scope=evidence_scope,
                evidence_guards=evidence_guards,
                supplemental=True,
            )
        )

    def _append_descriptor_finding(
        self,
        relation: Relation,
        *,
        target_expression: str,
        evidence_line: int,
        conditional: bool,
        evidence_scope: str | None = None,
        evidence_guards: tuple[str, ...] = (),
    ) -> None:
        kinds = (
            relation.upstream_descriptor_kind,
            relation.downstream_descriptor_kind,
            relation.installed_descriptor_kind,
        )
        if kinds == (None, None, None):
            return
        if conditional:
            reason_code = "conditional_descriptor_kind"
            reason = (
                "the callable has different descriptor kinds on normally "
                "completing source paths; no single binding kind was guessed"
            )
        elif "unknown" in kinds:
            reason_code = "unknown_descriptor_kind"
            reason = "a dynamic decorator or binding prevents an exact descriptor kind from being proven statically"
        elif relation.upstream_descriptor_kind != relation.installed_descriptor_kind:
            reason_code = "descriptor_kind_mismatch"
            reason = "the downstream binding installs a different callable kind from the upstream member"
        elif None in kinds:
            reason_code = "unknown_descriptor_kind"
            reason = "a dynamic decorator or binding prevents an exact descriptor kind from being proven statically"
        else:
            return
        status = "risk" if reason_code == "descriptor_kind_mismatch" else "review"
        self.findings.append(
            CandidateFinding(
                relation=relation.relation,
                downstream_file=relation.downstream_file,
                downstream_owner=relation.downstream_owner,
                downstream_name=relation.downstream_name,
                target_expression=target_expression,
                evidence_line=evidence_line,
                reason=reason,
                status=status,
                reason_code=reason_code,
                generator_issue=False,
                evidence_scope=evidence_scope,
                evidence_guards=evidence_guards,
                supplemental=True,
                upstream_descriptor_kind=relation.upstream_descriptor_kind,
                downstream_descriptor_kind=relation.downstream_descriptor_kind,
                installed_descriptor_kind=relation.installed_descriptor_kind,
            )
        )

    def _final_bindings(
        self,
        qualified_name: str,
    ) -> tuple[_ScopeBinding, ...]:
        qualified_name = self._canonical_reference(qualified_name)
        if qualified_name.startswith("vllm_ascend."):
            return self.downstream.find_final_bindings(qualified_name)
        if qualified_name.startswith("vllm."):
            return self.upstream.find_final_bindings(qualified_name)
        for package, index in self.externals.items():
            if qualified_name == package or qualified_name.startswith(f"{package}."):
                return index.find_final_bindings(qualified_name)
        return ()

    def _final_binding_kinds(
        self,
        qualified_name: str,
    ) -> set[str]:
        return {binding.kind for binding in self._final_bindings(qualified_name)}

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
        index: RepositoryIndex | None = None
        if qualified_name.startswith("vllm_ascend."):
            index = self.downstream
        elif qualified_name.startswith("vllm."):
            index = self.upstream
        else:
            index = next(
                (
                    candidate
                    for package, candidate in self.externals.items()
                    if qualified_name == package or qualified_name.startswith(f"{package}.")
                ),
                None,
            )
        if index is not None and qualified_name in index.class_base_conflicts:
            return [], [f"conditional class variants have different bases: {qualified_name}"]
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

    def _conditional_class_dependency(
        self,
        qualified_name: str,
        seen: frozenset[str] = frozenset(),
    ) -> str | None:
        """Return the first base that is a class only on some live paths."""

        if qualified_name in seen:
            return None
        class_info = self._class_info(qualified_name)
        if class_info is None:
            return None
        next_seen = frozenset((*seen, qualified_name))
        for base in class_info.resolved_bases:
            base = self._canonical_reference(base)
            kinds = self._final_binding_kinds(base)
            if "class" in kinds and kinds != {"class"}:
                return base
            nested = self._conditional_class_dependency(base, next_seen)
            if nested is not None:
                return nested
        return None

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

    def _timed_linearized_mro(self, qualified_name: str) -> MroResult:
        """Resolve one relation MRO and account for only the lazy MRO work."""

        started = time.perf_counter()
        try:
            return self._linearized_mro(qualified_name)
        finally:
            self._mro_resolution_seconds += time.perf_counter() - started

    def _collect_verified_overrides(self) -> None:
        """Collect downstream overrides with a statically proven upstream owner."""
        for class_info in self.downstream.classes.values():
            if self._conditional_class_dependency(class_info.qualified_name) is not None:
                continue
            mro_result = self._timed_linearized_mro(class_info.qualified_name)
            mro = mro_result.owners
            if mro_result.complete and not any(owner.startswith("vllm.") for owner in mro[1:]):
                continue
            for method_name, method_node in class_info.methods.items():
                resolution = self._effective_method_resolution(
                    mro[1:],
                    method_name,
                )
                downstream_name = f"{class_info.qualified_name}.{method_name}"
                downstream_kinds = self._final_binding_kinds(downstream_name)
                downstream_callable_kinds = downstream_kinds & {"function"}
                downstream_other_kinds = downstream_kinds - {"function"}
                upstream_target_owners = (
                    *resolution.callable_owners,
                    *resolution.blocking_owners,
                )
                if downstream_callable_kinds and downstream_other_kinds and upstream_target_owners:
                    target_owner = upstream_target_owners[0]
                    self.findings.append(
                        CandidateFinding(
                            relation="override",
                            downstream_file=class_info.file,
                            downstream_owner=class_info.name,
                            downstream_name=method_name,
                            target_expression=f"{target_owner}.{method_name}",
                            evidence_line=getattr(method_node, "lineno", 0),
                            reason=(
                                "downstream member is callable only on some "
                                "normally completing class paths; other final "
                                f"bindings: {', '.join(sorted(downstream_other_kinds))}"
                            ),
                            status="review",
                            reason_code="conditional_callable_presence",
                            generator_issue=False,
                        )
                    )
                    continue
                effective_owners = resolution.callable_owners if resolution.is_total_callable else ()
                if not effective_owners:
                    conditional_owners = (
                        *resolution.callable_owners,
                        *resolution.blocking_owners,
                    )
                    if conditional_owners:
                        target_owner = conditional_owners[0]
                        self.findings.append(
                            CandidateFinding(
                                relation="override",
                                downstream_file=class_info.file,
                                downstream_owner=class_info.name,
                                downstream_name=method_name,
                                target_expression=f"{target_owner}.{method_name}",
                                evidence_line=getattr(method_node, "lineno", 0),
                                reason=(
                                    "the effective upstream member is callable only on "
                                    "some normally completing lookup paths"
                                ),
                                status="review",
                                reason_code="conditional_callable_presence",
                                generator_issue=False,
                            )
                        )
                        continue
                    historical_lookup_root = (
                        next(
                            (owner for owner in mro[1:] if owner.startswith("vllm.")),
                            None,
                        )
                        if mro_result.complete
                        and resolution.may_be_missing
                        and not resolution.may_be_non_callable
                        and not resolution.has_unresolved_value
                        and not resolution.blocking_owners
                        and not hasattr(object, method_name)
                        else None
                    )
                    if historical_lookup_root is not None:
                        self.historical_override_candidates.append(
                            HistoricalOverrideCandidate(
                                lookup_root=historical_lookup_root,
                                downstream_file=class_info.file,
                                downstream_owner=class_info.name,
                                downstream_qualified_owner=class_info.qualified_name,
                                downstream_name=method_name,
                                evidence_line=getattr(method_node, "lineno", 0),
                            )
                        )
                    super_target = (
                        next(
                            (owner for owner in mro[1:] if owner.startswith("vllm.")),
                            None,
                        )
                        if mro_result.complete
                        and not hasattr(object, method_name)
                        and self._calls_same_method_on_super(
                            method_node,
                            method_name,
                        )
                        else None
                    )
                    if super_target is not None:
                        self.findings.append(
                            CandidateFinding(
                                relation="override",
                                downstream_file=class_info.file,
                                downstream_owner=class_info.name,
                                downstream_name=method_name,
                                target_expression=f"{super_target}.{method_name}",
                                evidence_line=getattr(method_node, "lineno", 0),
                                reason=(
                                    "downstream method directly calls the same "
                                    "method through super(), but no upstream "
                                    "implementation exists in the complete MRO"
                                ),
                                status="risk",
                                reason_code="missing_upstream_super_target",
                                generator_issue=False,
                            )
                        )
                        continue
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
                    for root_owner, owner_path in self._override_root_paths(
                        effective_owner,
                        method_name,
                    ):
                        self._record_verified_override_owner(
                            class_info,
                            method_name,
                            method_node,
                            root_owner,
                            mro,
                            override_path=(downstream_name, *owner_path),
                        )

    def _override_root_paths(
        self,
        effective_owner: str,
        method_name: str,
        seen: frozenset[str] = frozenset(),
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Resolve a downstream-owned override to its ultimate source root.

        Attribute lookup stops at the first effective implementation.  That
        implementation can itself belong to another vllm-ascend subclass,
        which still substitutes for the later vLLM method contract.  Follow
        only total-callable lookup prefixes and reuse the existing MRO and
        final-binding caches.  The full MRO may remain incomplete after a
        callable owner has already stopped lookup; an ambiguous or blocked
        intermediate owner is never guessed.
        """

        cache_key = (effective_owner, method_name)
        if effective_owner in seen:
            return ()
        if cache_key in self._override_root_path_cache:
            return self._override_root_path_cache[cache_key]

        qualified_method = f"{effective_owner}.{method_name}"
        if effective_owner.startswith("vllm.") or self._is_external_owner(
            effective_owner,
        ):
            result = ((effective_owner, (qualified_method,)),)
        elif not effective_owner.startswith("vllm_ascend."):
            result = ()
        else:
            mro_result = self._timed_linearized_mro(effective_owner)
            resolution = self._effective_method_resolution(
                mro_result.owners[1:],
                method_name,
            )
            if not resolution.is_total_callable:
                result = ()
            else:
                paths = {
                    (
                        root_owner,
                        (qualified_method, *parent_path),
                    )
                    for parent_owner in resolution.callable_owners
                    for root_owner, parent_path in self._override_root_paths(
                        parent_owner,
                        method_name,
                        frozenset((*seen, effective_owner)),
                    )
                }
                result = tuple(sorted(paths))

        self._override_root_path_cache[cache_key] = result
        return result

    def _calls_same_method_on_super(
        self,
        method_node: ast.AST,
        method_name: str,
    ) -> bool:
        """Whether this method directly evaluates ``super().<same-name>(...)``."""

        if not isinstance(
            method_node,
            (ast.AsyncFunctionDef, ast.FunctionDef),
        ):
            return False
        tag_guard_names = _tag_guard_names(method_node.body)
        for statement in _main_module_statements(
            method_node.body,
            tag_guard_names,
        ):
            for expression in self._statement_expressions(statement):
                for candidate in _main_expression_calls(
                    expression,
                    tag_guard_names,
                ):
                    function = candidate.func
                    if not (
                        isinstance(function, ast.Attribute)
                        and function.attr == method_name
                        and isinstance(function.value, ast.Call)
                    ):
                        continue
                    super_call = function.value
                    if (
                        isinstance(super_call.func, ast.Name)
                        and super_call.func.id == "super"
                        and not super_call.args
                        and not super_call.keywords
                    ):
                        return True
        return False

    def _record_verified_override_owner(
        self,
        class_info: ClassInfo,
        method_name: str,
        method_node: ast.AST,
        effective_owner: str,
        mro: Sequence[str],
        *,
        override_path: tuple[str, ...],
    ) -> None:
        """Record one override after validating its owner and installed contract."""
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
        upstream_descriptor_kind, upstream_descriptor_conditional = self._aggregate_descriptor_kinds(
            upstream_variants or (upstream_callable,)
        )
        downstream_candidates = downstream_variants or (
            (downstream_callable,) if downstream_callable is not None else ()
        )
        downstream_descriptor_kind, downstream_descriptor_conditional = self._aggregate_descriptor_kinds(
            downstream_candidates
        )
        evidence_line = (
            downstream_callable.binding_line
            if downstream_callable and downstream_callable.binding_line is not None
            else getattr(method_node, "lineno", 0)
        )
        upstream_signature_contract = self._signature_contract(
            upstream_callable,
            descriptor_kind=upstream_descriptor_kind,
        )
        downstream_signature_contract = (
            self._signature_contract(
                downstream_callable,
                descriptor_kind=downstream_descriptor_kind,
            )
            if downstream_callable is not None
            else None
        )
        relation = Relation(
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
            upstream_descriptor_kind=upstream_descriptor_kind,
            downstream_descriptor_kind=downstream_descriptor_kind,
            installed_descriptor_kind=downstream_descriptor_kind,
            upstream_property_accessors=upstream_callable.property_accessors,
            downstream_property_accessors=(
                downstream_callable.property_accessors if downstream_callable is not None else None
            ),
            installed_property_accessors=(
                downstream_callable.property_accessors if downstream_callable is not None else None
            ),
            upstream_signature_contract=upstream_signature_contract,
            downstream_signature_contract=downstream_signature_contract,
            installed_signature_contract=downstream_signature_contract,
            override_paths=(override_path,),
        )
        self.relations.append(relation)
        self._append_descriptor_finding(
            relation,
            target_expression=upstream_name,
            evidence_line=evidence_line,
            conditional=(upstream_descriptor_conditional or downstream_descriptor_conditional),
        )
        self._append_signature_finding(
            relation,
            target_expression=upstream_name,
            evidence_line=evidence_line,
        )
        self._append_signature_compatibility_finding(
            relation,
            target_expression=upstream_name,
            evidence_line=evidence_line,
        )

    def _effective_method_resolution(
        self,
        mro: Sequence[str],
        method_name: str,
    ) -> EffectiveMethodResolution:
        owners: list[str] = []
        blocking_owners: list[str] = []
        may_be_non_callable = False
        has_unresolved_value = False
        fallthrough = True
        for owner in mro:
            if not fallthrough:
                break
            class_info = self._class_info(owner)
            if class_info is None:
                continue
            qualified_name = f"{owner}.{method_name}"
            alternatives = self._final_bindings(qualified_name)
            if not alternatives:
                if method_name in class_info.methods:
                    owners.append(owner)
                    fallthrough = False
                continue

            kinds = {alternative.kind for alternative in alternatives}
            if "function" in kinds:
                owners.append(owner)
            bound_non_functions = [
                alternative for alternative in alternatives if alternative.kind not in {"function", "unbound"}
            ]
            if bound_non_functions:
                blocking_owners.append(owner)
                for alternative in bound_non_functions:
                    value_node = alternative.node
                    if isinstance(value_node, (ast.Assign, ast.AnnAssign)):
                        value_node = value_node.value
                    if alternative.kind == "value" and self._definitely_non_callable(value_node):
                        may_be_non_callable = True
                    else:
                        has_unresolved_value = True
            fallthrough = "unbound" in kinds

        return EffectiveMethodResolution(
            callable_owners=tuple(dict.fromkeys(owners)),
            may_be_missing=fallthrough,
            may_be_non_callable=may_be_non_callable,
            has_unresolved_value=has_unresolved_value,
            blocking_owners=tuple(dict.fromkeys(blocking_owners)),
        )

    def _effective_method_owners(
        self,
        mro: Sequence[str],
        method_name: str,
    ) -> tuple[str, ...]:
        resolution = self._effective_method_resolution(mro, method_name)
        return resolution.callable_owners if resolution.is_total_callable else ()

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

    def _class_line(self, class_info: ClassInfo) -> int:
        node = self.downstream.find_callable(class_info.qualified_name)
        return getattr(node.node, "lineno", 0) if node else 0
