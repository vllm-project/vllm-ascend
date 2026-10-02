# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Separate source annotation imports from executable module bindings."""

from __future__ import annotations

import ast
import threading
from typing import Any

from .generator import (
    MAX_SCOPE_STATE_CACHE_ENTRIES,
    _scope_reference_variants,
    _scope_state_before,
    _ScopeBinding,
    _ScopePrefixCache,
    _tag_guard_names,
)
from .module_attributes import annotation_module_body, runtime_module_body
from .source_facts import SourceFacts


class AnnotationNamespace:
    """Resolve unique type names without importing or evaluating source code.

    TYPE_CHECKING-only imports can supply annotation evidence, not runtime
    callables. Conflicting or unknown runtime bindings invalidate a type hint.
    Both views reuse the ordinary scope interpreter, including rebinding,
    deletes and branch joins. The original AST and repository index stay intact.
    """

    def __init__(self, tree: ast.Module, module: str, is_package: bool, *, source_facts: SourceFacts | None = None):
        self._lock = threading.RLock()
        self.module = module
        self.is_package = is_package
        facts = source_facts or SourceFacts()
        self.annotation_body = facts.get("annotation_module_body", tree, lambda: annotation_module_body(tree))
        self.runtime_body = facts.get("runtime_module_body", tree, lambda: runtime_module_body(tree))
        self._annotation_prefixes = _ScopePrefixCache()
        self._runtime_prefixes = _ScopePrefixCache()
        self.final_line = max((getattr(n, "end_lineno", 0) or 0 for n in tree.body), default=0) + 1
        self._cache: dict[tuple[str, int, bool], str | None] = {}
        self._resolved_cache: dict[tuple[str, int], str | None] = {}
        self._runtime_bound_roots: dict[int, frozenset[str]] = {}
        self._annotation_states: dict[int, dict[str, tuple[_ScopeBinding, ...]]] = {}
        self._runtime_states: dict[int, dict[str, tuple[_ScopeBinding, ...]]] = {}
        self._annotation_guards = _tag_guard_names(self.annotation_body)
        self._runtime_guards = _tag_guard_names(self.runtime_body)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def _reference(self, expression: str, line: int, *, annotation: bool) -> str | None:
        key = (expression, line, annotation)
        if key in self._cache:
            return self._cache[key]
        body = self.annotation_body if annotation else self.runtime_body
        try:
            node = ast.parse(expression, mode="eval").body
        except SyntaxError:
            return None
        variants = _scope_reference_variants(
            node,
            statements=body,
            line=line,
            tag_guard_names=self._annotation_guards if annotation else self._runtime_guards,
            module=self.module,
            is_package=self.is_package,
            state_cache=self._annotation_states if annotation else self._runtime_states,
            prefix_cache=self._annotation_prefixes if annotation else self._runtime_prefixes,
        )
        result = next(iter(variants)) if len(variants) == 1 else None
        self._cache[key] = result
        return result

    def runtime(self, expression: str, line: int | None = None) -> str | None:
        with self._lock:
            return self._reference(expression, self.final_line if line is None else line, annotation=False)

    def resolve(self, expression: str, line: int | None = None) -> str | None:
        with self._lock:
            return self._resolve(expression, line)

    def _resolve(self, expression: str, line: int | None = None) -> str | None:
        point = self.final_line if line is None else line
        key = (expression, point)
        if key in self._resolved_cache:
            return self._resolved_cache[key]
        result = self._reference(expression, point, annotation=True)
        if result is not None:
            if point not in self._runtime_bound_roots:
                if point not in self._runtime_states:
                    if len(self._runtime_states) >= MAX_SCOPE_STATE_CACHE_ENTRIES:
                        self._runtime_states.pop(next(iter(self._runtime_states)))
                    self._runtime_states[point] = _scope_state_before(
                        self.runtime_body, point, self._runtime_guards, prefix_cache=self._runtime_prefixes
                    )
                state = self._runtime_states[point]
                if len(self._runtime_bound_roots) >= MAX_SCOPE_STATE_CACHE_ENTRIES:
                    self._runtime_bound_roots.pop(next(iter(self._runtime_bound_roots)))
                self._runtime_bound_roots[point] = frozenset(
                    name for name, bindings in state.items() if any(binding.kind != "unbound" for binding in bindings)
                )
            root = expression.split(".", 1)[0]
            if root in self._runtime_bound_roots[point] and self.runtime(expression, point) != result:
                result = None
        # Namespace instances describe one immutable source tree. Cache both
        # positive and unresolved results, keeping statement positions distinct.
        self._resolved_cache[key] = result
        return result
