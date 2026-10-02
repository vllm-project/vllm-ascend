# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run-local, source-only facts shared without sharing interpretation contexts."""

from __future__ import annotations

import ast
import threading
import time
from collections.abc import Callable
from typing import TypeVar, cast

T = TypeVar("T")


class SourceFacts:
    """Own immutable-source facts and counters for one repository index.

    Returned containers are read-only by convention. Never store a mutable
    execution flow, endpoint-dependent proof or final finding here. The owning
    index excludes this object from pickle and recreates it after loading.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._values: dict[tuple[str, object], object] = {}
        self.counts: dict[str, dict[str, float]] = {}

    def get(self, kind: str, key: object, build: Callable[[], T]) -> T:
        requested = time.perf_counter()
        with self._lock:
            acquired = time.perf_counter()
            counter = self.counts.setdefault(
                kind, {"requests": 0, "builds": 0, "hits": 0, "build_seconds": 0, "lock_wait_seconds": 0}
            )
            counter["lock_wait_seconds"] += acquired - requested
            counter["requests"] += 1
            identity = (kind, key)
            if identity in self._values:
                counter["hits"] += 1
                return cast(T, self._values[identity])
            started = time.perf_counter()
            value = build()
            counter["builds"] += 1
            counter["build_seconds"] += time.perf_counter() - started
            self._values[identity] = value
            return value

    def nodes(self, tree: ast.AST) -> tuple[ast.AST, ...]:
        return self.get("ast_nodes", tree, lambda: tuple(ast.walk(tree)))

    def parents(self, tree: ast.AST) -> dict[int, ast.AST]:
        return self.get(
            "ast_parents",
            tree,
            lambda: {id(child): parent for parent in self.nodes(tree) for child in ast.iter_child_nodes(parent)},
        )

    def metrics(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {kind: dict(values) for kind, values in self.counts.items()}
