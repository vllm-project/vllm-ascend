# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable identities for engine-proven method calls and endpoint replay.

These records are internal analysis evidence, not a public trust boundary.
Only a successful body proof may emit one. Fixed-body proofs retain their
source hash; factory invocations instead evaluate the effective method anew
at each endpoint. Both require an independently validated contextual lookup.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .dataclass_contracts import ClassSource

if TYPE_CHECKING:
    from .type_flow import FlowValue

MAX_METHOD_CONTEXTS = 16


def method_body_hash(node: ast.FunctionDef) -> str:
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


@dataclass(frozen=True, order=True)
class MethodCallProof:
    file: str
    line: int
    column: int
    end_line: int
    end_column: int
    receiver: str
    method: str
    body_sha256: str

    def encode(self) -> list[object]:
        return [
            self.file,
            self.line,
            self.column,
            self.end_line,
            self.end_column,
            self.receiver,
            self.method,
            self.body_sha256,
        ]

    @classmethod
    def decode(cls, value: object) -> MethodCallProof | None:
        if not isinstance(value, list) or len(value) != 8:
            return None
        if not all(isinstance(value[index], str) for index in (0, 5, 6, 7)):
            return None
        if not all(type(value[index]) is int and value[index] >= 0 for index in (1, 2, 3, 4)):
            return None
        if (
            value[1] < 1
            or value[3] < value[1]
            or not value[0].startswith("vllm_ascend/")
            or not value[5].startswith("vllm_ascend.")
            or not value[6].startswith("vllm_ascend.")
            or len(value[7]) != 64
        ):
            return None
        return cls(*value)

    def call_in(self, tree: ast.Module) -> ast.Call | None:
        matches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)
            == (self.line, self.column, self.end_line, self.end_column)
        ]
        return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True)
class FactoryCallProof:
    """Fixed invocation identity whose effective method is re-read per SHA.

    Unlike MethodCallProof this is not evidence of an invariant method body.
    It may be published only after the endpoint evaluator proves the call.
    """

    file: str
    line: int
    column: int
    end_line: int
    end_column: int
    receiver: str
    member: str

    @property
    def method(self) -> str:
        return f"{self.receiver}.{self.member}"

    def encode(self) -> list[object]:
        return [self.file, self.line, self.column, self.end_line, self.end_column, self.receiver, self.member]

    @classmethod
    def decode(cls, value: object) -> FactoryCallProof | None:
        if not isinstance(value, list) or len(value) != 7:
            return None
        if not all(isinstance(value[index], str) for index in (0, 5, 6)):
            return None
        if not all(type(value[index]) is int and value[index] >= 0 for index in (1, 2, 3, 4)):
            return None
        if (
            value[1] < 1
            or value[3] < value[1]
            or not value[0].startswith("vllm_ascend/")
            or not value[5].startswith(("vllm.", "vllm_ascend."))
            or not value[6].isidentifier()
        ):
            return None
        return cls(*value)

    def call_in(self, tree: ast.Module) -> ast.Call | None:
        # Share the exact four-coordinate call locator with fixed-body proofs.
        locator = MethodCallProof(
            self.file, self.line, self.column, self.end_line, self.end_column, self.receiver, self.method, ""
        )
        return locator.call_in(tree)


MethodProof = MethodCallProof | FactoryCallProof
FactoryInputResolver = Callable[[FactoryCallProof], tuple["FlowValue | None", ...] | None]
FactoryEvaluator = Callable[
    [FactoryCallProof, tuple["FlowValue | None", ...], int, tuple[str, ...]], "FlowValue | None"
]


@dataclass(frozen=True)
class MethodContextLookup:
    lookup: Callable[[str], ClassSource | None]
    proofs: frozenset[MethodProof]
    factory_evaluator: FactoryEvaluator | None = None

    def __call__(self, reference: str) -> ClassSource | None:
        return self.lookup(reference)
