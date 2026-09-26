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
# This file is a part of the vllm-ascend project.
"""Exact static contracts for downstream calls, member reads, and returns.

This module intentionally lives in the shared engine.  The main2main skill is
only an orchestration layer and must not grow a second AST implementation.
The analysis here is conservative: a dependency is returned only when its
callee is uniquely resolved, and dynamic argument/return shapes stay unknown.
"""

from __future__ import annotations

import ast
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Any

from .annotation_names import AnnotationNamespace
from .construction_effects import construction_effects, has_construction_effect, has_storage_effect
from .dataclass_contracts import ClassSource, dataclass_storage
from .factory_returns import factory_body_supported, proven_factory_return
from .generator import (
    _TRITON_JIT_DECORATOR,
    _TRITON_KERNEL_PROTOCOL,
    InterfaceBoundaryGenerator,
    ModuleInfo,
    RepositoryIndex,
    _expression_name,
    _function_local_names,
    _function_scope_nodes,
    _inspect_signature,
    _scope_final_bindings,
    _scope_reference_variants,
    _ScopeBinding,
    _ScopePrefixCache,
    _statements_must_terminate,
    _tag_guard_names,
)
from .helper_effects import HelperEffects, HelperSummary
from .method_context import (
    MAX_METHOD_CONTEXTS,
    FactoryCallProof,
    FactoryEvaluator,
    FactoryInputResolver,
    MethodCallProof,
    method_body_hash,
)
from .receiver_constraints import ReceiverConstraints
from .source_facts import SourceFacts
from .type_flow import (
    TYPE_BUILTINS,
    ContainerFlow,
    FlowValue,
    HelperCallEffects,
    HelperResolver,
    Lookup,
    TypeShape,
    factory_context_value,
    instance_field_origin,
    method_context_value,
    method_path_proofs,
)

MAX_FACTORY_RETURN_DEPTH = 16
MAX_RESOLVER_SCOPE_CACHES = 8


@dataclass(frozen=True)
class CallShape:
    """One concrete Python call expression, before runtime expansion."""

    positional_count: int
    keyword_names: tuple[str, ...]
    dynamic_starargs: bool = False
    dynamic_kwargs: bool = False

    @property
    def exact(self) -> bool:
        return not self.dynamic_starargs and not self.dynamic_kwargs

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReturnShape:
    """A structural shape that callers can observe without executing code."""

    kind: str
    arity: int | None = None
    keys: tuple[str, ...] = ()
    type_ref: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReturnContract:
    """The observable call protocol and possible successful return shapes."""

    protocol: str
    variants: tuple[ReturnShape, ...]
    status: str
    provenance: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "variants": [item.as_dict() for item in self.variants],
            "status": self.status,
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class ReturnUse:
    """How one downstream callsite immediately consumes a return value."""

    kind: str
    awaited: bool = False
    arity: int | None = None
    minimum_arity: int | None = None
    key: str | None = None
    index: int | None = None
    attribute: str | None = None
    status: str = "exact"

    @property
    def constrains_return(self) -> bool:
        return self.kind not in {"ignored", "passthrough"}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DirectCallDependency:
    """One uniquely resolved vLLM call made by vllm-ascend."""

    target: str
    access_kind: str
    file: str
    line: int
    column: int
    owner: str | None
    scope: str | None
    callee: str
    call_shape: CallShape
    return_use: ReturnUse
    receiver_type: str | None = None
    member: str | None = None
    invocation_kind: str = "python_call"
    lookup_root: str | None = None
    resolution_basis: str = "new_exact"
    receiver_path: tuple[str, ...] = ()
    receiver_binding: dict[str, Any] | None = None
    constructor_mutations: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["call_shape"] = self.call_shape.as_dict()
        payload["return_use"] = self.return_use.as_dict()
        return payload


@dataclass(frozen=True)
class DirectAttributeDependency:
    """One uniquely resolved vLLM member read made by vllm-ascend."""

    target: str
    access_kind: str
    file: str
    line: int
    column: int
    owner: str | None
    scope: str | None
    expression: str
    receiver_type: str | None = None
    member: str | None = None
    lookup_root: str | None = None
    resolution_basis: str = "new_exact"
    receiver_path: tuple[str, ...] = ()
    receiver_binding: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def call_shape(node: ast.Call) -> CallShape:
    positional_count = 0
    dynamic_starargs = False
    keyword_names: list[str] = []
    dynamic_kwargs = False
    for argument in node.args:
        if not isinstance(argument, ast.Starred):
            positional_count += 1
            continue
        if isinstance(argument.value, (ast.List, ast.Tuple)) and not any(
            isinstance(item, ast.Starred) for item in argument.value.elts
        ):
            positional_count += len(argument.value.elts)
        else:
            dynamic_starargs = True
    for keyword in node.keywords:
        if keyword.arg is not None:
            keyword_names.append(keyword.arg)
            continue
        if isinstance(keyword.value, ast.Dict) and all(
            isinstance(key, ast.Constant) and isinstance(key.value, str) for key in keyword.value.keys
        ):
            # A dict literal resolves duplicate keys before ``**`` expansion;
            # duplicates across separate expansions still remain visible.
            keyword_names.extend(
                dict.fromkeys(
                    key.value
                    for key in keyword.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                )
            )
        else:
            dynamic_kwargs = True
    return CallShape(
        positional_count=positional_count,
        keyword_names=tuple(keyword_names),
        dynamic_starargs=dynamic_starargs,
        dynamic_kwargs=dynamic_kwargs,
    )


def bind_call_shape(signature: list[object] | None, shape: CallShape) -> tuple[bool | None, str]:
    """Bind one actual call shape, not a replacement substitutability set."""

    if signature is None:
        return None, "callable signature could not be resolved"
    candidate = _inspect_signature(signature)
    if candidate is None:
        return None, "callable signature is not representable"
    if len(set(shape.keyword_names)) != len(shape.keyword_names):
        return False, "the call supplies the same keyword more than once"
    args = [object() for _ in range(shape.positional_count)]
    kwargs = {name: object() for name in shape.keyword_names}
    try:
        if shape.exact:
            candidate.bind(*args, **kwargs)
        else:
            candidate.bind_partial(*args, **kwargs)
    except TypeError as error:
        return False, f"call arguments do not bind: {error}"
    if not shape.exact:
        return None, "dynamic *args or **kwargs prevents exact binding"
    return True, "call arguments bind to the callable contract"


def _shape_key(shape: ReturnShape) -> str:
    return json.dumps(shape.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ordered_shapes(shapes: Iterable[ReturnShape]) -> tuple[ReturnShape, ...]:
    unique = {_shape_key(item): item for item in shapes}
    return tuple(unique[key] for key in sorted(unique))


def _resolved_name(node: ast.AST | None, resolver: Callable[[str], str | None] | None) -> str | None:
    name = _expression_name(node)
    if name is None:
        return None
    return resolver(name) if resolver is not None else name


def _annotation_shapes(
    node: ast.AST | None,
    resolver: Callable[[str], str | None] | None,
) -> tuple[ReturnShape, ...] | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        if node.value is None:
            return (ReturnShape("none"),)
        if isinstance(node.value, str):
            # Forward-reference strings are deliberately not evaluated.
            return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _annotation_shapes(node.left, resolver)
        right = _annotation_shapes(node.right, resolver)
        return _ordered_shapes((*left, *right)) if left is not None and right is not None else None
    if isinstance(node, ast.Subscript):
        base = (_expression_name(node.value) or "").rsplit(".", 1)[-1]
        elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if base in {"Optional"} and len(elements) == 1:
            nested = _annotation_shapes(elements[0], resolver)
            return _ordered_shapes((*nested, ReturnShape("none"))) if nested is not None else None
        if base in {"Union"}:
            variants: list[ReturnShape] = []
            for element in elements:
                nested = _annotation_shapes(element, resolver)
                if nested is None:
                    return None
                variants.extend(nested)
            return _ordered_shapes(variants)
        if base in {"tuple", "Tuple"}:
            if len(elements) == 2 and isinstance(elements[1], ast.Constant) and elements[1].value is Ellipsis:
                return (ReturnShape("tuple_variadic"),)
            return (ReturnShape("tuple", arity=len(elements)),)
        if base in {"list", "List", "Sequence"}:
            return (ReturnShape("sequence"),)
        if base in {"dict", "Dict", "Mapping"}:
            return (ReturnShape("mapping"),)
        if base in {"Iterator", "Iterable", "Generator"}:
            return (ReturnShape("opaque", type_ref="iterator_item"),)
        if base in {"AsyncIterator", "AsyncIterable", "AsyncGenerator"}:
            return (ReturnShape("opaque", type_ref="async_iterator_item"),)
        if base in {"ContextManager"}:
            return (ReturnShape("opaque", type_ref="context_value"),)
        if base in {"AsyncContextManager"}:
            return (ReturnShape("opaque", type_ref="async_context_value"),)
        return None
    name = _resolved_name(node, resolver)
    if name is None or name.rsplit(".", 1)[-1] in {"Any", "NoReturn", "Never"}:
        return None
    short = name.rsplit(".", 1)[-1]
    if short in {"Iterator", "Iterable", "Generator"}:
        return (ReturnShape("opaque", type_ref="iterator_item"),)
    if short in {"AsyncIterator", "AsyncIterable", "AsyncGenerator"}:
        return (ReturnShape("opaque", type_ref="async_iterator_item"),)
    if short == "ContextManager":
        return (ReturnShape("opaque", type_ref="context_value"),)
    if short == "AsyncContextManager":
        return (ReturnShape("opaque", type_ref="async_context_value"),)
    if short in {"None", "NoneType"}:
        return (ReturnShape("none"),)
    if short in {"bool", "bytes", "complex", "float", "int", "str"}:
        return (ReturnShape("scalar", type_ref=f"builtins.{short}"),)
    return (ReturnShape("object", type_ref=name),)


def _annotation_protocol(node: ast.AST | None) -> str | None:
    """Return an exact observable protocol declared by one annotation.

    A protocol hidden inside ``Optional`` or a heterogeneous union is not
    exact: callers cannot assume that every successful return supplies it.
    """

    if node is None:
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _annotation_protocol(node.left)
        right = _annotation_protocol(node.right)
        return left if left is not None and left == right else None
    if isinstance(node, ast.Subscript):
        base = (_expression_name(node.value) or "").rsplit(".", 1)[-1]
        if base in {"Optional", "Union"}:
            return None
    else:
        base = (_expression_name(node) or "").rsplit(".", 1)[-1]
    return {
        "Iterator": "iterator",
        "Iterable": "iterator",
        "Generator": "iterator",
        "AsyncIterator": "async_iterator",
        "AsyncIterable": "async_iterator",
        "AsyncGenerator": "async_iterator",
        "ContextManager": "context_manager",
        "AsyncContextManager": "async_context_manager",
        "Awaitable": "awaitable",
        "Coroutine": "awaitable",
    }.get(base)


def _assignment_value(
    function: ast.AsyncFunctionDef | ast.FunctionDef,
    name: str,
    before_line: int,
) -> ast.AST | None:
    values: list[ast.AST] = []
    for node in _function_scope_nodes(function):
        if getattr(node, "lineno", before_line) >= before_line:
            continue
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                values.append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            if node.value is not None:
                values.append(node.value)
    return values[0] if len(values) == 1 else None


def _expression_shapes(
    node: ast.AST | None,
    *,
    function: ast.AsyncFunctionDef | ast.FunctionDef,
    resolver: Callable[[str], str | None] | None,
    forward_name: str | None,
    seen_names: frozenset[str] = frozenset(),
) -> tuple[ReturnShape, ...] | None:
    if node is None or (isinstance(node, ast.Constant) and node.value is None):
        return (ReturnShape("none"),)
    if isinstance(node, ast.IfExp):
        body = _expression_shapes(
            node.body,
            function=function,
            resolver=resolver,
            forward_name=forward_name,
            seen_names=seen_names,
        )
        other = _expression_shapes(
            node.orelse,
            function=function,
            resolver=resolver,
            forward_name=forward_name,
            seen_names=seen_names,
        )
        return _ordered_shapes((*body, *other)) if body is not None and other is not None else None
    if isinstance(node, ast.Tuple) and not any(isinstance(item, ast.Starred) for item in node.elts):
        return (ReturnShape("tuple", arity=len(node.elts)),)
    if isinstance(node, ast.List) and not any(isinstance(item, ast.Starred) for item in node.elts):
        return (ReturnShape("list", arity=len(node.elts)),)
    if isinstance(node, ast.Dict) and all(
        isinstance(key, ast.Constant) and isinstance(key.value, str) for key in node.keys
    ):
        return (
            ReturnShape(
                "mapping",
                keys=tuple(
                    sorted(
                        key.value for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)
                    )
                ),
            ),
        )
    if isinstance(node, ast.Constant):
        return (ReturnShape("scalar", type_ref=f"builtins.{type(node.value).__name__}"),)
    if isinstance(node, ast.Name) and node.id not in seen_names:
        value = _assignment_value(function, node.id, getattr(node, "lineno", 0))
        if value is not None:
            return _expression_shapes(
                value,
                function=function,
                resolver=resolver,
                forward_name=forward_name,
                seen_names=frozenset((*seen_names, node.id)),
            )
        return None
    if isinstance(node, ast.Call):
        callee = _expression_name(node.func)
        if (
            forward_name
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == forward_name
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "super"
        ):
            return (ReturnShape("forward", type_ref=forward_name),)
        resolved = _resolved_name(node.func, resolver)
        short = (resolved or callee or "").rsplit(".", 1)[-1]
        if short == "tuple":
            return (ReturnShape("tuple_variadic"),)
        if short == "list":
            return (ReturnShape("sequence"),)
        if short == "dict":
            return (ReturnShape("mapping"),)
        if resolved and short[:1].isupper():
            return (ReturnShape("object", type_ref=resolved),)
        return None
    return None


def infer_return_contract(
    node: ast.AST | None,
    *,
    resolver: Callable[[str], str | None] | None = None,
    forward_name: str | None = None,
) -> ReturnContract | None:
    """Infer a conservative return contract for one Python function."""

    if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
        return None
    scope_nodes = list(_function_scope_nodes(node))
    has_yield = any(isinstance(item, (ast.Yield, ast.YieldFrom)) for item in scope_nodes)
    decorators = [_expression_name(item.func if isinstance(item, ast.Call) else item) for item in node.decorator_list]
    resolved_decorators = [resolver(name) if resolver is not None and name is not None else name for name in decorators]
    known_origins = {
        "abc.abstractmethod",
        "contextlib.asynccontextmanager",
        "contextlib.contextmanager",
        "typing.override",
        "typing_extensions.override",
    }
    builtin_descriptors = {"builtins.classmethod", "builtins.property", "builtins.staticmethod"}
    fallback_known = {
        "abstractmethod",
        "asynccontextmanager",
        "contextmanager",
        "override",
    }
    unknown_decorator = any(
        raw is None
        or not (
            resolved in builtin_descriptors
            or (resolver is None and raw in {"classmethod", "property", "staticmethod"})
            or resolved in known_origins
            or (resolver is None and raw.rsplit(".", 1)[-1] in fallback_known)
        )
        for raw, resolved in zip(decorators, resolved_decorators, strict=True)
    )

    def exact_status(status: str) -> str:
        return "unknown" if status == "exact" and unknown_decorator else status

    annotation_protocol = _annotation_protocol(node.returns)
    if "contextlib.asynccontextmanager" in resolved_decorators or (
        resolver is None and any((name or "").endswith("asynccontextmanager") for name in decorators)
    ):
        protocol = "async_context_manager"
    elif "contextlib.contextmanager" in resolved_decorators or (
        resolver is None and any((name or "").endswith("contextmanager") for name in decorators)
    ):
        protocol = "context_manager"
    elif has_yield and isinstance(node, ast.AsyncFunctionDef):
        protocol = "async_iterator"
    elif has_yield:
        protocol = "iterator"
    elif isinstance(node, ast.AsyncFunctionDef):
        protocol = "awaitable"
    elif annotation_protocol is not None:
        protocol = annotation_protocol
    else:
        protocol = "value"

    declared = _annotation_shapes(node.returns, resolver)
    returns = [item for item in scope_nodes if isinstance(item, ast.Return)]
    observed: list[ReturnShape] = []
    observed_exact = True
    if not has_yield:
        for item in returns:
            shapes = _expression_shapes(
                item.value,
                function=node,
                resolver=resolver,
                forward_name=forward_name,
            )
            if shapes is None:
                observed_exact = False
            else:
                observed.extend(shapes)
        if not _statements_must_terminate(node.body):
            observed.append(ReturnShape("none"))

    provenance: list[str] = []
    if declared is not None:
        provenance.append("return_annotation")
    if observed:
        provenance.append("return_statements")
    if has_yield:
        provenance.append("yield_protocol")
    if unknown_decorator:
        provenance.append("unknown_return_transform")

    if has_yield:
        variants = declared or (ReturnShape("opaque", type_ref="yield_item"),)
        return ReturnContract(
            protocol,
            _ordered_shapes(variants),
            exact_status("exact" if declared else "unknown"),
            tuple(provenance),
        )
    if not returns and _statements_must_terminate(node.body):
        if declared is not None:
            return ReturnContract(
                protocol,
                _ordered_shapes(declared),
                exact_status("exact"),
                tuple(provenance),
            )
        return ReturnContract(protocol, (), "bottom", ("no_normal_return",))
    if observed_exact and observed:
        observed_variants = _ordered_shapes(observed)
        if all(candidate.kind == "forward" for candidate in observed_variants):
            # A transparent ``return super().same_method(...)`` override keeps
            # following the selected upstream endpoint at each snapshot.  A
            # stale annotation must not freeze the replacement to the old
            # return shape and create a runtime false positive.
            return ReturnContract(
                protocol,
                observed_variants,
                exact_status("exact"),
                (*provenance, "transparent_super_forward_precedes_annotation"),
            )
        if declared is not None:
            # A precise annotation is the public contract.  Keep the body as
            # corroborating evidence, but do not turn value-level differences
            # into interface breaks.
            declared_variants = _ordered_shapes(declared)
            conflicts = [
                candidate
                for candidate in observed_variants
                if all(_replacement_shape_accepted(expected, candidate) is False for expected in declared_variants)
            ]
            if conflicts:
                return ReturnContract(
                    protocol,
                    declared_variants,
                    "unknown",
                    (*provenance, "annotation_body_conflict"),
                )
            return ReturnContract(
                protocol,
                declared_variants,
                exact_status("exact"),
                tuple(provenance),
            )
        return ReturnContract(
            protocol,
            observed_variants,
            exact_status("exact"),
            tuple(provenance),
        )
    if declared is not None:
        return ReturnContract(
            protocol,
            _ordered_shapes(declared),
            exact_status("exact"),
            tuple(provenance),
        )
    return ReturnContract(protocol, _ordered_shapes(observed), "unknown", tuple(provenance or ["dynamic_return"]))


def return_contract_from_dict(payload: dict[str, Any] | None) -> ReturnContract | None:
    if payload is None:
        return None
    try:
        return ReturnContract(
            protocol=str(payload["protocol"]),
            variants=tuple(
                ReturnShape(
                    kind=str(item["kind"]),
                    arity=item.get("arity"),
                    keys=tuple(str(key) for key in item.get("keys", ())),
                    type_ref=item.get("type_ref"),
                )
                for item in payload.get("variants", ())
            ),
            status=str(payload["status"]),
            provenance=tuple(str(item) for item in payload.get("provenance", ())),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _replacement_shape_accepted(expected: ReturnShape, candidate: ReturnShape) -> bool | None:
    if candidate.kind == "forward":
        return True
    if expected.kind == "object" and expected.type_ref in {"builtins.object", "object"}:
        return candidate.kind != "none"
    if expected.kind == "sequence":
        return candidate.kind in {"list", "sequence", "tuple", "tuple_variadic"}
    if expected.kind == "tuple_variadic":
        return candidate.kind in {"tuple", "tuple_variadic"}
    if candidate.kind == "tuple_variadic":
        # A variable-length tuple cannot promise any one fixed outer arity.
        return False if expected.kind == "tuple" else None
    if expected.kind != candidate.kind:
        return False
    if expected.kind in {"list", "tuple"}:
        return expected.arity == candidate.arity
    if expected.kind == "mapping":
        if not expected.keys:
            return True
        if not candidate.keys:
            return None
        return set(expected.keys).issubset(candidate.keys)
    if expected.kind in {"object", "scalar"}:
        if expected.type_ref is None or candidate.type_ref is None:
            return None
        return expected.type_ref == candidate.type_ref
    return True


def replacement_return_compatible(
    upstream: ReturnContract | None,
    downstream: ReturnContract | None,
) -> tuple[bool | None, str]:
    """Check covariant return substitutability for patch/override code."""

    if upstream is None or downstream is None:
        return None, "return contract could not be resolved"
    if upstream.status == "bottom":
        return None, "upstream has no observable normal return contract"
    if downstream.status == "bottom":
        return None, "replacement has no observable normal return contract"
    if upstream.status != "exact" or downstream.status != "exact":
        return None, "return contract contains a dynamic or ambiguous shape"
    if upstream.protocol != downstream.protocol:
        return False, f"return protocol changed from {upstream.protocol} to {downstream.protocol}"
    uncertain = False
    for candidate in downstream.variants:
        results = [_replacement_shape_accepted(expected, candidate) for expected in upstream.variants]
        if True in results:
            continue
        if None in results:
            uncertain = True
            continue
        return False, f"replacement return shape {candidate.kind} is outside the upstream contract"
    if uncertain:
        return None, "nominal return compatibility could not be proven"
    return True, "replacement return values satisfy the upstream return contract"


def _use_shape_compatible(shape: ReturnShape, use: ReturnUse) -> bool | None:
    if shape.kind == "forward":
        return None
    if use.kind == "unpack":
        if shape.kind not in {"list", "tuple"}:
            return None if shape.kind in {"sequence", "tuple_variadic"} else False
        if use.arity is not None:
            return shape.arity == use.arity
        return shape.arity is not None and shape.arity >= (use.minimum_arity or 0)
    if use.kind == "iterate":
        return shape.kind in {"list", "mapping", "sequence", "tuple", "tuple_variadic"}
    if use.kind == "subscript_index":
        if shape.kind not in {"list", "tuple"}:
            return None if shape.kind in {"sequence", "tuple_variadic"} else False
        if shape.arity is None or use.index is None:
            return None
        return -shape.arity <= use.index < shape.arity
    if use.kind == "subscript_key":
        if shape.kind != "mapping":
            return False
        if not shape.keys or use.key is None:
            return None
        return use.key in shape.keys
    if use.kind == "attribute":
        if shape.kind == "none":
            return False
        return None
    return True


def return_use_compatible(
    contract: ReturnContract | None,
    use: ReturnUse,
) -> tuple[bool | None, str]:
    if not use.constrains_return:
        return True, "the call result is not structurally consumed"
    if use.status != "exact":
        return None, "return value escapes or is consumed dynamically"
    if contract is None or contract.status != "exact":
        return None, "upstream return contract could not be proven"
    if use.awaited:
        if contract.protocol != "awaitable":
            return False, "the downstream awaits a non-awaitable return protocol"
    elif contract.protocol == "awaitable" and use.kind != "ignored":
        return False, "the downstream consumes an awaitable without awaiting it"
    if use.kind == "async_iterate":
        compatible = contract.protocol == "async_iterator"
        return compatible, "async iteration protocol matches" if compatible else "return is not an async iterator"
    if use.kind == "iterate" and contract.protocol == "iterator":
        return True, "iterator protocol matches"
    if use.kind == "context":
        compatible = contract.protocol == "context_manager"
        return compatible, "context-manager protocol matches" if compatible else "return is not a context manager"
    if use.kind == "async_context":
        compatible = contract.protocol == "async_context_manager"
        return compatible, (
            "async context-manager protocol matches" if compatible else "return is not an async context manager"
        )
    if use.kind == "await_only":
        return True, "awaitable protocol matches"
    if contract.protocol not in {"value", "awaitable"}:
        return False, f"{contract.protocol} does not provide the consumed value protocol"
    results = [_use_shape_compatible(shape, use) for shape in contract.variants]
    if False in results:
        return False, "at least one upstream return shape violates the downstream use"
    if results and all(result is True for result in results):
        return True, "all upstream return shapes satisfy the downstream use"
    return None, "the downstream return use could not be proven for every shape"


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    return {id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _nearest(node: ast.AST, parents: dict[int, ast.AST], kinds: tuple[type[ast.AST], ...]) -> ast.AST | None:
    current = parents.get(id(node))
    while current is not None:
        if isinstance(current, kinds):
            return current
        current = parents.get(id(current))
    return None


def _unpack_use(target: ast.AST, *, awaited: bool) -> ReturnUse | None:
    if not isinstance(target, (ast.List, ast.Tuple)):
        return None
    starred = sum(isinstance(item, ast.Starred) for item in target.elts)
    if starred == 0:
        return ReturnUse("unpack", awaited=awaited, arity=len(target.elts))
    if starred == 1:
        return ReturnUse("unpack", awaited=awaited, minimum_arity=len(target.elts) - 1)
    return ReturnUse("unknown", awaited=awaited, status="unknown")


def _same_scope_nodes(scope: ast.AST) -> Iterable[ast.AST]:
    if isinstance(scope, (ast.AsyncFunctionDef, ast.FunctionDef)):
        yield from _function_scope_nodes(scope)
    else:
        for child in ast.walk(scope):
            if child is not scope:
                yield child


def infer_return_use(node: ast.Call, parents: dict[int, ast.AST], scope: ast.AST) -> ReturnUse:
    value: ast.AST = node
    awaited = False
    parent = parents.get(id(value))
    if isinstance(parent, ast.Await) and parent.value is value:
        awaited = True
        value = parent
        parent = parents.get(id(value))

    if isinstance(parent, ast.Assign) and parent.value is value and len(parent.targets) == 1:
        unpack = _unpack_use(parent.targets[0], awaited=awaited)
        if unpack is not None:
            return unpack
        if isinstance(parent.targets[0], ast.Name):
            alias = parent.targets[0].id
            loads = [
                child
                for child in _same_scope_nodes(scope)
                if isinstance(child, ast.Name)
                and isinstance(child.ctx, ast.Load)
                and child.id == alias
                and getattr(child, "lineno", 0) >= getattr(parent, "lineno", 0)
            ]
            stores = [
                child
                for child in _same_scope_nodes(scope)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store) and child.id == alias
            ]
            if len(loads) == 1 and len(stores) == 1:
                value = loads[0]
                parent = parents.get(id(value))
            elif not loads:
                return ReturnUse("ignored", awaited=awaited)
            else:
                return ReturnUse("unknown", awaited=awaited, status="unknown")
    elif isinstance(parent, ast.NamedExpr) and parent.value is value:
        return ReturnUse("unknown", awaited=awaited, status="unknown")

    if isinstance(parent, ast.Assign) and parent.value is value:
        unpack = _unpack_use(parent.targets[0], awaited=awaited) if len(parent.targets) == 1 else None
        return unpack or ReturnUse("unknown", awaited=awaited, status="unknown")
    if isinstance(parent, ast.Attribute) and parent.value is value:
        return ReturnUse("attribute", awaited=awaited, attribute=parent.attr)
    if isinstance(parent, ast.Subscript) and parent.value is value:
        if isinstance(parent.slice, ast.Constant) and isinstance(parent.slice.value, int):
            return ReturnUse("subscript_index", awaited=awaited, index=parent.slice.value)
        if isinstance(parent.slice, ast.Constant) and isinstance(parent.slice.value, str):
            return ReturnUse("subscript_key", awaited=awaited, key=parent.slice.value)
        return ReturnUse("unknown", awaited=awaited, status="unknown")
    if isinstance(parent, (ast.For, ast.comprehension)) and parent.iter is value:
        return ReturnUse("iterate", awaited=awaited)
    if isinstance(parent, ast.AsyncFor) and parent.iter is value:
        return ReturnUse("async_iterate", awaited=awaited)
    if isinstance(parent, ast.withitem) and parent.context_expr is value:
        container = parents.get(id(parent))
        return ReturnUse("async_context" if isinstance(container, ast.AsyncWith) else "context", awaited=awaited)
    if awaited:
        return ReturnUse("await_only", awaited=True)
    if isinstance(parent, ast.Expr):
        return ReturnUse("ignored")
    if isinstance(parent, ast.Return):
        return ReturnUse("passthrough")
    if parent is None:
        return ReturnUse("ignored")
    return ReturnUse("unknown", status="unknown")


def _under_version_guard(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    current = parents.get(id(node))
    while current is not None:
        if isinstance(current, ast.If) and any(
            isinstance(item, ast.Call) and (_expression_name(item.func) or "").rsplit(".", 1)[-1] == "vllm_version_is"
            for item in ast.walk(current.test)
        ):
            return True
        current = parents.get(id(current))
    return False


def _inside_annotation(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    current = node
    parent = parents.get(id(current))
    while parent is not None:
        if isinstance(parent, ast.arg) and parent.annotation is current:
            return True
        if isinstance(parent, ast.AnnAssign) and parent.annotation is current:
            return True
        if isinstance(parent, (ast.AsyncFunctionDef, ast.FunctionDef)) and parent.returns is current:
            return True
        current = parent
        parent = parents.get(id(current))
    return False


def _under_attribute_fallback(node: ast.Attribute, parents: dict[int, ast.AST]) -> bool:
    """Return whether source explicitly handles this member being absent."""

    receiver = ast.dump(node.value, include_attributes=False)

    def positively_guards(test: ast.AST) -> bool:
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
            return any(positively_guards(value) for value in test.values)
        return (
            isinstance(test, ast.Call)
            and (_expression_name(test.func) or "").rsplit(".", 1)[-1] == "hasattr"
            and len(test.args) == 2
            and ast.dump(test.args[0], include_attributes=False) == receiver
            and isinstance(test.args[1], ast.Constant)
            and test.args[1].value == node.attr
        )

    current: ast.AST = node
    parent = parents.get(id(current))
    while parent is not None:
        if isinstance(parent, ast.If) and current in parent.body and positively_guards(parent.test):
            return True
        if isinstance(parent, ast.Try) and current in parent.body:
            if any(
                (_expression_name(handler.type) or "").rsplit(".", 1)[-1] == "AttributeError"
                for handler in parent.handlers
            ):
                return True
        current = parent
        parent = parents.get(id(current))
    return False


def _attribute_is_read(node: ast.Attribute, parents: dict[int, ast.AST]) -> bool:
    if isinstance(node.ctx, ast.Load):
        return True
    parent = parents.get(id(node))
    return isinstance(parent, ast.AugAssign) and parent.target is node


def _attribute_is_call_target(node: ast.Attribute, parents: dict[int, ast.AST]) -> bool:
    current: ast.AST = node
    parent = parents.get(id(current))
    while (
        isinstance(parent, ast.Attribute)
        and parent.value is current
        or isinstance(parent, ast.Subscript)
        and parent.value is current
    ):
        current = parent
        parent = parents.get(id(current))
    return isinstance(parent, ast.Call) and parent.func is current


def _member_access(node: ast.Call | ast.Attribute) -> ast.Attribute | None:
    candidate = node.func if isinstance(node, ast.Call) else node
    return candidate if isinstance(candidate, ast.Attribute) else None


def _annotation_reference(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        candidates = {_annotation_reference(node.left), _annotation_reference(node.right)} - {None, "None"}
        return next(iter(candidates)) if len(candidates) == 1 else None
    if isinstance(node, ast.Subscript):
        base = (_expression_name(node.value) or "").rsplit(".", 1)[-1]
        if base == "Optional":
            return _annotation_reference(node.slice)
        return None
    return _expression_name(node)


@dataclass(frozen=True)
class _ReceiverFlowSummary:
    """Read-only receiver proof detached from one resolver's callbacks."""

    receivers: Mapping[int, FlowValue]
    invalidated_receivers: frozenset[int]


class _DirectDependencyResolver:
    """Shared exact name, receiver, and MRO resolution for downstream uses."""

    _reference_prefixes: tuple[str, ...] = ("vllm.",)

    def __init__(
        self,
        engine: InterfaceBoundaryGenerator,
        *,
        historical_type_source: Lookup | None = None,
        factory_evaluator: FactoryEvaluator | None = None,
        historical_factory_evaluator: FactoryEvaluator | None = None,
        factory_input_resolver: FactoryInputResolver | None = None,
        historical_factory_input_resolver: FactoryInputResolver | None = None,
        receiver_flow_cache_key: object | None = None,
        historical_receiver_flow_cache_key: object | None = None,
    ):
        self.engine = engine
        self._facts = getattr(getattr(engine, "downstream", None), "source_facts", None) or SourceFacts()
        self._prefix_cache = _ScopePrefixCache()
        self._factory_evaluator = factory_evaluator
        self._factory_input_resolver = factory_input_resolver
        self._upstream_type_source: Lookup | None = None
        self._receiver_flow_cache_key = receiver_flow_cache_key
        self._historical_flow_resolver = (
            _DirectDependencyResolver(
                engine,
                receiver_flow_cache_key=historical_receiver_flow_cache_key,
            )
            if historical_type_source is not None
            else None
        )
        if self._historical_flow_resolver is not None:
            self._historical_flow_resolver._upstream_type_source = historical_type_source
            self._historical_flow_resolver._factory_evaluator = historical_factory_evaluator
            self._historical_flow_resolver._factory_input_resolver = historical_factory_input_resolver
        self.historical_candidates: list[DirectCallDependency] = []
        self._candidate_roots: dict[tuple[str, int], frozenset[str]] = {}
        self._constructed_instances: dict[
            tuple[str, int],
            dict[str, ast.AnnAssign | ast.Call | None],
        ] = {}
        self._function_locals: dict[int, frozenset[str]] = {}
        self._scope_tag_guards: dict[int, set[str]] = {}
        self._scope_states: dict[int, dict[int, dict[str, tuple[_ScopeBinding, ...]]]] = {}
        self._instance_member_types: dict[tuple[str, str], frozenset[str] | None] = {}
        self._container_flows: dict[
            tuple[int, bool, tuple[tuple[str, FlowValue | None], ...] | None], ContainerFlow
        ] = {}
        self._receiver_flow_summaries: dict[int, _ReceiverFlowSummary] = {}
        self._factory_returns: dict[tuple[str, tuple[tuple[str, FlowValue | None], ...]], FlowValue | None] = {}
        self._factory_stack: set[str] = set()
        self._annotation_namespaces: dict[tuple[int, str], AnnotationNamespace] = {}
        self._helper_summaries: dict[tuple[str, tuple[tuple[str, TypeShape | None], ...]], HelperSummary] = {}
        self._parent_maps: dict[str, dict[int, ast.AST]] = {}
        self._stored_field_origins: dict[tuple[str, str], tuple[str, TypeShape, str, bool] | None] = {}
        self._receiver_constraints = ReceiverConstraints(engine)
        self._storage_module_effects: dict[tuple[int, str], frozenset[str]] = {}
        self._storage_downstream_effects: frozenset[str] | None = None
        self._storage_references: dict[str, bool] = {}
        self._replacement_roots: dict[str, set[str]] = {}
        self._method_proofs: frozenset[MethodCallProof] = frozenset()
        self._flow_metrics: dict[str, dict[str, float]] = {}

    def _source_module_effects(self, index: RepositoryIndex, module: ModuleInfo) -> frozenset[str]:
        return self._facts.get(
            "construction_effects",
            (module.tree, module.name, module.is_package),
            lambda: construction_effects(module.tree, module.name, module.is_package),
        )

    def _function_nodes(self, function: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[ast.AST, ...]:
        return self._facts.get("function_scope_nodes", function, lambda: tuple(_function_scope_nodes(function)))

    def metrics(self) -> dict[str, Any]:
        result: dict[str, Any] = {"flows": self._flow_metrics, "scope_prefixes": self._prefix_cache.metrics()}
        if self._historical_flow_resolver is not None:
            result["historical"] = self._historical_flow_resolver.metrics()
        return result

    def _bind_flow_method(
        self,
        call: ast.Call,
        receiver: FlowValue | None,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        module: ModuleInfo,
    ) -> HelperResolver | None:
        fixed = self._bound_factory_method(call, receiver, function, module)
        if self._factory_evaluator is None or receiver is not None or not isinstance(call.func, ast.Attribute):
            return fixed
        expression = _expression_name(call.func.value)
        if expression is None or expression.split(".")[0] in self._local_names(function):
            return fixed
        parents = self._parent_maps[module.name]
        root = expression.split(".")[0]
        if self._outer_function_shadows(call, root, parents, function) or (
            DownstreamConstructorDetector._expression_scope_shadows(call, root, parents)
        ):
            return fixed
        reference = self._resolve_in_scope(
            call.func.value,
            function=function,
            module_info=module,
            line=call.lineno,
            reference_prefixes=("vllm.", "vllm_ascend."),
        )
        if reference is None:
            return fixed
        proof = FactoryCallProof.decode(
            [
                module.file.replace("\\", "/"),
                call.lineno,
                call.col_offset,
                call.end_lineno,
                call.end_col_offset,
                reference,
                call.func.attr,
            ]
        )
        if proof is None or proof.call_in(module.tree) is not call:
            return fixed
        evaluate = self._factory_evaluator

        def invoke(node: ast.Call, values: tuple[FlowValue | None, ...]) -> HelperCallEffects | None:
            # The fixed binder may bind successfully but fail its body proof.
            # Fall back after argument evaluation, without evaluating it twice.
            prior = fixed(node, values) if fixed is not None else None
            if prior is not None:
                return prior
            if any(isinstance(arg, ast.Starred) for arg in node.args) or any(kw.arg is None for kw in node.keywords):
                return None
            positional = len(node.args)
            keywords = tuple(kw.arg for kw in node.keywords if kw.arg is not None)
            actual = values
            if any(value is None or method_path_proofs(value.path, require_complete=True) is None for value in actual):
                repaired = self._factory_input_resolver(proof) if self._factory_input_resolver is not None else None
                if repaired is None or len(repaired) != len(values):
                    return None
                actual = repaired
            result = evaluate(proof, actual, positional, keywords)
            if result is None:
                return None
            result = factory_context_value(result, proof, actual, positional, keywords)
            if result is None:
                return None
            return HelperCallEffects(
                frozenset(range(len(values))),
                frozenset(),
                None,
                None,
                result if result.owned is None else None,
                result if result.owned is not None else None,
            )

        return invoke

    def _bound_factory_method(
        self,
        call: ast.Call,
        receiver: FlowValue | None,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        module: ModuleInfo,
    ) -> HelperResolver | None:
        """Bind only a unique fixed downstream method, never an annotation owner."""
        if not isinstance(call.func, ast.Attribute):
            return None
        member = call.func
        instance = receiver is not None
        if receiver is not None:
            if not receiver.constructed or member.attr in dict(receiver.stored_fields):
                return None
            class_reference = receiver.shape.reference
        else:
            expression = _expression_name(member.value)
            if expression is None or expression.split(".")[0] in self._local_names(function):
                return None
            parents = self._parent_maps[module.name]
            root = expression.split(".")[0]
            if self._outer_function_shadows(call, root, parents, function) or (
                DownstreamConstructorDetector._expression_scope_shadows(call, root, parents)
            ):
                return None
            resolved_class = self._resolve_in_scope(
                member.value,
                function=function,
                module_info=module,
                line=call.lineno,
                reference_prefixes=("vllm_ascend.",),
            )
            if resolved_class is None:
                return None
            class_reference = resolved_class
        if not class_reference.startswith("vllm_ascend."):
            return None
        mro = self.engine._linearized_mro(class_reference)
        if not mro.complete:
            return None
        resolution = self.engine._effective_method_resolution(mro.owners, member.attr)
        if (
            len(resolution.callable_owners) != 1
            or resolution.may_be_missing
            or resolution.may_be_non_callable
            or resolution.has_unresolved_value
        ):
            return None
        owner = resolution.callable_owners[0]
        # An upstream owner preceding the selected method may change at old.
        if any(not item.startswith("vllm_ascend.") for item in mro.owners[: mro.owners.index(owner) + 1]):
            return None
        for item in mro.owners:
            source = self._type_source(item)
            if source is None or source.node.keywords:
                return None
            if "__init_subclass__" in _scope_final_bindings(source.node.body, set()):
                # Class creation can replace the apparent method before any
                # callsite executes; its lexical body is not a runtime proof.
                return None
            for decorator in source.node.decorator_list:
                expression = _expression_name(decorator.func if isinstance(decorator, ast.Call) else decorator)
                if expression is None or source.resolve(expression) != "dataclasses.dataclass":
                    return None
        reference = f"{owner}.{member.attr}"
        info = self.engine.downstream.find_callable(reference)
        bindings = self.engine.downstream.find_final_bindings(reference)
        if (
            info is None
            or not isinstance(info.node, ast.FunctionDef)
            or len(bindings) != 1
            or bindings[0].node is not info.node
            or info.node.args.vararg
            or info.node.args.kwarg
        ):
            return None
        descriptor = info.descriptor_kind
        if descriptor not in {"ordinary", "staticmethod", "classmethod"}:
            return None
        if descriptor == "ordinary" and (not instance or info.node.decorator_list):
            return None
        if descriptor != "ordinary" and (
            len(info.node.decorator_list) != 1
            or info.decorator_references not in {(descriptor,), (f"builtins.{descriptor}",)}
        ):
            return None
        signature = _inspect_signature(info.signature) if info.signature is not None else None
        if signature is None:
            return None
        proof = MethodCallProof.decode(
            [
                module.file.replace("\\", "/"),
                call.lineno,
                call.col_offset,
                call.end_lineno,
                call.end_col_offset,
                class_reference,
                reference,
                method_body_hash(info.node),
            ]
        )
        if proof is None or proof.call_in(module.tree) is not call:
            return None
        proofs = self._method_proofs | {proof}
        if len(proofs) > MAX_METHOD_CONTEXTS:
            return None
        # Keep every other class call, write and escape. The one excluded call
        # is accepted below only if its actual body has a complete pure return.
        effects: set[str] = set()
        context_effects: dict[tuple[int, str], frozenset[str]] = {}
        for effect_module in self.engine.downstream.modules.values():
            key = (id(self.engine.downstream), effect_module.name)
            module_proofs = [item for item in proofs if item.file == effect_module.file.replace("\\", "/")]
            if module_proofs:
                calls = [item.call_in(effect_module.tree) for item in module_proofs]
                if any(item is None for item in calls):
                    return None
                context_effects[key] = construction_effects(
                    effect_module.tree,
                    effect_module.name,
                    effect_module.is_package,
                    bound_receiver_calls=frozenset(id(item) for item in calls),
                )
            else:
                if key not in self._storage_module_effects:
                    self._storage_module_effects[key] = self._source_module_effects(
                        self.engine.downstream, effect_module
                    )
                context_effects[key] = self._storage_module_effects[key]
            effects.update(context_effects[key])
        if (
            any(has_construction_effect(item, frozenset(effects)) for item in mro.owners)
            or has_construction_effect(reference, frozenset(effects))
            or any(effect.startswith(reference + ".") for effect in effects)
        ):
            return None
        method_node = info.node
        method_module = self.engine.downstream.modules[info.module]
        # A separate resolver owns the prospective context and all its caches.
        # Nothing escapes it until the complete fixed method proof succeeds.
        contextual = _DirectDependencyResolver(self.engine)
        contextual._upstream_type_source = self._upstream_type_source
        contextual._reference_prefixes = self._reference_prefixes
        contextual._method_proofs = frozenset(proofs)
        contextual._storage_module_effects = context_effects
        contextual._storage_downstream_effects = frozenset(effects)

        def resolve_bound(node: ast.Call, values: tuple[FlowValue | None, ...]) -> HelperCallEffects | None:
            if any(isinstance(arg, ast.Starred) for arg in node.args) or any(kw.arg is None for kw in node.keywords):
                return None
            implicit: tuple[FlowValue | None, ...] = ()
            if descriptor == "ordinary":
                implicit = (receiver,)
            elif descriptor == "classmethod":
                implicit = (FlowValue(TypeShape("class", (TypeShape(class_reference),)), (class_reference,)),)
            try:
                bound = signature.bind(
                    *implicit,
                    *values[: len(node.args)],
                    **{kw.arg: values[len(node.args) + offset] for offset, kw in enumerate(node.keywords) if kw.arg},
                )
            except TypeError:
                return None
            contextual._factory_stack = set(self._factory_stack)
            result = contextual._factory_return(reference, method_node, method_module, dict(bound.arguments))
            if result is None:
                return None
            result = method_context_value(result, proof)
            if result is None:
                return None
            return HelperCallEffects(
                frozenset(range(len(values))),
                frozenset(),
                None,
                None,
                result if result.owned is None else None,
                result if result.owned is not None else None,
            )

        return resolve_bound

    def _helper_call_effects(
        self, call: ast.Call, values: tuple[FlowValue | None, ...], function: ast.AST, module: ModuleInfo
    ) -> HelperCallEffects | None:
        expression = _expression_name(call.func)
        if expression is None or not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        if expression.split(".", 1)[0] in self._local_names(function):
            return None
        parents = self._parent_maps[module.name]
        root = expression.split(".", 1)[0]
        if self._outer_function_shadows(call, root, parents, function) or (
            DownstreamConstructorDetector._expression_scope_shadows(call, root, parents)
        ):
            return None
        if any(isinstance(arg, ast.Starred) for arg in call.args) or any(kw.arg is None for kw in call.keywords):
            return None
        index = self.engine.downstream
        reference = self._annotation_namespace(index, module).runtime(expression)
        if reference is None or not reference.startswith("vllm_ascend."):
            return None
        info = index.find_callable(reference)
        bindings = index.find_final_bindings(reference)
        if (
            info is None
            or info.owner is not None
            or not isinstance(info.node, ast.FunctionDef)
            or info.node.decorator_list
            or info.node.args.vararg
            or info.node.args.kwarg
            or len(bindings) != 1
            or bindings[0].node is not info.node
        ):
            return None
        signature = _inspect_signature(info.signature) if info.signature is not None else None
        if signature is None:
            return None
        try:
            bound = signature.bind(
                *range(len(call.args)),
                **{kw.arg: len(call.args) + offset for offset, kw in enumerate(call.keywords) if kw.arg is not None},
            )
        except TypeError:
            return None
        positions = {name: position for name, position in bound.arguments.items() if isinstance(position, int)}
        shapes = {
            name: value.shape if value is not None else None
            for name, position in positions.items()
            for value in (values[position],)
        }
        exact_return = self._factory_return(
            reference,
            info.node,
            index.modules[info.module],
            {name: values[position] for name, position in positions.items()},
        )
        if exact_return is None and not any(
            value is not None and ("vllm." in value.shape.render() or any("vllm." in root for root in value.roots))
            for value in values
        ):
            return None
        key = (reference, tuple(sorted(shapes.items())))
        if key not in self._helper_summaries:
            helper_module = index.modules[info.module]
            namespace = self._annotation_namespace(index, helper_module)

            def resolve(name: str) -> str | None:
                if name in TYPE_BUILTINS and not index.find_final_bindings(f"{info.module}.{name}"):
                    return name
                return namespace.runtime(name)

            self._helper_summaries[key] = HelperEffects(info.node, shapes, resolve, self._type_source).summary
        summary = self._helper_summaries[key]
        unsafe_roots = frozenset(
            root
            for name, position in positions.items()
            if name in summary.unsafe
            for value in (values[position],)
            if value is not None
            for root in value.roots
        )
        if exact_return is not None and exact_return.roots & unsafe_roots:
            exact_return = None
        mutated_return = bool(summary.returned & summary.unsafe)
        return_path = None
        if not mutated_return and summary.return_path is not None and summary.return_path[0] in positions:
            value = values[positions[summary.return_path[0]]]
            if value is not None:
                return_path = (*value.path, *summary.return_path[1:])
        return HelperCallEffects(
            frozenset(position for name, position in positions.items() if name not in summary.unsafe),
            frozenset(position for name, position in positions.items() if name in summary.returned),
            None if mutated_return else summary.return_shape,
            return_path,
            exact_return if exact_return is not None and exact_return.owned is None else None,
            exact_return if exact_return is not None and exact_return.owned is not None else None,
        )

    def _factory_return(
        self,
        reference: str,
        function: ast.FunctionDef,
        module: ModuleInfo,
        parameters: dict[str, FlowValue | None],
    ) -> FlowValue | None:
        """Keep a replayable allocation only when all supported paths agree.

        New allocations discard their helper-local outer identity; returned
        inputs retain caller identities and alias origins. Closures, escapes and
        unknown runtime protocols cannot be justified by annotations alone.
        """
        if reference in self._factory_stack or len(self._factory_stack) >= MAX_FACTORY_RETURN_DEPTH:
            return None
        if any(value is None for value in parameters.values()):
            return None
        key = (reference, tuple(sorted(parameters.items())))
        if key in self._factory_returns:
            return self._factory_returns[key]
        self._factory_returns[key] = None
        if not factory_body_supported(function):
            return None
        self._factory_stack.add(reference)
        try:
            flow = self._function_flow(function, module, factory_parameters=parameters)
            if has_construction_effect(reference, self._storage_downstream_effects or frozenset()):
                # A source-visible function-object write/escape invalidates the
                # body proof even when its name still resolves to one FunctionDef.
                return None
            self._factory_returns[key] = proven_factory_return(flow, parameters)
            return self._factory_returns[key]
        finally:
            self._factory_stack.remove(reference)

    def _annotation_namespace(self, index: RepositoryIndex, module: ModuleInfo) -> AnnotationNamespace:
        key = (id(index), module.name)
        if key not in self._annotation_namespaces:
            self._annotation_namespaces[key] = self._facts.get(
                "annotation_namespace",
                (index, module.tree, module.name, module.is_package),
                lambda: AnnotationNamespace(module.tree, module.name, module.is_package, source_facts=self._facts),
            )
        return self._annotation_namespaces[key]

    def _context_source(self, source: ClassSource | None, reference: str) -> ClassSource | None:
        if source is None or not self._method_proofs:
            return source
        effects = self._storage_downstream_effects or frozenset()
        return replace(
            source,
            storage_effects=source.storage_effects
            or has_storage_effect(reference, effects)
            or has_storage_effect(source.qualified_name, effects),
        )

    def _type_source(self, reference: str) -> ClassSource | None:
        if reference.startswith("vllm.") and self._upstream_type_source is not None:
            return self._context_source(self._upstream_type_source(reference), reference)
        index = self.engine.upstream if reference.startswith("vllm.") else self.engine.downstream
        info = index.find_callable(reference)
        if info is None or not isinstance(info.node, ast.ClassDef):
            return None
        namespace = self._annotation_namespace(index, index.modules[info.module])

        def resolve(expression: str) -> str | None:
            if expression in TYPE_BUILTINS and not index.find_final_bindings(f"{info.module}.{expression}"):
                return expression
            target = namespace.resolve(expression)
            return index.canonical_name(target) if target is not None else None

        effect_key = (id(index), info.module)
        if effect_key not in self._storage_module_effects:
            module = index.modules[info.module]
            self._storage_module_effects[effect_key] = self._source_module_effects(index, module)
        effects = self._storage_module_effects[effect_key]
        return self._context_source(
            ClassSource(
                info.node,
                resolve,
                info.file,
                reference,
                has_storage_effect(reference, effects)
                or has_storage_effect(f"{info.module}.{info.node.name}", effects),
            ),
            reference,
        )

    def _storage_constructor(
        self,
        call: ast.Call,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        module: ModuleInfo,
    ) -> str | None:
        expression = _expression_name(call.func)
        if expression is None:
            return None
        parents = self._parent_maps[module.name]
        root = expression.split(".")[0]
        if self._outer_function_shadows(call, root, parents, function) or (
            DownstreamConstructorDetector._expression_scope_shadows(call, root, parents)
        ):
            return None
        reference = self._resolve_in_scope(
            call.func,
            function=function,
            module_info=module,
            line=call.lineno,
            reference_prefixes=("vllm.", "vllm_ascend."),
        )
        if reference is None:
            return None
        if reference not in self._storage_references:
            # Source layout is a cheap filter before collecting global downstream
            # class writes/escapes. The ordinary index and old snapshot remain
            # separate, so a new-only storage layout cannot stand in for old.
            if dataclass_storage(reference, self._type_source) is None:
                self._storage_references[reference] = False
            else:
                if self._storage_downstream_effects is None:
                    effects: set[str] = set()
                    for item in self.engine.downstream.modules.values():
                        key = (id(self.engine.downstream), item.name)
                        if key not in self._storage_module_effects:
                            self._storage_module_effects[key] = self._source_module_effects(
                                self.engine.downstream, item
                            )
                        effects.update(self._storage_module_effects[key])
                    self._storage_downstream_effects = frozenset(effects)
                downstream_effects = self._storage_downstream_effects

                def lookup(name: str) -> ClassSource | None:
                    if has_construction_effect(name, downstream_effects):
                        return None
                    return self._type_source(name)

                self._storage_references[reference] = dataclass_storage(reference, lookup) is not None
        return reference if self._storage_references[reference] else None

    def _replacement_callable(
        self, call: ast.Call, function: ast.FunctionDef | ast.AsyncFunctionDef, module: ModuleInfo
    ) -> bool:
        if module.name not in self._replacement_roots:
            self._replacement_roots[module.name] = DataclassReplaceDetector._replace_roots(
                module.tree, self._facts.nodes(module.tree)
            )
        expression = _expression_name(call.func)
        if expression is None or expression.split(".")[0] not in self._replacement_roots[module.name]:
            return False
        root = expression.split(".")[0]
        parents = self._parent_maps[module.name]
        if self._outer_function_shadows(call, root, parents, function) or (
            DownstreamConstructorDetector._expression_scope_shadows(call, root, parents)
        ):
            return False
        if (
            self._resolve_in_scope(
                call.func,
                function=function,
                module_info=module,
                line=call.lineno,
                reference_prefixes=("dataclasses.",),
            )
            != "dataclasses.replace"
        ):
            return False
        effects = self._storage_downstream_effects or frozenset()
        for module_effects in self._storage_module_effects.values():
            effects |= module_effects
        return not has_construction_effect("dataclasses.replace", effects)

    def _function_flow(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        module: ModuleInfo,
        *,
        capture_call_inputs: bool = False,
        factory_parameters: dict[str, FlowValue | None] | None = None,
    ) -> ContainerFlow:
        # Actual arguments and return capture must never reuse an
        # annotation-only flow cached for ordinary receiver discovery.
        key = (
            id(function),
            capture_call_inputs,
            tuple(sorted(factory_parameters.items())) if factory_parameters is not None else None,
        )
        mode = "factory" if factory_parameters is not None else "call_inputs" if capture_call_inputs else "receivers"
        metrics = self._flow_metrics.setdefault(mode, {"requests": 0, "builds": 0, "hits": 0, "build_seconds": 0})
        metrics["requests"] += 1
        if key not in self._container_flows:
            started = time.perf_counter()
            namespace = self._annotation_namespace(self.engine.downstream, module)

            def resolve(expression: str) -> str | None:
                if expression in TYPE_BUILTINS and not self.engine.downstream.find_final_bindings(
                    f"{module.name}.{expression}"
                ):
                    return expression
                return namespace.resolve(expression, function.lineno)

            def runtime(expression: str) -> str | None:
                if expression in TYPE_BUILTINS and not self.engine.downstream.find_final_bindings(
                    f"{module.name}.{expression}"
                ):
                    return expression
                return namespace.runtime(expression, function.lineno)

            instance_fields = {}
            if module.name not in self._parent_maps:
                self._parent_maps[module.name] = self._facts.parents(module.tree)
            parents = self._parent_maps[module.name]
            owner = self._class_name(function, parents, module.name)
            args = [*function.args.posonlyargs, *function.args.args]
            if (
                owner is not None
                and factory_parameters is None
                and args
                and not function.decorator_list
                and isinstance(parents.get(id(function)), ast.ClassDef)
            ):
                receiver = args[0].arg
                for child in self._facts.nodes(function):
                    if (
                        isinstance(child, ast.Attribute)
                        and isinstance(child.value, ast.Name)
                        and child.value.id == receiver
                    ):
                        origin_key = (owner, child.attr)
                        if origin_key not in self._stored_field_origins:
                            self._stored_field_origins[origin_key] = instance_field_origin(
                                owner, child.attr, self._type_source
                            )
                        origin = self._stored_field_origins[origin_key]
                        if origin is None:
                            continue
                        declaring_owner, shape, parameter, owned = origin
                        path = (
                            (declaring_owner, f"instance_field:{child.attr}")
                            if declaring_owner.startswith("vllm.")
                            else (shape.render(),)
                        )
                        name = f"{receiver}.{child.attr}"
                        instance_fields[name] = FlowValue(
                            shape,
                            ("unknown",) if owned else path,
                            frozenset() if owned else frozenset({f"{receiver}:{declaring_owner}.{parameter}"}),
                            f"owned:{receiver}:{declaring_owner}.{parameter}" if owned else None,
                        )
            self._container_flows[key] = ContainerFlow(
                function,
                resolve,
                self._type_source,
                runtime_resolve=runtime,
                helper_resolve=lambda call, values: self._helper_call_effects(call, values, function, module),
                method_bind=lambda call, receiver: self._bind_flow_method(call, receiver, function, module),
                constructor_resolve=lambda call: self._storage_constructor(call, function, module),
                replacement_resolve=lambda call: self._replacement_callable(call, function, module),
                capture_call_inputs=capture_call_inputs,
                capture_returns=factory_parameters is not None,
                parameters=factory_parameters,
                instance_fields=instance_fields,
                instance_owner=owner,
            )
            metrics["builds"] += 1
            metrics["build_seconds"] += time.perf_counter() - started
        else:
            metrics["hits"] += 1
        if factory_parameters is not None:
            return self._container_flows.pop(key)
        return self._container_flows[key]

    def _receiver_flow(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        module: ModuleInfo,
    ) -> ContainerFlow | _ReceiverFlowSummary:
        if self._receiver_flow_cache_key is None:
            return self._function_flow(function, module)
        local_key = id(function)
        if local_key in self._receiver_flow_summaries:
            return self._receiver_flow_summaries[local_key]

        def build() -> _ReceiverFlowSummary:
            flow = self._function_flow(function, module)
            return _ReceiverFlowSummary(
                MappingProxyType(dict(flow.receivers)),
                frozenset(flow.invalidated_receivers),
            )

        summary = self._facts.get(
            "receiver_flow_summary",
            (self._receiver_flow_cache_key, function),
            build,
        )
        self._receiver_flow_summaries[local_key] = summary
        return summary

    def _container_receiver(
        self,
        member: ast.Attribute,
        function: ast.FunctionDef | ast.AsyncFunctionDef | None,
        module: ModuleInfo,
    ) -> FlowValue | None:
        if function is None:
            return None
        value = self._receiver_flow(function, module).receivers.get(id(member))
        if (
            value is None
            or (len(value.path) < 2 and not value.constructed)
            or not (
                value.shape.reference.startswith("vllm.")
                or value.constructed
                and value.shape.reference.startswith("vllm_ascend.")
            )
        ):
            if self._historical_flow_resolver is not None:
                # Keep an independent old-source interpretation. A field removed
                # at new may otherwise erase types needed to discover later uses.
                # The returned path must still resolve independently at both SHAs
                # before range comparison can promote a break.
                return self._historical_flow_resolver._container_receiver(member, function, module)
            return None
        return value

    def _receiver_proof_invalidated(
        self,
        member: ast.Attribute,
        function: ast.FunctionDef | ast.AsyncFunctionDef | None,
        module: ModuleInfo,
    ) -> bool:
        """Veto nominal fallback only after flow explicitly lost a receiver."""
        if function is None:
            return False
        if id(member) in self._receiver_flow(function, module).invalidated_receivers:
            return True
        return (
            self._historical_flow_resolver is not None
            and self._historical_flow_resolver._receiver_proof_invalidated(member, function, module)
        )

    def _local_names(
        self,
        function: ast.AsyncFunctionDef | ast.FunctionDef | None,
    ) -> frozenset[str]:
        if function is None:
            return frozenset()
        key = id(function)
        if key not in self._function_locals:
            self._function_locals[key] = self._facts.get(
                "function_local_names", function, lambda: frozenset(_function_local_names(function))
            )
        return self._function_locals[key]

    def _tag_guards(self, statements: Sequence[ast.stmt]) -> set[str]:
        key = id(statements)
        if key not in self._scope_tag_guards:
            self._scope_tag_guards[key] = _tag_guard_names(statements)
        return self._scope_tag_guards[key]

    @staticmethod
    def _assignment_targets(node: ast.Assign | ast.AnnAssign) -> set[str]:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return {
            child.id
            for target in targets
            for child in ast.walk(target)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
        }

    def _scope_candidate_roots(
        self,
        function: ast.AsyncFunctionDef | ast.FunctionDef | None,
        module_info: ModuleInfo,
    ) -> frozenset[str]:
        cache_key = (module_info.name, id(function) if function is not None else 0)
        if cache_key in self._candidate_roots:
            return self._candidate_roots[cache_key]
        roots = {
            local
            for local, target in module_info.imports.items()
            if target == "vllm" or target.startswith(self._reference_prefixes)
        }
        if "vllm_ascend." in self._reference_prefixes:
            roots.update(node.name for node in module_info.tree.body if isinstance(node, ast.ClassDef))
        nodes = self._function_nodes(function) if function is not None else tuple(module_info.tree.body)
        assignments: list[tuple[set[str], str | None]] = []
        for node in nodes:
            if isinstance(node, ast.Import):
                roots.update(
                    alias.asname or alias.name.split(".", 1)[0]
                    for alias in node.names
                    if alias.name == "vllm" or alias.name.startswith(self._reference_prefixes)
                )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and (node.module == "vllm" or node.module.startswith(self._reference_prefixes))
            ):
                roots.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                expression_node = value.func if isinstance(value, ast.Call) else value
                assignments.append((self._assignment_targets(node), _expression_name(expression_node)))
                if isinstance(node, ast.AnnAssign):
                    annotation = _annotation_reference(node.annotation)
                    annotation_root = annotation.split(".", 1)[0] if annotation else None
                    assignments.append((self._assignment_targets(node), annotation_root))
        changed = True
        while changed:
            changed = False
            for targets, expression in assignments:
                if expression is None or expression.split(".", 1)[0] not in roots:
                    continue
                additions = targets - roots
                if additions:
                    roots.update(additions)
                    changed = True
        result = frozenset(roots)
        self._candidate_roots[cache_key] = result
        return result

    def _constructed_instance_bindings(
        self,
        function: ast.AsyncFunctionDef | ast.FunctionDef,
        module_info: ModuleInfo,
    ) -> dict[str, ast.AnnAssign | ast.Call | None]:
        cache_key = (module_info.name, id(function))
        if cache_key in self._constructed_instances:
            return self._constructed_instances[cache_key]
        candidates: dict[str, list[ast.AnnAssign | ast.Call | None]] = {}
        instance_candidates: set[str] = set()
        for node in self._function_nodes(function):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = self._assignment_targets(node)
            if isinstance(node, ast.AnnAssign) and node.value is not None and node in function.body:
                reference = _annotation_reference(node.annotation)
                if reference is not None:
                    for target in targets:
                        candidates.setdefault(target, []).append(node)
                        instance_candidates.add(target)
                    continue
            if isinstance(node.value, ast.Call):
                for target in targets:
                    candidates.setdefault(target, []).append(node.value)
                    instance_candidates.add(target)
            else:
                for target in targets:
                    candidates.setdefault(target, []).append(None)
        bindings = {
            name: values[0] if len(values) == 1 else None
            for name, values in candidates.items()
            if name in instance_candidates
        }
        self._constructed_instances[cache_key] = bindings
        return bindings

    @staticmethod
    def _no_fallback(_node: ast.AST) -> set[str | None]:
        return {None}

    def _scope_state_cache(self, statements: Sequence[ast.stmt]) -> dict[int, dict[str, tuple[_ScopeBinding, ...]]]:
        # Source bodies live for this resolver's lifetime. Cache only immutable
        # scope states, never fallback results that depend on caller shadowing.
        key = id(statements)
        if key not in self._scope_states:
            if len(self._scope_states) >= MAX_RESOLVER_SCOPE_CACHES:
                self._scope_states.pop(next(iter(self._scope_states)))
            self._scope_states[key] = {}
        return self._scope_states[key]

    def _module_reference(
        self,
        expression: ast.AST,
        *,
        module_info: ModuleInfo,
        line: int,
        reference_prefixes: tuple[str, ...] | None = None,
    ) -> str | None:
        """Resolve a name from module flow at one exact program point.

        ``ModuleInfo.imports`` is an index of discovered imports, not a proof
        that the imported binding still owns the name at this callsite.  Use
        the shared normal-path scope interpreter so assignment, ``del`` and
        conditional rebinding remain fail-closed.
        """

        variants = _scope_reference_variants(
            expression,
            statements=module_info.tree.body,
            line=line,
            tag_guard_names=self._tag_guards(module_info.tree.body),
            module=module_info.name,
            is_package=module_info.is_package,
            fallback=self._no_fallback,
            state_cache=self._scope_state_cache(module_info.tree.body),
            prefix_cache=self._prefix_cache,
        )
        concrete = {item for item in variants if item is not None}
        if len(variants) != 1 or len(concrete) != 1:
            return None
        result = next(iter(concrete))
        return result if result.startswith(reference_prefixes or self._reference_prefixes) else None

    def _module_fallback(
        self,
        module_info: ModuleInfo,
        blocked_names: set[str] | frozenset[str],
        reference_prefixes: tuple[str, ...] | None = None,
    ) -> Callable[[ast.AST], set[str | None]]:
        final_line = (
            max(
                (
                    getattr(statement, "end_lineno", getattr(statement, "lineno", 0))
                    for statement in module_info.tree.body
                ),
                default=0,
            )
            + 1
        )

        def resolve(node: ast.AST) -> set[str | None]:
            expression = _expression_name(node)
            if expression is None or expression.split(".", 1)[0] in blocked_names:
                return {None}
            resolved = self._module_reference(
                node,
                module_info=module_info,
                line=final_line,
                reference_prefixes=reference_prefixes,
            )
            return {resolved} if resolved is not None else {None}

        return resolve

    def _name_reassigned_before(
        self,
        function: ast.AsyncFunctionDef | ast.FunctionDef,
        name: str,
        point: ast.AST,
    ) -> bool:
        point_position = (
            getattr(point, "lineno", 0),
            getattr(point, "col_offset", 0),
        )

        def writes() -> dict[str, tuple[ast.AST, ...]]:
            result: dict[str, list[ast.AST]] = {}
            for candidate in self._function_nodes(function):
                names: set[str] = set()
                if isinstance(candidate, ast.Name) and isinstance(candidate.ctx, (ast.Del, ast.Store)):
                    names.add(candidate.id)
                elif isinstance(candidate, (ast.Import, ast.ImportFrom)):
                    names = (
                        {alias.asname or alias.name.split(".", 1)[0] for alias in candidate.names}
                        if isinstance(candidate, ast.Import)
                        else {alias.asname or alias.name for alias in candidate.names if alias.name != "*"}
                    )
                for bound in names:
                    result.setdefault(bound, []).append(candidate)
            return {bound: tuple(nodes) for bound, nodes in result.items()}

        index = self._facts.get("function_writes", function, writes)
        for candidate in index.get(name, ()):
            candidate_position = (
                getattr(candidate, "lineno", point_position[0]),
                getattr(candidate, "col_offset", 0),
            )
            if candidate_position < point_position:
                return True
        return False

    def _outer_function_shadows(
        self,
        node: ast.AST,
        root: str,
        parents: dict[int, ast.AST],
        nearest_function: ast.AsyncFunctionDef | ast.FunctionDef | None,
    ) -> bool:
        current = parents.get(id(nearest_function)) if nearest_function is not None else parents.get(id(node))
        while current is not None:
            if isinstance(current, (ast.AsyncFunctionDef, ast.FunctionDef)) and root in self._local_names(current):
                return True
            current = parents.get(id(current))
        return False

    def _class_name(self, node: ast.AST, parents: dict[int, ast.AST], module: str) -> str | None:
        classes: list[str] = []
        current = parents.get(id(node))
        while current is not None:
            if isinstance(current, ast.ClassDef):
                classes.append(current.name)
            current = parents.get(id(current))
        return f"{module}.{'.'.join(reversed(classes))}" if classes else None

    @staticmethod
    def _receiver_member_assignment(
        node: ast.AST,
        receiver: str,
        member: str,
    ) -> tuple[ast.AST | None, ast.AST | None] | None:
        """Return the value and annotation for one ``receiver.member`` write."""

        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            return None
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        if not any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == receiver
            and target.attr == member
            for target in targets
        ):
            return None
        annotation = node.annotation if isinstance(node, ast.AnnAssign) else None
        return node.value, annotation

    def _owner_instance_member_types(
        self,
        owner: str,
        member: str,
    ) -> frozenset[str] | None:
        """Infer a unique constructed type for an instance member.

        An empty set means the owner does not assign the member. ``None``
        means it does assign the member, but the assigned type is not exact.
        This deliberately accepts assignments outside ``__init__`` because
        guarded initialization helpers are common in vLLM. Call discovery
        still requires one unique resolved type across every such write.
        """

        cache_key = (owner, member)
        if cache_key in self._instance_member_types:
            return self._instance_member_types[cache_key]
        index = self.engine.upstream if owner.startswith("vllm.") else self.engine.downstream
        class_info = index.find_class(owner)
        if class_info is None:
            self._instance_member_types[cache_key] = frozenset()
            return frozenset()

        resolved_types: set[str] = set()
        found_assignment = False
        unresolved_assignment = False
        method_nodes = {
            id(node): node
            for variants in class_info.method_variants.values()
            for node in variants
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        }
        method_nodes.update(
            {
                id(node): node
                for node in class_info.methods.values()
                if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
            }
        )
        for method in method_nodes.values():
            positional = [*method.args.posonlyargs, *method.args.args]
            receiver = positional[0].arg if positional else None
            if receiver is None:
                continue
            for candidate in _function_scope_nodes(method):
                assignment = self._receiver_member_assignment(candidate, receiver, member)
                if assignment is None:
                    continue
                found_assignment = True
                value, annotation = assignment
                reference_node: ast.AST | None = None
                if isinstance(value, ast.Call):
                    reference_node = value.func
                elif annotation is not None:
                    reference = _annotation_reference(annotation)
                    if reference is not None:
                        try:
                            reference_node = ast.parse(reference, mode="eval").body
                        except SyntaxError:
                            reference_node = None
                reference = _expression_name(reference_node) if reference_node is not None else None
                if reference is None:
                    unresolved_assignment = True
                    continue
                resolved = index.canonical_name(index.resolve_reference(class_info.module, reference))
                if not resolved.startswith(("vllm.", "vllm_ascend.")):
                    unresolved_assignment = True
                    continue
                resolved_types.add(resolved)

        result: frozenset[str] | None
        if not found_assignment:
            result = frozenset()
        elif unresolved_assignment or len(resolved_types) != 1:
            result = None
        else:
            result = frozenset(resolved_types)
        self._instance_member_types[cache_key] = result
        return result

    def _self_member_receiver_target(
        self,
        node: ast.Call,
        parents: dict[int, ast.AST],
        module: str,
    ) -> tuple[str, str, str, str, str | None, str] | None:
        """Resolve ``self.<field>.<method>()`` through a proven field type."""

        member_access = _member_access(node)
        if member_access is None or not isinstance(member_access.value, ast.Attribute):
            return None
        field_access = member_access.value
        function = _nearest(node, parents, (ast.FunctionDef, ast.AsyncFunctionDef))
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        positional = [*function.args.posonlyargs, *function.args.args]
        receiver = positional[0].arg if positional else None
        if not (
            receiver
            and isinstance(field_access.value, ast.Name)
            and field_access.value.id == receiver
            and not self._name_reassigned_before(function, receiver, node)
        ):
            return None
        class_name = self._class_name(node, parents, module)
        if class_name is None:
            return None
        owner_mro = self.engine._linearized_mro(class_name)
        if not owner_mro.complete:
            return None

        field_type: str | None = None
        for owner in owner_mro.owners:
            candidates = self._owner_instance_member_types(owner, field_access.attr)
            if candidates is None:
                return None
            if candidates:
                field_type = next(iter(candidates))
                break
        if field_type is None:
            return None

        field_mro = self.engine._linearized_mro(field_type)
        if not field_mro.complete:
            return None
        resolution = self.engine._effective_method_resolution(field_mro.owners, member_access.attr)
        if (
            len(resolution.callable_owners) == 1
            and not resolution.may_be_missing
            and not resolution.may_be_non_callable
            and not resolution.has_unresolved_value
        ):
            target = f"{resolution.callable_owners[0]}.{member_access.attr}"
            return (
                (
                    target,
                    "instance",
                    field_type,
                    member_access.attr,
                    None,
                    "new_exact",
                )
                if target.startswith("vllm.")
                else None
            )

        definitely_missing = (
            not resolution.callable_owners
            and resolution.may_be_missing
            and not resolution.may_be_non_callable
            and not resolution.has_unresolved_value
            and not resolution.blocking_owners
            and not hasattr(object, member_access.attr)
        )
        lookup_root = next((owner for owner in field_mro.owners if owner.startswith("vllm.")), None)
        if not definitely_missing or lookup_root is None:
            return None
        return (
            f"{lookup_root}.{member_access.attr}",
            "instance",
            field_type,
            member_access.attr,
            lookup_root,
            "old_fallback_instance_field",
        )

    def _self_or_super_target(
        self,
        node: ast.Call | ast.Attribute,
        parents: dict[int, ast.AST],
        module: str,
    ) -> tuple[str, str, str, str, str | None, str] | None:
        member_access = _member_access(node)
        if member_access is None:
            return None
        is_super = (
            isinstance(member_access.value, ast.Call)
            and isinstance(member_access.value.func, ast.Name)
            and member_access.value.func.id == "super"
            and not member_access.value.args
            and not member_access.value.keywords
        )
        function = _nearest(node, parents, (ast.FunctionDef, ast.AsyncFunctionDef))
        receiver = None
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            positional = [*function.args.posonlyargs, *function.args.args]
            receiver = positional[0].arg if positional else None
        is_receiver = isinstance(member_access.value, ast.Name) and member_access.value.id == receiver
        if not is_super and not is_receiver:
            return None
        if (
            is_receiver
            and isinstance(function, (ast.AsyncFunctionDef, ast.FunctionDef))
            and receiver is not None
            and self._name_reassigned_before(function, receiver, node)
        ):
            return None
        class_name = self._class_name(node, parents, module)
        if class_name is None:
            return None
        mro = self.engine._linearized_mro(class_name)
        if not mro.complete:
            return None
        owners = mro.owners[1:] if is_super else mro.owners
        resolution = self.engine._effective_method_resolution(owners, member_access.attr)
        if (
            len(resolution.callable_owners) == 1
            and not resolution.may_be_missing
            and not resolution.may_be_non_callable
            and not resolution.has_unresolved_value
        ):
            target = f"{resolution.callable_owners[0]}.{member_access.attr}"
            return (
                (
                    target,
                    "instance",
                    class_name,
                    member_access.attr,
                    None,
                    "new_exact",
                )
                if target.startswith("vllm.")
                else None
            )
        definitely_missing = (
            not resolution.callable_owners
            and resolution.may_be_missing
            and not resolution.may_be_non_callable
            and not resolution.has_unresolved_value
            and not resolution.blocking_owners
            and not hasattr(object, member_access.attr)
        )
        if not definitely_missing:
            return None
        lookup_root = next(
            (owner for owner in owners if owner.startswith("vllm.")),
            None,
        )
        if lookup_root is None:
            return None
        return (
            f"{lookup_root}.{member_access.attr}",
            "instance",
            class_name,
            member_access.attr,
            lookup_root,
            "old_fallback_super" if is_super else "old_fallback_self",
        )

    def _annotated_instance_target(
        self,
        node: ast.Call | ast.Attribute,
        function: ast.AsyncFunctionDef | ast.FunctionDef | None,
        module_info: ModuleInfo,
    ) -> tuple[str, str, str, str, str | None, str] | None:
        member_access = _member_access(node)
        if function is None or member_access is None or not isinstance(member_access.value, ast.Name):
            return None
        root = member_access.value.id
        arguments = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
        argument = next((item for item in arguments if item.arg == root), None)
        reference = _annotation_reference(argument.annotation) if argument is not None else None
        if reference is None or self._name_reassigned_before(function, root, node):
            return None
        try:
            annotation_expression = ast.parse(reference, mode="eval").body
        except SyntaxError:
            return None
        resolved = self._module_reference(
            annotation_expression,
            module_info=module_info,
            line=getattr(function, "lineno", 0),
        )
        if resolved is None:
            return None
        target = f"{resolved}.{member_access.attr}"
        return (target, "instance", resolved, member_access.attr, None, "parameter_annotation")

    def _receiver_binding_evidence(
        self,
        node: ast.Call | ast.Attribute,
        function: ast.FunctionDef | ast.AsyncFunctionDef | None,
        module: ModuleInfo,
        receiver_type: str | None,
        resolution_basis: str,
        receiver_path: tuple[str, ...] = (),
    ) -> dict[str, Any] | None:
        member = _member_access(node)
        if (
            resolution_basis not in {"parameter_annotation", "typed_container_flow"}
            or receiver_type is None
            or function is None
            or member is None
        ):
            return None
        parameter = (
            member.value.id
            if resolution_basis == "parameter_annotation" and isinstance(member.value, ast.Name)
            else None
        )
        if parameter is not None and self._receiver_proof_invalidated(member, function, module):
            return {
                "kind": "invalidated_parameter_receiver",
                "declared_type": receiver_type,
                "parameter": parameter,
                "receiver_expression": ast.unparse(member.value),
                "file": module.file,
                "line": member.lineno,
                "source": "ordered_container_flow",
                "reason": "preceding effects invalidated receiver provenance; the annotation is not an exact binding",
            }
        evidence = self._receiver_constraints.evidence(receiver_type, member.attr, module, function, parameter)
        if evidence is not None and resolution_basis == "typed_container_flow":
            # A path through annotated container elements/fields is still a
            # declared type. Do not invent a runtime subclass or pass a local
            # alias as if it were the helper's formal parameter.
            evidence = dict(
                evidence,
                kind="container_annotation_with_local_member_alternatives",
                receiver_path=receiver_path,
                receiver_expression=ast.unparse(member.value),
            )
        return evidence

    def _resolve_in_scope(
        self,
        expression: ast.AST,
        *,
        function: ast.AsyncFunctionDef | ast.FunctionDef | None,
        module_info: ModuleInfo,
        line: int,
        reference_prefixes: tuple[str, ...] | None = None,
    ) -> str | None:
        expression_name = _expression_name(expression)
        if expression_name is None:
            return None
        local_names = self._local_names(function)
        if function is None:
            return self._module_reference(
                expression,
                module_info=module_info,
                line=line,
                reference_prefixes=reference_prefixes,
            )
        statements: Sequence[ast.stmt] = function.body
        variants = _scope_reference_variants(
            expression,
            statements=statements,
            line=line,
            tag_guard_names=self._tag_guards(statements),
            module=module_info.name,
            is_package=module_info.is_package,
            fallback=self._module_fallback(module_info, local_names, reference_prefixes),
            state_cache=self._scope_state_cache(statements),
            prefix_cache=self._prefix_cache,
        )
        concrete = {item for item in variants if item is not None}
        if len(variants) != 1 or len(concrete) != 1:
            return None
        result = next(iter(concrete))
        return result if result.startswith(reference_prefixes or self._reference_prefixes) else None

    def _constructed_instance_target(
        self,
        node: ast.Call | ast.Attribute,
        function: ast.AsyncFunctionDef | ast.FunctionDef | None,
        module_info: ModuleInfo,
    ) -> tuple[str, str, str, str, str | None, str] | None:
        member_access = _member_access(node)
        if member_access is None:
            return None
        annotation_reference: str | None = None
        if isinstance(member_access.value, ast.Call):
            reference = self._resolve_in_scope(
                member_access.value.func,
                function=function,
                module_info=module_info,
                line=getattr(member_access.value, "lineno", getattr(node, "lineno", 0)),
            )
            if reference is None:
                return None
            # This remains a candidate receiver path, not proof that the
            # symbol is a class.  Old/new endpoint resolution must each prove
            # the class and inherited member independently.
            return (f"{reference}.{member_access.attr}", "instance", reference, member_access.attr, None, "new_exact")
        if function is not None and isinstance(member_access.value, ast.Name):
            root = member_access.value.id
            binding = self._constructed_instance_bindings(function, module_info).get(root)
            if isinstance(binding, ast.AnnAssign):
                binding_end = (
                    getattr(binding, "end_lineno", getattr(binding, "lineno", 0)),
                    getattr(binding, "end_col_offset", getattr(binding, "col_offset", 0)),
                )
                call_start = (
                    getattr(node, "lineno", 0),
                    getattr(node, "col_offset", 0),
                )
                if binding_end > call_start:
                    return None
                annotation_reference = _annotation_reference(binding.annotation)
            elif isinstance(binding, ast.Call):
                binding_end = (
                    getattr(binding, "end_lineno", getattr(binding, "lineno", 0)),
                    getattr(binding, "end_col_offset", getattr(binding, "col_offset", 0)),
                )
                call_start = (
                    getattr(node, "lineno", 0),
                    getattr(node, "col_offset", 0),
                )
                if binding_end > call_start:
                    return None
                reference = self._resolve_in_scope(
                    member_access.value,
                    function=function,
                    module_info=module_info,
                    line=getattr(node, "lineno", 0),
                )
                if reference is None and binding_end[0] == call_start[0]:
                    reference = self._resolve_in_scope(
                        binding.func,
                        function=function,
                        module_info=module_info,
                        line=getattr(binding, "lineno", getattr(node, "lineno", 0)),
                    )
                if reference is None:
                    return None
                return (
                    f"{reference}.{member_access.attr}",
                    "instance",
                    reference,
                    member_access.attr,
                    None,
                    "new_exact",
                )
        if annotation_reference is None:
            return None
        try:
            annotation_expression = ast.parse(annotation_reference, mode="eval").body
        except SyntaxError:
            return None
        reference = self._module_reference(
            annotation_expression,
            module_info=module_info,
            line=getattr(function, "lineno", 0),
        )
        if reference is None:
            return None
        target = f"{reference}.{member_access.attr}"
        return (target, "instance", reference, member_access.attr, None, "new_exact")

    def _resolved_access_kind(
        self,
        node: ast.Call | ast.Attribute,
        target: str,
        function: ast.AsyncFunctionDef | ast.FunctionDef | None,
        module_info: ModuleInfo,
    ) -> str | None:
        member_access = _member_access(node)
        if function is None or member_access is None or not isinstance(member_access.value, ast.Name):
            return self._access_kind(node, target)
        root = member_access.value.id
        bindings = self._constructed_instance_bindings(function, module_info)
        if root not in bindings:
            return self._access_kind(node, target)
        binding = bindings[root]
        if isinstance(binding, ast.AnnAssign):
            return "instance"
        return None

    @staticmethod
    def _access_kind(_node: ast.Call | ast.Attribute, _target: str) -> str:
        # A bare imported symbol can be a function or a class, and an
        # attribute can belong to either a module or a class.  Preserve that
        # syntax-neutral fact; each old/new snapshot derives its own binding.
        return "direct"

    def _triton_launch_target(self, target: str) -> bool:
        """Prove that one subscript launch resolves to a plain Triton JIT kernel."""

        callable_info = self.engine.upstream.find_callable(target)
        return callable_info is not None and callable_info.decorator_references == (_TRITON_JIT_DECORATOR,)


class DirectCallDetector(_DirectDependencyResolver):
    """Discover exact downstream-to-upstream calls without changing golden relations."""

    def discover(self) -> list[DirectCallDependency]:
        """Discover exact downstream calls and constrained return uses."""
        dependencies: list[DirectCallDependency] = []
        self.historical_candidates = []
        for module_info in self.engine.downstream.modules.values():
            tree = module_info.tree
            parents = self._facts.parents(tree)
            for node in self._facts.nodes(tree):
                if not isinstance(node, ast.Call) or _under_version_guard(node, parents):
                    continue
                invocation_kind = "python_call"
                callable_node = node.func
                if isinstance(node.func, ast.Subscript):
                    invocation_kind = _TRITON_KERNEL_PROTOCOL
                    callable_node = node.func.value
                function_node = _nearest(node, parents, (ast.FunctionDef, ast.AsyncFunctionDef))
                function = function_node if isinstance(function_node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
                special: tuple[str, str, str, str, str | None, str] | None = None
                if invocation_kind == "python_call":
                    special = self._self_or_super_target(node, parents, module_info.name)
                    special = special or self._self_member_receiver_target(node, parents, module_info.name)
                    special = special or self._annotated_instance_target(node, function, module_info)
                expression = _expression_name(callable_node)
                root = expression.split(".", 1)[0] if expression is not None else None
                candidate_roots = self._scope_candidate_roots(function, module_info)
                if root is not None and self._outer_function_shadows(
                    node,
                    root,
                    parents,
                    function,
                ):
                    continue
                may_be_constructed = invocation_kind == "python_call" and (
                    (
                        isinstance(callable_node, ast.Attribute)
                        and isinstance(callable_node.value, ast.Call)
                        and (_expression_name(callable_node.value.func) or "").split(".", 1)[0] in candidate_roots
                    )
                    or root in candidate_roots
                )
                if special is None and may_be_constructed:
                    special = self._constructed_instance_target(node, function, module_info)
                flow = None
                if invocation_kind == "python_call" and isinstance(callable_node, ast.Attribute):
                    flow = self._container_receiver(callable_node, function, module_info)
                    if (
                        flow is None
                        and self._receiver_proof_invalidated(callable_node, function, module_info)
                        and (special is None or special[5] != "parameter_annotation")
                    ):
                        continue
                    if flow is not None:
                        receiver = flow.shape.reference
                        special = (
                            f"{receiver}.{callable_node.attr}",
                            "instance",
                            receiver,
                            callable_node.attr,
                            None,
                            "constructed_storage" if flow.constructed else "typed_container_flow",
                        )
                receiver_type: str | None
                member: str | None
                lookup_root: str | None
                access_kind: str
                if special is not None:
                    targets = {special[0]}
                    access_kind = special[1]
                    receiver_type = special[2]
                    member = special[3]
                    lookup_root = special[4]
                    resolution_basis = special[5]
                else:
                    if root not in candidate_roots:
                        continue
                    resolved = self._resolve_in_scope(
                        callable_node,
                        function=function,
                        module_info=module_info,
                        line=getattr(node, "lineno", 0),
                    )
                    if resolved is None or not resolved.startswith("vllm."):
                        continue
                    if invocation_kind == _TRITON_KERNEL_PROTOCOL and not self._triton_launch_target(resolved):
                        continue
                    targets = {resolved}
                    resolved_access_kind = self._resolved_access_kind(
                        node,
                        resolved,
                        function,
                        module_info,
                    )
                    if resolved_access_kind is None:
                        continue
                    access_kind = resolved_access_kind
                    receiver_type = None
                    member = callable_node.attr if isinstance(callable_node, ast.Attribute) else None
                    lookup_root = None
                    resolution_basis = "new_exact"
                target = next(iter(targets))
                if not (
                    target.startswith("vllm.")
                    or flow is not None
                    and flow.constructed
                    and target.startswith("vllm_ascend.")
                ):
                    continue
                scope_node = function or tree
                owner = self._class_name(node, parents, module_info.name)
                dependency = DirectCallDependency(
                    target=target,
                    access_kind=access_kind,
                    file=module_info.file,
                    line=getattr(node, "lineno", 0),
                    column=getattr(node, "col_offset", 0),
                    owner=owner.rsplit(".", 1)[-1] if owner else None,
                    scope=function.name if function is not None else None,
                    callee=ast.unparse(node.func),
                    call_shape=call_shape(node),
                    return_use=infer_return_use(node, parents, scope_node),
                    receiver_type=receiver_type,
                    member=member,
                    invocation_kind=invocation_kind,
                    lookup_root=lookup_root,
                    resolution_basis=resolution_basis,
                    receiver_path=flow.path if flow is not None else (),
                    receiver_binding=self._receiver_binding_evidence(
                        node,
                        function,
                        module_info,
                        receiver_type,
                        resolution_basis,
                        flow.path if flow is not None else (),
                    ),
                )
                if lookup_root is not None:
                    self.historical_candidates.append(dependency)
                else:
                    dependencies.append(dependency)
        dependencies.extend(DownstreamConstructorDetector(self.engine).discover())
        historical_source = (
            self._historical_flow_resolver._upstream_type_source if self._historical_flow_resolver is not None else None
        )
        replacements = DataclassReplaceDetector(
            self.engine,
            historical_type_source=historical_source,
            factory_evaluator=self._factory_evaluator,
            factory_input_resolver=self._factory_input_resolver,
            historical_factory_input_resolver=(
                self._historical_flow_resolver._factory_input_resolver
                if self._historical_flow_resolver is not None
                else None
            ),
            historical_factory_evaluator=(
                self._historical_flow_resolver._factory_evaluator
                if self._historical_flow_resolver is not None
                else None
            ),
        )
        dependencies.extend(replacements.discover())
        unique = {
            (
                item.file,
                item.line,
                item.column,
                item.target,
                item.callee,
                item.access_kind,
                json.dumps(item.call_shape.as_dict(), sort_keys=True, separators=(",", ":")),
                json.dumps(item.return_use.as_dict(), sort_keys=True, separators=(",", ":")),
                item.receiver_type,
                item.member,
                item.invocation_kind,
            ): item
            for item in dependencies
        }
        return [unique[key] for key in sorted(unique)]


class DataclassReplaceDetector(_DirectDependencyResolver):
    """Trace stdlib replacement to a proven concrete dataclass instance."""

    _reference_prefixes = ("vllm.", "vllm_ascend.", "dataclasses.")

    @staticmethod
    def _replace_roots(tree: ast.Module, nodes: Iterable[ast.AST] | None = None) -> set[str]:
        roots: set[str] = set()
        assignments = []
        for node in ast.walk(tree) if nodes is None else nodes:
            if isinstance(node, ast.Import):
                roots.update(alias.asname or alias.name for alias in node.names if alias.name == "dataclasses")
            elif isinstance(node, ast.ImportFrom) and node.module == "dataclasses":
                roots.update(alias.asname or alias.name for alias in node.names if alias.name == "replace")
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                assignments.append(node)
        changed = True
        while changed:
            changed = False
            for assignment in assignments:
                name = _expression_name(assignment.value)
                if name is not None and name.split(".")[0] in roots:
                    additions = _DirectDependencyResolver._assignment_targets(assignment) - roots
                    roots.update(additions)
                    changed = changed or bool(additions)
        return roots

    def discover(self) -> list[DirectCallDependency]:
        dependencies: list[DirectCallDependency] = []
        if self._historical_flow_resolver is not None:
            self._historical_flow_resolver._reference_prefixes = self._reference_prefixes
        for module in self.engine.downstream.modules.values():
            roots = self._replace_roots(module.tree, self._facts.nodes(module.tree))
            if not roots:
                continue
            parents = self._facts.parents(module.tree)
            for node in self._facts.nodes(module.tree):
                if not isinstance(node, ast.Call) or len(node.args) != 1 or _under_version_guard(node, parents):
                    continue
                function = _nearest(node, parents, (ast.FunctionDef, ast.AsyncFunctionDef))
                if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                expression = _expression_name(node.func)
                if expression is None:
                    continue
                root = expression.split(".")[0]
                if root not in roots:
                    continue
                if self._outer_function_shadows(node, root, parents, function) or (
                    DownstreamConstructorDetector._expression_scope_shadows(node, root, parents)
                ):
                    continue
                if self._resolve_in_scope(node.func, function=function, module_info=module, line=node.lineno) != (
                    "dataclasses.replace"
                ):
                    continue
                values = self._function_flow(function, module, capture_call_inputs=True).call_inputs.get(id(node), ())
                value = values[0] if values else None
                if (value is None or not value.constructed) and self._historical_flow_resolver is not None:
                    values = self._historical_flow_resolver._function_flow(
                        function, module, capture_call_inputs=True
                    ).call_inputs.get(id(node), ())
                    value = values[0] if values else None
                if (
                    value is None
                    or not value.constructed
                    or not value.shape.reference.startswith(("vllm.", "vllm_ascend."))
                ):
                    continue
                effects = self._storage_downstream_effects or frozenset()
                for module_effects in self._storage_module_effects.values():
                    effects |= module_effects
                if self._historical_flow_resolver is not None:
                    effects |= self._historical_flow_resolver._storage_downstream_effects or frozenset()
                if has_construction_effect("dataclasses.replace", effects):
                    continue
                target = value.shape.reference
                if target.startswith("vllm_ascend.") and not any(
                    owner.startswith("vllm.") for owner in self.engine._linearized_mro(target).owners
                ):
                    continue
                owner = self._class_name(node, parents, module.name)
                # The object argument chooses the runtime class; remaining
                # keywords bind to that class's replacement field protocol.
                shape = call_shape(ast.Call(func=node.func, args=[], keywords=node.keywords))
                dependencies.append(
                    DirectCallDependency(
                        target=target,
                        access_kind="constructor",
                        file=module.file,
                        line=node.lineno,
                        column=node.col_offset,
                        owner=owner.rsplit(".", 1)[-1] if owner else None,
                        scope=function.name,
                        callee=ast.unparse(node.func),
                        call_shape=shape,
                        return_use=infer_return_use(node, parents, function),
                        resolution_basis="dataclass_replace",
                        receiver_path=value.path,
                    )
                )
        return dependencies


class DownstreamConstructorDetector(_DirectDependencyResolver):
    """Find concrete local class calls whose inheritance includes upstream code."""

    _reference_prefixes = ("vllm.", "vllm_ascend.")

    @staticmethod
    def _expression_scope_shadows(node: ast.AST, root: str, parents: dict[int, ast.AST]) -> bool:
        current = parents.get(id(node))
        while current is not None:
            if isinstance(current, ast.Lambda) and any(child is node for child in ast.walk(current.body)):
                arguments = [*current.args.posonlyargs, *current.args.args, *current.args.kwonlyargs]
                arguments.extend(arg for arg in (current.args.vararg, current.args.kwarg) if arg is not None)
                if any(arg.arg == root for arg in arguments):
                    return True
            if isinstance(current, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                # Python evaluates the outermost iterable in the enclosing scope.
                in_outer_iterable = bool(current.generators) and any(
                    child is node for child in ast.walk(current.generators[0].iter)
                )
                if not in_outer_iterable and any(
                    isinstance(child, ast.Name) and child.id == root
                    for generator in current.generators
                    for child in ast.walk(generator.target)
                ):
                    return True
            current = parents.get(id(current))
        return False

    def _mutations(self) -> dict[str, list[dict[str, Any]]]:
        """Record class-call protocol writes using the shared scope resolver."""
        mutations: dict[str, list[dict[str, Any]]] = {}
        members = {"__init__", "__new__", "__bases__", "__call__", "__dataclass_fields__"}
        for module in self.engine.downstream.modules.values():
            parents = self._facts.parents(module.tree)
            for node in self._facts.nodes(module.tree):
                owner_node: ast.AST | None = None
                replacement: ast.AST | None = None
                member: str | None = None
                statement = parents.get(id(node))
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.ctx, (ast.Store, ast.Del))
                    and node.attr in members
                ):
                    owner_node, member = node.value, node.attr
                    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                        replacement = statement.value
                elif isinstance(node, ast.Call) and _expression_name(node.func) in {"setattr", "delattr"}:
                    if (
                        len(node.args) >= 2
                        and isinstance(node.args[1], ast.Constant)
                        and isinstance(node.args[1].value, str)
                        and node.args[1].value in members
                    ):
                        owner_node, member = node.args[0], node.args[1].value
                        replacement = node.args[2] if len(node.args) == 3 and not node.keywords else None
                if (
                    not isinstance(node, (ast.Attribute, ast.Call))
                    or owner_node is None
                    or member is None
                    or _under_version_guard(node, parents)
                ):
                    continue
                owner_name = _expression_name(owner_node)
                if owner_name is None or self._expression_scope_shadows(node, owner_name.split(".", 1)[0], parents):
                    continue
                function_node = _nearest(node, parents, (ast.FunctionDef, ast.AsyncFunctionDef))
                function = function_node if isinstance(function_node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
                target = self._resolve_in_scope(owner_node, function=function, module_info=module, line=node.lineno)
                if target is None:
                    continue
                target = self.engine.downstream.canonical_name(target)
                replacement_ref = (
                    self._resolve_in_scope(replacement, function=function, module_info=module, line=node.lineno)
                    if replacement is not None
                    else None
                )
                direct_module_statement = statement is not None and parents.get(id(statement)) is module.tree
                stable_replacement = replacement_ref is not None
                if replacement_ref is not None and replacement_ref.startswith("vllm_ascend."):
                    bindings = self.engine.downstream.find_final_bindings(replacement_ref)
                    info = self.engine.downstream.find_callable(replacement_ref)
                    stable_replacement = (
                        len(bindings) == 1
                        and info is not None
                        and (info.file != module.file or bindings[0].line < node.lineno)
                    )
                if isinstance(node, ast.Call):
                    # A shadowed setter is an unknown consumer, not the builtin.
                    setter = _expression_name(node.func)
                    direct_module_statement = direct_module_statement and (
                        setter not in self._local_names(function)
                        and self._resolve_in_scope(node.func, function=function, module_info=module, line=node.lineno)
                        is None
                    )
                mutations.setdefault(target, []).append(
                    {
                        "target": target,
                        "member": member,
                        "file": module.file,
                        "line": node.lineno,
                        "scope": function.name if function is not None else None,
                        "expression": ast.unparse(statement or node),
                        "replacement": replacement_ref,
                        "exact_module_assignment": bool(direct_module_statement and stable_replacement),
                    }
                )
        return mutations

    def discover(self) -> list[DirectCallDependency]:
        dependencies: list[DirectCallDependency] = []
        eligible: dict[str, bool] = {}
        mutations = self._mutations()
        for module in self.engine.downstream.modules.values():
            parents = self._facts.parents(module.tree)
            roots = set(module.imports)
            # This is only a prefilter. Lazy imports still need the shared
            # statement-order resolver below to prove their actual binding.
            for imported in self._facts.nodes(module.tree):
                if isinstance(imported, (ast.Import, ast.ImportFrom)):
                    roots.update(alias.asname or alias.name.split(".", 1)[0] for alias in imported.names)
            roots.update(node.name for node in self._facts.nodes(module.tree) if isinstance(node, ast.ClassDef))
            assignments = [
                node for node in self._facts.nodes(module.tree) if isinstance(node, (ast.Assign, ast.AnnAssign))
            ]
            changed = True
            while changed:
                changed = False
                for assignment in assignments:
                    reference = _expression_name(assignment.value)
                    if reference is not None and reference.split(".", 1)[0] in roots:
                        additions = self._assignment_targets(assignment) - roots
                        roots.update(additions)
                        changed = changed or bool(additions)
            for node in self._facts.nodes(module.tree):
                if not isinstance(node, ast.Call) or _under_version_guard(node, parents):
                    continue
                expression = _expression_name(node.func)
                if expression is None or expression.split(".", 1)[0] not in roots:
                    continue
                if self._expression_scope_shadows(node, expression.split(".", 1)[0], parents):
                    continue
                function_node = _nearest(node, parents, (ast.FunctionDef, ast.AsyncFunctionDef))
                function = function_node if isinstance(function_node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
                if self._outer_function_shadows(node, expression.split(".", 1)[0], parents, function):
                    continue
                resolved = self._resolve_in_scope(node.func, function=function, module_info=module, line=node.lineno)
                if resolved is None or not resolved.startswith("vllm_ascend."):
                    continue
                target = self.engine.downstream.canonical_name(resolved)
                if target not in eligible:
                    info = self.engine.downstream.find_callable(target)
                    bindings = self.engine.downstream.find_final_bindings(target)
                    eligible[target] = (
                        info is not None
                        and isinstance(info.node, ast.ClassDef)
                        and len(bindings) == 1
                        and bindings[0].node is info.node
                        and any(owner.startswith("vllm.") for owner in self.engine._linearized_mro(target).owners)
                    )
                if not eligible[target]:
                    continue
                owner = self._class_name(node, parents, module.name)
                relevant_mutations = tuple(
                    mutation
                    for class_owner in self.engine._linearized_mro(target).owners
                    for mutation in mutations.get(class_owner, ())
                    if not (
                        function is None
                        and mutation["scope"] is None
                        and mutation["file"] == module.file
                        and mutation["line"] > node.lineno
                    )
                )
                if function is not None and relevant_mutations:
                    invocations = []
                    for statement in module.tree.body:
                        value = (
                            statement.value if isinstance(statement, (ast.Expr, ast.Assign, ast.AnnAssign)) else None
                        )
                        if not isinstance(value, ast.Call):
                            continue
                        reference = self._resolve_in_scope(
                            value.func, function=None, module_info=module, line=value.lineno
                        )
                        called = self.engine.downstream.find_callable(reference) if reference is not None else None
                        if called is not None and called.node is function:
                            invocations.append(value.lineno)
                    # A module-final initializer cannot describe both the
                    # helper's early invocation and its possible later calls.
                    # Preserve this temporal ambiguity instead of silently
                    # treating the later initializer as retroactive.
                    relevant_mutations = tuple(
                        dict(
                            mutation,
                            exact_module_assignment=False,
                            earlier_module_invocations=[line for line in invocations if line < mutation["line"]],
                        )
                        if mutation["file"] == module.file
                        and mutation["scope"] is None
                        and any(line < mutation["line"] for line in invocations)
                        else mutation
                        for mutation in relevant_mutations
                    )
                dependencies.append(
                    DirectCallDependency(
                        target=target,
                        access_kind="constructor",
                        file=module.file,
                        line=node.lineno,
                        column=node.col_offset,
                        owner=owner.rsplit(".", 1)[-1] if owner else None,
                        scope=function.name if function is not None else None,
                        callee=ast.unparse(node.func),
                        call_shape=call_shape(node),
                        return_use=infer_return_use(node, parents, function or module.tree),
                        resolution_basis="downstream_constructor",
                        constructor_mutations=relevant_mutations,
                    )
                )
        return dependencies


class DirectAttributeDetector(_DirectDependencyResolver):
    """Discover exact downstream reads of upstream members."""

    def __init__(
        self,
        engine: InterfaceBoundaryGenerator,
        *,
        historical_type_source: Lookup | None = None,
        factory_evaluator: FactoryEvaluator | None = None,
        historical_factory_evaluator: FactoryEvaluator | None = None,
        factory_input_resolver: FactoryInputResolver | None = None,
        historical_factory_input_resolver: FactoryInputResolver | None = None,
        receiver_flow_cache_key: object | None = None,
        historical_receiver_flow_cache_key: object | None = None,
    ):
        super().__init__(
            engine,
            historical_type_source=historical_type_source,
            factory_evaluator=factory_evaluator,
            historical_factory_evaluator=historical_factory_evaluator,
            factory_input_resolver=factory_input_resolver,
            historical_factory_input_resolver=historical_factory_input_resolver,
            receiver_flow_cache_key=receiver_flow_cache_key,
            historical_receiver_flow_cache_key=historical_receiver_flow_cache_key,
        )
        self.historical_attribute_candidates: list[DirectAttributeDependency] = []
        self._defined_member_cache: dict[tuple[int, str, str], bool] = {}
        self._patch_receivers: dict[int, set[str]] = {}
        for relation in engine.relations:
            if (
                relation.relation != "monkey_patch"
                or relation.upstream_package != "vllm"
                or relation.upstream_owner is None
                or relation.installed_descriptor_kind != "ordinary"
            ):
                continue
            module = relation.downstream_file.removesuffix(".py").replace("/", ".")
            target = ".".join(p for p in (module, relation.downstream_owner, relation.downstream_name) if p)
            replacement = engine.downstream.find_callable(target)
            upstream_module = relation.upstream_file.removesuffix(".py").replace("/", ".")
            upstream_module = upstream_module.removesuffix(".__init__")
            owner = f"{upstream_module}.{relation.upstream_owner}"
            if replacement is not None and replacement.node is not None and engine.upstream.find_class(owner):
                self._patch_receivers.setdefault(id(replacement.node), set()).add(owner)

    def _index_owner_defines_member(self, index: RepositoryIndex, owner: str, member: str) -> bool:
        cache_key = (id(index), owner, member)
        if cache_key in self._defined_member_cache:
            return self._defined_member_cache[cache_key]
        if index.find_value(f"{owner}.{member}") is not None:
            self._defined_member_cache[cache_key] = True
            return True
        if index.find_callable(f"{owner}.{member}") is not None:
            self._defined_member_cache[cache_key] = True
            return True
        class_info = index.classes.get(owner)
        if class_info is None:
            self._defined_member_cache[cache_key] = False
            return False
        for method in class_info.methods.values():
            if not isinstance(method, (ast.AsyncFunctionDef, ast.FunctionDef)):
                continue
            positional = [*method.args.posonlyargs, *method.args.args]
            receiver = positional[0].arg if positional else None
            if receiver is None:
                continue
            for candidate in _function_scope_nodes(method):
                if isinstance(candidate, (ast.Assign, ast.AnnAssign)):
                    if isinstance(candidate, ast.AnnAssign) and candidate.value is None:
                        continue
                    targets = candidate.targets if isinstance(candidate, ast.Assign) else (candidate.target,)
                    if any(
                        isinstance(target, ast.Attribute)
                        and target.attr == member
                        and isinstance(target.value, ast.Name)
                        and target.value.id == receiver
                        for target in targets
                    ):
                        self._defined_member_cache[cache_key] = True
                        return True
                if (
                    isinstance(candidate, ast.Call)
                    and (_expression_name(candidate.func) or "").rsplit(".", 1)[-1] == "setattr"
                    and len(candidate.args) >= 2
                    and isinstance(candidate.args[0], ast.Name)
                    and candidate.args[0].id == receiver
                    and isinstance(candidate.args[1], ast.Constant)
                    and candidate.args[1].value == member
                ):
                    self._defined_member_cache[cache_key] = True
                    return True
        self._defined_member_cache[cache_key] = False
        return False

    @staticmethod
    def _node_position(node: ast.AST, *, end: bool = False) -> tuple[int, int]:
        line_name = "end_lineno" if end else "lineno"
        column_name = "end_col_offset" if end else "col_offset"
        return (
            getattr(node, line_name, getattr(node, "lineno", 0)),
            getattr(node, column_name, getattr(node, "col_offset", 0)),
        )

    def _local_owner_defines_member_before_read(
        self,
        owner: str,
        member: str,
        read: ast.Attribute,
        parents: dict[int, ast.AST],
    ) -> bool:
        """Prove that downstream owns a member before this exact read.

        A write in an unrelated method, or later in the same method, does not
        initialize a field before it is read. Constructor writes retain their
        class-wide meaning for reads in other methods.
        """

        index = self.engine.downstream
        if index.find_value(f"{owner}.{member}") is not None:
            return True
        if index.find_callable(f"{owner}.{member}") is not None:
            return True
        class_info = index.classes.get(owner)
        if class_info is None:
            return False
        current = _nearest(read, parents, (ast.FunctionDef, ast.AsyncFunctionDef))
        read_position = self._node_position(read)
        method_nodes = {
            id(node): node
            for variants in class_info.method_variants.values()
            for node in variants
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        }
        method_nodes.update(
            {
                id(node): node
                for node in class_info.methods.values()
                if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
            }
        )
        for method in method_nodes.values():
            if method is not current and method.name != "__init__":
                continue
            positional = [*method.args.posonlyargs, *method.args.args]
            receiver = positional[0].arg if positional else None
            if receiver is None:
                continue
            for candidate in _function_scope_nodes(method):
                assignment = self._receiver_member_assignment(candidate, receiver, member)
                if assignment is None:
                    continue
                value, _annotation = assignment
                if isinstance(candidate, ast.AnnAssign) and value is None:
                    continue
                if method is current and self._node_position(candidate, end=True) >= read_position:
                    continue
                return True
            # ``setattr`` is handled separately because it does not carry a
            # normal assignment target.
            for candidate in _function_scope_nodes(method):
                if not (
                    isinstance(candidate, ast.Call)
                    and (_expression_name(candidate.func) or "").rsplit(".", 1)[-1] == "setattr"
                    and len(candidate.args) >= 2
                    and isinstance(candidate.args[0], ast.Name)
                    and candidate.args[0].id == receiver
                    and isinstance(candidate.args[1], ast.Constant)
                    and candidate.args[1].value == member
                ):
                    continue
                if method is current and self._node_position(candidate, end=True) >= read_position:
                    continue
                return True
        return False

    def _self_or_super_attribute_target(
        self,
        node: ast.Attribute,
        parents: dict[int, ast.AST],
        module: str,
    ) -> tuple[str, str, str, str, str | None, str] | None:
        is_super = (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "super"
            and not node.value.args
            and not node.value.keywords
        )
        function = _nearest(node, parents, (ast.FunctionDef, ast.AsyncFunctionDef))
        receiver = None
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            positional = [*function.args.posonlyargs, *function.args.args]
            receiver = positional[0].arg if positional else None
        is_receiver = isinstance(node.value, ast.Name) and node.value.id == receiver
        if not is_super and not is_receiver:
            return None
        if (
            is_receiver
            and isinstance(function, (ast.AsyncFunctionDef, ast.FunctionDef))
            and receiver is not None
            and self._name_reassigned_before(function, receiver, node)
        ):
            return None
        class_name = self._class_name(node, parents, module)
        if class_name is None:
            patch_receivers = self._patch_receivers.get(id(function), set())
            if is_receiver and len(patch_receivers) == 1:
                patch_receiver = next(iter(patch_receivers))
                return (
                    f"{patch_receiver}.{node.attr}",
                    "instance",
                    patch_receiver,
                    node.attr,
                    patch_receiver,
                    "verified_method_patch_receiver",
                )
            return None
        mro = self.engine._linearized_mro(class_name)
        if not mro.complete:
            return None
        owners = mro.owners[1:] if is_super else mro.owners
        lookup_root = next((owner for owner in owners if owner.startswith("vllm.")), None)
        if lookup_root is None:
            return None
        for owner in owners:
            if owner.startswith("vllm."):
                if self._index_owner_defines_member(self.engine.upstream, owner, node.attr):
                    return (
                        f"{owner}.{node.attr}",
                        "instance",
                        class_name,
                        node.attr,
                        lookup_root,
                        "new_exact",
                    )
            elif self._local_owner_defines_member_before_read(
                owner,
                node.attr,
                node,
                parents,
            ):
                return None
        if hasattr(object, node.attr):
            return None
        return (
            f"{lookup_root}.{node.attr}",
            "instance",
            class_name,
            node.attr,
            lookup_root,
            "old_fallback_super" if is_super else "old_fallback_self",
        )

    def discover(self) -> list[DirectAttributeDependency]:
        dependencies: list[DirectAttributeDependency] = []
        for module_info in self.engine.downstream.modules.values():
            tree = module_info.tree
            parents = self._facts.parents(tree)
            for node in self._facts.nodes(tree):
                if (
                    not isinstance(node, ast.Attribute)
                    or not _attribute_is_read(node, parents)
                    or _under_version_guard(node, parents)
                    or _inside_annotation(node, parents)
                    or _under_attribute_fallback(node, parents)
                ):
                    continue
                parent = parents.get(id(node))
                if (
                    isinstance(parent, ast.Attribute)
                    and parent.value is node
                    or _attribute_is_call_target(node, parents)
                ):
                    continue
                function_node = _nearest(node, parents, (ast.FunctionDef, ast.AsyncFunctionDef))
                function = function_node if isinstance(function_node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
                special = self._self_or_super_attribute_target(node, parents, module_info.name)
                special = special or self._annotated_instance_target(node, function, module_info)
                expression = _expression_name(node)
                root = expression.split(".", 1)[0] if expression is not None else None
                candidate_roots = self._scope_candidate_roots(function, module_info)
                if root is not None and self._outer_function_shadows(
                    node,
                    root,
                    parents,
                    function,
                ):
                    continue
                may_be_constructed = (
                    isinstance(node.value, ast.Call)
                    and (_expression_name(node.value.func) or "").split(".", 1)[0] in candidate_roots
                ) or root in candidate_roots
                if special is None and may_be_constructed:
                    special = self._constructed_instance_target(node, function, module_info)

                flow = self._container_receiver(node, function, module_info)
                if (
                    flow is None
                    and self._receiver_proof_invalidated(node, function, module_info)
                    and (special is None or special[5] != "parameter_annotation")
                ):
                    continue
                if flow is not None:
                    receiver = flow.shape.reference
                    special = (
                        f"{receiver}.{node.attr}",
                        "instance",
                        receiver,
                        node.attr,
                        None,
                        "constructed_storage" if flow.constructed else "typed_container_flow",
                    )
                receiver_type: str | None
                member: str | None
                lookup_root: str | None
                access_kind: str
                if special is not None:
                    target = special[0]
                    access_kind = special[1]
                    receiver_type = special[2]
                    member = special[3]
                    lookup_root = special[4]
                    resolution_basis = special[5]
                else:
                    if root not in candidate_roots:
                        continue
                    resolved = self._resolve_in_scope(
                        node,
                        function=function,
                        module_info=module_info,
                        line=getattr(node, "lineno", 0),
                    )
                    if resolved is None or not resolved.startswith("vllm."):
                        continue
                    resolved_access_kind = self._resolved_access_kind(
                        node,
                        resolved,
                        function,
                        module_info,
                    )
                    if resolved_access_kind is None:
                        continue
                    target = resolved
                    access_kind = resolved_access_kind
                    receiver_type = None
                    member = node.attr
                    lookup_root = None
                    resolution_basis = "new_exact"
                owner = self._class_name(node, parents, module_info.name)
                dependency = DirectAttributeDependency(
                    target=target,
                    access_kind=access_kind,
                    file=module_info.file,
                    line=getattr(node, "lineno", 0),
                    column=getattr(node, "col_offset", 0),
                    owner=owner.rsplit(".", 1)[-1] if owner else None,
                    scope=function.name if function is not None else None,
                    expression=ast.unparse(node),
                    receiver_path=flow.path if flow is not None else (),
                    receiver_binding=self._receiver_binding_evidence(
                        node,
                        function,
                        module_info,
                        receiver_type,
                        resolution_basis,
                        flow.path if flow is not None else (),
                    ),
                    receiver_type=receiver_type,
                    member=member,
                    lookup_root=lookup_root,
                    resolution_basis=resolution_basis,
                )
                # A proven self/super read keeps its downstream receiver for
                # evidence, but snapshots must look up its upstream MRO root.
                # Only absent-at-new candidates need historical discovery.
                if resolution_basis.startswith("old_fallback"):
                    self.historical_attribute_candidates.append(dependency)
                else:
                    dependencies.append(dependency)
        unique = {
            (
                item.file,
                item.line,
                item.column,
                item.target,
                item.expression,
                item.access_kind,
                item.receiver_type,
                item.member,
            ): item
            for item in dependencies
        }
        return [unique[key] for key in sorted(unique)]


__all__ = [
    "CallShape",
    "DirectAttributeDependency",
    "DirectAttributeDetector",
    "DirectCallDependency",
    "DirectCallDetector",
    "ReturnContract",
    "ReturnShape",
    "ReturnUse",
    "bind_call_shape",
    "call_shape",
    "infer_return_contract",
    "replacement_return_compatible",
    "return_contract_from_dict",
    "return_use_compatible",
]
