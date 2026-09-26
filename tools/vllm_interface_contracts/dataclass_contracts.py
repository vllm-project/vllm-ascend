# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-only dataclass field ordering and generated argument protocols.

No repository classes or decorators are executed. Unknown decorators, dynamic
field options, conditional declarations and multiple inheritance stay unknown.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassSource:
    node: ast.ClassDef
    resolve: Callable[[str], str | None]
    file: str
    qualified_name: str
    storage_effects: bool = False


@dataclass(frozen=True)
class DataclassField:
    name: str
    required: bool
    keyword_only: bool
    included: bool
    owner: str
    file: str
    line: int
    init_variable: bool = False


@dataclass(frozen=True)
class DataclassLayout:
    fields: tuple[DataclassField, ...]
    generates_initializer: bool

    def initializer(self) -> ast.FunctionDef | None:
        """Represent only the generated argument protocol, without executing code."""
        if not self.generates_initializer or self.ordering_error() is not None:
            return None
        fields = [field for field in self.fields if field.included]
        positional = [field for field in fields if not field.keyword_only]
        keywords = [field for field in fields if field.keyword_only]
        node = ast.parse("def __init__(self): pass").body[0]
        assert isinstance(node, ast.FunctionDef)
        receiver = "__dataclass_self__" if any(field.name == "self" for field in fields) else "self"
        node.args = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=receiver), *(ast.arg(arg=field.name) for field in positional)],
            vararg=None,
            kwonlyargs=[ast.arg(arg=field.name) for field in keywords],
            kw_defaults=[None if field.required else ast.Constant(None) for field in keywords],
            kwarg=None,
            defaults=[ast.Constant(None) for field in positional if not field.required],
        )
        return ast.fix_missing_locations(node)

    def ordering_error(self) -> tuple[DataclassField, DataclassField] | None:
        if not self.generates_initializer:
            return None
        default: DataclassField | None = None
        for field in self.fields:
            if not field.included or field.keyword_only:
                continue
            if field.required and default is not None:
                return default, field
            if not field.required:
                default = field
        return None


@dataclass(frozen=True)
class DataclassStorage:
    """Generated initialization which stores supplied arguments unchanged.

    This is a class-source proof, not a callsite or lifetime proof. Consumers
    must still verify class bindings, protocol mutations, argument binding and
    subsequent writes/escapes independently in each source snapshot.
    """

    initializer_owner: str
    layout: DataclassLayout
    stored_parameters: frozenset[str]

    def literal_defaults(self, lookup: Callable[[str], ClassSource | None]) -> dict[str, ast.Constant]:
        """Read literal defaults captured by the proven generated initializer.

        Start at its owner, not an init=False/plain subclass. An annotation
        without a value may inherit a default; a field() declaration does not.
        Existing storage checks have already excluded factories and hooks.
        """
        remaining = {
            field.name for field in self.layout.fields if not field.required and field.name in self.stored_parameters
        }
        defaults: dict[str, ast.Constant] = {}
        current: str | None = self.initializer_owner
        seen: set[str] = set()
        while current is not None and current not in seen and remaining:
            seen.add(current)
            source = lookup(current)
            if source is None or source.storage_effects:
                return {}
            for statement in reversed(source.node.body):
                if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
                    continue
                name = statement.target.id
                if name not in remaining or statement.value is None:
                    continue
                value = statement.value
                remaining.remove(name)
                if isinstance(value, ast.Call) and source.resolve(ast.unparse(value.func)) == "dataclasses.field":
                    candidate = next((item.value for item in value.keywords if item.arg == "default"), None)
                    if isinstance(candidate, ast.Constant):
                        defaults[name] = candidate
                elif isinstance(value, ast.Constant):
                    defaults[name] = value
            current = source.resolve(ast.unparse(source.node.bases[0])) if len(source.node.bases) == 1 else None
        return defaults

    def bind_call(self, call: ast.Call) -> dict[str, int] | None:
        """Map supplied arguments to parameters using the existing field layout.

        Values are positions in the call's positional-then-keyword value list,
        not inferred Python values. Omitted defaults are deliberately absent.
        Star expansions need independent call-shape evidence and stay unknown.
        InitVars participate in binding even though they are not stored.
        """
        if any(isinstance(argument, ast.Starred) for argument in call.args):
            return None
        keywords = [keyword.arg for keyword in call.keywords]
        if any(name is None for name in keywords) or len(set(keywords)) != len(keywords):
            return None
        fields = [field for field in self.layout.fields if field.included]
        parameters = [
            inspect.Parameter(
                field.name,
                inspect.Parameter.KEYWORD_ONLY if field.keyword_only else inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=inspect.Parameter.empty if field.required else None,
            )
            for field in sorted(fields, key=lambda field: field.keyword_only)
        ]
        try:
            bound = inspect.Signature(parameters).bind(
                *range(len(call.args)),
                **{name: len(call.args) + offset for offset, name in enumerate(keywords) if name is not None},
            )
        except (TypeError, ValueError):
            return None
        return dict(bound.arguments)


def dataclass_storage(
    reference: str,
    lookup: Callable[[str], ClassSource | None],
) -> DataclassStorage | None:
    """Prove ordinary generated stores without executing repository code.

    An argument protocol alone is insufficient: InitVar is not stored,
    post-init may replace values, and descriptors/default factories can run
    arbitrary code. Reject those effects rather than infer storage from field
    annotations. Plain children and init=False children may inherit a proven
    generated initializer; their additional annotations do not add parameters.
    """
    hierarchy: list[tuple[str, ClassSource, DataclassLayout]] = []
    seen: set[str] = set()
    current: str | None = reference
    while current is not None and current not in {"object", "builtins.object"}:
        if current in seen:
            return None
        seen.add(current)
        source = lookup(current)
        layout = dataclass_layout(current, lookup)
        if (
            source is None
            or layout is None
            or source.storage_effects
            or source.node.keywords
            or len(source.node.bases) > 1
        ):
            return None
        hierarchy.append((current, source, layout))
        if not source.node.bases:
            break
        current = source.resolve(ast.unparse(source.node.bases[0]))
        if current is None:
            return None
    generated = next(((name, layout) for name, _, layout in hierarchy if layout.generates_initializer), None)
    if generated is None or generated[1].initializer() is None:
        return None
    owner, initializer_layout = generated
    parameter_names = {field.name for field in initializer_layout.fields if field.included}
    # Only the effective initializer's field definitions determine its stores.
    # A decorated child may override an inherited InitVar with an ordinary
    # field; annotations on an init=False/plain child do not rewrite __init__.
    init_variables = {field.name for field in initializer_layout.fields if field.init_variable}
    hooks = {
        "__init__",
        "__new__",
        "__post_init__",
        "__setattr__",
        "__delattr__",
        "__getattribute__",
        "__getattr__",
        "__init_subclass__",
        "__class__",
        "__dict__",
        "__slots__",
        "__weakref__",
    }
    for _, source, _ in hierarchy:
        for statement in source.node.body:
            if isinstance(statement, ast.Pass) or (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                continue
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if statement.name in hooks or statement.name in parameter_names:
                    return None
                if any(
                    source.resolve(ast.unparse(decorator))
                    not in {
                        "property",
                        "builtins.property",
                        "staticmethod",
                        "builtins.staticmethod",
                        "classmethod",
                        "builtins.classmethod",
                    }
                    for decorator in statement.decorator_list
                ):
                    return None
                continue
            if isinstance(statement, ast.Assign):
                # An ordinary constant class member (including a non-callable
                # method blocker such as run=None) does not alter generated
                # field stores. Never extend this to namespace/protocol hooks,
                # inherited field defaults, descriptors or evaluated factories.
                field_names = {field.name for field in initializer_layout.fields}
                if isinstance(statement.value, ast.Constant) and all(
                    isinstance(target, ast.Name) and target.id not in field_names and not target.id.startswith("__")
                    for target in statement.targets
                ):
                    continue
                return None
            if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
                # Even an unrelated class-body call or assignment may install a
                # descriptor or replace an inherited construction hook.
                return None
            name = statement.target.id
            if name in hooks:
                return None
            annotation = statement.annotation
            if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
                try:
                    annotation = ast.parse(annotation.value, mode="eval").body
                except SyntaxError:
                    return None
            annotation_root = annotation.value if isinstance(annotation, ast.Subscript) else annotation
            kind = source.resolve(ast.unparse(annotation_root))
            if name in parameter_names and kind == "typing.ClassVar":
                return None
            value = statement.value
            if value is None or isinstance(value, ast.Constant):
                continue
            if source.resolve(ast.unparse(value)) == "dataclasses.MISSING":
                continue
            if (
                not isinstance(value, ast.Call)
                or source.resolve(ast.unparse(value.func)) != "dataclasses.field"
                or value.args
            ):
                return None
            for keyword in value.keywords:
                if keyword.arg in {None, "default_factory"} or not isinstance(keyword.value, ast.Constant):
                    return None
    return DataclassStorage(owner, initializer_layout, frozenset(parameter_names - init_variables))


@dataclass(frozen=True)
class DataclassReplacement:
    """Shared metadata inputs for replacement binding and returned storage."""

    layout: DataclassLayout
    storage: DataclassStorage

    @property
    def copied_fields(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.layout.fields if field.included)

    @property
    def excluded_fields(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.layout.fields if not field.included)

    @property
    def required_keywords(self) -> tuple[str, ...]:
        return tuple(
            field.name
            for field in self.layout.fields
            if field.included
            and field.required
            and (field.init_variable or field.name not in self.storage.stored_parameters)
        )


def dataclass_replacement(reference: str, lookup: Callable[[str], ClassSource | None]) -> DataclassReplacement | None:
    storage = dataclass_storage(reference, lookup)
    layout = dataclass_layout(reference, lookup) if storage is not None else None
    return DataclassReplacement(layout, storage) if layout is not None and storage is not None else None


def dataclass_layout(
    reference: str,
    lookup: Callable[[str], ClassSource | None],
    seen: frozenset[str] = frozenset(),
) -> DataclassLayout | None:
    """Resolve a unique source hierarchy and its ordered init fields."""
    if reference in seen or (source := lookup(reference)) is None:
        return None
    node = source.node
    if node.keywords or len(node.bases) > 1:
        return None
    init, default_kw_only = True, False
    decorated = False
    for decorator in node.decorator_list:
        call = decorator if isinstance(decorator, ast.Call) else None
        target = ast.unparse(call.func if call else decorator)
        if decorated or source.resolve(target) != "dataclasses.dataclass":
            return None
        decorated = True
        if call is not None:
            if call.args:
                return None
            for keyword in call.keywords:
                if keyword.arg not in {
                    "init",
                    "repr",
                    "eq",
                    "order",
                    "unsafe_hash",
                    "frozen",
                    "match_args",
                    "kw_only",
                    "slots",
                    "weakref_slot",
                }:
                    return None
                if keyword.arg is None or not isinstance(keyword.value, ast.Constant):
                    return None
                if not isinstance(keyword.value.value, bool):
                    return None
                if keyword.arg == "init":
                    init = keyword.value.value
                elif keyword.arg == "kw_only":
                    default_kw_only = keyword.value.value

    fields: dict[str, DataclassField] = {}
    for base in node.bases:
        base_ref = source.resolve(ast.unparse(base))
        if base_ref in {"object", "builtins.object"}:
            continue
        if base_ref is None:
            return None
        parent = dataclass_layout(base_ref, lookup, seen | {reference})
        if parent is None:
            return None
        fields.update((field.name, field) for field in parent.fields)
    if not decorated:
        return DataclassLayout(tuple(fields.values()), False)

    keyword_only = default_kw_only
    for statement in node.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if statement.name == "__init__":
                init = False
            continue
        if isinstance(statement, (ast.If, ast.Try, ast.For, ast.While, ast.With)):
            # Choosing both paths would silently create an impossible layout.
            if any(isinstance(child, (ast.AnnAssign, ast.Assign)) for child in ast.walk(statement)):
                return None
            continue
        if isinstance(statement, ast.Assign):
            if any(
                isinstance(target, ast.Name) and (target.id in fields or target.id == "__init__")
                for target in statement.targets
            ):
                return None
            continue
        if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
            continue
        name = statement.target.id
        annotation = statement.annotation
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            try:
                annotation = ast.parse(annotation.value, mode="eval").body
            except SyntaxError:
                return None
        root = annotation.value if isinstance(annotation, ast.Subscript) else annotation
        annotation_ref = source.resolve(ast.unparse(root))
        if annotation_ref == "dataclasses.KW_ONLY":
            keyword_only = True
            continue
        included = annotation_ref != "typing.ClassVar"
        field_kw_only = keyword_only
        value = statement.value
        required = value is None or source.resolve(ast.unparse(value)) == "dataclasses.MISSING"
        if isinstance(value, ast.Call) and source.resolve(ast.unparse(value.func)) == "dataclasses.field":
            if value.args:
                return None
            required = True
            defaults = 0
            for keyword in value.keywords:
                if keyword.arg in {"default", "default_factory"}:
                    if source.resolve(ast.unparse(keyword.value)) != "dataclasses.MISSING":
                        required = False
                        defaults += 1
                        if keyword.arg == "default" and isinstance(keyword.value, (ast.List, ast.Dict, ast.Set)):
                            return None
                elif keyword.arg in {"init", "kw_only"}:
                    if not isinstance(keyword.value, ast.Constant) or not isinstance(keyword.value.value, bool):
                        return None
                    if keyword.arg == "init":
                        included = included and keyword.value.value
                    else:
                        field_kw_only = keyword.value.value
                elif keyword.arg not in {"repr", "hash", "compare", "metadata"}:
                    return None
            if defaults > 1:
                return None
        elif isinstance(value, (ast.Call, ast.List, ast.Dict, ast.Set)):
            # A descriptor factory can change requiredness through __get__;
            # mutable defaults can prevent the dataclass from being defined.
            return None
        elif value is None and name in fields:
            # An annotation without a value does not erase an inherited default.
            required = fields[name].required
        fields[name] = DataclassField(
            name,
            required,
            field_kw_only,
            included,
            reference,
            source.file,
            statement.lineno,
            annotation_ref == "dataclasses.InitVar",
        )
    return DataclassLayout(tuple(fields.values()), init)
