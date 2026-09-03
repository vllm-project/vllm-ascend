#!/usr/bin/env python3
#
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
#
"""Forbid Ascend hardware-identity decisions outside the HAL boundary.

Production code should select hardware-dependent behavior through
``HardwareCapability`` or a profile policy/family. Only detection, identity
mapping, and profile registration may interpret a concrete Ascend family.
Generic platform introspection may report a device name, but business code
must not use that value to choose an implementation.

This check intentionally targets explicit identity access. Deliberately
disguised or inter-procedural identity logic still requires code review.
"""

import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_PACKAGE = "vllm_ascend"

# These paths are generated build metadata or staged vendor operator sources,
# not Python business logic. They are ignored by Git and may appear only after
# a local build, so they must not make the full-tree pre-commit check unstable.
NON_BUSINESS_PYTHON_FILES = frozenset({"vllm_ascend/_build_info.py"})
NON_BUSINESS_PYTHON_PREFIXES = ("vllm_ascend/_cann_ops_custom/",)

# Keep this boundary file-specific. Allowing a directory would let unrelated
# business logic placed beside the HAL silently bypass the guard.
ALLOWED_IDENTITY_FILES = frozenset(
    {
        "vllm_ascend/device/device_config.py",
        "vllm_ascend/device/hardware.py",
        "vllm_ascend/device/hardware_profile.py",
    }
)

ENV_CONFIG_FILE = "vllm_ascend/envs.py"
SOC_VERSION_ENV_KEY = "SOC_VERSION"

RAW_IDENTITY_NAMES = frozenset(
    {
        "AscendDeviceType",
        "get_ascend_device_type",
        "is_310p",
        "is_950",
        "device_type_from_runtime_soc",
        "device_type_from_soc_version",
    }
)
IDENTITY_MODULES = frozenset(
    {
        "vllm_ascend.device.device_config",
        "vllm_ascend.device.hardware",
    }
)
NON_IDENTITY_NAMES = frozenset({"check_ascend_device_type"})

_DEVICE_PREDICATE_RE = re.compile(
    r"^is_(?:ascend_?)?(?:310p\d*(?:vir\d+)?|950[a-z0-9]*|910[a-z0-9]*|a[235])(?:_[a-z0-9_]+)?$",
    re.IGNORECASE,
)
_SOC_NAME_RE = re.compile(r"(?:^|_)soc(?:_|$)", re.IGNORECASE)
_DEVICE_IDENTITY_LITERAL_RE = re.compile(
    r"^(?:_310p|(?:ascend)?(?:310p\d*(?:vir\d+)?|910[a-z0-9_-]*|950[a-z0-9_-]*|a[235]))$",
    re.IGNORECASE,
)
_STRING_IDENTITY_PREDICATES = frozenset({"startswith", "endswith"})


@dataclass(frozen=True, order=True)
class Violation:
    line: int
    column: int
    code: str
    message: str

    def format(self, path: str) -> str:
        return f"{path}:{self.line}:{self.column}: {self.code} {self.message}"


def _normalized_repo_path(filepath: str | Path) -> str | None:
    path = Path(filepath)
    try:
        resolved = path.resolve()
        relative = resolved.relative_to(REPO_ROOT)
    except (OSError, ValueError):
        return None
    return relative.as_posix()


def _normalized_source_path(relative_path: str | Path) -> str:
    return str(relative_path).replace("\\", "/").lstrip("./")


def _is_production_python(path: str) -> bool:
    parts = path.split("/")
    return (
        len(parts) > 1
        and parts[0] == PRODUCTION_PACKAGE
        and path.endswith(".py")
        and path not in NON_BUSINESS_PYTHON_FILES
        and not path.startswith(NON_BUSINESS_PYTHON_PREFIXES)
    )


def _default_python_paths() -> list[Path]:
    """Return tracked production Python sources, with a source-tree fallback."""
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "-z", "--", PRODUCTION_PACKAGE],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
        )
    except OSError:
        tracked = None

    if tracked is not None and tracked.returncode == 0:
        relative_paths = tracked.stdout.decode("utf-8", errors="surrogateescape").split("\0")
        return [REPO_ROOT / path for path in relative_paths if _is_production_python(path)]

    return sorted(
        path
        for path in (REPO_ROOT / PRODUCTION_PACKAGE).rglob("*.py")
        if _is_production_python(_normalized_repo_path(path) or "")
    )


def _identifier_code(identifier: str) -> str | None:
    if identifier in NON_IDENTITY_NAMES:
        return None

    lowered = identifier.casefold()
    if lowered in {"_device_type", "__device_type__"}:
        return "HDI003"
    if (
        identifier in RAW_IDENTITY_NAMES
        or "ascend_device_type" in lowered
        or _DEVICE_PREDICATE_RE.fullmatch(identifier)
    ):
        return "HDI001"
    if "soc_version" in lowered or _SOC_NAME_RE.search(identifier) or lowered in {"get_soc_version", "__soc_version__"}:
        return "HDI002"
    return None


def _message(code: str, identifier: str) -> str:
    if code == "HDI001":
        subject = "raw Ascend device identity"
    elif code == "HDI002":
        subject = "raw SoC identity"
    elif code == "HDI003":
        subject = "private detected device identity"
    elif code == "HDI005":
        subject = "concrete device-family literal"
    else:
        subject = "dynamic raw hardware identity"
    return (
        f"{subject} `{identifier}` is forbidden outside the HAL boundary; "
        "query HardwareCapability or a profile policy/family instead"
    )


def _dotted_name(node: ast.expr) -> str | None:
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _string_literal(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _device_identity_literals(node: ast.AST) -> list[ast.Constant]:
    return [
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Constant)
        and isinstance(candidate.value, str)
        and _DEVICE_IDENTITY_LITERAL_RE.fullmatch(candidate.value.strip())
    ]


def _resolve_import_from(node: ast.ImportFrom, source_path: str) -> str:
    if node.level == 0:
        return node.module or ""

    path_parts = source_path.split("/")
    package_parts = path_parts[:-1]
    if path_parts[-1] == "__init__.py":
        package_parts = path_parts[:-1]
    trim = node.level - 1
    if trim:
        package_parts = package_parts[:-trim]
    if node.module:
        package_parts.extend(node.module.split("."))
    return ".".join(package_parts)


def _is_exact_env_declaration(node: ast.Call, tree: ast.AST, path: str) -> bool:
    """Allow only envs.py's existing lazy declaration of SOC_VERSION."""
    if path != ENV_CONFIG_FILE or _dotted_name(node.func) != "os.getenv":
        return False
    if not node.args or _string_literal(node.args[0]) != SOC_VERSION_ENV_KEY:
        return False

    if not isinstance(tree, ast.Module):
        return False
    for statement in tree.body:
        if (
            not isinstance(statement, ast.AnnAssign)
            or not isinstance(statement.target, ast.Name)
            or statement.target.id != "env_variables"
            or not isinstance(statement.value, ast.Dict)
        ):
            continue
        for key, value in zip(statement.value.keys, statement.value.values):
            if (
                key is not None
                and _string_literal(key) == SOC_VERSION_ENV_KEY
                and isinstance(value, ast.Lambda)
                and value.body is node
            ):
                return True
    return False


@dataclass
class _EnvironmentAliasState:
    os_aliases: set[str]
    getenv_aliases: set[str]
    environ_aliases: set[str]

    def copy(self) -> "_EnvironmentAliasState":
        return _EnvironmentAliasState(
            set(self.os_aliases),
            set(self.getenv_aliases),
            set(self.environ_aliases),
        )


def _bound_target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.List, ast.Tuple)):
        return {name for element in target.elts for name in _bound_target_names(element)}
    return set()


class _ScopeAliasCollector(ast.NodeVisitor):
    """Collect aliases in one lexical scope without entering nested scopes."""

    def __init__(
        self,
        parameter_names: set[str],
        parameter_defaults: list[tuple[str, ast.expr]],
    ) -> None:
        self.bound_names = set(parameter_names)
        self.parameter_defaults = list(parameter_defaults)
        self.os_imports: set[str] = set()
        self.getenv_imports: set[str] = set()
        self.environ_imports: set[str] = set()
        self.assignments: list[tuple[str, ast.expr]] = []

    def _record_assignment(self, target: ast.expr, value: ast.expr) -> None:
        if isinstance(value, ast.NamedExpr):
            self._record_assignment(value.target, value.value)
            value = value.value
        names = _bound_target_names(target)
        self.bound_names.update(names)
        if len(names) == 1 and isinstance(target, ast.Name):
            self.assignments.append((target.id, value))
        elif (
            isinstance(target, (ast.List, ast.Tuple))
            and isinstance(value, (ast.List, ast.Tuple))
            and len(target.elts) == len(value.elts)
        ):
            for element, element_value in zip(target.elts, value.elts):
                self._record_assignment(element, element_value)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound_name = alias.asname or alias.name.split(".", 1)[0]
            self.bound_names.add(bound_name)
            if alias.name == "os" or (alias.asname is None and alias.name.startswith("os.")):
                self.os_imports.add(bound_name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                if node.level == 0 and node.module == "os":
                    self.getenv_imports.add("getenv")
                    self.environ_imports.add("environ")
                continue
            bound_name = alias.asname or alias.name
            self.bound_names.add(bound_name)
            if node.level == 0 and node.module == "os":
                if alias.name == "getenv":
                    self.getenv_imports.add(bound_name)
                elif alias.name == "environ":
                    self.environ_imports.add(bound_name)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._record_assignment(target, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.bound_names.update(_bound_target_names(node.target))
        if node.value is not None:
            self._record_assignment(node.target, node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._record_assignment(node.target, node.value)

    def _visit_for(self, node: ast.For | ast.AsyncFor) -> None:
        self.bound_names.update(_bound_target_names(node.target))
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._visit_for(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_for(node)

    def _visit_with(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self.bound_names.update(_bound_target_names(item.optional_vars))
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self._visit_with(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._visit_with(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self.bound_names.add(node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.bound_names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.bound_names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bound_names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _parameter_names(arguments: ast.arguments) -> set[str]:
    parameters = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
    if arguments.vararg is not None:
        parameters.append(arguments.vararg)
    if arguments.kwarg is not None:
        parameters.append(arguments.kwarg)
    return {parameter.arg for parameter in parameters}


def _parameter_defaults(arguments: ast.arguments) -> list[tuple[str, ast.expr]]:
    positional = [*arguments.posonlyargs, *arguments.args]
    defaults = list(zip(positional[-len(arguments.defaults) :], arguments.defaults)) if arguments.defaults else []
    defaults.extend(
        (parameter, default)
        for parameter, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
        if default is not None
    )
    return [(parameter.arg, default) for parameter, default in defaults]


def _collect_scope_aliases(
    nodes: list[ast.stmt] | list[ast.expr],
    inherited: _EnvironmentAliasState,
    *,
    parameter_names: set[str] | None = None,
    parameter_defaults: list[tuple[str, ast.expr]] | None = None,
) -> _EnvironmentAliasState:
    collector = _ScopeAliasCollector(
        parameter_names or set(),
        parameter_defaults or [],
    )
    for node in nodes:
        collector.visit(node)

    def is_os_module(
        node: ast.expr,
        aliases: _EnvironmentAliasState,
    ) -> bool:
        if isinstance(node, ast.NamedExpr):
            return is_os_module(node.value, aliases)
        return isinstance(node, ast.Name) and node.id in aliases.os_aliases

    def is_environ(
        node: ast.expr,
        aliases: _EnvironmentAliasState,
    ) -> bool:
        if isinstance(node, ast.NamedExpr):
            return is_environ(node.value, aliases)
        return (
            isinstance(node, ast.Name)
            and node.id in aliases.environ_aliases
            or (isinstance(node, ast.Attribute) and node.attr == "environ" and is_os_module(node.value, aliases))
        )

    def is_getenv(
        node: ast.expr,
        aliases: _EnvironmentAliasState,
    ) -> bool:
        if isinstance(node, ast.NamedExpr):
            return is_getenv(node.value, aliases)
        return (
            isinstance(node, ast.Name)
            and node.id in aliases.getenv_aliases
            or (isinstance(node, ast.Attribute) and node.attr == "getenv" and is_os_module(node.value, aliases))
            or (isinstance(node, ast.Attribute) and node.attr == "get" and is_environ(node.value, aliases))
        )

    state = inherited.copy()
    state.os_aliases.difference_update(collector.bound_names)
    state.getenv_aliases.difference_update(collector.bound_names)
    state.environ_aliases.difference_update(collector.bound_names)
    state.os_aliases.update(collector.os_imports)
    state.getenv_aliases.update(collector.getenv_imports)
    state.environ_aliases.update(collector.environ_imports)

    # Function defaults are evaluated in the enclosing scope, before their
    # parameter names shadow an outer alias with the same spelling.
    for target, value in collector.parameter_defaults:
        if is_os_module(value, inherited):
            state.os_aliases.add(target)
        elif is_environ(value, inherited):
            state.environ_aliases.add(target)
        elif is_getenv(value, inherited):
            state.getenv_aliases.add(target)

    changed = True
    while changed:
        changed = False
        for target, value in collector.assignments:
            alias_set: set[str] | None = None
            if is_os_module(value, state):
                alias_set = state.os_aliases
            elif is_environ(value, state):
                alias_set = state.environ_aliases
            elif is_getenv(value, state):
                alias_set = state.getenv_aliases
            if alias_set is not None and target not in alias_set:
                alias_set.add(target)
                changed = True
    return state


class HardwareIdentityVisitor(ast.NodeVisitor):
    def __init__(self, tree: ast.AST, source_path: str) -> None:
        self._tree = tree
        self._source_path = source_path
        self._violations: list[Violation] = []
        self._seen: set[tuple[int, int, str, str]] = set()
        self._aliases = _EnvironmentAliasState({"os"}, set(), set())
        self._function_parent_aliases = self._aliases

    @property
    def violations(self) -> list[Violation]:
        return sorted(self._violations)

    def _add(
        self,
        node: ast.AST,
        code: str,
        identifier: str,
        *,
        dynamic: bool = False,
    ) -> None:
        line = getattr(node, "lineno", 1)
        column = getattr(node, "col_offset", 0) + 1
        output_code = "HDI004" if dynamic else code
        key = (line, column, output_code, identifier)
        if key in self._seen:
            return
        self._seen.add(key)
        self._violations.append(Violation(line, column, output_code, _message(output_code, identifier)))

    def _check_identifier(self, node: ast.AST, identifier: str) -> None:
        if code := _identifier_code(identifier):
            self._add(node, code, identifier)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            imported_name = alias.name.rsplit(".", 1)[-1]
            if code := _identifier_code(imported_name):
                self._add(node, code, imported_name)
            elif alias.asname and (code := _identifier_code(alias.asname)):
                self._add(node, code, alias.asname)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = _resolve_import_from(node, self._source_path)
        for alias in node.names:
            if alias.name == "*" and module in IDENTITY_MODULES:
                self._add(node, "HDI001", f"{module}.*")
            elif code := _identifier_code(alias.name):
                self._add(node, code, alias.name)
            elif alias.asname and (code := _identifier_code(alias.asname)):
                self._add(node, code, alias.asname)

    def _visit_scope(
        self,
        nodes: list[ast.stmt] | list[ast.expr],
        inherited: _EnvironmentAliasState,
        *,
        parameter_names: set[str] | None = None,
        parameter_defaults: list[tuple[str, ast.expr]] | None = None,
        nested_function_parent: _EnvironmentAliasState | None = None,
    ) -> None:
        previous_aliases = self._aliases
        previous_function_parent = self._function_parent_aliases
        self._aliases = _collect_scope_aliases(
            nodes,
            inherited,
            parameter_names=parameter_names,
            parameter_defaults=parameter_defaults,
        )
        self._function_parent_aliases = nested_function_parent or self._aliases
        try:
            for node in nodes:
                self.visit(node)
        finally:
            self._aliases = previous_aliases
            self._function_parent_aliases = previous_function_parent

    def visit_Module(self, node: ast.Module) -> None:
        self._visit_scope(node.body, self._aliases)

    def visit_Name(self, node: ast.Name) -> None:
        self._check_identifier(node, node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._check_identifier(node, node.attr)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self._check_identifier(node, node.arg)
        self.generic_visit(node)

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        self._check_identifier(node, node.name)
        self.visit(node.args)
        for decorator in node.decorator_list:
            self.visit(decorator)
        if node.returns is not None:
            self.visit(node.returns)
        for type_parameter in getattr(node, "type_params", []):
            self.visit(type_parameter)
        self._visit_scope(
            node.body,
            self._function_parent_aliases,
            parameter_names=_parameter_names(node.args),
            parameter_defaults=_parameter_defaults(node.args),
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._check_identifier(node, node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword)
        for type_parameter in getattr(node, "type_params", []):
            self.visit(type_parameter)
        previous_aliases = self._aliases
        previous_function_parent = self._function_parent_aliases
        self._aliases = self._function_parent_aliases.copy()
        try:
            # A class body executes sequentially and is not an enclosing
            # lexical scope for nested classes or methods. Check each statement
            # before applying the aliases and shadows that it binds.
            for statement in node.body:
                self.visit(statement)
                previous_statement_aliases = self._aliases
                self._aliases = _collect_scope_aliases(
                    [statement],
                    previous_statement_aliases,
                )
                if not isinstance(
                    statement,
                    (
                        ast.AnnAssign,
                        ast.Assign,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                        ast.FunctionDef,
                        ast.Import,
                        ast.ImportFrom,
                    ),
                ):
                    # Conditional/loop/try bodies may not execute. Preserve
                    # prior taint while also retaining aliases they may add.
                    self._aliases.os_aliases.update(previous_statement_aliases.os_aliases)
                    self._aliases.getenv_aliases.update(previous_statement_aliases.getenv_aliases)
                    self._aliases.environ_aliases.update(previous_statement_aliases.environ_aliases)
        finally:
            self._aliases = previous_aliases
            self._function_parent_aliases = previous_function_parent

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit(node.args)
        self._visit_scope(
            [node.body],
            self._function_parent_aliases,
            parameter_names=_parameter_names(node.args),
            parameter_defaults=_parameter_defaults(node.args),
        )

    def _visit_comprehension(
        self,
        generators: list[ast.comprehension],
        result_nodes: list[ast.expr],
    ) -> None:
        if not generators:
            for result_node in result_nodes:
                self.visit(result_node)
            return

        # The outermost iterable is evaluated in the enclosing scope. Targets,
        # filters, later iterables, and result expressions use the comprehension's
        # own scope, which must not inherit a shadowed class namespace.
        self.visit(generators[0].iter)
        bound_names = {name for generator in generators for name in _bound_target_names(generator.target)}
        scoped_nodes: list[ast.expr] = []
        for index, generator in enumerate(generators):
            scoped_nodes.append(generator.target)
            if index:
                scoped_nodes.append(generator.iter)
            scoped_nodes.extend(generator.ifs)
        scoped_nodes.extend(result_nodes)
        self._visit_scope(
            scoped_nodes,
            self._function_parent_aliases,
            parameter_names=bound_names,
        )

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node.generators, [node.elt])

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node.generators, [node.key, node.value])

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg is not None:
            self._check_identifier(node, node.arg)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            self._check_identifier(node, name)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for name in node.names:
            self._check_identifier(node, name)

    def _is_soc_env_call(self, node: ast.Call) -> bool:
        key_node = (
            node.args[0]
            if node.args
            else next(
                (keyword.value for keyword in node.keywords if keyword.arg == "key"),
                None,
            )
        )
        if key_node is None or _string_literal(key_node) != SOC_VERSION_ENV_KEY:
            return False
        return self._is_getenv_callable(node.func)

    def _is_getenv_callable(self, node: ast.expr) -> bool:
        if isinstance(node, ast.NamedExpr):
            return self._is_getenv_callable(node.value)
        if isinstance(node, ast.Name):
            return node.id in self._aliases.getenv_aliases
        if not isinstance(node, ast.Attribute):
            return False
        if node.attr == "getenv":
            return self._is_os_module(node.value)
        return node.attr == "get" and self._is_os_environ(node.value)

    def _is_os_module(self, node: ast.expr) -> bool:
        if isinstance(node, ast.NamedExpr):
            return self._is_os_module(node.value)
        return isinstance(node, ast.Name) and node.id in self._aliases.os_aliases

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _dotted_name(node.func)
        tail = call_name.rsplit(".", 1)[-1] if call_name else None
        if tail in {"getattr", "hasattr", "setattr", "delattr"} and len(node.args) >= 2:
            if identifier := _string_literal(node.args[1]):
                if code := _identifier_code(identifier):
                    self._add(node, code, identifier, dynamic=True)

        attribute_name = node.func.attr if isinstance(node.func, ast.Attribute) else tail
        if attribute_name in _STRING_IDENTITY_PREDICATES:
            for argument in node.args:
                for literal in _device_identity_literals(argument):
                    self._add(literal, "HDI005", literal.value)

        if (
            attribute_name == "get"
            and isinstance(node.func, ast.Attribute)
            and self._is_dynamic_namespace(node.func.value)
            and node.args
        ):
            if identifier := _string_literal(node.args[0]):
                if code := _identifier_code(identifier):
                    self._add(node, code, identifier, dynamic=True)

        if self._is_soc_env_call(node) and not _is_exact_env_declaration(node, self._tree, self._source_path):
            self._add(node, "HDI002", "SOC_VERSION environment value")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and _DEVICE_IDENTITY_LITERAL_RE.fullmatch(node.value.strip()):
            self._add(node, "HDI005", node.value)

    def visit_Compare(self, node: ast.Compare) -> None:
        for operand in (node.left, *node.comparators):
            for literal in _device_identity_literals(operand):
                self._add(literal, "HDI005", literal.value)
        self.generic_visit(node)

    def visit_MatchValue(self, node: ast.MatchValue) -> None:
        for literal in _device_identity_literals(node.value):
            self._add(literal, "HDI005", literal.value)
        self.generic_visit(node)

    def _is_os_environ(self, node: ast.expr) -> bool:
        if isinstance(node, ast.NamedExpr):
            return self._is_os_environ(node.value)
        if isinstance(node, ast.Name):
            return node.id in self._aliases.environ_aliases
        return isinstance(node, ast.Attribute) and node.attr == "environ" and self._is_os_module(node.value)

    @staticmethod
    def _is_dynamic_namespace(node: ast.expr) -> bool:
        if isinstance(node, ast.Attribute):
            return node.attr == "__dict__"
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"globals", "locals", "vars"}
        )

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if identifier := _string_literal(node.slice):
            if identifier == SOC_VERSION_ENV_KEY and self._is_os_environ(node.value):
                self._add(node, "HDI002", "SOC_VERSION environment value")
            elif self._is_dynamic_namespace(node.value) and (code := _identifier_code(identifier)):
                self._add(node, code, identifier, dynamic=True)
        self.generic_visit(node)


def check_source(source: str, relative_path: str | Path) -> list[Violation]:
    source_path = _normalized_source_path(relative_path)
    if not _is_production_python(source_path) or source_path in ALLOWED_IDENTITY_FILES:
        return []

    try:
        tree = ast.parse(source, filename=source_path)
    except SyntaxError as exc:
        return [
            Violation(
                exc.lineno or 1,
                exc.offset or 1,
                "HDI000",
                f"cannot parse file for hardware identity checks: {exc.msg}",
            )
        ]

    visitor = HardwareIdentityVisitor(tree, source_path)
    visitor.visit(tree)
    return visitor.violations


def check_file(filepath: str | Path) -> list[str]:
    source_path = _normalized_repo_path(filepath)
    if source_path is None or not _is_production_python(source_path):
        return []

    try:
        source = Path(filepath).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"{source_path}:1:1: HDI000 cannot read file: {exc}"]

    return [violation.format(source_path) for violation in check_source(source, source_path)]


def main(argv: list[str] | None = None) -> int:
    paths: list[str | Path] = list(sys.argv[1:] if argv is None else argv)
    if not paths:
        paths = _default_python_paths()

    violations = [message for path in paths for message in check_file(path)]
    if violations:
        print("Raw hardware identity must stay inside the HAL boundary.\n")
        for violation in sorted(violations):
            print(f"  {violation}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
