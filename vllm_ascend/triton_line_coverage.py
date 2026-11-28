#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Record lines that a Triton kernel actually executes during coverage runs.

``@triton.jit`` bodies are not Python bytecode, so ``coverage run`` never sees
them. When ``VLLM_ASCEND_TRITON_COVERAGE=1``, the first launch of a kernel
rewrites that kernel and the ``@triton.jit`` functions it calls. Each statement
stores ``1`` into a device buffer when that statement runs. ``constexpr``
branches that the compiler drops lose their stores with them. After the launch,
non-zero buffer slots are written to ``triton_lines.json`` beside ``COVERAGE_FILE``
for the precision selector.

Normal launches do not rewrite kernels. Set the variable only for coverage
collection.
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
import os
import tempfile
import textwrap
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENV_VAR = "VLLM_ASCEND_TRITON_COVERAGE"
COV_PTR = "_vllm_cov_ptr"
SIDECAR_NAME = "triton_lines.json"

_LOCK = threading.Lock()
_INSTALLED = False
_BUFFERS: dict[Any, Any] = {}
_ACCUMULATED: dict[str, set[int]] = {}


class LineTable:
    """Stable map from an executed probe id back to a source line."""

    def __init__(self) -> None:
        self._ids: dict[tuple[str, int], int] = {}
        self.entries: list[tuple[str, int, str]] = []

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, filename: str, lineno: int) -> int:
        relative = repo_relative_path(filename)
        key = (relative, lineno)
        existing = self._ids.get(key)
        if existing is not None:
            return existing
        index = len(self.entries)
        self._ids[key] = index
        self.entries.append((relative, lineno, filename.replace("\\", "/")))
        return index


_LINES = LineTable()


def repo_relative_path(filename: str) -> str:
    """Return the path under ``vllm_ascend/`` that the precision selector uses."""
    normalized = filename.replace("\\", "/")
    marker = "vllm_ascend/"
    index = normalized.rfind(marker)
    if index >= 0:
        return normalized[index + len(marker) :]
    return normalized


def lines_from_hits(
    values: list[int],
    entries: list[tuple[str, int, str]],
) -> dict[str, set[int]]:
    """Map a device buffer back to the source lines whose probes ran."""
    covered: dict[str, set[int]] = {}
    for index, value in enumerate(values):
        if not value or index >= len(entries):
            continue
        relative, lineno, _absolute = entries[index]
        covered.setdefault(relative, set()).add(lineno)
    return covered


def transform_source(
    source: str,
    first_lineno: int,
    resolve_jit: Any,
    line_table: LineTable,
    filename: str,
) -> str:
    """Insert execution probes and thread the coverage pointer into ``source``."""
    tree = ast.parse(source)
    rewriter = _Rewriter(first_lineno, resolve_jit, line_table, filename, _local_names(tree))
    rewritten = rewriter.visit(tree)
    ast.fix_missing_locations(rewritten)
    return ast.unparse(rewritten) + "\n"


def install() -> bool:
    """Patch ``JITFunction.run`` when coverage collection asked for kernel lines."""
    global _INSTALLED
    if os.environ.get(ENV_VAR) != "1":
        return False
    if _INSTALLED:
        return True
    try:
        from triton.runtime.jit import JITFunction
    except Exception:
        logger.warning("Triton line coverage disabled: triton.runtime.jit is unavailable")
        return False

    original_run = JITFunction.run
    if getattr(original_run, "_vllm_cov_wrapped", False):
        _INSTALLED = True
        return True

    def run(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _covered_run(self, original_run, args, kwargs)

    setattr(run, "_vllm_cov_wrapped", True)
    JITFunction.run = run
    _INSTALLED = True
    logger.info("Triton line coverage installed")
    return True


def pytest_configure(config: Any) -> None:
    """Pytest plugin entry. Coverage runs load this module with ``-p``."""
    del config
    install()


class _Rewriter(ast.NodeTransformer):
    def __init__(
        self,
        first_lineno: int,
        resolve_jit: Any,
        line_table: LineTable,
        filename: str,
        local_names: set[str],
    ) -> None:
        self._first_lineno = first_lineno
        self._resolve_jit = resolve_jit
        self._line_table = line_table
        self._filename = filename
        self._local_names = local_names
        self._rewritten_def = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if self._rewritten_def:
            return node
        self._rewritten_def = True
        node.decorator_list = []
        if not any(arg.arg == COV_PTR for arg in node.args.args):
            node.args.args.append(ast.arg(arg=COV_PTR))
        node.body = self._block(node.body)
        return node

    def visit_If(self, node: ast.If) -> ast.If:
        node.test = self.visit(node.test)
        node.body = self._block(node.body)
        node.orelse = self._block(node.orelse)
        return node

    def visit_For(self, node: ast.For) -> ast.For:
        node.target = self.visit(node.target)
        node.iter = self.visit(node.iter)
        node.body = self._block(node.body)
        node.orelse = self._block(node.orelse)
        return node

    def visit_While(self, node: ast.While) -> ast.While:
        node.test = self.visit(node.test)
        node.body = self._block(node.body)
        node.orelse = self._block(node.orelse)
        return node

    def visit_With(self, node: ast.With) -> ast.With:
        node.items = [self.visit(item) for item in node.items]
        node.body = self._block(node.body)
        return node

    def visit_Try(self, node: ast.Try) -> ast.Try:
        node.body = self._block(node.body)
        for handler in node.handlers:
            handler.body = self._block(handler.body)
        node.orelse = self._block(node.orelse)
        node.finalbody = self._block(node.finalbody)
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        node = self.generic_visit(node)
        if not _call_targets_jit(node.func, self._resolve_jit, self._local_names):
            return node
        if any(isinstance(arg, ast.Name) and arg.id == COV_PTR for arg in node.args):
            return node
        node.args.append(ast.Name(id=COV_PTR, ctx=ast.Load()))
        return node

    def _block(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        rewritten: list[ast.stmt] = []
        for statement in statements:
            if _is_docstring(statement):
                rewritten.append(statement)
                continue
            rewritten.append(self._store(self._first_lineno + statement.lineno - 1))
            rewritten.append(self.visit(statement))
        return rewritten

    def _store(self, lineno: int) -> ast.Expr:
        probe = self._line_table.add(self._filename, lineno)
        return ast.Expr(
            value=ast.Call(
                func=ast.Attribute(value=ast.Name(id="tl", ctx=ast.Load()), attr="store", ctx=ast.Load()),
                args=[
                    ast.BinOp(
                        left=ast.Name(id=COV_PTR, ctx=ast.Load()),
                        op=ast.Add(),
                        right=ast.Constant(probe),
                    ),
                    ast.Constant(1),
                ],
                keywords=[],
            )
        )


def _local_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                names.add(arg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def _is_docstring(statement: ast.stmt) -> bool:
    return isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(
        statement.value.value, str
    )


def _is_jit_function(value: Any) -> bool:
    return value.__class__.__name__ == "JITFunction" and hasattr(value, "fn") and hasattr(value, "src")


def _resolve_jit(func: ast.expr, globals_dict: dict[str, Any]) -> Any:
    if isinstance(func, ast.Name):
        value = globals_dict.get(func.id)
        return value if _is_jit_function(value) else None
    if not isinstance(func, ast.Attribute):
        return None
    base: Any = func.value
    parts = [func.attr]
    while isinstance(base, ast.Attribute):
        parts.append(base.attr)
        base = base.value
    if not isinstance(base, ast.Name):
        return None
    value = globals_dict.get(base.id)
    for part in reversed(parts):
        if value is None:
            return None
        value = getattr(value, part, None)
    return value if _is_jit_function(value) else None


def _call_targets_jit(func: ast.expr, resolve_jit: Any, local_names: set[str]) -> bool:
    if isinstance(func, ast.Name) and func.id in local_names:
        return False
    return bool(resolve_jit(func))


def _load_source(jit_fn: Any) -> tuple[str, int]:
    lines, first_lineno = inspect.getsourcelines(jit_fn.fn)
    source = textwrap.dedent("".join(lines))
    return source, first_lineno


def _collect(jit_fn: Any, seen: set[int]) -> list[Any]:
    if id(jit_fn) in seen:
        return []
    seen.add(id(jit_fn))
    if getattr(jit_fn, "_vllm_cov_state", None) == "on":
        return []
    source, _first_lineno = _load_source(jit_fn)
    tree = ast.parse(source)
    locals_in_fn = _local_names(tree)
    globals_dict = jit_fn.__globals__
    ordered: list[Any] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in locals_in_fn:
            continue
        callee = _resolve_jit(node.func, globals_dict)
        if callee is not None:
            ordered.extend(_collect(callee, seen))
    ordered.append(jit_fn)
    return ordered


def _rebind(jit_fn: Any, source: str) -> None:
    from triton.runtime.jit import KernelParam

    filename = _dump_source(jit_fn, source)
    namespace = dict(jit_fn.__globals__)
    preserved = {name: value for name, value in namespace.items() if _is_jit_function(value)}
    exec(compile(source, filename, "exec"), namespace)
    new_fn = namespace[jit_fn.__name__]
    namespace.update(preserved)
    namespace[jit_fn.__name__] = jit_fn
    new_fn.__module__ = jit_fn.__module__
    new_fn.__qualname__ = getattr(jit_fn, "__qualname__", jit_fn.__name__)

    jit_fn.fn = new_fn
    jit_fn.__globals__ = namespace
    jit_fn.signature = inspect.signature(new_fn)
    do_not_specialize = getattr(jit_fn, "do_not_specialize", []) or []
    do_not_specialize_on_alignment = getattr(jit_fn, "do_not_specialize_on_alignment", []) or []
    jit_fn.params = []
    for index, param in enumerate(jit_fn.signature.parameters.values()):
        skip_specialize = index in do_not_specialize or param.name in do_not_specialize
        skip_alignment = index in do_not_specialize_on_alignment or param.name in do_not_specialize_on_alignment
        jit_fn.params.append(KernelParam(index, param, skip_specialize, skip_alignment))
    jit_fn.arg_names = [param.name for param in jit_fn.params]
    jit_fn.constexprs = [param.num for param in jit_fn.params if param.is_constexpr]
    jit_fn.binder = None
    jit_fn.starting_line_number = 1
    for cache_name in ("cache", "device_caches"):
        cache = getattr(jit_fn, cache_name, None)
        if hasattr(cache, "clear"):
            cache.clear()
    jit_fn.src = source
    jit_fn._vllm_cov_state = "on"


def _dump_source(jit_fn: Any, source: str) -> str:
    directory = Path(tempfile.gettempdir()) / "vllm_ascend_triton_cov"
    directory.mkdir(parents=True, exist_ok=True)
    safe_module = str(jit_fn.__module__).replace("<", "_").replace(">", "_")
    safe_name = str(jit_fn.__name__).replace("<", "_").replace(">", "_")
    path = directory / f"{safe_module}.{safe_name}.{id(jit_fn)}.py"
    path.write_text(source, encoding="utf-8")
    return str(path)


def _instrument(jit_fn: Any) -> bool:
    if getattr(jit_fn, "_vllm_cov_state", None) == "on":
        return True
    if getattr(jit_fn, "_vllm_cov_state", None) == "failed":
        return False
    try:
        pending = _collect(jit_fn, set())
        rewritten: list[tuple[Any, str]] = []
        for item in pending:
            source, first_lineno = _load_source(item)
            globals_dict = item.__globals__

            def resolve(func: ast.expr, globals_for_item: dict[str, Any] = globals_dict) -> Any:
                return _resolve_jit(func, globals_for_item)

            rewritten.append(
                (
                    item,
                    transform_source(source, first_lineno, resolve, _LINES, item.fn.__code__.co_filename),
                )
            )
        for item, source in rewritten:
            _rebind(item, source)
    except Exception:
        logger.exception("Failed to instrument Triton kernel %s", getattr(jit_fn, "__name__", jit_fn))
        jit_fn._vllm_cov_state = "failed"
        return False
    return True


def _device_of(value: Any) -> Any:
    device = getattr(value, "device", None)
    if device is None or getattr(device, "type", None) in (None, "cpu"):
        return None
    return device


def _find_device(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    for value in (*args, *kwargs.values()):
        device = _device_of(value)
        if device is not None:
            return device
    try:
        import torch
    except Exception:
        return None
    npu = getattr(torch, "npu", None)
    if npu is None:
        return None
    try:
        if not npu.is_available():
            return None
        return torch.device("npu", npu.current_device())
    except Exception:
        return None


def _buffer(device: Any, size: int) -> Any:
    import torch

    current = _BUFFERS.get(device)
    if current is None or current.numel() < size:
        current = torch.zeros(size, dtype=torch.int32, device=device)
        _BUFFERS[device] = current
    else:
        current[:size].zero_()
    return current


def _sidecar_path() -> Path | None:
    coverage_file = os.environ.get("COVERAGE_FILE")
    if not coverage_file:
        return None
    return Path(coverage_file).parent / SIDECAR_NAME


def _flush_sidecar() -> None:
    path = _sidecar_path()
    if path is None or not _ACCUMULATED:
        return
    existing: dict[str, list[int]] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = {}
        if isinstance(loaded, dict):
            for filename, line_list in loaded.items():
                existing[str(filename)] = [int(line) for line in line_list]
    for filename, line_set in _ACCUMULATED.items():
        merged = set(existing.get(filename, []))
        merged.update(line_set)
        existing[filename] = sorted(merged)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(existing), encoding="utf-8")
    temporary.replace(path)


def _record(values: list[int], entries: list[tuple[str, int, str]]) -> None:
    covered = lines_from_hits(values, entries)
    if not covered:
        return
    for filename, line_set in covered.items():
        _ACCUMULATED.setdefault(filename, set()).update(line_set)
    _flush_sidecar()


def _covered_run(jit_fn: Any, original_run: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    with _LOCK:
        device = _find_device(args, kwargs)
        instrumented = getattr(jit_fn, "_vllm_cov_state", None) == "on"
        if not instrumented:
            if device is None or not _instrument(jit_fn):
                return original_run(jit_fn, *args, **kwargs)
        elif device is None:
            raise RuntimeError(f"{jit_fn.__name__} was instrumented but this launch has no device tensor")
        size = max(len(_LINES), 1)
        entries = list(_LINES.entries)
        buffer = _buffer(device, size)
        call_kwargs = dict(kwargs)
        call_kwargs[COV_PTR] = buffer
        try:
            return original_run(jit_fn, *args, **call_kwargs)
        finally:
            try:
                values = buffer[:size].detach().to("cpu").tolist()
                _record(values, entries)
            except Exception:
                logger.exception("Failed to record Triton line coverage for %s", jit_fn.__name__)


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    del session, exitstatus
    with _LOCK:
        _flush_sidecar()


if os.environ.get(ENV_VAR) == "1":
    install()
