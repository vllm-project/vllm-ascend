# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Count ``@triton.jit`` kernels in the coverage.py data for a test run.

Triton compiles a kernel and launches it on device. The Python tracer never
steps through the kernel body, so those lines stay uncovered even when the
test launched the kernel. On each launch this plugin copies the kernel's
statement and branch arcs, including ``@triton.jit`` helpers it calls by name,
into the active CoverageData.

Device-side branch counters are not available, so one launch credits every
static branch in that kernel. The measured lines then match the kernel source
that the test actually compiled and launched. Kernels outside ``vllm_ascend``
are ignored.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import textwrap
import threading
from pathlib import PurePosixPath
from types import CodeType
from typing import Any

_LAUNCH_CLASS_NAMES = frozenset({"JITFunction", "Autotuner"})
_LAUNCH_METHODS = ("run", "__call__")
_TRITON_MODULES = frozenset({"triton.runtime.jit", "triton.runtime.autotuner"})

_lock = threading.Lock()
_arcs: dict[str, set[tuple[int, int]]] = {}
_recorded_ids: set[int] = set()
_parsed_arcs: dict[str, set[tuple[int, int]]] = {}
_warned = False


def coverage_init(_reg: Any, _options: Any) -> None:
    """coverage.py entry point declared by ``tests/coveragerc``."""
    install()


def install() -> None:
    """Install the launch hook and the coverage flush hook."""
    _install_flush_hook()
    _install_import_hook()
    _try_patch_loaded_triton()


def clear_recorded_state() -> None:
    """Drop kernels recorded in this process. Tests use this for isolation."""
    with _lock:
        _arcs.clear()
        _recorded_ids.clear()
        _parsed_arcs.clear()


def record_launched_kernel(kernel: object) -> None:
    """Remember the source lines of *kernel* and its static jit callees."""
    functions, non_project_ids = _functions_to_record(kernel)
    updates: list[tuple[str, set[tuple[int, int]]]] = []
    committed_ids: list[int] = []
    for function in functions:
        try:
            arcs = _arcs_for_function(function)
        except Exception as exc:
            _warn_once(exc)
            continue
        committed_ids.append(id(function))
        if arcs:
            updates.append((function.__code__.co_filename, arcs))
    if not committed_ids and not non_project_ids:
        return
    with _lock:
        _recorded_ids.update(committed_ids)
        _recorded_ids.update(non_project_ids)
        for filename, arcs in updates:
            _arcs.setdefault(filename, set()).update(arcs)


def _functions_to_record(kernel: object) -> tuple[list[Any], list[int]]:
    with _lock:
        already = set(_recorded_ids)
    found: list[Any] = []
    non_project_ids: list[int] = []
    stack = [kernel]
    seen_objects: set[int] = set()
    while stack:
        current = stack.pop()
        if id(current) in seen_objects:
            continue
        seen_objects.add(id(current))
        function = _python_function(current)
        if function is None:
            continue
        function_id = id(function)
        if function_id in already:
            continue
        already.add(function_id)
        filename = function.__code__.co_filename
        if not _is_project_file(filename):
            non_project_ids.append(function_id)
            continue
        found.append(function)
        stack.extend(_static_callees(function))
    return found, non_project_ids


def _python_function(kernel: object) -> Any | None:
    seen: set[int] = set()
    current: Any = kernel
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _is_jit_kernel(current):
            current = getattr(current, "fn", None)
            continue
        code = getattr(current, "__code__", None)
        if isinstance(code, CodeType):
            return current
        current = getattr(current, "fn", None)
    return None


def _is_jit_kernel(obj: object) -> bool:
    cls = type(obj)
    module = cls.__module__ or ""
    return cls.__name__ in _LAUNCH_CLASS_NAMES and module.startswith("triton")


def _is_project_file(filename: str) -> bool:
    if not filename or filename.startswith("<"):
        return False
    parts = PurePosixPath(filename.replace("\\", "/")).parts
    return "vllm_ascend" in parts


def _static_callees(function: Any) -> list[object]:
    try:
        source = textwrap.dedent(inspect.getsource(function))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return []
    callees: list[object] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = _resolve_call_target(function, node.func)
        if target is not None and _is_jit_kernel(target):
            callees.append(target)
    return callees


def _resolve_call_target(function: Any, node: ast.expr) -> object | None:
    if isinstance(node, ast.Name):
        return _lookup_name(function, node.id)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        base = _lookup_name(function, node.value.id)
        if base is not None:
            return getattr(base, node.attr, None)
    return None


def _lookup_name(function: Any, name: str) -> object | None:
    freevars = function.__code__.co_freevars
    closure = function.__closure__
    if closure is not None and name in freevars:
        try:
            return closure[freevars.index(name)].cell_contents
        except ValueError:
            return None
    return function.__globals__.get(name)


def _arcs_for_function(function: Any) -> set[tuple[int, int]]:
    filename = function.__code__.co_filename
    span = _function_span(function)
    parsed = _parsed_file_arcs(filename)
    if span is not None and parsed:
        start, end = span
        selected = {arc for arc in parsed if _arc_in_span(arc, start, end)}
        if selected:
            return selected
    return _linear_arcs(function)


def _function_span(function: Any) -> tuple[int, int] | None:
    try:
        lines, start = inspect.getsourcelines(function)
    except (OSError, TypeError):
        return None
    if not lines:
        return None
    return start, start + len(lines) - 1


def _arc_in_span(arc: tuple[int, int], start: int, end: int) -> bool:
    positive = [line for line in arc if line > 0]
    if not positive:
        return False
    return all(start <= line <= end for line in positive)


def _parsed_file_arcs(filename: str) -> set[tuple[int, int]]:
    cached = _parsed_arcs.get(filename)
    if cached is not None:
        return cached
    parsed: set[tuple[int, int]] = set()
    try:
        from coverage.parser import PythonParser

        parser = PythonParser(filename=filename)
        parser.parse_source()
        parsed = set(parser.arcs())
    except Exception:
        parsed = set()
    _parsed_arcs[filename] = parsed
    return parsed


def _linear_arcs(function: Any) -> set[tuple[int, int]]:
    code = function.__code__
    lines = sorted({lineno for _, _, lineno in code.co_lines() if lineno and lineno > 0})
    if not lines:
        return set()
    entry = -code.co_firstlineno
    arcs: set[tuple[int, int]] = set()
    previous = entry
    for line in lines:
        arcs.add((previous, line))
        previous = line
    arcs.add((previous, entry))
    return arcs


def _merge_into(collector: object) -> bool:
    covdata = getattr(collector, "covdata", None)
    if covdata is None:
        return False
    with _lock:
        arcs = {filename: set(pairs) for filename, pairs in _arcs.items() if pairs}
    if not arcs:
        return False
    if getattr(collector, "branch", True):
        covdata.add_arcs(arcs)
    else:
        lines = {
            filename: {line for pair in pairs for line in pair if line > 0} for filename, pairs in arcs.items()
        }
        covdata.add_lines(lines)
    return True


def _install_flush_hook() -> None:
    from coverage.collector import Collector

    current = Collector.flush_data
    if getattr(current, "_triton_coverage", False):
        return

    def flush_data(self: Collector, _original: Any = current) -> bool:
        had_data = _original(self)
        merged = False
        try:
            merged = _merge_into(self)
        except Exception as exc:
            _warn_once(exc)
        return bool(had_data or merged)

    flush_data._triton_coverage = True  # type: ignore[attr-defined]
    Collector.flush_data = flush_data  # type: ignore[method-assign]


class _RecordingLoader:
    """Run the real loader, then patch Triton launch methods it just defined."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)

    def exec_module(self, module: Any) -> None:
        self._loader.exec_module(module)
        try:
            _try_patch_loaded_triton()
        except Exception as exc:
            _warn_once(exc)


class _TritonLaunchFinder:
    """Wrap the Triton JIT modules so a later import still gets the launch hook."""

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        del path, target
        if fullname not in _TRITON_MODULES:
            return None
        try:
            sys.meta_path.remove(self)
        except ValueError:
            return None
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None or isinstance(spec.loader, _RecordingLoader):
            return spec
        spec.loader = _RecordingLoader(spec.loader)
        return spec


def _install_import_hook() -> None:
    if any(isinstance(finder, _TritonLaunchFinder) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _TritonLaunchFinder())


def _try_patch_loaded_triton() -> None:
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith("triton"):
            continue
        for class_name in _LAUNCH_CLASS_NAMES:
            cls = getattr(module, class_name, None)
            if isinstance(cls, type):
                _patch_launch_class(cls)


def _patch_launch_class(cls: type) -> None:
    for method_name in _LAUNCH_METHODS:
        if method_name not in cls.__dict__:
            continue
        original = cls.__dict__[method_name]
        if not callable(original) or getattr(original, "_triton_coverage", False):
            continue

        def launch(self: Any, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            try:
                record_launched_kernel(self)
            except Exception as exc:
                _warn_once(exc)
            return _original(self, *args, **kwargs)

        launch._triton_coverage = True  # type: ignore[attr-defined]
        setattr(cls, method_name, launch)


def _warn_once(exc: BaseException) -> None:
    global _warned
    if _warned:
        return
    _warned = True
    print(f"triton coverage: failed to record a kernel launch ({exc})", file=sys.stderr)
