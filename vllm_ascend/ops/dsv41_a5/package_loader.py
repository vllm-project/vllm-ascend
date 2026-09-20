# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Targeted loader for the prebuilt A5 operator wheel.

The wheel's public package initializer eagerly discovers every bundled
operator.  Most of those operators own C++ glue sources, so importing one
Python-DSL A5 operator unexpectedly JIT-builds unrelated extensions.  The A5
adapters need only the explicitly named DSL modules; create lightweight
package namespaces and let Python execute that leaf module directly.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.machinery
import importlib.util
import os
import sys
import threading
import types
from importlib import metadata

_import_lock = threading.Lock()
_opapi_handle = None


def _prepend_env_path(name: str, path: str) -> None:
    entries = [entry for entry in os.environ.get(name, "").split(":") if entry]
    if path not in entries:
        os.environ[name] = ":".join((path, *entries))


def _bootstrap_packaged_op_runtime() -> None:
    """Expose the installed A5 OPP vendors and op-api symbols to this process.

    The standalone operator packages are installed under CANN rather than the
    vLLM wheel.  Some container entrypoints do not source the vendor set-env
    scripts, which otherwise presents as a misleading "operator not found"
    failure.  ``LD_PRELOAD=.../libopapi_nn.so`` remains the preferred launch
    setting; RTLD_GLOBAL here makes targeted/eager use deterministic as well.
    """
    global _opapi_handle

    homes = []
    for variable in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        value = os.environ.get(variable)
        if value:
            homes.append(value)
    homes.append("/usr/local/Ascend/ascend-toolkit/latest")

    for home in dict.fromkeys(homes):
        opp_root = os.path.join(home, "opp")
        for vendor in ("custom_transformer", "customize"):
            vendor_path = os.path.join(opp_root, "vendors", vendor)
            if os.path.isdir(vendor_path):
                _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_path)
        if _opapi_handle is None:
            opapi_path = os.path.join(home, "lib64", "libopapi_nn.so")
            if os.path.isfile(opapi_path):
                _opapi_handle = ctypes.CDLL(opapi_path, mode=getattr(ctypes, "RTLD_GLOBAL", 0))


def _namespace_package(name: str, path: str, origin: str | None = None):
    module = types.ModuleType(name)
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.origin = origin
    spec.submodule_search_locations = [path]
    module.__package__ = name
    module.__path__ = [path]
    module.__spec__ = spec
    return module


def _load_payload_package(package_path: str):
    payload_path = os.path.join(os.path.dirname(package_path), "ops")
    payload_init = os.path.join(payload_path, "__init__.py")
    if not os.path.isfile(payload_init):
        # The transformer wheel can live in CANN's Python directory while the
        # DSL payload is provided by a regular site-packages distribution.
        # Resolve the distribution that owns ``ops`` instead of accepting an
        # unrelated module with the same generic top-level name.
        for distribution_name in metadata.packages_distributions().get("ops", ()):
            candidate = metadata.distribution(distribution_name).locate_file("ops")
            candidate_path = os.fspath(candidate)
            candidate_init = os.path.join(candidate_path, "__init__.py")
            if os.path.isfile(candidate_init):
                payload_path = candidate_path
                payload_init = candidate_init
                break
    if not os.path.isfile(payload_init):
        raise ModuleNotFoundError(f"A5 operator payload package is absent: {payload_init}")
    current = sys.modules.get("ops")
    if current is not None and payload_path in tuple(getattr(current, "__path__", ())):
        return current
    spec = importlib.util.spec_from_file_location("ops", payload_init, submodule_search_locations=[payload_path])
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"A5 operator payload package is absent: {payload_init}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("ops")
    sys.modules["ops"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop("ops", None)
        else:
            sys.modules["ops"] = previous
        raise
    return module


def import_packaged_a5_module(module_name: str):
    """Import one ``cann_ops_transformer`` leaf without all-op discovery."""
    package = "cann_ops_transformer"
    if not module_name.startswith(f"{package}.ops."):
        raise ValueError(f"Not a packaged A5 operator module: {module_name}")

    with _import_lock:
        _bootstrap_packaged_op_runtime()
        if package not in sys.modules:
            spec = importlib.util.find_spec(package)
            if spec is None or not spec.submodule_search_locations:
                raise ModuleNotFoundError(package)
            package_path = os.fspath(next(iter(spec.submodule_search_locations)))
            sys.modules[package] = _namespace_package(package, package_path, spec.origin)
        else:
            package_path = os.fspath(next(iter(sys.modules[package].__path__)))

        ops_name = f"{package}.ops"
        if ops_name not in sys.modules:
            ops_path = os.path.join(package_path, "ops")
            sys.modules[ops_name] = _namespace_package(ops_name, ops_path)

        # The .run package installs the AOT/DSL payload as a separate top-level
        # ``ops`` resource package.  Load it explicitly from the wheel sibling
        # path: test runners and applications can themselves own a module named
        # ``ops``, which must not silently shadow this ABI dependency.
        _load_payload_package(package_path)
        return importlib.import_module(module_name)
