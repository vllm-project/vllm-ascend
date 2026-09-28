# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Targeted loader for the A5 operator wrappers and local PythonDSL sources.

The transformer wheel's public initializer eagerly discovers every bundled
operator and JIT-builds unrelated extensions.  We load only the requested
wrapper.  Its ``ops.*`` imports resolve to the 0923 DSL sources vendored under
``vllm_ascend.ops.pythondsl`` rather than the arena operator wheel.
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


def _load_payload_package():
    payload = importlib.import_module("vllm_ascend.ops.pythondsl")
    payload_path = os.path.dirname(payload.__file__)
    current = sys.modules.get("ops")
    if current is not None and payload_path in tuple(getattr(current, "__path__", ())):
        return current

    # cann_ops_transformer wrappers import ``ops.<dsl_module>`` at call time.
    # Point that namespace at our local sources without executing the arena
    # wheel's initializer or registering its precompiled native resources.
    for name in (
        "ops.mixed_quant_sparse_flash_mla",
        "ops.mixed_quant_sparse_flash_mla_metadata",
        "ops.quant_lightning_indexer_dsl",
        "ops.quant_lightning_indexer_metadata_dsl",
        "ops.quant_sparse_lightning_indexer_dsl",
        "ops.quant_sparse_lightning_indexer_metadata_dsl",
    ):
        sys.modules.pop(name, None)
    module = _namespace_package("ops", payload_path, payload.__file__)
    sys.modules["ops"] = module
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

        _load_payload_package()
        return importlib.import_module(module_name)
