# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM Ascend-owned metric handlers, one module per feature.

Shared building blocks live in :mod:`.common`; feature modules only depend on
them. Handler functions are resolved lazily from the feature submodules below
(PEP 562), so adding a feature only means dropping in ``handlers/<feature>.py``
plus its YAML: the reference
``vllm_ascend.observability.ms_metrics.handlers:function_name`` keeps working
without editing this file. The former static re-export list was a merge
hotspot every provider branch touched in the same lines, and it silently broke
whenever a feature module renamed a handler.
"""

import importlib
import pkgutil

# ``common`` holds shared helpers (metric specs, level gates), not handlers.
_NON_FEATURE_MODULES = frozenset({"common"})


def _feature_modules():
    """Yield the feature handler modules, importing them on first use."""
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name in _NON_FEATURE_MODULES or module_info.ispkg:
            continue
        yield importlib.import_module(f"{__name__}.{module_info.name}")


def __getattr__(name: str):
    """Resolve ``handlers:function_name`` references lazily.

    Only reached after the normal attribute lookup fails. A feature module is
    imported once (from ``sys.modules`` afterwards) and the first module
    defining ``name`` wins; handler names are unique across features by
    convention (each is prefixed with its feature name).
    """
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    for module in _feature_modules():
        value = getattr(module, name, None)
        if value is not None:
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    names = set(globals())
    for module in _feature_modules():
        names.update(key for key in vars(module) if key.endswith("_handler"))
    return sorted(names)
