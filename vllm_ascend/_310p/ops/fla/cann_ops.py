# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from functools import lru_cache
from importlib import import_module
from types import ModuleType


@lru_cache(maxsize=1)
def _load_cann_ops_transformer() -> ModuleType | None:
    """Load the optional extension once, including an unavailable result."""
    try:
        return import_module("cann_ops_transformer.ops")
    except (ImportError, OSError, RuntimeError):
        return None


@lru_cache(maxsize=2)
def get_gdn_op(name: str) -> Callable | None:
    """Resolve and cache each GDN operator independently."""
    ops = _load_cann_ops_transformer()
    return None if ops is None else getattr(ops, name, None)
