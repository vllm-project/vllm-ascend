# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# This file is a part of the vllm-ascend project.

"""
Triton Kernel Dispatcher for multi-platform support (310P V2 shim).

This module mirrors ``vllm/model_executor/triton_dispatcher.py`` from upstream
vLLM PR #43048 ([Feature] Triton kernel dispatcher,
https://github.com/vllm-project/vllm/pull/43048). That PR is **not yet merged**
into the vLLM version pinned by this repository (v0.26.x), so the 310P Model
Runner V2 adaptation vendors a copy with the same semantics:

    1. In vLLM Ascend's own code, decorate the default Triton kernel:
       ```python
       from vllm_ascend.worker.v2.triton_dispatcher import pluggable_kernel

       @pluggable_kernel
       @triton.jit
       def my_kernel(...):
           ...
       ```
    2. On 310P, register a non-Triton implementation:
       ```python
       from vllm_ascend.worker.v2.triton_dispatcher import register_kernel

       @register_kernel("vllm_ascend.worker.v2.block_table._compute_slot_mappings_kernel")
       def my_kernel_impl(*args, grid=None, **kwargs):
           ...
       ```

Once upstream vLLM merges PR #43048 and this repository upgrades vLLM, delete
this module and change the two import lines above to
``from vllm.model_executor.triton_dispatcher import ...``. The call sites keep
the standard ``kernel[grid](...)`` syntax and need no other changes.
"""

from collections.abc import Callable
from typing import Any, Protocol

from vllm.logger import init_logger


class SubscriptableCallable(Protocol):
    """Protocol for callables that support subscript notation (e.g. Triton kernels)."""

    __name__: str

    def __getitem__(self, grid: Any) -> Callable: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


logger = init_logger(__name__)

# Global registry: { kernel_name: implementation_fn }
_kernel_registry: dict[str, Callable] = {}


def _get_kernel_impl(name: str) -> Callable | None:
    """Return the registered kernel implementation, or None to fall back to default."""
    return _kernel_registry.get(name)


class _KernelLauncher:
    """Holds the grid configuration and kernel implementation.

    Returned by ``KernelDispatcher.__getitem__(grid)`` for custom
    implementations. The grid is passed as a keyword argument, mirroring the
    upstream PR #43048 ABI.
    """

    __slots__ = ("_impl", "_grid")

    def __init__(self, impl: Callable, grid: Any):
        self._impl = impl
        self._grid = grid

    def __call__(self, *args, **kwargs):
        return self._impl(*args, grid=self._grid, **kwargs)


class KernelDispatcher:
    """Wraps a default Triton kernel and allows platform overrides.

    Maintains the standard ``kernel[grid](...)`` syntax so existing call sites
    keep working when a platform registers an implementation.
    """

    def __init__(self, name: str, default_impl: SubscriptableCallable):
        self.name = name
        self.default_impl = default_impl
        self.__name__ = default_impl.__name__
        self.__module__ = default_impl.__module__

    def __call__(self, *args, **kwargs):
        impl = _get_kernel_impl(self.name)
        if impl is not None:
            return impl(*args, **kwargs)
        return self.default_impl(*args, **kwargs)

    def __getitem__(self, grid):
        impl = _get_kernel_impl(self.name)
        if impl is not None:
            # Custom implementation registered: wrap in a launcher that passes
            # the grid as a keyword argument.
            return _KernelLauncher(impl, grid)
        # No custom implementation: fall back to the default Triton kernel,
        # which natively supports the [grid] syntax.
        return self.default_impl[grid]


def pluggable_kernel(jit_decorated_func: SubscriptableCallable) -> KernelDispatcher:
    """Decorator to automatically register a Triton kernel with the dispatcher.

    Apply **after** ``@triton.jit``:

        @pluggable_kernel
        @triton.jit
        def my_kernel(...):
            pass

    The fully qualified name (module + function name) is used as the registry
    key so that out-of-tree registrations can target it.
    """
    kernel_name = f"{jit_decorated_func.__module__}.{jit_decorated_func.__name__}"
    logger.debug("Auto-registered kernel %s", kernel_name)
    return KernelDispatcher(kernel_name, jit_decorated_func)


def register_kernel(name: str) -> Callable:
    """Decorator to register a platform-specific kernel implementation."""

    def decorator(func: Callable) -> Callable:
        if name in _kernel_registry:
            logger.warning("Kernel %s is already registered. Overwriting.", name)
        _kernel_registry[name] = func
        logger.debug("Registered kernel %s", name)
        return func

    return decorator
