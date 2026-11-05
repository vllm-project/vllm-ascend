# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Scoped adapters for upstream suffix launches, which have no instance hooks.

The lock lives for the worker process and serializes Ascend proposal/ingestion
entry points because the replaced names belong to an upstream Python module.
Direct upstream suffix calls must not run concurrently in that same process.
Bindings can be restored as soon as launches have been submitted; this does not
wait for device completion. Remove these overrides when upstream provides
overridable launch methods or a device kernel provider.
"""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from threading import RLock
from types import ModuleType

from vllm_ascend.ops.triton.spec_decode.suffix import (
    launch_suffix_finalize,
    launch_suffix_ingest,
    launch_suffix_local_scan,
    launch_suffix_observe,
)

# Shared only for the lifetime of this worker process. An RLock also permits
# nested overrides to restore the immediately preceding bindings correctly.
_SUFFIX_KERNEL_LOCK = RLock()


@dataclass(frozen=True)
class _KernelLaunchAdapter:
    """Expose a host launch function through the upstream kernel[grid](...) ABI."""

    launch: Callable

    def __getitem__(self, grid: tuple[int, ...]) -> Callable:
        return partial(self.launch, grid)


@contextmanager
def _override_suffix_kernels(module: ModuleType, launches: Mapping[str, Callable]) -> Iterator[None]:
    with _SUFFIX_KERNEL_LOCK:
        originals = {name: getattr(module, name) for name in launches}
        try:
            for name, launch in launches.items():
                setattr(module, name, _KernelLaunchAdapter(launch))
            yield
        finally:
            for name, original in originals.items():
                setattr(module, name, original)


@contextmanager
def ascend_suffix_proposal(module: ModuleType) -> Iterator[None]:
    with _override_suffix_kernels(
        module,
        {
            "_suffix_local_scan_kernel": launch_suffix_local_scan,
            "_suffix_finalize_kernel": launch_suffix_finalize,
            "_suffix_observe_kernel": launch_suffix_observe,
        },
    ):
        yield


@contextmanager
def ascend_suffix_ingestion(module: ModuleType) -> Iterator[None]:
    with _override_suffix_kernels(module, {"_suffix_ingest_kernel": launch_suffix_ingest}):
        yield
