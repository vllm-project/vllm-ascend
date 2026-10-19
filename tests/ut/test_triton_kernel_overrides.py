#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Unit tests for the NPU Triton kernel override registration table.

Deliberately plain pytest functions (no ``TestBase``): importing this module
must not trigger ``adapt_patch`` so the table is validated in isolation.
"""

import importlib
from typing import Any

import pytest

from vllm_ascend.triton_kernel_overrides import (
    _DEVICE_FUNCTION_SWAPS,
    TRITON_KERNEL_OVERRIDES,
    register_triton_kernels,
)

# The retired patch_v2/patch_triton.py carried 27 assignments; the
# dispatcher's identity sweep collapses the fan-out into 17 registrations
# plus one device-function swap (device functions are inlined by the
# Triton compiler and cannot be wrapped -- see _DEVICE_FUNCTION_SWAPS).
EXPECTED_OVERRIDE_COUNT = 17

# Device functions (helpers called inside a @triton.jit kernel body) are
# inlined by the Triton compiler, which resolves the callee from the
# defining module's globals and rejects the dispatcher's KernelOverride
# wrapper -- so they cannot go through register_kernels and stay manual.
# Dropping or adding an entry must be a conscious, audited change -- hence
# a named lock, same as EXPECTED_OVERRIDE_COUNT.
EXPECTED_DEVICE_FUNCTION_SWAP_COUNT = 1


def _resolve(key: str):
    """Resolve one table key the way the dispatcher does it.

    Returns (host, attr, defining_module). The last dotted segment is the
    attribute name; a module part ending in an uppercase name is a class
    mount (module + class), otherwise the whole part is the module.
    """
    module_part, attr = key.rsplit(".", 1)
    parts = module_part.split(".")
    if parts[-1][:1].isupper():
        module = importlib.import_module(".".join(parts[:-1]))
        return getattr(module, parts[-1]), attr, ".".join(parts[:-1])
    return importlib.import_module(module_part), attr, module_part


@pytest.mark.parametrize("key", sorted(TRITON_KERNEL_OVERRIDES))
def test_key_targets_the_definition_module(key):
    """Every key must point at the kernel's defining module.

    A pure re-export copy resolves to the same object and happens to sweep
    correctly, but a copy that wraps or rebinds does not: registration
    succeeds with no warning while the real kernel keeps running upstream.
    ``__module__`` is the machine-checkable definition site, so the rule
    holds for every key.
    """
    host, attr, defining_module = _resolve(key)
    original = getattr(host, attr)
    # Unwrap classmethod/bound-method shells so __module__ is comparable.
    func = getattr(original, "__func__", original)
    assert func.__module__ == defining_module, (
        f"{key!r} targets a copy site; the kernel is defined in "
        f"{func.__module__!r}. Point the key at the defining module."
    )


@pytest.mark.parametrize("key", sorted(TRITON_KERNEL_OVERRIDES))
def test_impl_is_callable(key):
    assert callable(TRITON_KERNEL_OVERRIDES[key])


def test_table_covers_the_retired_patch():
    assert len(TRITON_KERNEL_OVERRIDES) == EXPECTED_OVERRIDE_COUNT


def test_device_function_swaps_cover_the_retired_patch():
    assert len(_DEVICE_FUNCTION_SWAPS) == EXPECTED_DEVICE_FUNCTION_SWAP_COUNT


@pytest.mark.parametrize("key", sorted(_DEVICE_FUNCTION_SWAPS))
def test_device_function_swap_targets_the_definition_module(key):
    """Same rule as test_key_targets_the_definition_module, applied to the
    manual swap table: the swap must land in the defining module, because
    the Triton inliner resolves device-function callees from the defining
    module's globals."""
    host, attr, defining_module = _resolve(key)
    original = getattr(host, attr)
    # Unwrap classmethod/bound-method shells so __module__ is comparable.
    func = getattr(original, "__func__", original)
    assert func.__module__ == defining_module, (
        f"{key!r} targets a copy site; the function is defined in "
        f"{func.__module__!r}. Point the key at the defining module."
    )


def test_register_triton_kernels_hands_over_the_whole_table(monkeypatch):
    import vllm.triton_utils.dispatcher as dispatcher

    import vllm_ascend.triton_kernel_overrides as overrides

    seen = {}
    monkeypatch.setattr(dispatcher, "register_kernels", lambda table: seen.update(table))
    # Reset the idempotency latch: import_kernels may already have run in
    # this session (vllm calls it at module level from _custom_ops.py etc.).
    monkeypatch.setattr(overrides, "_kernels_registered", False)
    register_triton_kernels()
    assert seen == TRITON_KERNEL_OVERRIDES


def _original_from_logits(dummy):
    return dummy


class Mount:
    """Class-mount fixture for the no-implicit-cls tripwire below.

    Must start uppercase: the dispatcher treats an uppercase-leading
    segment in a key's module part as a class name, so an underscore
    prefix would be misparsed as a submodule path and fail to import.
    """

    # Type-checkers only: the tripwire below installs a KernelOverride
    # here at runtime, and a bare annotation declares the shape without
    # creating a runtime attribute.
    from_logits: Any


def test_class_mount_receives_no_implicit_cls():
    """Fallback tripwire for the one class-mount override.

    KernelOverride is not a descriptor, so a class-mount override must not
    receive an implicit cls/class argument at call time. The wrapper is
    constructed directly rather than via register_kernels: the tripwire
    targets call-time binding semantics, while register_kernels' identity
    sweep is environment-sensitive (llvmlite's ffi.lib answers "__dict__"
    lookups with a non-dict wrapper, crashing the sweep whenever
    numba/llvmlite is loaded in the session). If this ever fails against
    the CI vLLM, keep ``SamplingMaskTensors.from_logits`` as the only
    manual assignment and migrate the other 17 entries only.
    """
    from vllm.triton_utils.dispatcher import KernelOverride

    received = []

    def impl(*args):
        received.append(args)
        return args

    Mount.from_logits = KernelOverride(_original_from_logits, impl)
    Mount.from_logits(42)
    # Exactly the explicit argument: no implicit class/host argument.
    assert received == [(42,)]
