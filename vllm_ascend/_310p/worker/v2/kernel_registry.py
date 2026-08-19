# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Optional bridge between 310P and the upstream Triton kernel dispatcher.

The dispatcher proposed by https://github.com/vllm-project/vllm/pull/43048
(``@pluggable_kernel`` / ``register_kernel``) is not part of vLLM main, so no
310P code path may depend on it. Every Triton-backed step the 310P V2 runner
reaches is replaced at class or module level instead:

* ``Ascend310PBlockTables`` computes gather and slot mapping in NumPy;
* ``Ascend310PRequestState`` owns staged writes on CPU;
* ``Ascend310PRopeState`` builds multimodal positions on CPU;
* ``Ascend310PGreedySampler`` avoids the upstream sampling kernels.

This module keeps the dispatcher path alive for the day that PR lands and the
relevant upstream kernel is marked pluggable: add its fully qualified name and
310P implementation to ``KERNEL_IMPLS``, then the override takes effect at the
shared ``kernel[grid](...)`` call site.
"""

from __future__ import annotations

from collections.abc import Callable

from vllm.logger import logger

try:
    from vllm.model_executor.triton_dispatcher import register_kernel

    HAS_TRITON_DISPATCHER = True
except ImportError:
    HAS_TRITON_DISPATCHER = False

# Fully qualified pluggable kernel name -> 310P implementation. Empty for the
# first release: none of the kernels the 310P V2 runner reaches are decorated
# with ``@pluggable_kernel``, upstream or in vLLM Ascend.
KERNEL_IMPLS: dict[str, Callable] = {}


def register_310p_kernels() -> tuple[str, ...]:
    """Register the 310P kernel implementations, if the dispatcher exists.

    Returns the registered kernel names, empty when the dispatcher is missing
    or when nothing needs to be overridden through it.
    """
    if not KERNEL_IMPLS:
        return ()
    if not HAS_TRITON_DISPATCHER:
        logger.debug(
            "vLLM has no Triton kernel dispatcher; 310P keeps its class-level Triton-free implementations."
        )
        return ()

    for kernel_name, impl in KERNEL_IMPLS.items():
        register_kernel(kernel_name)(impl)
    logger.info("Registered %d 310P kernel implementations with the Triton kernel dispatcher.", len(KERNEL_IMPLS))
    return tuple(KERNEL_IMPLS)
