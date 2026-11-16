# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Capability-gated host registration for the native CPU primary tier."""

from typing import Any

import torch
from vllm.logger import logger

_PINNED_CAPABILITY = 1
_SDMA_CAPABILITY = 2
_DRAM_BACKENDS = ("auto", "memcpy", "sdma")


class RegisteredDram:
    def __init__(self, memory: torch.Tensor, use_sdma: bool) -> None:
        self.memory = torch.classes._C_ascend.KVOffloadHostMemory(memory, use_sdma)
        self.use_sdma = use_sdma

    def prepare(self, src: torch.Tensor, dst: torch.Tensor, sizes: torch.Tensor, direction: int) -> Any:
        if not self.use_sdma:
            return None
        return torch.classes._C_ascend.KVOffloadSdmaBatch(self.memory, src, dst, sizes, direction)

    def close(self) -> None:
        self.memory.close()


def register_dram(region: Any, backend: str) -> RegisteredDram | None:
    if backend not in _DRAM_BACKENDS:
        raise ValueError(f"dram_transfer_backend must be one of {_DRAM_BACKENDS}, got {backend!r}")
    capability_op = getattr(torch.ops._C_ascend, "kv_offload_dram_capabilities", None)
    capabilities = int(capability_op()) if capability_op is not None else 0
    use_sdma = backend != "memcpy" and bool(capabilities & _SDMA_CAPABILITY)
    if backend == "sdma" and not use_sdma:
        raise RuntimeError("dram_transfer_backend=sdma requires an A3 build with FFTS and host-mapping support")
    if use_sdma:
        try:
            registered = RegisteredDram(region._base, use_sdma=True)
            logger.info("Tiering DRAM transfers use mapped host memory and FFTS SDMA")
            return registered
        except RuntimeError:
            if backend == "sdma":
                raise
            logger.warning("Tiering DRAM SDMA registration failed; trying the memcpy backend", exc_info=True)
    if capabilities & _PINNED_CAPABILITY:
        try:
            registered = RegisteredDram(region._base, use_sdma=False)
            logger.info("Tiering DRAM transfers use pinned mmap memory and memcpy")
            return registered
        except RuntimeError:
            logger.warning("Tiering DRAM pinning failed; using the original mmap memcpy path", exc_info=True)
    else:
        logger.info("Tiering DRAM transfers use mmap memcpy; host pinning is unavailable in this build")
    return None
