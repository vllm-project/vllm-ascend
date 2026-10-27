# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed, worker-local resources for current FlashMLA / history-pack overlap."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

# The supported Ascend 950 FlashMLA MIX kernel uses two vector cores per cube.
FLASHMLA_VECTOR_PER_CUBE = 2
FLASHMLA_ATTENTION_CORE_DIVISOR = 2
FLASHMLA_MAX_PACK_ROWS_PER_CORE = 16


@dataclass(frozen=True)
class FlashMLACoreBudget:
    stream: torch.npu.Stream
    cube_cores: int
    vector_cores: int
    pack_vector_cores: int

    def allows_pack(self, rows: int) -> bool:
        # Bound work per available AIV, rather than a token count tied to one
        # DCP width or SKU. Large packs need their full grid to avoid regression.
        available_vector = self.vector_cores + self.pack_vector_cores
        return rows <= FLASHMLA_MAX_PACK_ROWS_PER_CORE * available_vector

    @property
    def cache_key(self) -> tuple[int, int, int]:
        return self.cube_cores, self.vector_cores, self.pack_vector_cores


def split_core_budget(cube: int, vector: int) -> tuple[int, int, int] | None:
    """Round to a legal MIX budget and leave at least one vector core for pack."""
    if cube < 1 or vector <= FLASHMLA_VECTOR_PER_CUBE:
        return None
    available_cube = min(cube, vector // FLASHMLA_VECTOR_PER_CUBE)
    attention_cube = max(1, available_cube // FLASHMLA_ATTENTION_CORE_DIVISOR)
    attention_cube = min(attention_cube, (vector - 1) // FLASHMLA_VECTOR_PER_CUBE)
    attention_vector = attention_cube * FLASHMLA_VECTOR_PER_CUBE
    return attention_cube, attention_vector, vector - attention_vector


def create_flashmla_core_budget(device: torch.device) -> FlashMLACoreBudget | None:
    """Run once before warmup/capture; never change a caller/shared stream.

    Limits are conservative ceilings, not reservations or a query for idle cores.
    A failed capability probe leaves the original execution path available.
    """
    from vllm_ascend.utils import is_950

    if not is_950():
        logger.warning("FlashMLA DCP core limits require Ascend 950; using unrestricted overlap")
        return None
    stream = None
    try:
        with torch.npu.device(device):
            index = torch.npu.current_device()
            props = torch.npu.get_device_properties(index)
            process = torch.npu.get_device_limit(index)
            parent = torch.npu.get_stream_limit(torch.npu.current_stream())
            # Do not expand an application's existing process or stream budget,
            # even on runtimes where a stream limit overrides a process limit.
            cubes = props.cube_core_num, process["cube_core_num"], parent["cube_core_num"]
            vectors = props.vector_core_num, process["vector_core_num"], parent["vector_core_num"]
            if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in (*cubes, *vectors)):
                raise ValueError("invalid reported core counts")
            cube, vector = min(cubes), min(vectors)
            allocation = split_core_budget(cube, vector)
            if allocation is None:
                raise ValueError("no room for both FlashMLA MIX and history pack")
            cube, vector, pack = allocation
            reset_limit = torch.npu.reset_stream_limit
            stream = torch.npu.Stream(device=device)
            torch.npu.set_stream_limit(stream, cube_num=cube, vector_num=vector)
            actual = torch.npu.get_stream_limit(stream)
            if actual["cube_core_num"] != cube or actual["vector_core_num"] != vector:
                raise ValueError(f"stream core limit readback differs: {actual}")
    except (AttributeError, RuntimeError, ValueError, TypeError, AssertionError, KeyError) as exc:
        if stream is not None:
            # A setter may have applied only one half of the MIX limit.
            # Do not leave a partially modified pooled stream behind.
            reset_limit(stream)
        logger.warning("FlashMLA DCP core limits unavailable; using unrestricted overlap: %s", exc)
        return None
    logger.info("FlashMLA DCP resources: attention AIC=%d AIV=%d, pack AIV<=%d", cube, vector, pack)
    return FlashMLACoreBudget(stream, cube, vector, pack)
