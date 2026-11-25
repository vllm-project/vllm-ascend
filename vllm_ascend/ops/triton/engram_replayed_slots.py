# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v41/nvidia/model_state.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel and JIT warmup wrapper for V2 replayed slot padding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    triton_scalar_specialization_rep,
)
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel

_PAD_V2_REPLAYED_SLOTS_BLOCK = 1024


@triton.jit
def _pad_v2_replayed_slots_kernel(
    slot_mappings_ptr,
    group_stride,
    cacheable_groups_ptr,
    query_start_loc_ptr,
    positions_ptr,
    replay_start_ptr,
    replay_window,
    pad_slot_id,
    NUM_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    start = tl.load(replay_start_ptr + req)
    if start <= 0:
        return
    begin = tl.load(query_start_loc_ptr + req)
    end = tl.load(query_start_loc_ptr + req + 1)
    for tok in range(begin, end, BLOCK):
        offsets = tok + tl.arange(0, BLOCK)
        pos = tl.load(positions_ptr + offsets, mask=offsets < end, other=0)
        replayed = (offsets < end) & (pos >= start) & (pos < start + replay_window)
        for i in tl.static_range(NUM_GROUPS):
            group = tl.load(cacheable_groups_ptr + i)
            tl.store(
                slot_mappings_ptr + group * group_stride + offsets,
                pad_slot_id,
                mask=replayed,
            )


class PadV2ReplayedSlotsKernel(AscendTritonJitKernel["PadV2ReplayedSlotsKernel.CompileKey"]):
    """JIT warmup wrapper for ``_pad_v2_replayed_slots_kernel``.

    The kernel only launches on requests whose prefix-cache hit triggers SWA
    bounded prefix replay, so startup dummy runs never compile it. Registering
    the deployment's replay geometry in ``kernel_warmup`` removes the cold-start
    JIT compilation stall on the first prefix-cache hit.
    """

    kernel = _pad_v2_replayed_slots_kernel

    @dataclass(frozen=True)
    class CompileKey:
        slot_mappings_dtype: torch.dtype
        group_stride: int
        positions_dtype: torch.dtype
        replay_window: int
        num_groups: int
        block: int = _PAD_V2_REPLAYED_SLOTS_BLOCK

    def dispatch(
        self,
        *,
        slot_mappings_dtype: torch.dtype,
        group_stride: int,
        positions_dtype: torch.dtype,
        replay_window: int,
        num_groups: int,
        block: int = _PAD_V2_REPLAYED_SLOTS_BLOCK,
    ) -> PadV2ReplayedSlotsKernel.CompileKey:
        return self.CompileKey(
            slot_mappings_dtype=slot_mappings_dtype,
            group_stride=triton_scalar_specialization_rep(int(group_stride)),
            positions_dtype=positions_dtype,
            replay_window=triton_scalar_specialization_rep(int(replay_window)),
            num_groups=int(num_groups),
            block=int(block),
        )

    def get_warmup_keys(
        self,
        *,
        slot_mappings_dtype: torch.dtype,
        group_stride: int,
        positions_dtype: torch.dtype,
        replay_window: int,
        num_groups: int,
        block: int = _PAD_V2_REPLAYED_SLOTS_BLOCK,
    ) -> list[PadV2ReplayedSlotsKernel.CompileKey]:
        return self._trace_dispatch(self.dispatch)(
            slot_mappings_dtype=slot_mappings_dtype,
            group_stride=int(group_stride),
            positions_dtype=positions_dtype,
            replay_window=int(replay_window),
            num_groups=int(num_groups),
            block=int(block),
        )

    def warmup_inputs(self, compile_key: PadV2ReplayedSlotsKernel.CompileKey) -> dict[str, Any]:
        return dict(
            slot_mappings=TritonWarmupTensor(
                compile_key.slot_mappings_dtype,
                shape=(1, 1),
                strides=(compile_key.group_stride, 1),
            ),
            cacheable_groups=TritonWarmupTensor(torch.int32, shape=(compile_key.num_groups,)),
            query_start_loc=TritonWarmupTensor(torch.int32, shape=(2,)),
            positions=TritonWarmupTensor(compile_key.positions_dtype, shape=(1,)),
            replay_start=TritonWarmupTensor(torch.int32, shape=(1,)),
            replay_window=compile_key.replay_window,
            pad_slot_id=PAD_SLOT_ID,
            NUM_GROUPS=compile_key.num_groups,
            BLOCK=compile_key.block,
            num_reqs=1,
        )

    @kernel_launcher
    def __call__(
        self,
        slot_mappings: torch.Tensor,
        cacheable_groups: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        replay_start: torch.Tensor,
        *,
        replay_window: int,
        pad_slot_id: int,
        NUM_GROUPS: int,
        BLOCK: int,
        num_reqs: int,
        kernel: Any = None,
    ) -> LaunchSpec:
        return (num_reqs,), dict(
            kernel=_pad_v2_replayed_slots_kernel if kernel is None else kernel,
            slot_mappings_ptr=slot_mappings,
            group_stride=slot_mappings.stride(0),
            cacheable_groups_ptr=cacheable_groups,
            query_start_loc_ptr=query_start_loc,
            positions_ptr=positions,
            replay_start_ptr=replay_start,
            replay_window=replay_window,
            pad_slot_id=pad_slot_id,
            NUM_GROUPS=NUM_GROUPS,
            BLOCK=BLOCK,
        )


_PAD_V2_REPLAYED_SLOTS_KERNEL = PadV2ReplayedSlotsKernel()


def register_pad_v2_replayed_slots_warmup(
    *,
    slot_mappings_dtype: torch.dtype,
    group_stride: int,
    positions_dtype: torch.dtype,
    replay_window: int,
    num_groups: int,
) -> bool:
    """Register ``_pad_v2_replayed_slots_kernel`` for one replay geometry.

    No-op outside an active ``JitWarmupRegistry.activate()`` window.
    """
    if not HAS_TRITON or replay_window <= 0 or num_groups <= 0 or group_stride <= 0:
        return False
    _PAD_V2_REPLAYED_SLOTS_KERNEL.register_warmup(
        slot_mappings_dtype=slot_mappings_dtype,
        group_stride=group_stride,
        positions_dtype=positions_dtype,
        replay_window=replay_window,
        num_groups=num_groups,
    )
    return True
