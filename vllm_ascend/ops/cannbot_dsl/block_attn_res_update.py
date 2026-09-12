# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""K3 Phase-2 ``block_attn_res_update`` partial update and online-softmax merge.

This operator is the fused CANNDSL counterpart of K3's
``_update_attn_res_online_softmax`` helper.  For one caller-sliced slot it:

1. computes ``updated_partial = partial_block + partial_delta`` in fp32;
2. computes the RMS-normalized partial score with the selected effective query;
3. merges that one-candidate partial state with the Phase-1 inter-block state;
4. returns both the final normalized output and ``updated_partial``.

The Phase-1 state uses the paper's stable representation ``(o_tilde, m, l)``.
For the partial candidate, ``m2=score``, ``l2=1`` and ``o2=partial_block``.
"""

from dataclasses import dataclass
from functools import lru_cache

import cannbotdsl
import torch

from cannbotdsl import dtypes
from cannbotdsl.arch import get_block_idx, get_block_num
from cannbotdsl.channel import Channel
from cannbotdsl.core.frontend.compiler import compile_function
from cannbotdsl.dtypes import (
    bfloat16 as BFloat16,
    float16 as Float16,
    float32 as Float32,
)
from cannbotdsl.integer import Int64
from cannbotdsl.jit_runner import jit
from cannbotdsl.kernel_launcher import kernel
from cannbotdsl.raw_reg import (
    PackMode,
    UnpackMode,
    full_mask,
    update_mask,
    vadd,
    vadds,
    vcast,
    vdiv,
    vdup_lane0,
    vdup_scalar,
    vexp_sub,
    vload,
    vload_brc,
    vload_unpack,
    vmax,
    vmem_bar,
    vmul,
    vmuls,
    vreduce_sum,
    vsqrt,
    vstore,
    vstore_pack,
)
from cannbotdsl.tensor import local_slice, mem_copy, tile_view
from cannbotdsl.typing.types import MemLoc, Tensor
from cannbotdsl.vf import vf


VL = 64
DEFAULT_BLOCK_NUM = 64
MAX_D = 8192

_TORCH_TO_DSL = {
    torch.bfloat16: BFloat16,
    torch.float16: Float16,
}
_COMPILED_KERNEL_CACHE: dict[tuple[object, ...], object] = {}
UB_BYTES = 240 * 1024


@dataclass(frozen=True, slots=True)
class UpdateStaticShape:
    """Static dimensions and dtype shared by launch planning and compilation."""

    tokens: int
    d: int
    delta_dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class UpdateLaunchPlan:
    """Named compile/launch configuration for one static Update shape."""

    shape: UpdateStaticShape
    block_num: int
    pipeline_rows: bool
    epsilon: float


@dataclass(frozen=True, slots=True)
class UpdateInputs:
    """Related whole-network inputs validated and launched as one named group."""

    partial_block: torch.Tensor
    partial_delta: torch.Tensor
    effective_query: torch.Tensor
    inter_max: torch.Tensor
    inter_exp_sum: torch.Tensor
    inter_numerator: torch.Tensor
    epsilon: float


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _require(condition: bool, message: str) -> None:
    """Raise a stable input-contract error."""
    if not condition:
        raise ValueError(message)


def _use_row_pipeline(tokens: int, d: int, block_num: int) -> bool:
    """Whether two input rows fit in UB and each core owns useful next work."""
    w = _ceil_div(int(d), VL) * VL
    # depth-1: four fp32 vectors + one 16-bit delta + one 16-bit h output.
    single_bytes = 20 * w + VL * 4
    # The pipeline adds a second partial/delta/inter/stats input slot.
    pipelined_bytes = single_bytes + 10 * w + VL * 4
    return int(tokens) > int(block_num) and pipelined_bytes <= UB_BYTES


def _launch_block_num(tokens: int) -> int:
    """Update owns token rows, so launching more blocks cannot add work."""
    return max(1, min(DEFAULT_BLOCK_NUM, int(tokens)))


# This VF callback is expanded by ``compile_function``. Its Buffer/scalar
# operands must remain flattened because dataclasses cannot cross the DSL
# kernel-region ABI boundary.
# pylint: disable=too-many-arguments,too-many-positional-arguments
def _block_attn_res_update_body(
        partial_ub, delta_ub, inter_ub, query_ub,
        stats_ub, partial_out_ub, h_out_ub,
        num_seg, d, avg, epsilon, h_dtype):
    """Compute in fp32, then cast the normalized h output to delta dtype."""
    with vf(mode="raw"):
        full = full_mask(32)

        for seg in range(Int64(0), Int64(num_seg)):
            off = seg * VL
            mask, _ = update_mask(Int64(d) - off, elem_bits=32)
            partial = vload(partial_ub, off)
            delta = vcast(
                vload_unpack(delta_ub, off, mode=UnpackMode.B16_TO_B32),
                Float32, mask=mask,
            )
            updated = vadd(partial, delta, mask=mask)
            vstore(partial_out_ub, off, updated, mask)
        vmem_bar(mode="vst_vld")

        dot = vdup_scalar(0.0, Float32, mask=full)
        ssq = vdup_scalar(0.0, Float32, mask=full)
        for seg in range(Int64(0), Int64(num_seg)):
            off = seg * VL
            mask, _ = update_mask(Int64(d) - off, elem_bits=32)
            partial = vload(partial_out_ub, off)
            query = vload(query_ub, off)
            dot = vadd(dot, vmul(partial, query, mask=mask), mask=full)
            ssq = vadd(ssq, vmul(partial, partial, mask=mask), mask=full)

        denom = vsqrt(
            vadds(
                vmuls(vreduce_sum(ssq, mask=full), avg, mask=full),
                epsilon, mask=full,
            ),
            mask=full,
        )
        partial_score = vdup_lane0(
            vdiv(vreduce_sum(dot, mask=full), denom, mask=full),
            mask=full,
        )

        inter_max = vload_brc(stats_ub, 0)
        inter_sum = vload_brc(stats_ub, 8)
        merged_max = vmax(inter_max, partial_score, mask=full)
        history_scale = vexp_sub(inter_max, merged_max, mask=full)
        partial_scale = vexp_sub(partial_score, merged_max, mask=full)
        merged_sum = vadd(
            vmul(history_scale, inter_sum, mask=full),
            partial_scale,
            mask=full,
        )
        # merged_sum is a scalar broadcast across D.  Compute its reciprocal
        # once and reuse it for every segment instead of issuing one vector
        # divide per segment.
        one = vdup_scalar(1.0, Float32, mask=full)
        inv_merged_sum = vdiv(one, merged_sum, mask=full)

        for seg in range(Int64(0), Int64(num_seg)):
            off = seg * VL
            mask, _ = update_mask(Int64(d) - off, elem_bits=32)
            partial = vload(partial_out_ub, off)
            inter = vload(inter_ub, off)
            numerator = vadd(
                vmul(inter, history_scale, mask=mask),
                vmul(partial, partial_scale, mask=mask),
                mask=mask,
            )
            merged = vmul(numerator, inv_merged_sum, mask=mask)
            vstore_pack(
                h_out_ub,
                off,
                vcast(merged, h_dtype, mask=mask),
                mask,
                mode=PackMode.B32_TO_B16,
            )
        vmem_bar(mode="vst_vld")


_block_attn_res_update_body = compile_function(
    _block_attn_res_update_body, enable_preprocessor=True,
).function
# pylint: enable=too-many-arguments,too-many-positional-arguments


# The vector object is instantiated inside a DSL kernel, so Buffer handles and
# GM operands remain explicit primitives rather than Python containers.
# pylint: disable=too-many-arguments,too-many-positional-arguments
class BlockAttnResUpdateVector:
    """Per-core UB state for the fused Phase-2 update."""

    def __init__(self, d: int, delta_dtype=Float16,
                 pipeline_rows: bool = False):
        self.d = int(d)
        self.delta_dtype = delta_dtype
        self.w = _ceil_div(self.d, VL) * VL
        self.num_seg = _ceil_div(self.d, VL)
        self.avg = 1.0 / self.d
        self.pipeline_rows = bool(pipeline_rows)
        input_depth = 2 if self.pipeline_rows else 1

        self.partial_ub = Channel(
            MemLoc.UB, shape=(1, self.w), dtype=Float32,
            depth=input_depth,
        )
        self.delta_ub = Channel(
            MemLoc.UB, shape=(1, self.w), dtype=delta_dtype,
            depth=input_depth,
        )
        self.inter_ub = Channel(
            MemLoc.UB, shape=(1, self.w), dtype=Float32,
            depth=input_depth,
        )
        self.query_ub = Channel(
            MemLoc.UB, shape=(1, self.w), dtype=Float32, depth=1,
        )
        # m and l occupy separate 32-byte aligned regions.
        self.stats_ub = Channel(
            MemLoc.UB, shape=(1, VL), dtype=Float32,
            depth=input_depth,
        )
        self.partial_out_ub = Channel(
            MemLoc.UB, shape=(1, self.w), dtype=Float32, depth=1,
        )
        self.h_out_ub = Channel(
            MemLoc.UB, shape=(1, self.w), dtype=delta_dtype, depth=1,
        )
    def load_query(self, gm_effective_query):
        slot = self.query_ub.acquire()
        mem_copy(
            local_slice(slot, (1, self.d), offset=0),
            tile_view(
                gm_effective_query, (1, self.d),
                (Int64(0), Int64(0)),
            ),
        )
        self.query_ub.commit(slot)

    def load_row(self, gm_partial, gm_delta, gm_inter_numerator,
                 gm_inter_max, gm_inter_sum, row):
        partial_slot = self.partial_ub.acquire()
        mem_copy(
            local_slice(partial_slot, (1, self.d), offset=0),
            tile_view(gm_partial, (1, self.d), (row, Int64(0))),
        )
        self.partial_ub.commit(partial_slot)

        delta_slot = self.delta_ub.acquire()
        mem_copy(
            local_slice(delta_slot, (1, self.d), offset=0),
            tile_view(gm_delta, (1, self.d), (row, Int64(0))),
        )
        self.delta_ub.commit(delta_slot)

        inter_slot = self.inter_ub.acquire()
        mem_copy(
            local_slice(inter_slot, (1, self.d), offset=0),
            tile_view(
                gm_inter_numerator, (1, self.d),
                (row, Int64(0)),
            ),
        )
        self.inter_ub.commit(inter_slot)

        stats_slot = self.stats_ub.acquire()
        mem_copy(
            local_slice(stats_slot, (1, 1), offset=0),
            tile_view(gm_inter_max, (1, 1), (row, Int64(0))),
        )
        mem_copy(
            local_slice(stats_slot, (1, 1), offset=32),
            tile_view(gm_inter_sum, (1, 1), (row, Int64(0))),
        )
        self.stats_ub.commit(stats_slot)

    def compute_row(self, query_cur, epsilon):
        partial_cur = self.partial_ub.wait()
        delta_cur = self.delta_ub.wait()
        inter_cur = self.inter_ub.wait()
        stats_cur = self.stats_ub.wait()
        partial_out_slot = self.partial_out_ub.acquire()
        h_out_slot = self.h_out_ub.acquire()

        _block_attn_res_update_body(
            partial_cur, delta_cur, inter_cur, query_cur,
            stats_cur, partial_out_slot, h_out_slot,
            self.num_seg, self.d, self.avg, epsilon, self.delta_dtype,
        )

        self.partial_ub.release(partial_cur)
        self.delta_ub.release(delta_cur)
        self.inter_ub.release(inter_cur)
        self.stats_ub.release(stats_cur)
        self.partial_out_ub.commit(partial_out_slot)
        self.h_out_ub.commit(h_out_slot)

    def store_row(self, gm_h, gm_partial_out, row):
        partial_cur = self.partial_out_ub.wait()
        mem_copy(
            tile_view(gm_partial_out, (1, self.d), (row, Int64(0))),
            local_slice(partial_cur, (1, self.d), offset=0),
        )
        self.partial_out_ub.release(partial_cur)

        h_cur = self.h_out_ub.wait()
        mem_copy(
            tile_view(gm_h, (1, self.d), (row, Int64(0))),
            local_slice(h_cur, (1, self.d), offset=0),
        )
        self.h_out_ub.release(h_cur)
# pylint: enable=too-many-arguments,too-many-positional-arguments


# Kernel entry operands mirror the flattened CANNBotDSL Tensor ABI.
# pylint: disable=too-many-arguments,too-many-positional-arguments
@kernel
class BlockAttnResUpdateKernel:
    """Fused K3 Phase-2 update; each core owns a contiguous token range."""

    def __init__(self, num_tokens: int, d: int, epsilon: float,
                 delta_dtype=Float16, pipeline_rows: bool = False):
        self.num_tokens = int(num_tokens)
        self.d = int(d)
        self.epsilon = float(epsilon)
        self.delta_dtype = delta_dtype
        self.pipeline_rows = bool(pipeline_rows)

    def __call__(self, gm_partial: Tensor, gm_delta: Tensor,
                 gm_effective_query: Tensor,
                 gm_inter_max: Tensor, gm_inter_sum: Tensor,
                 gm_inter_numerator: Tensor,
                 gm_h: Tensor, gm_partial_out: Tensor):
        vector = BlockAttnResUpdateVector(
            self.d, self.delta_dtype, self.pipeline_rows,
        )
        block_idx = get_block_idx()
        block_num = get_block_num()
        num_tokens = Int64(self.num_tokens)
        logical = block_num
        if logical > num_tokens:
            logical = num_tokens
        rows_per_core = (num_tokens + logical - Int64(1)) // logical
        row_start = block_idx * rows_per_core
        row_end = row_start + rows_per_core
        if row_end > num_tokens:
            row_end = num_tokens

        if block_idx < logical:
            vector.load_query(gm_effective_query)
            query_cur = vector.query_ub.wait()
            if self.pipeline_rows:
                vector.load_row(
                    gm_partial, gm_delta, gm_inter_numerator,
                    gm_inter_max, gm_inter_sum, row_start,
                )
                for row in range(row_start, row_end):
                    next_row = row + Int64(1)
                    if next_row < row_end:
                        # Preload row+1 into the second FIFO slot while the
                        # Vector pipe consumes row.
                        vector.load_row(
                            gm_partial, gm_delta, gm_inter_numerator,
                            gm_inter_max, gm_inter_sum, next_row,
                        )
                    vector.compute_row(query_cur, self.epsilon)
                    vector.store_row(gm_h, gm_partial_out, row)
            else:
                for row in range(row_start, row_end):
                    vector.load_row(
                        gm_partial, gm_delta, gm_inter_numerator,
                        gm_inter_max, gm_inter_sum, row,
                    )
                    vector.compute_row(query_cur, self.epsilon)
                    vector.store_row(gm_h, gm_partial_out, row)
            vector.query_ub.release(query_cur)
# pylint: enable=too-many-arguments,too-many-positional-arguments


class BlockAttnResUpdate:
    def __init__(self, plan: UpdateLaunchPlan):
        shape = plan.shape
        self.num_tokens = int(shape.tokens)
        self.d = int(shape.d)
        self.epsilon = float(plan.epsilon)
        self.delta_dtype = _TORCH_TO_DSL[shape.delta_dtype]
        self.block_num = int(plan.block_num)
        self.pipeline_rows = bool(plan.pipeline_rows)

    # The JIT entry signature is the compiled kernel's flattened Tensor ABI.
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    @jit
    def run(self, gm_partial, gm_delta, gm_effective_query,
            gm_inter_max, gm_inter_sum, gm_inter_numerator,
            gm_h, gm_partial_out):
        op = BlockAttnResUpdateKernel(
            self.num_tokens, self.d, self.epsilon,
            self.delta_dtype, self.pipeline_rows,
        )
        op[self.block_num](
            gm_partial, gm_delta, gm_effective_query,
            gm_inter_max, gm_inter_sum, gm_inter_numerator,
            gm_h, gm_partial_out,
        )

    # pylint: enable=too-many-arguments,too-many-positional-arguments


def _compiled_update_kernel(plan: UpdateLaunchPlan):
    """Compile once per static Update shape and reuse it across all slots."""
    shape = plan.shape
    key = (
        shape,
        int(plan.block_num),
        bool(plan.pipeline_rows),
        float(plan.epsilon),
    )
    compiled = _COMPILED_KERNEL_CACHE.get(key)
    if compiled is not None:
        return compiled

    op = BlockAttnResUpdate(plan)
    fake = cannbotdsl.TensorSpec
    compiled = op.run.compile(
        fake((shape.tokens, shape.d), dtypes.float32),
        fake(
            (shape.tokens, shape.d),
            _TORCH_TO_DSL[shape.delta_dtype],
        ),
        fake((1, shape.d), dtypes.float32),
        fake((shape.tokens, 1), dtypes.float32),
        fake((shape.tokens, 1), dtypes.float32),
        fake((shape.tokens, shape.d), dtypes.float32),
        fake(
            (shape.tokens, shape.d),
            _TORCH_TO_DSL[shape.delta_dtype],
        ),
        fake((shape.tokens, shape.d), dtypes.float32),
    )
    _COMPILED_KERNEL_CACHE[key] = compiled
    return compiled


@lru_cache(maxsize=None)
def _update_launch_plan(
    shape: UpdateStaticShape,
    epsilon: float,
):
    """Cache static core count, pipeline choice, and compiled callable."""
    block_num = _launch_block_num(shape.tokens)
    pipeline_rows = _use_row_pipeline(shape.tokens, shape.d, block_num)
    plan = UpdateLaunchPlan(
        shape=shape,
        block_num=block_num,
        pipeline_rows=pipeline_rows,
        epsilon=float(epsilon),
    )
    return _compiled_update_kernel(plan)


def _block_attn_res_update_eager(
    inputs: UpdateInputs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused equivalent of K3 ``_update_attn_res_online_softmax``.

    Args:
        inputs: Named group containing the partial state, delta, selected query,
            Phase-1 statistics/numerator, and positive Python epsilon.
    Returns:
        ``(h, partial_blocks)``. ``h`` is ``[tokens,D]`` and follows
        ``partial_delta.dtype``; ``partial_blocks`` remains fp32 and equals
        ``partial_block + partial_delta.float()``.
    """
    partial_block = inputs.partial_block
    partial_delta = inputs.partial_delta
    effective_query = inputs.effective_query
    inter_max = inputs.inter_max
    inter_exp_sum = inputs.inter_exp_sum
    inter_numerator = inputs.inter_numerator
    epsilon = inputs.epsilon
    _require(
        partial_block.dtype == torch.float32,
        f"partial_block 必须为 fp32,实际 {partial_block.dtype}",
    )
    _require(
        partial_block.dim() == 2 and int(partial_block.shape[1]) >= 1,
        "partial_block 必须为 [tokens,D] 且 D>=1",
    )
    _require(
        partial_delta.dtype in (torch.bfloat16, torch.float16),
        f"partial_delta 必须为 bf16/fp16,实际 {partial_delta.dtype}",
    )
    _require(
        tuple(partial_delta.shape) == tuple(partial_block.shape),
        "partial_delta 必须与 partial_block 同 shape",
    )
    _require(
        partial_block.is_contiguous() and partial_delta.is_contiguous(),
        "partial_block / partial_delta 必须 contiguous",
    )
    _require(
        partial_block.device == partial_delta.device,
        "partial_block / partial_delta 必须同 device",
    )

    tokens, d = (int(size) for size in partial_block.shape)
    _require(d <= MAX_D, f"D={d} 超过当前上限 {MAX_D}")
    _require(
        effective_query.dtype == torch.float32
        and effective_query.dim() == 1
        and int(effective_query.shape[0]) == d,
        f"effective_query 必须为 [{d}] fp32",
    )
    _require(
        effective_query.is_contiguous()
        and effective_query.device == partial_block.device,
        "effective_query 必须 contiguous 且与 partial_block 同 device",
    )

    stats_shape = (tokens,)
    for name, value in (
        ("inter_max", inter_max),
        ("inter_exp_sum", inter_exp_sum),
    ):
        _require(
            value.dtype == torch.float32
            and tuple(value.shape) == stats_shape,
            f"{name} 必须为 [tokens] fp32",
        )
        _require(
            value.is_contiguous() and value.device == partial_block.device,
            f"{name} 必须 contiguous 且与 partial_block 同 device",
        )

    _require(
        inter_numerator.dtype == torch.float32,
        "inter_numerator 必须为 fp32",
    )
    _require(
        tuple(inter_numerator.shape) == (tokens, d),
        "inter_numerator 必须为 [tokens,D]",
    )
    _require(
        inter_numerator.is_contiguous()
        and inter_numerator.device == partial_block.device,
        "inter_numerator 必须 contiguous 且与 partial_block 同 device",
    )

    _require(
        isinstance(epsilon, (float, int)) and float(epsilon) > 0.0,
        "epsilon 必须为正 Python float",
    )
    epsilon_value = float(epsilon)

    h = torch.empty(
        partial_block.shape,
        dtype=partial_delta.dtype,
        device=partial_block.device,
    )
    partial_blocks = torch.empty_like(partial_block)

    if tokens == 0:
        return h, partial_blocks

    static_shape = UpdateStaticShape(
        tokens=tokens,
        d=d,
        delta_dtype=partial_delta.dtype,
    )
    compiled = _update_launch_plan(static_shape, epsilon_value)
    compiled(
        partial_block,
        partial_delta,
        effective_query.reshape(1, d),
        inter_max.reshape(tokens, 1),
        inter_exp_sum.reshape(tokens, 1),
        inter_numerator,
        h,
        partial_blocks,
    )
    return h, partial_blocks


# Keep the CANNBotDSL provider call outside TorchDynamo by representing the
# functional whole-network call as one dispatcher node. CANNBotDSL 0.3 accepts
# provider-owned Torch tensors directly and owns Host ABI/stream resolution.
_UPDATE_TORCH_LIBRARY = torch.library.Library("cannbot_attn_res", "FRAGMENT")
_UPDATE_TORCH_LIBRARY.define(
    "block_attn_res_update(Tensor partial_block, Tensor partial_delta, "
    "Tensor effective_query, Tensor inter_max, Tensor inter_exp_sum, "
    "Tensor inter_numerator, float epsilon=1e-6) -> (Tensor, Tensor)"
)


# Dispatcher implementations and the public function must exactly match the
# stable seven-input whole-network schema. Internally they immediately create
# ``UpdateInputs`` so the related values are handled as one named group.
# pylint: disable=too-many-arguments,too-many-positional-arguments
@torch.library.impl(_UPDATE_TORCH_LIBRARY, "block_attn_res_update", "Meta")
def _block_attn_res_update_meta(
    partial_block: torch.Tensor,
    partial_delta: torch.Tensor,
    effective_query: torch.Tensor,
    inter_max: torch.Tensor,
    inter_exp_sum: torch.Tensor,
    inter_numerator: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        effective_query,
        inter_max,
        inter_exp_sum,
        inter_numerator,
        epsilon,
    )
    return (
        partial_delta.new_empty(partial_block.shape),
        partial_block.new_empty(partial_block.shape),
    )


@torch.library.impl(
    _UPDATE_TORCH_LIBRARY, "block_attn_res_update", "PrivateUse1",
)
def _block_attn_res_update_npu(
    partial_block: torch.Tensor,
    partial_delta: torch.Tensor,
    effective_query: torch.Tensor,
    inter_max: torch.Tensor,
    inter_exp_sum: torch.Tensor,
    inter_numerator: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _block_attn_res_update_eager(
        UpdateInputs(
            partial_block=partial_block,
            partial_delta=partial_delta,
            effective_query=effective_query,
            inter_max=inter_max,
            inter_exp_sum=inter_exp_sum,
            inter_numerator=inter_numerator,
            epsilon=epsilon,
        )
    )


def block_attn_res_update(
    partial_block: torch.Tensor,
    partial_delta: torch.Tensor,
    effective_query: torch.Tensor,
    inter_max: torch.Tensor,
    inter_exp_sum: torch.Tensor,
    inter_numerator: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Graph-safe public entry for the fused online-softmax update."""
    _require(
        isinstance(epsilon, (float, int)) and float(epsilon) > 0.0,
        "epsilon 必须为正 Python float",
    )
    epsilon_value = float(epsilon)
    if partial_block.device.type != "npu":
        return _block_attn_res_update_eager(
            UpdateInputs(
                partial_block=partial_block,
                partial_delta=partial_delta,
                effective_query=effective_query,
                inter_max=inter_max,
                inter_exp_sum=inter_exp_sum,
                inter_numerator=inter_numerator,
                epsilon=epsilon_value,
            )
        )

    return torch.ops.cannbot_attn_res.block_attn_res_update.default(
        partial_block,
        partial_delta,
        effective_query,
        inter_max,
        inter_exp_sum,
        inter_numerator,
        epsilon_value,
    )
# pylint: enable=too-many-arguments,too-many-positional-arguments
