# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Fused recurrent Kimi Delta Attention decode."""

from __future__ import annotations
from cannbotdsl.buffer import Buffer

from typing import Optional, Tuple

import torch
import cannbotdsl

from cannbotdsl.channel import Channel
from cannbotdsl.arch import get_block_idx, get_block_num
from cannbotdsl.constexpr import const_expr
from cannbotdsl.control_flow import range as dsl_range
from cannbotdsl import dtypes
from cannbotdsl.integer import Int64
from cannbotdsl.jit_runner import jit
from cannbotdsl.kernel_launcher import kernel
from cannbotdsl.math import cast
from cannbotdsl.tensor import _layout_op_wrapper, idx2crd, local_slice, make_layout, mem_copy, tile_view
from cannbotdsl.typing.types import MemLoc, Tensor
from cannbotdsl.vf import vf
from cannbotdsl.raw_reg import (
    UnpackMode,
    update_mask,
    vadd,
    vadds,
    vcast,
    vdiv,
    vdup_lane0,
    vdup_scalar,
    vexp,
    vload,
    vload_brc,
    vload_unpack,
    vmul,
    vmuls,
    vneg,
    vreduce_sum,
    vsqrt,
    vstore,
    vstore_first,
    vsub,
)


SUPPORTED_HEAD_DIM = 128
VL = 64
D_SEGMENTS = SUPPORTED_HEAD_DIM // VL
DEFAULT_BLOCK_NUM = 56
_DYNAMIC_KERNEL_CACHE: dict[tuple[int, int, str, int, object, int, float], object] = {}


def gqa_group_size(num_value_heads: int, num_kv_heads: int) -> int:
    """Return the number of value heads mapped to each Q/K head."""
    if num_value_heads <= 0 or num_kv_heads <= 0 or num_value_heads % num_kv_heads:
        raise ValueError(
            f"GQA requires positive Nv % Nk == 0, got Nv={num_value_heads}, Nk={num_kv_heads}"
        )
    return num_value_heads // num_kv_heads


def _sequence_shape(tensor: torch.Tensor, layout_qkv: str) -> tuple[int, int, int, int]:
    """Return a sequence tensor's logical ``(B, N, S, D)`` shape."""
    assert tensor.dim() == 4, "query/key/value must use a rank-4 layout"
    if layout_qkv == "BSND":
        batch, seq_len, heads, dim = tensor.shape
        return batch, heads, seq_len, dim
    return tuple(tensor.shape)


def _bsnd_to_bnsd(t: Tensor) -> Tensor:
    """Create a logical BNSD view of a contiguous BSND sequence tensor."""
    def _swap_1_2(layout):
        shape, stride = layout.shape, layout.stride
        return make_layout((shape[0], shape[2], shape[1], shape[3]),
                           stride=(stride[0], stride[2], stride[1], stride[3]))

    return _layout_op_wrapper(_swap_1_2, t)


def get_row_block_config(batch_heads: int, _seq_len: int) -> int:
    return 32 if batch_heads <= 16 else 64


def _device_block_num(ref: torch.Tensor) -> int:
    if ref.device.type not in {"npu", "privateuseone"}:
        return DEFAULT_BLOCK_NUM
    npu = getattr(torch, "npu", None)
    if npu is None:
        return DEFAULT_BLOCK_NUM
    device_index = ref.device.index
    if device_index is None:
        device_index = npu.current_device()
    vector_core_num = int(getattr(npu.get_device_properties(device_index), "vector_core_num", DEFAULT_BLOCK_NUM))
    if vector_core_num <= 0:
        raise RuntimeError(f"NPU {device_index} reports invalid vector_core_num={vector_core_num}")
    return vector_core_num


def _dynamic_sequence_tensor(dtype, batch_size, seq_len, num_heads, head_dim, layout_qkv):
    if layout_qkv == "BNSD":
        return cannbotdsl.TensorSpec((batch_size, num_heads, seq_len, head_dim), dtype)
    return cannbotdsl.TensorSpec((batch_size, seq_len, num_heads, head_dim), dtype)


def _static_tensor_spec(dtype, tensor):
    return cannbotdsl.TensorSpec(tuple(tensor.shape), dtype, stride=tuple(tensor.stride()))


class FusedRecurrentKDAVector:
    def __init__(self, row_block: int, dk: int, state_dtype=dtypes.bfloat16):
        self.row_block = row_block
        self.dk = dk
        self.state_is_bf16 = state_dtype == dtypes.bfloat16
        value_width = max(row_block, VL)
        self.state = Channel(MemLoc.UB, shape=(row_block, dk), dtype=dtypes.float32, depth=2)
        self.key = Buffer(MemLoc.UB, (1, dk), dtypes.float32)
        self.query = Buffer(MemLoc.UB, (1, dk), dtypes.float32)
        self.decay = Buffer(MemLoc.UB, (1, dk), dtypes.float32)
        self.ub_value = Buffer(MemLoc.UB, (1, value_width), dtypes.float32)
        self.ub_beta = Buffer(MemLoc.UB, (1, VL), dtypes.float32)
        self.out = Buffer(MemLoc.UB, (1, row_block), dtypes.float32)
        if self.state_is_bf16:
            self.state_bf16 = Channel(
                MemLoc.UB, shape=(row_block, dk), dtype=dtypes.bfloat16, depth=2)
            self.state_bf16_output = Channel(
                MemLoc.UB, shape=(row_block, dk), dtype=dtypes.bfloat16, depth=2)
        self.value = Channel(
            MemLoc.UB, shape=(1, max(row_block, VL)), dtype=dtypes.bfloat16, depth=2)
        self.output = Channel(MemLoc.UB, shape=(1, row_block), dtype=dtypes.bfloat16, depth=2)
        self.value_real = local_slice(self.ub_value, (1, row_block), offset=0)
        self.beta = local_slice(self.ub_beta, (1, VL), offset=0)
        self.qk_norm_sums = Buffer(MemLoc.UB, (1, D_SEGMENTS * 2), dtypes.float32)
        self.state_key_sums = Buffer(MemLoc.UB, (1, VL), dtypes.float32)
        self.delta_row = Buffer(MemLoc.UB, (1, VL), dtypes.float32)
        self.raw_query = Channel(MemLoc.UB, shape=(1, dk), dtype=dtypes.bfloat16, depth=2)
        self.raw_key = Channel(MemLoc.UB, shape=(1, dk), dtype=dtypes.bfloat16, depth=2)
        self.raw_g = Channel(MemLoc.UB, shape=(1, dk), dtype=dtypes.bfloat16, depth=2)
        self.raw_beta = Channel(MemLoc.UB, shape=(1, VL), dtype=dtypes.bfloat16, depth=2)
        self.dt_bias = Channel(MemLoc.UB, shape=(1, dk), dtype=dtypes.float32, depth=2)
        self.a_log = Channel(MemLoc.UB, shape=(1,), dtype=dtypes.float32, depth=2)

    def load_state(self, gm_state_tile):
        if self.state_is_bf16:
            mem_copy(self.state_bf16, gm_state_tile)
            with vf(outputs=[self.state]):
                cast(self.state, self.state_bf16)
        else:
            mem_copy(self.state, gm_state_tile)

    def load_qk(self, raw_key_gm, raw_query_gm):
        mem_copy(self.raw_query, raw_query_gm)
        mem_copy(self.raw_key, raw_key_gm)

    @jit
    def normalize_qk(self, scale_value: float):
        with vf(mode="raw"):
            full, _ = update_mask(VL, elem_bits=32)
            query_raw_pre = vload_unpack(self.raw_query, 0, mode=UnpackMode.B16_TO_B32)
            query_raw_post = vload_unpack(self.raw_query, VL, mode=UnpackMode.B16_TO_B32)
            key_raw_pre = vload_unpack(self.raw_key, 0, mode=UnpackMode.B16_TO_B32)
            key_raw_post = vload_unpack(self.raw_key, VL, mode=UnpackMode.B16_TO_B32)
            query_pre = vcast(query_raw_pre, dtypes.float32, mask=full)
            query_post = vcast(query_raw_post, dtypes.float32, mask=full)
            key_pre = vcast(key_raw_pre, dtypes.float32, mask=full)
            key_post = vcast(key_raw_post, dtypes.float32, mask=full)
            query_square_pre = vmul(query_pre, query_pre, mask=full)
            query_square_post = vmul(query_post, query_post, mask=full)
            key_square_pre = vmul(key_pre, key_pre, mask=full)
            key_square_post = vmul(key_post, key_post, mask=full)
            query_sum_pre = vreduce_sum(query_square_pre, mask=full)
            query_sum_post = vreduce_sum(query_square_post, mask=full)
            key_sum_pre = vreduce_sum(key_square_pre, mask=full)
            key_sum_post = vreduce_sum(key_square_post, mask=full)
            vstore(self.query, 0, query_pre, full)
            vstore(self.query, VL, query_post, full)
            vstore(self.key, 0, key_pre, full)
            vstore(self.key, VL, key_post, full)
            vstore_first(self.qk_norm_sums, 0, query_sum_pre)
            vstore_first(self.qk_norm_sums, 1, query_sum_post)
            vstore_first(self.qk_norm_sums, D_SEGMENTS, key_sum_pre)
            vstore_first(self.qk_norm_sums, D_SEGMENTS + 1, key_sum_post)

        with vf(mode="raw"):
            lane0, _ = update_mask(1, elem_bits=32)
            query_sum_pre = vload_brc(self.qk_norm_sums, 0)
            query_sum_post = vload_brc(self.qk_norm_sums, 1)
            key_sum_pre = vload_brc(self.qk_norm_sums, D_SEGMENTS)
            key_sum_post = vload_brc(self.qk_norm_sums, D_SEGMENTS + 1)
            query_sum = vadd(query_sum_pre, query_sum_post, mask=lane0)
            key_sum = vadd(key_sum_pre, key_sum_post, mask=lane0)
            query_sum_eps = vadds(query_sum, 1e-6, mask=lane0)
            key_sum_eps = vadds(key_sum, 1e-6, mask=lane0)
            query_root = vsqrt(query_sum_eps, mask=lane0)
            key_root = vsqrt(key_sum_eps, mask=lane0)
            vstore_first(self.qk_norm_sums, 0, query_root)
            vstore_first(self.qk_norm_sums, 1, key_root)

        with vf(mode="raw"):
            full, _ = update_mask(VL, elem_bits=32)
            query_root = vload_brc(self.qk_norm_sums, 0)
            key_root = vload_brc(self.qk_norm_sums, 1)
            query_denom = vdup_lane0(query_root, mask=full)
            key_denom = vdup_lane0(key_root, mask=full)
            query_pre = vload(self.query, 0)
            query_post = vload(self.query, VL)
            key_pre = vload(self.key, 0)
            key_post = vload(self.key, VL)
            normalized_query_pre = vdiv(query_pre, query_denom, mask=full)
            normalized_query_post = vdiv(query_post, query_denom, mask=full)
            normalized_key_pre = vdiv(key_pre, key_denom, mask=full)
            normalized_key_post = vdiv(key_post, key_denom, mask=full)
            scaled_query_pre = vmuls(normalized_query_pre, scale_value, mask=full)
            scaled_query_post = vmuls(normalized_query_post, scale_value, mask=full)
            vstore(self.query, 0, scaled_query_pre, full)
            vstore(self.query, VL, scaled_query_post, full)
            vstore(self.key, 0, normalized_key_pre, full)
            vstore(self.key, VL, normalized_key_post, full)

    def load_g_beta_value(self, raw_g_gm, value_gm, raw_beta_gm, a_log_gm, dt_bias_gm):
        mem_copy(self.raw_g, raw_g_gm)
        mem_copy(local_slice(self.raw_beta, (1, 1), offset=0), raw_beta_gm)
        mem_copy(local_slice(self.value, (1, self.row_block), offset=0), value_gm)
        mem_copy(self.a_log, a_log_gm)
        mem_copy(self.dt_bias, dt_bias_gm)

    @jit
    def activate_g_beta(self, lower_bound: float):
        with vf(mode="raw"):
            full, _ = update_mask(VL, elem_bits=32)
            one = vdup_scalar(1.0, dtypes.float32)
            a_log = vload_brc(self.a_log, 0)
            alpha = vexp(a_log, mask=full)
            negative_alpha = vneg(alpha, mask=full)
            raw_gate_pre = vload_unpack(self.raw_g, 0, mode=UnpackMode.B16_TO_B32)
            raw_gate_post = vload_unpack(self.raw_g, VL, mode=UnpackMode.B16_TO_B32)
            gate_value_pre = vcast(raw_gate_pre, dtypes.float32, mask=full)
            gate_value_post = vcast(raw_gate_post, dtypes.float32, mask=full)
            dt_bias_pre = vload(self.dt_bias, 0)
            dt_bias_post = vload(self.dt_bias, VL)
            gate_input_pre = vadd(gate_value_pre, dt_bias_pre, mask=full)
            gate_input_post = vadd(gate_value_post, dt_bias_post, mask=full)
            gate_pre = vmul(negative_alpha, gate_input_pre, mask=full)
            gate_post = vmul(negative_alpha, gate_input_post, mask=full)
            gate_exp_pre = vexp(gate_pre, mask=full)
            gate_exp_post = vexp(gate_post, mask=full)
            gate_denom_pre = vadds(gate_exp_pre, 1.0, mask=full)
            gate_denom_post = vadds(gate_exp_post, 1.0, mask=full)
            gate_sigmoid_pre = vdiv(one, gate_denom_pre, mask=full)
            gate_sigmoid_post = vdiv(one, gate_denom_post, mask=full)
            activated_gate_pre = vmuls(gate_sigmoid_pre, lower_bound, mask=full)
            activated_gate_post = vmuls(gate_sigmoid_post, lower_bound, mask=full)
            vstore(self.decay, 0, activated_gate_pre, full)
            vstore(self.decay, VL, activated_gate_post, full)

            raw_beta_value = vload_unpack(self.raw_beta, 0, mode=UnpackMode.B16_TO_B32)
            beta_logit = vcast(raw_beta_value, dtypes.float32, mask=full)
            beta_neg_logit = vneg(beta_logit, mask=full)
            beta_exp = vexp(beta_neg_logit, mask=full)
            beta_denom = vadds(beta_exp, 1.0, mask=full)
            activated_beta = vdiv(one, beta_denom, mask=full)
            vstore(self.ub_beta, 0, activated_beta, full)
            value_raw = vload_unpack(self.value, 0, mode=UnpackMode.B16_TO_B32)
            value_f32 = vcast(value_raw, dtypes.float32, mask=full)
            vstore(self.ub_value, 0, value_f32, full)

    @jit
    def recur_step(self):
        row_block, dk = self.row_block, self.dk
        state, out = self.state, self.out
        key, query, decay = self.key, self.query, self.decay
        value_real, beta = self.value_real, self.beta
        state_key_sums, delta_row = self.state_key_sums, self.delta_row

        with vf(mode="raw"):
            mask, _ = update_mask(VL, elem_bits=32)
            gate_pre = vload(decay, 0)
            gate_post = vload(decay, VL)
            decay_pre = vexp(gate_pre, mask=mask)
            decay_post = vexp(gate_post, mask=mask)
            key_pre = vload(key, 0)
            key_post = vload(key, VL)
            for dv in dsl_range(Int64(0), Int64(row_block), Int64(1), unroll=2):
                offset_pre = dv * dk
                offset_post = offset_pre + VL
                state_pre = vload(state, offset_pre)
                state_post = vload(state, offset_post)
                decayed_pre = vmul(state_pre, decay_pre, mask=mask)
                decayed_post = vmul(state_post, decay_post, mask=mask)
                product_pre = vmul(decayed_pre, key_pre, mask=mask)
                product_post = vmul(decayed_post, key_post, mask=mask)
                product = vadd(product_pre, product_post, mask=mask)
                state_key_sum = vreduce_sum(product, mask=mask)
                vstore_first(state_key_sums, dv, state_key_sum)

        with vf(mode="raw"):
            mask, _ = update_mask(VL, elem_bits=32)
            beta_brc = vload_brc(beta, 0)
            state_key_vec = vload(state_key_sums, 0)
            value_vec = vload(value_real, 0)
            residual_vec = vsub(value_vec, state_key_vec, mask=mask)
            delta_vec = vmul(residual_vec, beta_brc, mask=mask)
            vstore(delta_row, 0, delta_vec, mask)

        with vf(mode="raw"):
            mask, _ = update_mask(VL, elem_bits=32)
            gate_pre = vload(decay, 0)
            gate_post = vload(decay, VL)
            decay_pre = vexp(gate_pre, mask=mask)
            decay_post = vexp(gate_post, mask=mask)
            key_pre = vload(key, 0)
            key_post = vload(key, VL)
            query_pre = vload(query, 0)
            query_post = vload(query, VL)
            for dv in dsl_range(Int64(0), Int64(row_block), Int64(1), unroll=2):
                offset_pre = dv * dk
                offset_post = offset_pre + VL
                state_pre = vload(state, offset_pre)
                state_post = vload(state, offset_post)
                decayed_pre = vmul(state_pre, decay_pre, mask=mask)
                decayed_post = vmul(state_post, decay_post, mask=mask)
                delta = vload_brc(delta_row, dv)
                delta_key_pre = vmul(delta, key_pre, mask=mask)
                delta_key_post = vmul(delta, key_post, mask=mask)
                state_new_pre = vadd(decayed_pre, delta_key_pre, mask=mask)
                state_new_post = vadd(decayed_post, delta_key_post, mask=mask)
                vstore(state, offset_pre, state_new_pre, mask)
                vstore(state, offset_post, state_new_post, mask)
                output_pre = vmul(state_new_pre, query_pre, mask=mask)
                output_post = vmul(state_new_post, query_post, mask=mask)
                output = vadd(output_pre, output_post, mask=mask)
                output_sum = vreduce_sum(output, mask=mask)
                vstore_first(out, dv, output_sum)

    def store_output(self, gm_out_tile):
        with vf(outputs=[self.output]):
            cast(self.output, self.out)
        mem_copy(gm_out_tile, self.output)

    def store_state(self, gm_state_tile):
        if self.state_is_bf16:
            with vf(outputs=[self.state_bf16_output]):
                cast(self.state_bf16_output, self.state)
            mem_copy(gm_state_tile, self.state_bf16_output)
        else:
            mem_copy(gm_state_tile, self.state)


@kernel
class fused_recurrent_kda_kernel:
    def __init__(
        self,
        head_dim: int = 128,
        state_dtype=dtypes.float32,
        row_block: int = 64,
        gqa_group: int = 1,
        scale_value: float = 1.0,
        lower_bound: float = -1.0,
    ):
        self.head_dim = head_dim
        self.state_dtype = state_dtype
        self.row_block = int(row_block)
        self.gqa_group = int(gqa_group)
        self.scale_value = float(scale_value)
        self.lower_bound = float(lower_bound)

    def __call__(
        self,
        gm_key: Tensor,
        gm_query: Tensor,
        gm_g: Tensor,
        gm_value: Tensor,
        gm_beta: Tensor,
        gm_a_log: Tensor,
        gm_dt_bias: Tensor,
        gm_out: Tensor,
        gm_initial_state: Tensor,
        gm_final_state: Tensor,
        gm_ssm: Tensor,
        gm_na: Tensor,
        seq_len: int,
    ):
        head_dim = self.head_dim
        row_block = self.row_block
        num_row_blocks = head_dim // row_block
        vector = FusedRecurrentKDAVector(row_block, head_dim, state_dtype=self.state_dtype)
        batch_dim = gm_key.shape[0]
        value_heads_dim = gm_value.shape[1]
        total_items = batch_dim * value_heads_dim * num_row_blocks
        logical_core_num = get_block_num()
        if logical_core_num > total_items:
            logical_core_num = total_items
        items_per_core = (total_items + logical_core_num - 1) // logical_core_num
        item_start = get_block_idx() * items_per_core
        item_end = item_start + items_per_core
        if item_end > total_items:
            item_end = total_items
        if get_block_idx() < logical_core_num:
            for item in range(item_start, item_end):
                batch_index, value_head, row_index = idx2crd(item, [batch_dim, value_heads_dim, num_row_blocks])
                kv_head = value_head // self.gqa_group
                base = batch_index * seq_len
                initial_state_index = gm_ssm[base + gm_na[batch_index] - 1]
                vector.load_state(tile_view(gm_initial_state[initial_state_index, value_head, None, None],
                                  (row_block, head_dim),(row_index, 0)))
                for token in range(0, seq_len, 1):
                    vector.load_qk(tile_view(gm_key[batch_index, kv_head, None, None], (1, head_dim), (token, 0)),
                                   tile_view(gm_query[batch_index, kv_head, None, None], (1, head_dim), (token, 0)))
                    vector.normalize_qk(self.scale_value)
                    vector.load_g_beta_value(tile_view(gm_g[batch_index, value_head, None, None], (1, head_dim), (token, 0)),
                                            tile_view(gm_value[batch_index, value_head, None, None], (1, row_block), (token, row_index)),
                                            tile_view(gm_beta[batch_index, value_head, None, None], (1, 1), (token, 0)),
                                            tile_view(gm_a_log, (1,), (value_head,)),
                                            tile_view(gm_dt_bias, (1, head_dim), (value_head, 0)),)
                    vector.activate_g_beta(self.lower_bound)
                    vector.recur_step()
                    vector.store_output(tile_view(gm_out[batch_index, value_head, None, None],
                                        (1, row_block),(token, row_index)))
                    snapshot_state_index = gm_ssm[base + token]
                    vector.store_state(tile_view(gm_final_state[snapshot_state_index, value_head, None, None],
                                       (row_block, head_dim),(row_index, 0)))


class FusedRecurrentKDA:
    def __init__(
        self,
        head_dim: int,
        state_dtype,
        row_block: int,
        layout_qkv: str,
        block_num: int,
        gqa_group: int,
        scale_value: float,
        lower_bound: float,
    ):
        self.head_dim = head_dim
        self.state_dtype = state_dtype
        self.row_block = int(row_block)
        self.layout_qkv = layout_qkv
        self.block_num = int(block_num)
        self.gqa_group = int(gqa_group)
        self.scale_value = float(scale_value)
        self.lower_bound = float(lower_bound)

    @jit
    def run(self, key: Tensor, query: Tensor, raw_g: Tensor, value: Tensor, raw_beta: Tensor,
        a_log: Tensor, dt_bias: Tensor, out: Tensor, initial_state: Tensor, final_state: Tensor, ssm: Tensor,
        na: Tensor, seq_len: int):
        if const_expr(self.layout_qkv == "BSND"):
            key, query, raw_g, value, raw_beta, out = (
                _bsnd_to_bnsd(t) for t in (key, query, raw_g, value, raw_beta, out))
        op = fused_recurrent_kda_kernel(self.head_dim, self.state_dtype, self.row_block, self.gqa_group, self.scale_value, self.lower_bound)
        op[self.block_num](
            key, query, raw_g, value, raw_beta, a_log, dt_bias, out,
            initial_state, final_state, ssm, na, seq_len,
        )


def fused_recurrent_kda_functional(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float],
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    layout_qkv: str,
    *,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run raw-input recurrent KDA decode and update ``state`` in place.

    Q/K, g, and beta are bf16 raw inputs.  The final kernel normalizes Q/K
    with epsilon ``1e-6``, activates gates with ``A_log``, ``dt_bias``, and
    ``lower_bound``, and applies sigmoid to beta before the recurrent update.
    ``A_log`` is fp32 ``[Nv]`` and ``dt_bias`` is fp32 ``[Nv, D]``.
    """
    assert layout_qkv in ("BNSD", "BSND"), f"layout_qkv must be BNSD or BSND, got {layout_qkv!r}"
    batch, num_kv_heads, seq_len, dim = _sequence_shape(query, layout_qkv)
    num_value_heads = _sequence_shape(value, layout_qkv)[1]
    gqa_group_size(num_value_heads, num_kv_heads)

    assert _sequence_shape(key, layout_qkv) == (batch, num_kv_heads, seq_len, dim), \
        "key shape must match query"
    assert dim == SUPPORTED_HEAD_DIM, f"only D={SUPPORTED_HEAD_DIM} is supported, got D={dim}"
    assert _sequence_shape(value, layout_qkv) == (batch, num_value_heads, seq_len, dim), \
        "value shape must match Nv/layout"
    beta_shape = ((batch, seq_len, num_value_heads, 1) if layout_qkv == "BSND"
                  else (batch, num_value_heads, seq_len, 1))
    assert tuple(beta.shape) == beta_shape, f"beta must have shape {beta_shape}"
    assert _sequence_shape(g, layout_qkv) == (batch, num_value_heads, seq_len, dim), \
        "g shape must match value"
    assert state.dim() == 4 and tuple(state.shape[1:]) == (num_value_heads, dim, dim), \
        f"state must have shape (pool, {num_value_heads}, {dim}, {dim})"
    assert state.shape[0] >= batch * seq_len, "state pool must provide at least B*S slots"
    assert tuple(A_log.shape) == (num_value_heads,), f"A_log must have shape ({num_value_heads},)"
    assert tuple(dt_bias.shape) == (num_value_heads, dim), \
        f"dt_bias must have shape ({num_value_heads}, {dim})"
    assert A_log.dtype == torch.float32, f"A_log must be fp32, got {A_log.dtype}"
    assert dt_bias.dtype == torch.float32, f"dt_bias must be fp32, got {dt_bias.dtype}"
    assert -5.0 <= float(lower_bound) <= 0.0, "lower_bound must be in [-5, 0]"

    for name, tensor in (("query", query), ("key", key), ("value", value), ("g", g), ("beta", beta)):
        assert tensor.dtype == torch.bfloat16, f"{name} must be bf16, got {tensor.dtype}"
    assert state.dtype in (torch.bfloat16, torch.float32), f"state must be bf16 or fp32, got {state.dtype}"
    tensors = (query, key, value, state, beta, g, A_log, dt_bias)
    assert all(tensor.device == state.device for tensor in tensors), "all inputs must share one device"
    assert all(tensor.is_contiguous() for tensor in tensors), "all inputs must be contiguous"

    device = state.device
    if ssm_state_indices is not None:
        assert ssm_state_indices.dtype == torch.int32, \
            f"ssm_state_indices must be int32, got {ssm_state_indices.dtype}"
        assert tuple(ssm_state_indices.shape) == (batch * seq_len,), \
            f"ssm_state_indices must have shape ({batch * seq_len},)"
        assert ssm_state_indices.device == device and ssm_state_indices.is_contiguous(), \
            "ssm_state_indices must be contiguous on the input device"
    if num_accepted_tokens is not None:
        assert num_accepted_tokens.dtype == torch.int32, \
            f"num_accepted_tokens must be int32, got {num_accepted_tokens.dtype}"
        assert tuple(num_accepted_tokens.shape) == (batch,), \
            f"num_accepted_tokens must have shape ({batch},)"
        assert num_accepted_tokens.device == device and num_accepted_tokens.is_contiguous(), \
            "num_accepted_tokens must be contiguous on the input device"

    if ssm_state_indices is None:
        ssm_i64 = torch.arange(batch * seq_len, dtype=torch.int64, device=device)
    else:
        ssm_i64 = ssm_state_indices.to(torch.int64)
    if num_accepted_tokens is None:
        na_i64 = torch.ones(batch, dtype=torch.int64, device=device)
    else:
        na_i64 = num_accepted_tokens.to(torch.int64)
    assert ssm_i64.is_contiguous() and na_i64.is_contiguous()

    scale_value = float(scale) if scale is not None else dim ** -0.5
    row_block = get_row_block_config(batch * num_value_heads, seq_len)
    assert row_block in (16, 32, 64) and dim % row_block == 0, \
        f"row_block must divide D and be one of 16, 32, 64, got {row_block}"
    state_dtype = dtypes.bfloat16 if state.dtype == torch.bfloat16 else dtypes.float32
    initial_state = state
    final_state = state
    state_block_num = int(state.shape[0])
    block_num = _device_block_num(query)
    if layout_qkv == "BSND":
        out = torch.zeros(batch, seq_len, num_value_heads, dim, dtype=torch.bfloat16, device=device)
    else:
        out = torch.zeros(batch, num_value_heads, seq_len, dim, dtype=torch.bfloat16, device=device)

    static_layout = None
    if layout_qkv == "BSND":
        static_layout = tuple(
            (tuple(tensor.shape), tuple(tensor.stride()))
            for tensor in (
                key, query, g, value, beta, A_log, dt_bias, out,
                initial_state, final_state, ssm_i64, na_i64,
            )
        )
    cache_key = (
        num_kv_heads, num_value_heads, layout_qkv, row_block, state_dtype, block_num, state_block_num,
        scale_value, float(lower_bound), static_layout,
    )
    fn = _DYNAMIC_KERNEL_CACHE.get(cache_key)
    if fn is None:
        compile_batch = batch if layout_qkv == "BSND" else cannbotdsl.Dim("B")
        compile_seq = seq_len if layout_qkv == "BSND" else cannbotdsl.Dim("S")
        op = FusedRecurrentKDA(
            dim, state_dtype, row_block, layout_qkv, block_num,
            gqa_group_size(num_value_heads, num_kv_heads), scale_value, float(lower_bound),
        )
        tensor_spec = cannbotdsl.TensorSpec
        if layout_qkv == "BSND":
            compile_args = (
                _static_tensor_spec(dtypes.bfloat16, key), _static_tensor_spec(dtypes.bfloat16, query),
                _static_tensor_spec(dtypes.bfloat16, g), _static_tensor_spec(dtypes.bfloat16, value),
                _static_tensor_spec(dtypes.bfloat16, beta), _static_tensor_spec(dtypes.float32, A_log),
                _static_tensor_spec(dtypes.float32, dt_bias), _static_tensor_spec(dtypes.bfloat16, out),
                _static_tensor_spec(state_dtype, initial_state), _static_tensor_spec(state_dtype, final_state),
                _static_tensor_spec(dtypes.int64, ssm_i64),
                _static_tensor_spec(dtypes.int64, na_i64),
            )
        else:
            compile_args = (
                _dynamic_sequence_tensor(dtypes.bfloat16, compile_batch, compile_seq, num_kv_heads, dim, layout_qkv),
                _dynamic_sequence_tensor(dtypes.bfloat16, compile_batch, compile_seq, num_kv_heads, dim, layout_qkv),
                _dynamic_sequence_tensor(dtypes.bfloat16, compile_batch, compile_seq, num_value_heads, dim, layout_qkv),
                _dynamic_sequence_tensor(dtypes.bfloat16, compile_batch, compile_seq, num_value_heads, dim, layout_qkv),
                _dynamic_sequence_tensor(dtypes.bfloat16, compile_batch, compile_seq, num_value_heads, 1, layout_qkv),
                tensor_spec((num_value_heads,), dtypes.float32),
                tensor_spec((num_value_heads, dim), dtypes.float32),
                _dynamic_sequence_tensor(dtypes.bfloat16, compile_batch, compile_seq, num_value_heads, dim, layout_qkv),
                tensor_spec((state_block_num, num_value_heads, dim, dim), state_dtype),
                tensor_spec((state_block_num, num_value_heads, dim, dim), state_dtype),
                tensor_spec((compile_batch * compile_seq,), dtypes.int64),
                tensor_spec((compile_batch,), dtypes.int64),
            )
        fn = op.run.compile(*compile_args, dtypes.int64)
        _DYNAMIC_KERNEL_CACHE[cache_key] = fn
    fn(
        key, query, g, value,
        beta, A_log, dt_bias, out,
        initial_state, final_state, ssm_i64,
        na_i64, seq_len,
    )
    return state, out


def fused_recurrent_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float],
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    layout_qkv: str,
    *,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run raw-input recurrent KDA decode and update ``state`` in place."""
    _, output = fused_recurrent_kda_functional(
        query, key, value, state, beta, g, scale, A_log, dt_bias,
        lower_bound, layout_qkv,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
    )
    return state, output


_GRAPH_LIBRARY = torch.library.Library("cannbotdsl_fused_recurrent_kda", "DEF")
_GRAPH_LIBRARY.define(
    "fused_recurrent_kda("
    "Tensor query, Tensor key, Tensor value, Tensor(a!) state, Tensor beta, Tensor g, "
    "float scale, Tensor A_log, Tensor dt_bias, float lower_bound, str layout_qkv, "
    "Tensor? ssm_state_indices=None, Tensor? num_accepted_tokens=None) -> Tensor"
)
_GRAPH_LIBRARY.define(
    "fused_recurrent_kda_functional("
    "Tensor query, Tensor key, Tensor value, Tensor state, Tensor beta, Tensor g, "
    "float scale, Tensor A_log, Tensor dt_bias, float lower_bound, str layout_qkv, "
    "Tensor? ssm_state_indices=None, Tensor? num_accepted_tokens=None) -> (Tensor, Tensor)"
)


@torch.library.impl(_GRAPH_LIBRARY, "fused_recurrent_kda", "Meta")
def _fused_recurrent_kda_meta(
    query,
    key,
    value,
    state,
    beta,
    g,
    scale,
    A_log,
    dt_bias,
    lower_bound,
    layout_qkv,
    ssm_state_indices=None,
    num_accepted_tokens=None,
):
    del query, key, beta, g, scale, A_log, dt_bias
    del lower_bound, ssm_state_indices, num_accepted_tokens, layout_qkv
    return torch.empty_like(value, device="meta")


@torch.library.impl(_GRAPH_LIBRARY, "fused_recurrent_kda_functional", "Meta")
def _fused_recurrent_kda_functional_meta(
    query,
    key,
    value,
    state,
    beta,
    g,
    scale,
    A_log,
    dt_bias,
    lower_bound,
    layout_qkv,
    ssm_state_indices=None,
    num_accepted_tokens=None,
):
    del query, key, beta, g, scale, A_log, dt_bias
    del lower_bound, ssm_state_indices, num_accepted_tokens, layout_qkv
    return torch.empty_like(value, device="meta"), torch.empty_like(state, device="meta")


@torch.library.impl(_GRAPH_LIBRARY, "fused_recurrent_kda", "PrivateUse1")
def _fused_recurrent_kda_privateuse1(
    query,
    key,
    value,
    state,
    beta,
    g,
    scale,
    A_log,
    dt_bias,
    lower_bound,
    layout_qkv,
    ssm_state_indices=None,
    num_accepted_tokens=None,
):
    _, output = fused_recurrent_kda(
        query,
        key,
        value,
        state,
        beta,
        g,
        scale,
        A_log,
        dt_bias,
        lower_bound,
        layout_qkv,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
    )
    return output


@torch.library.impl(_GRAPH_LIBRARY, "fused_recurrent_kda_functional", "PrivateUse1")
def _fused_recurrent_kda_functional_privateuse1(
    query,
    key,
    value,
    state,
    beta,
    g,
    scale,
    A_log,
    dt_bias,
    lower_bound,
    layout_qkv,
    ssm_state_indices=None,
    num_accepted_tokens=None,
):
    _, output = fused_recurrent_kda_functional(
        query,
        key,
        value,
        state,
        beta,
        g,
        scale,
        A_log,
        dt_bias,
        lower_bound,
        layout_qkv,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
    )
    return output, state


fused_recurrent_kda_op = torch.ops.cannbotdsl_fused_recurrent_kda.fused_recurrent_kda
fused_recurrent_kda_functional_op = (
    torch.ops.cannbotdsl_fused_recurrent_kda.fused_recurrent_kda_functional
)
