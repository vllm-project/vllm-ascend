#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#


import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend import envs
from vllm_ascend.ops.triton.triton_utils import extract_slice, get_ub_size_bytes, get_vectorcore_num, insert_slice

from .ac_dispatch_policy import ShapeKey, layout_valid, select_dispatch


@triton.jit(
    do_not_specialize=["num_tokens", "front_core_num", "num_tokens_each_front_core", "num_tokens_each_tail_core"]
)
def split_qkv_rmsnorm_mrope_kernel(
    in_qkv_ptr: torch.Tensor,
    q_weight_ptr: torch.Tensor,
    q_bias_ptr: torch.Tensor,
    k_weight_ptr: torch.Tensor,
    k_bias_ptr: torch.Tensor,
    cos_sin_ptr: torch.Tensor,
    out_q_ptr: torch.Tensor,
    out_k_ptr: torch.Tensor,
    out_v_ptr: torch.Tensor,
    out_gate_ptr: torch.Tensor,
    num_tokens,
    front_core_num,
    num_tokens_each_front_core,
    num_tokens_each_tail_core,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_size: tl.constexpr,
    q_size: tl.constexpr,
    kv_size: tl.constexpr,
    eps: tl.constexpr,
    mrope_section_t,
    mrope_section_h,
    mrope_section_w,
    has_bias: tl.constexpr,
    is_interleaved: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope_dim: tl.constexpr,
    IS_PARTIAL_ROPE: tl.constexpr,
    gate_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    tl.static_assert(BLOCK_M == 1 or BLOCK_M == 2, "BLOCK_M must be 1 or 2")
    block_idx = tl.program_id(0)

    loop_num = num_tokens_each_front_core
    if block_idx >= front_core_num:
        loop_num = num_tokens_each_tail_core

    block_offset = num_tokens_each_front_core * block_idx
    if block_idx >= front_core_num:
        block_offset = (
            num_tokens_each_front_core * front_core_num + (block_idx - front_core_num) * num_tokens_each_tail_core
        )

    q_rmsnorm_weight = tl.load(q_weight_ptr + tl.arange(0, head_size))
    k_rmsnorm_weight = tl.load(k_weight_ptr + tl.arange(0, head_size))

    if has_bias:
        q_bias = tl.load(q_bias_ptr + tl.arange(0, head_size))
        k_bias = tl.load(k_bias_ptr + tl.arange(0, head_size))

    if BLOCK_M == 1:
        for index in range(loop_num):
            ## load ##
            # q
            in_q_offset = in_qkv_ptr + (block_offset + index) * (q_size + gate_size + 2 * kv_size)
            if gate_size > 0:
                in_q_gate_tensor = (
                    tl.load(in_q_offset + tl.arange(0, q_size + gate_size))
                    .to(tl.float32)
                    .reshape(num_q_heads, head_size * 2)
                )
                in_q_tensor = extract_slice(
                    in_q_gate_tensor,
                    offsets=(0, 0),
                    sizes=(num_q_heads, head_size),
                    strides=(1, 1),
                )
                in_gate_tensor = extract_slice(
                    in_q_gate_tensor,
                    offsets=(0, head_size),
                    sizes=(num_q_heads, head_size),
                    strides=(1, 1),
                ).reshape(q_size)
            else:
                in_q_tensor = tl.load(in_q_offset + tl.arange(0, q_size)).to(tl.float32).reshape(num_q_heads, head_size)

            # k
            in_k_offset = in_q_offset + q_size + gate_size
            in_k_tensor = tl.load(in_k_offset + tl.arange(0, kv_size)).to(tl.float32).reshape(num_kv_heads, head_size)
            # v
            in_v_offset = in_k_offset + kv_size
            in_v_tensor = tl.load(in_v_offset + tl.arange(0, kv_size))

            # cos, sin
            cos_offsets = tl.arange(0, half_rope_dim)
            if is_interleaved:
                h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
                w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
                t_mask = ~(h_mask | w_mask)
            else:
                t_mask = cos_offsets < mrope_section_t
                h_mask = (mrope_section_t - 1 < cos_offsets) & (cos_offsets < mrope_section_t + mrope_section_h)
                w_mask = (mrope_section_t + mrope_section_h - 1 < cos_offsets) & (
                    cos_offsets < mrope_section_t + mrope_section_h + mrope_section_w
                )

            t_cos_offset = cos_sin_ptr + (block_offset + index) * rope_dim
            h_cos_offset = t_cos_offset + num_tokens * rope_dim
            w_cos_offset = h_cos_offset + num_tokens * rope_dim

            t_sin_offset = cos_sin_ptr + (block_offset + index) * rope_dim + half_rope_dim
            h_sin_offset = t_sin_offset + num_tokens * rope_dim
            w_sin_offset = h_sin_offset + num_tokens * rope_dim

            t_cos_tensor = tl.load(t_cos_offset + cos_offsets, mask=t_mask, other=0)
            h_cos_tensor = tl.load(h_cos_offset + cos_offsets, mask=h_mask, other=0)
            w_cos_tensor = tl.load(w_cos_offset + cos_offsets, mask=w_mask, other=0)
            t_sin_tensor = tl.load(t_sin_offset + cos_offsets, mask=t_mask, other=0)
            h_sin_tensor = tl.load(h_sin_offset + cos_offsets, mask=h_mask, other=0)
            w_sin_tensor = tl.load(w_sin_offset + cos_offsets, mask=w_mask, other=0)

            cos_half = (t_cos_tensor + h_cos_tensor + w_cos_tensor).to(tl.float32)
            sin_half = (t_sin_tensor + h_sin_tensor + w_sin_tensor).to(tl.float32)

            ## compute ##
            # q-rmsnorm
            squares = in_q_tensor * in_q_tensor
            variances = tl.sum(squares, axis=1) / head_size
            reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(num_q_heads, 1)
            q_normalized = in_q_tensor * reciprocal_std
            q_normalized = q_normalized * q_rmsnorm_weight
            if has_bias:
                q_normalized = q_normalized + q_bias

            # k-rmsnorm
            squares = in_k_tensor * in_k_tensor
            variances = tl.sum(squares, axis=1) / head_size
            reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(num_kv_heads, 1)
            k_normalized = in_k_tensor * reciprocal_std
            k_normalized = k_normalized * k_rmsnorm_weight
            if has_bias:
                k_normalized = k_normalized + k_bias

            # q-mrope
            x1 = extract_slice(
                q_normalized,
                offsets=(0, 0),
                sizes=(num_q_heads, half_rope_dim),
                strides=(1, 1),
            )
            x2 = extract_slice(
                q_normalized,
                offsets=(0, half_rope_dim),
                sizes=(num_q_heads, half_rope_dim),
                strides=(1, 1),
            )
            roped_q = insert_slice(
                q_normalized,
                x1 * cos_half - x2 * sin_half,
                offsets=(0, 0),
                sizes=(num_q_heads, half_rope_dim),
                strides=(1, 1),
            )
            roped_q = insert_slice(
                roped_q,
                x2 * cos_half + x1 * sin_half,
                offsets=(0, half_rope_dim),
                sizes=(num_q_heads, half_rope_dim),
                strides=(1, 1),
            )

            # k-mrope
            y1 = extract_slice(
                k_normalized,
                offsets=(0, 0),
                sizes=(num_kv_heads, half_rope_dim),
                strides=(1, 1),
            )
            y2 = extract_slice(
                k_normalized,
                offsets=(0, half_rope_dim),
                sizes=(num_kv_heads, half_rope_dim),
                strides=(1, 1),
            )
            roped_k = insert_slice(
                k_normalized,
                y1 * cos_half - y2 * sin_half,
                offsets=(0, 0),
                sizes=(num_kv_heads, half_rope_dim),
                strides=(1, 1),
            )
            roped_k = insert_slice(
                roped_k,
                y2 * cos_half + y1 * sin_half,
                offsets=(0, half_rope_dim),
                sizes=(num_kv_heads, half_rope_dim),
                strides=(1, 1),
            )

            q_normalized = roped_q
            k_normalized = roped_k

            ## store ##
            # out_q
            out_q_offset = out_q_ptr + (block_offset + index) * q_size
            out_q_indices = tl.arange(0, q_size)
            tl.store(out_q_offset + out_q_indices, q_normalized.reshape(q_size))

            # out_k
            out_k_offset = out_k_ptr + (block_offset + index) * kv_size
            out_k_indices = tl.arange(0, kv_size)
            tl.store(out_k_offset + out_k_indices, k_normalized.reshape(kv_size))

            # out_v
            out_v_offset = out_v_ptr + (block_offset + index) * kv_size
            tl.store(out_v_offset + tl.arange(0, kv_size), in_v_tensor)

            # out_gate
            if gate_size > 0:
                out_gate_offset = out_gate_ptr + (block_offset + index) * gate_size
                tl.store(out_gate_offset + tl.arange(0, gate_size), in_gate_tensor)
    elif BLOCK_M == 2:
        # Process paired rows with bounded Q head tiles; keep an independent
        # M1-style single-row body for the odd last token.
        num_pairs = loop_num // 2
        n_tail = loop_num - num_pairs * 2
        NK: tl.constexpr = BLOCK_M * num_kv_heads
        # Constexpr selection uses AnnAssign first-definitions inside each
        # static-if branch: a plain re-assignment of a tl.constexpr name
        # demotes it to a scalar tensor on Triton-Ascend 3.2.1 and breaks
        # the constexpr shape contract of the tl.arange/reshape uses below.
        # BLOCK_H_Q = min(num_q_heads, 12)
        if num_q_heads < 12:
            BLOCK_H_Q: tl.constexpr = num_q_heads
        else:
            BLOCK_H_Q: tl.constexpr = 12  # type: ignore[no-redef]
        # Per-head GM slot width inside the q(+gate) segment of each row:
        # [q_h | gate_h] = head_size * 2 with gate, [q_h] = head_size without.
        if gate_size == 0:
            HEAD_SLOT: tl.constexpr = head_size
        else:
            HEAD_SLOT: tl.constexpr = head_size * 2  # type: ignore[no-redef]
        NQ_TILE: tl.constexpr = BLOCK_M * BLOCK_H_Q
        N_FULL_TILES: tl.constexpr = num_q_heads // BLOCK_H_Q
        N_BOUNDARY_HEADS: tl.constexpr = num_q_heads - N_FULL_TILES * BLOCK_H_Q
        for index in range(num_pairs):
            ## load ##
            tok = index * BLOCK_M + tl.arange(0, BLOCK_M)
            valid = tok < loop_num
            rows = (block_offset + tok)[:, None]
            row_stride = q_size + gate_size + 2 * kv_size

            # cos, sin — per-plane (BLOCK_M, half_rope_dim) 2D masked loads
            cos_offsets = tl.arange(0, half_rope_dim)
            if is_interleaved:
                h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
                w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
                t_mask = ~(h_mask | w_mask)
            else:
                t_mask = cos_offsets < mrope_section_t
                h_mask = (mrope_section_t - 1 < cos_offsets) & (cos_offsets < mrope_section_t + mrope_section_h)
                w_mask = (mrope_section_t + mrope_section_h - 1 < cos_offsets) & (
                    cos_offsets < mrope_section_t + mrope_section_h + mrope_section_w
                )

            # rows already carries the full per-token row offset
            # (block_offset + index*BLOCK_M + arange); applying the extra
            # (block_offset + index*BLOCK_M) term again would read the wrong
            # cos/sin rows for every block except the first. Use rows * rope_dim.
            cos_base = cos_sin_ptr + rows * rope_dim
            t_cos_tensor = tl.load(
                cos_base + cos_offsets[None, :],
                mask=valid[:, None] & t_mask[None, :],
                other=0,
            )
            h_cos_tensor = tl.load(
                cos_base + num_tokens * rope_dim + cos_offsets[None, :],
                mask=valid[:, None] & h_mask[None, :],
                other=0,
            )
            w_cos_tensor = tl.load(
                cos_base + 2 * num_tokens * rope_dim + cos_offsets[None, :],
                mask=valid[:, None] & w_mask[None, :],
                other=0,
            )
            t_sin_tensor = tl.load(
                cos_base + half_rope_dim + cos_offsets[None, :],
                mask=valid[:, None] & t_mask[None, :],
                other=0,
            )
            h_sin_tensor = tl.load(
                cos_base + num_tokens * rope_dim + half_rope_dim + cos_offsets[None, :],
                mask=valid[:, None] & h_mask[None, :],
                other=0,
            )
            w_sin_tensor = tl.load(
                cos_base + 2 * num_tokens * rope_dim + half_rope_dim + cos_offsets[None, :],
                mask=valid[:, None] & w_mask[None, :],
                other=0,
            )

            cos_half = (t_cos_tensor + h_cos_tensor + w_cos_tensor).to(tl.float32)
            sin_half = (t_sin_tensor + h_sin_tensor + w_sin_tensor).to(tl.float32)

            ## Q/Gate full head tiles — no head-dim mask, token-tail valid only ##
            for h_tile in range(N_FULL_TILES):
                tile_cols = tl.arange(0, BLOCK_H_Q * HEAD_SLOT)
                in_q_gate_tile = tl.load(
                    in_qkv_ptr + rows * row_stride + h_tile * BLOCK_H_Q * HEAD_SLOT + tile_cols[None, :],
                    mask=valid[:, None],
                    other=0,
                )
                if gate_size > 0:
                    q_gate = in_q_gate_tile.to(tl.float32).reshape(NQ_TILE, head_size * 2)
                    in_q_tensor = extract_slice(
                        q_gate,
                        offsets=(0, 0),
                        sizes=(NQ_TILE, head_size),
                        strides=(1, 1),
                    )
                    in_gate_tensor = extract_slice(
                        q_gate,
                        offsets=(0, head_size),
                        sizes=(NQ_TILE, head_size),
                        strides=(1, 1),
                    )
                else:
                    in_q_tensor = in_q_gate_tile.to(tl.float32).reshape(NQ_TILE, head_size)

                # head replication of cos/sin for the current tile
                cos_q = tl.broadcast_to(cos_half[:, None, :], (BLOCK_M, BLOCK_H_Q, half_rope_dim)).reshape(
                    NQ_TILE, half_rope_dim
                )
                sin_q = tl.broadcast_to(sin_half[:, None, :], (BLOCK_M, BLOCK_H_Q, half_rope_dim)).reshape(
                    NQ_TILE, half_rope_dim
                )

                # q-rmsnorm
                squares = in_q_tensor * in_q_tensor
                variances = tl.sum(squares, axis=1) / head_size
                reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(NQ_TILE, 1)
                q_normalized = in_q_tensor * reciprocal_std
                q_normalized = q_normalized * q_rmsnorm_weight
                if has_bias:
                    q_normalized = q_normalized + q_bias

                # q-mrope
                x1 = extract_slice(
                    q_normalized,
                    offsets=(0, 0),
                    sizes=(NQ_TILE, half_rope_dim),
                    strides=(1, 1),
                )
                x2 = extract_slice(
                    q_normalized,
                    offsets=(0, half_rope_dim),
                    sizes=(NQ_TILE, half_rope_dim),
                    strides=(1, 1),
                )
                roped_q = insert_slice(
                    q_normalized,
                    x1 * cos_q - x2 * sin_q,
                    offsets=(0, 0),
                    sizes=(NQ_TILE, half_rope_dim),
                    strides=(1, 1),
                )
                roped_q = insert_slice(
                    roped_q,
                    x2 * cos_q + x1 * sin_q,
                    offsets=(0, half_rope_dim),
                    sizes=(NQ_TILE, half_rope_dim),
                    strides=(1, 1),
                )
                q_normalized = roped_q

                ## store ##
                store_cols = tl.arange(0, BLOCK_H_Q * head_size)
                tl.store(
                    out_q_ptr + rows * q_size + h_tile * BLOCK_H_Q * head_size + store_cols[None, :],
                    roped_q.reshape(BLOCK_M, BLOCK_H_Q * head_size),
                    mask=valid[:, None],
                )
                if gate_size > 0:
                    tl.store(
                        out_gate_ptr + rows * gate_size + h_tile * BLOCK_H_Q * head_size + store_cols[None, :],
                        in_gate_tensor.reshape(BLOCK_M, BLOCK_H_Q * head_size),
                        mask=valid[:, None],
                    )

            ## boundary tile — the only head-masked Q/Gate tile ##
            if N_BOUNDARY_HEADS > 0:
                tile_cols = tl.arange(0, BLOCK_H_Q * HEAD_SLOT)
                tile_head = N_FULL_TILES * BLOCK_H_Q + tile_cols // HEAD_SLOT
                head_ok = tile_head < num_q_heads
                in_q_gate_tile = tl.load(
                    in_qkv_ptr + rows * row_stride + N_FULL_TILES * BLOCK_H_Q * HEAD_SLOT + tile_cols[None, :],
                    mask=valid[:, None] & head_ok[None, :],
                    other=0,
                )
                if gate_size > 0:
                    q_gate = in_q_gate_tile.to(tl.float32).reshape(NQ_TILE, head_size * 2)
                    in_q_tensor = extract_slice(
                        q_gate,
                        offsets=(0, 0),
                        sizes=(NQ_TILE, head_size),
                        strides=(1, 1),
                    )
                    in_gate_tensor = extract_slice(
                        q_gate,
                        offsets=(0, head_size),
                        sizes=(NQ_TILE, head_size),
                        strides=(1, 1),
                    )
                else:
                    in_q_tensor = in_q_gate_tile.to(tl.float32).reshape(NQ_TILE, head_size)

                # head replication of cos/sin for the current tile
                cos_q = tl.broadcast_to(cos_half[:, None, :], (BLOCK_M, BLOCK_H_Q, half_rope_dim)).reshape(
                    NQ_TILE, half_rope_dim
                )
                sin_q = tl.broadcast_to(sin_half[:, None, :], (BLOCK_M, BLOCK_H_Q, half_rope_dim)).reshape(
                    NQ_TILE, half_rope_dim
                )

                # q-rmsnorm
                squares = in_q_tensor * in_q_tensor
                variances = tl.sum(squares, axis=1) / head_size
                reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(NQ_TILE, 1)
                q_normalized = in_q_tensor * reciprocal_std
                q_normalized = q_normalized * q_rmsnorm_weight
                if has_bias:
                    q_normalized = q_normalized + q_bias

                # q-mrope
                x1 = extract_slice(
                    q_normalized,
                    offsets=(0, 0),
                    sizes=(NQ_TILE, half_rope_dim),
                    strides=(1, 1),
                )
                x2 = extract_slice(
                    q_normalized,
                    offsets=(0, half_rope_dim),
                    sizes=(NQ_TILE, half_rope_dim),
                    strides=(1, 1),
                )
                roped_q = insert_slice(
                    q_normalized,
                    x1 * cos_q - x2 * sin_q,
                    offsets=(0, 0),
                    sizes=(NQ_TILE, half_rope_dim),
                    strides=(1, 1),
                )
                roped_q = insert_slice(
                    roped_q,
                    x2 * cos_q + x1 * sin_q,
                    offsets=(0, half_rope_dim),
                    sizes=(NQ_TILE, half_rope_dim),
                    strides=(1, 1),
                )
                q_normalized = roped_q

                ## store ##
                store_cols = tl.arange(0, BLOCK_H_Q * head_size)
                store_head = N_FULL_TILES * BLOCK_H_Q + store_cols // head_size
                store_head_ok = store_head < num_q_heads
                tl.store(
                    out_q_ptr + rows * q_size + N_FULL_TILES * BLOCK_H_Q * head_size + store_cols[None, :],
                    roped_q.reshape(BLOCK_M, BLOCK_H_Q * head_size),
                    mask=valid[:, None] & store_head_ok[None, :],
                )
                if gate_size > 0:
                    tl.store(
                        out_gate_ptr + rows * gate_size + N_FULL_TILES * BLOCK_H_Q * head_size + store_cols[None, :],
                        in_gate_tensor.reshape(BLOCK_M, BLOCK_H_Q * head_size),
                        mask=valid[:, None] & store_head_ok[None, :],
                    )
            ## K — after all Q tiles, once per token group ##
            in_k_block = tl.load(
                in_qkv_ptr + rows * row_stride + (q_size + gate_size) + tl.arange(0, kv_size)[None, :],
                mask=valid[:, None],
                other=0,
            )
            in_k_tensor = in_k_block.to(tl.float32).reshape(NK, head_size)

            cos_k = tl.broadcast_to(cos_half[:, None, :], (BLOCK_M, num_kv_heads, half_rope_dim)).reshape(
                NK, half_rope_dim
            )
            sin_k = tl.broadcast_to(sin_half[:, None, :], (BLOCK_M, num_kv_heads, half_rope_dim)).reshape(
                NK, half_rope_dim
            )

            # k-rmsnorm
            squares = in_k_tensor * in_k_tensor
            variances = tl.sum(squares, axis=1) / head_size
            reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(NK, 1)
            k_normalized = in_k_tensor * reciprocal_std
            k_normalized = k_normalized * k_rmsnorm_weight
            if has_bias:
                k_normalized = k_normalized + k_bias

            # k-mrope
            y1 = extract_slice(
                k_normalized,
                offsets=(0, 0),
                sizes=(NK, half_rope_dim),
                strides=(1, 1),
            )
            y2 = extract_slice(
                k_normalized,
                offsets=(0, half_rope_dim),
                sizes=(NK, half_rope_dim),
                strides=(1, 1),
            )
            roped_k = insert_slice(
                k_normalized,
                y1 * cos_k - y2 * sin_k,
                offsets=(0, 0),
                sizes=(NK, half_rope_dim),
                strides=(1, 1),
            )
            roped_k = insert_slice(
                roped_k,
                y2 * cos_k + y1 * sin_k,
                offsets=(0, half_rope_dim),
                sizes=(NK, half_rope_dim),
                strides=(1, 1),
            )
            k_normalized = roped_k

            # out_k
            tl.store(
                out_k_ptr + rows * kv_size + tl.arange(0, kv_size)[None, :],
                roped_k.reshape(BLOCK_M, kv_size),
                mask=valid[:, None],
            )

            ## V — pure move, once per token group ##
            in_v_block = tl.load(
                in_qkv_ptr + rows * row_stride + (q_size + gate_size + kv_size) + tl.arange(0, kv_size)[None, :],
                mask=valid[:, None],
                other=0,
            )
            tl.store(
                out_v_ptr + rows * kv_size + tl.arange(0, kv_size)[None, :],
                in_v_block,
                mask=valid[:, None],
            )
        # An odd final token uses the independent single-row dataflow.
        for tail_index in range(n_tail):
            tail_row = block_offset + num_pairs * 2 + tail_index
            ## load ##
            # q
            tail_in_q_offset = in_qkv_ptr + tail_row * (q_size + gate_size + 2 * kv_size)
            if gate_size > 0:
                tail_in_q_gate_tensor = (
                    tl.load(tail_in_q_offset + tl.arange(0, q_size + gate_size))
                    .to(tl.float32)
                    .reshape(num_q_heads, head_size * 2)
                )
                tail_in_q_tensor = extract_slice(
                    tail_in_q_gate_tensor,
                    offsets=(0, 0),
                    sizes=(num_q_heads, head_size),
                    strides=(1, 1),
                )
                tail_in_gate_tensor = extract_slice(
                    tail_in_q_gate_tensor,
                    offsets=(0, head_size),
                    sizes=(num_q_heads, head_size),
                    strides=(1, 1),
                ).reshape(q_size)
            else:
                tail_in_q_tensor = (
                    tl.load(tail_in_q_offset + tl.arange(0, q_size)).to(tl.float32).reshape(num_q_heads, head_size)
                )

            # k
            tail_in_k_offset = tail_in_q_offset + q_size + gate_size
            tail_in_k_tensor = (
                tl.load(tail_in_k_offset + tl.arange(0, kv_size)).to(tl.float32).reshape(num_kv_heads, head_size)
            )
            # v
            tail_in_v_offset = tail_in_k_offset + kv_size
            tail_in_v_tensor = tl.load(tail_in_v_offset + tl.arange(0, kv_size))

            # cos, sin
            tail_cos_offsets = tl.arange(0, half_rope_dim)
            if is_interleaved:
                tail_h_mask = ((tail_cos_offsets % 3) == 1) & (tail_cos_offsets <= 3 * mrope_section_h)
                tail_w_mask = ((tail_cos_offsets % 3) == 2) & (tail_cos_offsets <= 3 * mrope_section_w)
                tail_t_mask = ~(tail_h_mask | tail_w_mask)
            else:
                tail_t_mask = tail_cos_offsets < mrope_section_t
                tail_h_mask = (mrope_section_t - 1 < tail_cos_offsets) & (
                    tail_cos_offsets < mrope_section_t + mrope_section_h
                )
                tail_w_mask = (mrope_section_t + mrope_section_h - 1 < tail_cos_offsets) & (
                    tail_cos_offsets < mrope_section_t + mrope_section_h + mrope_section_w
                )

            tail_t_cos_offset = cos_sin_ptr + tail_row * rope_dim
            tail_h_cos_offset = tail_t_cos_offset + num_tokens * rope_dim
            tail_w_cos_offset = tail_h_cos_offset + num_tokens * rope_dim

            tail_t_sin_offset = cos_sin_ptr + tail_row * rope_dim + half_rope_dim
            tail_h_sin_offset = tail_t_sin_offset + num_tokens * rope_dim
            tail_w_sin_offset = tail_h_sin_offset + num_tokens * rope_dim

            tail_t_cos_tensor = tl.load(tail_t_cos_offset + tail_cos_offsets, mask=tail_t_mask, other=0)
            tail_h_cos_tensor = tl.load(tail_h_cos_offset + tail_cos_offsets, mask=tail_h_mask, other=0)
            tail_w_cos_tensor = tl.load(tail_w_cos_offset + tail_cos_offsets, mask=tail_w_mask, other=0)
            tail_t_sin_tensor = tl.load(tail_t_sin_offset + tail_cos_offsets, mask=tail_t_mask, other=0)
            tail_h_sin_tensor = tl.load(tail_h_sin_offset + tail_cos_offsets, mask=tail_h_mask, other=0)
            tail_w_sin_tensor = tl.load(tail_w_sin_offset + tail_cos_offsets, mask=tail_w_mask, other=0)

            tail_cos_half = (tail_t_cos_tensor + tail_h_cos_tensor + tail_w_cos_tensor).to(tl.float32)
            tail_sin_half = (tail_t_sin_tensor + tail_h_sin_tensor + tail_w_sin_tensor).to(tl.float32)

            ## compute ##
            # q-rmsnorm
            tail_squares = tail_in_q_tensor * tail_in_q_tensor
            tail_variances = tl.sum(tail_squares, axis=1) / head_size
            tail_reciprocal_std = (1 / tl.sqrt(tail_variances + eps)).reshape(num_q_heads, 1)
            tail_q_normalized = tail_in_q_tensor * tail_reciprocal_std
            tail_q_normalized = tail_q_normalized * q_rmsnorm_weight
            if has_bias:
                tail_q_normalized = tail_q_normalized + q_bias

            # k-rmsnorm
            tail_squares = tail_in_k_tensor * tail_in_k_tensor
            tail_variances = tl.sum(tail_squares, axis=1) / head_size
            tail_reciprocal_std = (1 / tl.sqrt(tail_variances + eps)).reshape(num_kv_heads, 1)
            tail_k_normalized = tail_in_k_tensor * tail_reciprocal_std
            tail_k_normalized = tail_k_normalized * k_rmsnorm_weight
            if has_bias:
                tail_k_normalized = tail_k_normalized + k_bias

            # q-mrope
            tail_x1 = extract_slice(
                tail_q_normalized,
                offsets=(0, 0),
                sizes=(num_q_heads, half_rope_dim),
                strides=(1, 1),
            )
            tail_x2 = extract_slice(
                tail_q_normalized,
                offsets=(0, half_rope_dim),
                sizes=(num_q_heads, half_rope_dim),
                strides=(1, 1),
            )
            tail_roped_q = insert_slice(
                tail_q_normalized,
                tail_x1 * tail_cos_half - tail_x2 * tail_sin_half,
                offsets=(0, 0),
                sizes=(num_q_heads, half_rope_dim),
                strides=(1, 1),
            )
            tail_roped_q = insert_slice(
                tail_roped_q,
                tail_x2 * tail_cos_half + tail_x1 * tail_sin_half,
                offsets=(0, half_rope_dim),
                sizes=(num_q_heads, half_rope_dim),
                strides=(1, 1),
            )

            # k-mrope
            tail_y1 = extract_slice(
                tail_k_normalized,
                offsets=(0, 0),
                sizes=(num_kv_heads, half_rope_dim),
                strides=(1, 1),
            )
            tail_y2 = extract_slice(
                tail_k_normalized,
                offsets=(0, half_rope_dim),
                sizes=(num_kv_heads, half_rope_dim),
                strides=(1, 1),
            )
            tail_roped_k = insert_slice(
                tail_k_normalized,
                tail_y1 * tail_cos_half - tail_y2 * tail_sin_half,
                offsets=(0, 0),
                sizes=(num_kv_heads, half_rope_dim),
                strides=(1, 1),
            )
            tail_roped_k = insert_slice(
                tail_roped_k,
                tail_y2 * tail_cos_half + tail_y1 * tail_sin_half,
                offsets=(0, half_rope_dim),
                sizes=(num_kv_heads, half_rope_dim),
                strides=(1, 1),
            )

            tail_q_normalized = tail_roped_q
            tail_k_normalized = tail_roped_k

            ## store ##
            # out_q
            tail_out_q_offset = out_q_ptr + tail_row * q_size
            tail_out_q_indices = tl.arange(0, q_size)
            tl.store(tail_out_q_offset + tail_out_q_indices, tail_q_normalized.reshape(q_size))
            # out_k
            tail_out_k_offset = out_k_ptr + tail_row * kv_size
            tail_out_k_indices = tl.arange(0, kv_size)
            tl.store(tail_out_k_offset + tail_out_k_indices, tail_k_normalized.reshape(kv_size))
            # out_v
            tail_out_v_offset = out_v_ptr + tail_row * kv_size
            tl.store(tail_out_v_offset + tl.arange(0, kv_size), tail_in_v_tensor)
            # out_gate
            if gate_size > 0:
                tail_out_gate_offset = out_gate_ptr + tail_row * gate_size
                tl.store(tail_out_gate_offset + tl.arange(0, gate_size), tail_in_gate_tensor)


def _get_ub_capacity() -> int | None:
    # The shared helper owns initialization, its default and its override.
    # This is routing capacity, not measured free UB or an allocator budget.
    try:
        return get_ub_size_bytes()
    except Exception:
        return None


_M2_KERNEL = triton.autotune(
    configs=[triton.Config({"multibuffer": False}, num_stages=1, num_warps=32)],
    key=[],
)(split_qkv_rmsnorm_mrope_kernel)


def triton_split_qkv_rmsnorm_mrope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    mrope_section: list[int],
    is_interleaved: bool,
    rope_dim: int | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    has_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    core_num = get_vectorcore_num()

    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    num_tokens = qkv.shape[0]

    gate_size = q_size if has_gate else 0

    if rope_dim is None:
        rope_dim = head_size
    IS_PARTIAL_ROPE = rope_dim != head_size

    front_core_num = core_num
    if num_tokens % core_num != 0:
        front_core_num = num_tokens % core_num

    num_tokens_each_front_core = (num_tokens + core_num - 1) // core_num

    tail_core_num = 0
    if num_tokens > core_num:
        tail_core_num = core_num - front_core_num

    num_tokens_each_tail_core = num_tokens // core_num

    q_output = torch.empty(num_tokens, q_size, device=qkv.device, dtype=qkv.dtype)
    k_output = torch.empty(num_tokens, kv_size, device=qkv.device, dtype=qkv.dtype)
    v_output = torch.empty(num_tokens, kv_size, device=qkv.device, dtype=qkv.dtype)
    gate_output = torch.empty(num_tokens, gate_size, device=qkv.device, dtype=qkv.dtype)

    total_core = front_core_num + tail_core_num
    block_dim = core_num
    if total_core < core_num:
        block_dim = total_core

    has_bias = q_bias is not None

    # The centralized env getter is evaluated on each call, preserving the
    # tested default request of 2 and the explicit M1 diagnostic path.
    block_m = envs.VLLM_ASCEND_SPLIT_QKV_RMSNORM_MROPE_BLOCK_M

    # Reuse the launch partition instead of constructing it again in the
    # selector. There is only one M2 instance; short/no-pair work uses M1.
    if block_m == 2:
        dtype_tag = str(qkv.dtype).replace("torch.", "")
        if dtype_tag not in ("bfloat16", "float16"):
            block_m = 1
        else:
            shape = ShapeKey(
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                has_gate=has_gate,
                rope_dim=rope_dim,
                has_bias=has_bias,
                is_interleaved=is_interleaved,
                mrope_section=(mrope_section[0], mrope_section[1], mrope_section[2]),
                eps=eps,
                dtype=dtype_tag,
            )
            decision = select_dispatch(
                shape=shape,
                min_active_tokens=num_tokens_each_tail_core if tail_core_num else num_tokens_each_front_core,
                max_active_tokens=num_tokens_each_front_core,
                current_layout_valid=layout_valid(
                    qkv,
                    q_weight,
                    k_weight,
                    cos_sin,
                    q_bias,
                    k_bias,
                    num_tokens,
                    q_size,
                    gate_size,
                    kv_size,
                    head_size,
                    rope_dim,
                ),
                capacity_bytes=_get_ub_capacity(),
            )
            block_m = decision.block_m

    launch_kernel = _M2_KERNEL if block_m == 2 else split_qkv_rmsnorm_mrope_kernel
    launch_kernel[(block_dim,)](
        qkv,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        cos_sin,
        q_output,
        k_output,
        v_output,
        gate_output,
        num_tokens,
        front_core_num,
        num_tokens_each_front_core,
        num_tokens_each_tail_core,
        num_q_heads,
        num_kv_heads,
        head_size,
        q_size,
        kv_size,
        eps,
        mrope_section[0],
        mrope_section[1],
        mrope_section[2],
        has_bias,
        is_interleaved,
        rope_dim,
        rope_dim // 2,
        IS_PARTIAL_ROPE,
        gate_size,
        block_m,
    )

    return q_output, k_output, v_output, gate_output


def triton_split_qkv_rmsnorm_mrope_fake(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    mrope_section: list[int],
    is_interleaved: bool,
    rope_dim: int | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    has_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = qkv.shape[0]
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    gate_size = q_size if has_gate else 0

    q_output = torch.empty(
        num_tokens,
        q_size,
        device=qkv.device,
        dtype=qkv.dtype,
    )

    k_output = torch.empty(
        num_tokens,
        kv_size,
        device=qkv.device,
        dtype=qkv.dtype,
    )

    v_output = torch.empty(
        num_tokens,
        kv_size,
        device=qkv.device,
        dtype=qkv.dtype,
    )

    gate_output = torch.empty(
        num_tokens,
        gate_size,
        device=qkv.device,
        dtype=qkv.dtype,
    )

    return q_output, k_output, v_output, gate_output


direct_register_custom_op(
    op_name="triton_split_qkv_rmsnorm_mrope",
    op_func=triton_split_qkv_rmsnorm_mrope,
    fake_impl=triton_split_qkv_rmsnorm_mrope_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
