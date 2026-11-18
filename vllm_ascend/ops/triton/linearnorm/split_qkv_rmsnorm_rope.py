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
import torch_npu
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import LayerNameType, _resolve_layer_name, direct_register_custom_op

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.triton.triton_utils import (
    extract_slice,
    get_element,
    get_vectorcore_num,
    init_device_properties_triton,
    insert_slice,
)

_get_raw_stream = getattr(getattr(torch_npu, "_C", None), "_npu_getCurrentRawStream", None)


@triton.jit
def split_qkv_rmsnorm_rope_kernel(
    input_gm_ptr,
    q_gm_ptr,
    k_gm_ptr,
    v_gm_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    k_bias_ptr,
    batch_size,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BIAS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_ROPE_DIM: tl.constexpr,
    IS_PARTIAL_ROPE: tl.constexpr,
    num_vectorcore: tl.constexpr,
    batch_size_per_iter_per_vec: tl.constexpr,
    qk_head_nums_per_iter_per_vec: tl.constexpr,
    q_head_num: tl.constexpr,
    kv_head_num: tl.constexpr,
    qk_head_num_sum: tl.constexpr,
    v_batch_size_per_iter_per_vec: tl.constexpr,
    positions_gm_ptr,
    cos_sin_cache_gm_ptr,
):
    row_pid = tl.program_id(0)

    q_weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM))
    k_weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM))

    batch_size_per_vec = tl.cdiv(batch_size, num_vectorcore)
    iter_num_per_vec = tl.cdiv(batch_size_per_vec, batch_size_per_iter_per_vec)
    v_iter_num_per_vec = tl.cdiv(batch_size_per_vec, v_batch_size_per_iter_per_vec)
    input_batch_offset = row_pid * batch_size_per_vec
    mblk_idx = tl.arange(0, batch_size_per_iter_per_vec) + input_batch_offset
    nblk_idx = tl.arange(0, q_hidden_size + kv_hidden_size)
    nmask = nblk_idx < total_hidden_size

    input_batch_offset_end = min(input_batch_offset + batch_size_per_vec, batch_size)

    pos_indices = input_batch_offset + tl.arange(0, batch_size_per_iter_per_vec)
    output_q_nblk_idx = tl.arange(0, q_hidden_size)
    output_q_nmask = output_q_nblk_idx < q_hidden_size
    output_kv_nblk_idx = tl.arange(0, kv_hidden_size)
    output_kv_nmask = output_kv_nblk_idx < kv_hidden_size
    sin_cos_range = tl.arange(0, ROPE_DIM)
    cos_sin_cache_offset = cos_sin_cache_gm_ptr + sin_cos_range

    for iter in tl.range(iter_num_per_vec):
        pos_offset = iter * batch_size_per_iter_per_vec
        x = tl.load(
            positions_gm_ptr + pos_indices + pos_offset, mask=(pos_indices + pos_offset) < input_batch_offset_end
        )
        mmask = (mblk_idx + pos_offset) < input_batch_offset_end
        mask = (mmask[:, None]) & (nmask[None, :])
        idx = (mblk_idx + pos_offset)[:, None] * total_hidden_size + nblk_idx[None, :]
        values_tmp1 = tl.load(input_gm_ptr + idx, mask=mask).reshape(qk_head_nums_per_iter_per_vec, HEAD_DIM)
        if BIAS:
            q_bias_values = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM))
            k_bias_values = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM))

        values_tmp3 = tl.zeros((batch_size_per_iter_per_vec, ROPE_DIM), dtype=tl.bfloat16)
        for i in tl.range(batch_size_per_iter_per_vec):
            pos = get_element(x, (i,))
            values_tmp3 = insert_slice(
                values_tmp3.reshape(batch_size_per_iter_per_vec, ROPE_DIM),
                tl.load(pos * ROPE_DIM + cos_sin_cache_offset[:, None]).reshape(1, ROPE_DIM),
                offsets=(i, 0),
                sizes=(1, ROPE_DIM),
                strides=(1, 1),
            )
        values_tmp3 = values_tmp3.reshape(batch_size_per_iter_per_vec, 1, ROPE_DIM)
        cos = extract_slice(
            values_tmp3,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, 1, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        sin = extract_slice(
            values_tmp3,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, 1, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )

        normalized_values = values_tmp1.to(tl.float32)
        normalized_values = normalized_values * normalized_values
        normalized_values = tl.sum(normalized_values, axis=1) / HEAD_DIM
        normalized_values = 1 / tl.sqrt(normalized_values + eps).reshape(qk_head_nums_per_iter_per_vec, 1)
        normalized_values = values_tmp1 * normalized_values

        normalized_values_tmp = extract_slice(
            normalized_values.reshape(batch_size_per_iter_per_vec, qk_head_num_sum, HEAD_DIM),
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HEAD_DIM),
            strides=(1, 1, 1),
        )

        if BIAS:
            normalized_values_tmp = (normalized_values_tmp * q_weight_values + q_bias_values).to(tl.bfloat16)
        else:
            normalized_values_tmp = (normalized_values_tmp * q_weight_values).to(tl.bfloat16)

        # q rope
        values_tmp = tl.zeros((batch_size_per_iter_per_vec, q_head_num, ROPE_DIM), dtype=tl.bfloat16)
        x1 = extract_slice(
            normalized_values_tmp,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        x2 = extract_slice(
            normalized_values_tmp,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp = insert_slice(
            values_tmp,
            x1 * cos - x2 * sin,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp = insert_slice(
            values_tmp,
            x2 * cos + x1 * sin,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        q_output_idx = output_q_nblk_idx[None, :] + (mblk_idx + pos_offset)[:, None] * q_hidden_size
        mask = (mmask[:, None]) & (output_q_nmask[None, :])
        if IS_PARTIAL_ROPE:
            normalized_values_tmp = insert_slice(
                normalized_values_tmp,
                values_tmp,
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, q_head_num, ROPE_DIM),
                strides=(1, 1, 1),
            )
            tl.store(
                q_gm_ptr + q_output_idx,
                normalized_values_tmp.reshape(batch_size_per_iter_per_vec, q_hidden_size),
                mask=mask,
            )
        else:
            tl.store(
                q_gm_ptr + q_output_idx,
                values_tmp.reshape(batch_size_per_iter_per_vec, q_hidden_size),
                mask=mask,
            )

        # k rope
        normalized_values_tmp1 = extract_slice(
            normalized_values.reshape(batch_size_per_iter_per_vec, qk_head_num_sum, HEAD_DIM),
            offsets=(0, q_head_num, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HEAD_DIM),
            strides=(1, 1, 1),
        )

        if BIAS:
            normalized_values_tmp1 = (normalized_values_tmp1 * k_weight_values + k_bias_values).to(tl.bfloat16)
        else:
            normalized_values_tmp1 = (normalized_values_tmp1 * k_weight_values).to(tl.bfloat16)

        values_tmp2 = tl.zeros((batch_size_per_iter_per_vec, kv_head_num, ROPE_DIM), dtype=tl.bfloat16)

        x1 = extract_slice(
            normalized_values_tmp1,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        x2 = extract_slice(
            normalized_values_tmp1,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp2 = insert_slice(
            values_tmp2,
            x1 * cos - x2 * sin,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp2 = insert_slice(
            values_tmp2,
            x2 * cos + x1 * sin,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )

        kv_output_idx = output_kv_nblk_idx[None, :] + (mblk_idx + pos_offset)[:, None] * kv_hidden_size
        mask = (mmask[:, None]) & (output_kv_nmask[None, :])
        if IS_PARTIAL_ROPE:
            normalized_values_tmp1 = insert_slice(
                normalized_values_tmp1,
                values_tmp2,
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, kv_head_num, ROPE_DIM),
                strides=(1, 1, 1),
            )
            tl.store(
                k_gm_ptr + kv_output_idx,
                normalized_values_tmp1.reshape(batch_size_per_iter_per_vec, kv_hidden_size),
                mask=mask,
            )
        else:
            tl.store(
                k_gm_ptr + kv_output_idx,
                values_tmp2.reshape(batch_size_per_iter_per_vec, kv_hidden_size),
                mask=mask,
            )

    mblk_idx = tl.arange(0, v_batch_size_per_iter_per_vec) + input_batch_offset
    nblk_idx = tl.arange(q_hidden_size + kv_hidden_size, total_hidden_size)
    nmask = nblk_idx < total_hidden_size
    out_nblk_idx = tl.arange(0, kv_hidden_size)
    out_nmask = out_nblk_idx < kv_hidden_size

    for _ in tl.range(v_iter_num_per_vec):
        mmask = mblk_idx < input_batch_offset_end
        mask = (mmask[:, None]) & (nmask[None, :])
        idx = mblk_idx[:, None] * total_hidden_size + nblk_idx[None, :]
        values = tl.load(input_gm_ptr + idx, mask=mask)
        out_idx = mblk_idx[:, None] * kv_hidden_size + out_nblk_idx[None, :]
        out_mask = (mmask[:, None]) & (out_nmask[None, :])
        tl.store(v_gm_ptr + out_idx, values, mask=out_mask)
        mblk_idx += v_batch_size_per_iter_per_vec


@triton.jit(do_not_specialize=["batch_size"])
def fast_split_qkv_rope_kernel(
    input_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    cos_sin_ptr,
    pos_ptr,
    qw_ptr,
    kw_ptr,
    batch_size,
    eps: tl.constexpr,
    q_hidden: tl.constexpr,
    kv_hidden: tl.constexpr,
    total_hidden: tl.constexpr,
    q_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope: tl.constexpr,
    num_vec: tl.constexpr,
):
    pid = tl.program_id(0)
    per = tl.cdiv(batch_size, num_vec)
    b0 = pid * per
    b1 = tl.minimum(b0 + per, batch_size)
    qn = tl.arange(0, q_hidden)
    kn = tl.arange(q_hidden, q_hidden + kv_hidden)
    vn = tl.arange(q_hidden + kv_hidden, total_hidden)
    wd = tl.arange(0, head_dim)
    qw = tl.load(qw_ptr + wd).to(tl.float32)
    kw = tl.load(kw_ptr + wd).to(tl.float32)
    out_qn = tl.arange(0, q_hidden)
    kvn = tl.arange(0, kv_hidden)
    half_off = tl.arange(0, half_rope)
    for m in tl.range(b0, b1):
        qv = tl.load(input_ptr + m * total_hidden + qn).to(tl.float32).reshape(q_heads, head_dim)
        qvar = tl.sum(qv * qv, axis=1, keep_dims=True) / head_dim
        qnorm = (qv * tl.math.rsqrt(qvar + eps) * qw[None, :]).to(tl.bfloat16)
        kv = tl.load(input_ptr + m * total_hidden + kn).to(tl.float32).reshape(kv_heads, head_dim)
        kvar = tl.sum(kv * kv, axis=1, keep_dims=True) / head_dim
        knorm = (kv * tl.math.rsqrt(kvar + eps) * kw[None, :]).to(tl.bfloat16)
        pos = tl.load(pos_ptr + m).to(tl.int64)
        csoff = pos * rope_dim + half_off
        cos = tl.load(cos_sin_ptr + csoff)[None, :]
        sin = tl.load(cos_sin_ptr + csoff + half_rope)[None, :]
        qx1 = extract_slice(qnorm, (0, 0), (q_heads, half_rope), (1, 1))
        qx2 = extract_slice(qnorm, (0, half_rope), (q_heads, half_rope), (1, 1))
        qo = tl.zeros((q_heads, rope_dim), dtype=tl.bfloat16)
        qo = insert_slice(qo, (qx1 * cos - qx2 * sin), (0, 0), (q_heads, half_rope), (1, 1))
        qo = insert_slice(qo, (qx2 * cos + qx1 * sin), (0, half_rope), (q_heads, half_rope), (1, 1))
        tl.store(q_ptr + m * q_hidden + out_qn, qo.reshape(q_hidden))

        kx1 = extract_slice(knorm, (0, 0), (kv_heads, half_rope), (1, 1))
        kx2 = extract_slice(knorm, (0, half_rope), (kv_heads, half_rope), (1, 1))
        ko = tl.zeros((kv_heads, rope_dim), dtype=tl.bfloat16)
        ko = insert_slice(ko, (kx1 * cos - kx2 * sin), (0, 0), (kv_heads, half_rope), (1, 1))
        ko = insert_slice(ko, (kx2 * cos + kx1 * sin), (0, half_rope), (kv_heads, half_rope), (1, 1))
        tl.store(k_ptr + m * kv_hidden + kvn, ko.reshape(kv_hidden))
        vv = tl.load(input_ptr + m * total_hidden + vn)
        tl.store(v_ptr + m * kv_hidden + kvn, vv)


@triton.jit(
    do_not_specialize=[
        "batch_size",
        "n_slots",
        "total_slots",
        "kc_block_stride",
        "vc_block_stride",
    ]
)
def fused_qkv_cache_bs1_kernel(
    input_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    kc_ptr,
    vc_ptr,
    slots_ptr,
    cos_sin_ptr,
    pos_ptr,
    qw_ptr,
    kw_ptr,
    batch_size,
    n_slots,
    total_slots,
    kc_block_stride,
    vc_block_stride,
    eps: tl.constexpr,
    q_hidden: tl.constexpr,
    kv_hidden: tl.constexpr,
    total_hidden: tl.constexpr,
    q_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope: tl.constexpr,
    block_size: tl.constexpr,
    num_vec: tl.constexpr,
    WRITE_KV_TENSORS: tl.constexpr,
):
    pid = tl.program_id(0)
    per = tl.cdiv(batch_size, num_vec)
    b0 = pid * per
    b1 = tl.minimum(b0 + per, batch_size)
    qn = tl.arange(0, q_hidden)
    kn = tl.arange(q_hidden, q_hidden + kv_hidden)
    vn = tl.arange(q_hidden + kv_hidden, total_hidden)
    wd = tl.arange(0, head_dim)
    qw = tl.load(qw_ptr + wd).to(tl.float32)
    kw = tl.load(kw_ptr + wd).to(tl.float32)
    out_qn = tl.arange(0, q_hidden)
    kvn = tl.arange(0, kv_hidden)
    half_off = tl.arange(0, half_rope)
    for m in tl.range(b0, b1):
        qv = tl.load(input_ptr + m * total_hidden + qn).to(tl.float32).reshape(q_heads, head_dim)
        qvar = tl.sum(qv * qv, axis=1, keep_dims=True) / head_dim
        qnorm = (qv * tl.math.rsqrt(qvar + eps) * qw[None, :]).to(tl.bfloat16)
        kv = tl.load(input_ptr + m * total_hidden + kn).to(tl.float32).reshape(kv_heads, head_dim)
        kvar = tl.sum(kv * kv, axis=1, keep_dims=True) / head_dim
        knorm = (kv * tl.math.rsqrt(kvar + eps) * kw[None, :]).to(tl.bfloat16)
        pos = tl.load(pos_ptr + m).to(tl.int64)
        csoff = pos * rope_dim + half_off
        cos = tl.load(cos_sin_ptr + csoff)[None, :]
        sin = tl.load(cos_sin_ptr + csoff + half_rope)[None, :]
        qx1 = extract_slice(qnorm, (0, 0), (q_heads, half_rope), (1, 1))
        qx2 = extract_slice(qnorm, (0, half_rope), (q_heads, half_rope), (1, 1))
        qo = tl.zeros((q_heads, rope_dim), dtype=tl.bfloat16)
        qo = insert_slice(qo, (qx1 * cos - qx2 * sin), (0, 0), (q_heads, half_rope), (1, 1))
        qo = insert_slice(qo, (qx2 * cos + qx1 * sin), (0, half_rope), (q_heads, half_rope), (1, 1))
        tl.store(q_ptr + m * q_hidden + out_qn, qo.reshape(q_hidden))

        kx1 = extract_slice(knorm, (0, 0), (kv_heads, half_rope), (1, 1))
        kx2 = extract_slice(knorm, (0, half_rope), (kv_heads, half_rope), (1, 1))
        ko = tl.zeros((kv_heads, rope_dim), dtype=tl.bfloat16)
        ko = insert_slice(ko, (kx1 * cos - kx2 * sin), (0, 0), (kv_heads, half_rope), (1, 1))
        ko = insert_slice(ko, (kx2 * cos + kx1 * sin), (0, half_rope), (kv_heads, half_rope), (1, 1))
        kflat = ko.reshape(kv_hidden)
        vv = tl.load(input_ptr + m * total_hidden + vn)
        if WRITE_KV_TENSORS:
            tl.store(k_ptr + m * kv_hidden + kvn, kflat)
            tl.store(v_ptr + m * kv_hidden + kvn, vv)
        if m < n_slots:
            slot = tl.load(slots_ptr + m).to(tl.int64)
            if (slot >= 0) & (slot < total_slots):
                blk = slot // block_size
                off = slot % block_size
                tl.store(kc_ptr + blk * kc_block_stride + off * kv_hidden + kvn, kflat)
                tl.store(vc_ptr + blk * vc_block_stride + off * kv_hidden + kvn, vv)


_VEC_CORES: int | None = None
_ATTN_STATE_CLS = None
_FAST_SPLIT_KERNELS: dict[tuple, object] = {}
_FUSED_QKV_CACHE_KERNELS: dict[tuple, object] = {}


def _get_safe_vec_cores() -> int:
    global _VEC_CORES
    if _VEC_CORES is None:
        init_device_properties_triton()
        _VEC_CORES = int(get_vectorcore_num())
    return _VEC_CORES


def _get_attn_state_cls():
    global _ATTN_STATE_CLS
    if _ATTN_STATE_CLS is None:
        from vllm_ascend.attention.attention_v1 import AscendAttentionState

        _ATTN_STATE_CLS = AscendAttentionState
    return _ATTN_STATE_CLS


def _is_pure_chunked_pfa_step(attn_metadata, attn_state_cls, batch_size: int) -> bool:
    if (
        attn_metadata.attn_state != attn_state_cls.ChunkedPrefill
        or batch_size < 1152
        or getattr(attn_metadata, "num_decodes", 0) != 0
        or getattr(attn_metadata, "num_prefills", 0) <= 0
        or getattr(attn_metadata, "attn_mask", None) is None
    ):
        return False
    cached = getattr(attn_metadata, "_m21_can_pure_pfa", None)
    if cached is not None:
        return cached
    aq = getattr(attn_metadata, "actual_seq_lengths_q", None)
    skv = getattr(attn_metadata, "seq_lens_list", None)
    if not aq or not skv or len(aq) != len(skv) or int(aq[-1]) < 1152:
        can = False
    else:
        can = True
        prev_q = 0
        for i in range(len(aq)):
            cur_q = int(aq[i])
            if cur_q - prev_q != int(skv[i]):
                can = False
                break
            prev_q = cur_q
    attn_metadata._m21_can_pure_pfa = can
    return can


def _resolve_qkv_cache_ctx(resolved_name: str):
    if not is_forward_context_available():
        return None, None, None, None
    fc = get_forward_context()
    no_compile = getattr(fc, "no_compile_layers", None)
    if no_compile is None:
        return None, None, None, None
    attn_layer = no_compile.get(resolved_name)
    if attn_layer is None:
        return None, None, None, None
    kv_cache = getattr(attn_layer, "kv_cache", None)
    raw_meta = getattr(fc, "attn_metadata", None)
    if isinstance(raw_meta, dict):
        attn_metadata = raw_meta.get(resolved_name)
    elif isinstance(raw_meta, list) and raw_meta and isinstance(raw_meta[0], dict):
        attn_metadata = raw_meta[0].get(resolved_name)
    else:
        attn_metadata = raw_meta
    fc_slots = getattr(fc, "slot_mapping", None)
    fc_slot_mapping = fc_slots.get(resolved_name) if isinstance(fc_slots, dict) else None
    return attn_metadata, attn_layer, kv_cache, fc_slot_mapping


def _can_use_fast_split(
    input: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    q_bias: torch.Tensor | None,
    k_bias: torch.Tensor | None,
) -> bool:
    total_hidden = q_hidden_size + 2 * kv_hidden_size
    return (
        q_bias is None
        and k_bias is None
        and 0 < head_dim <= 256
        and (head_dim & (head_dim - 1)) == 0
        and 0 < q_hidden_size <= 8192
        and (q_hidden_size & (q_hidden_size - 1)) == 0
        and 0 < kv_hidden_size <= 2048
        and (kv_hidden_size & (kv_hidden_size - 1)) == 0
        and q_hidden_size % head_dim == 0
        and kv_hidden_size % head_dim == 0
        and cos_sin_cache.shape[-1] == head_dim
        and input.shape[0] > 0
        and input.ndim == 2
        and input.shape[1] == total_hidden
        and input.dtype == torch.bfloat16
        and cos_sin_cache.dtype == torch.bfloat16
        and q_weight.dtype == torch.bfloat16
        and k_weight.dtype == torch.bfloat16
        and cos_sin_cache.is_contiguous()
        and q_weight.is_contiguous()
        and k_weight.is_contiguous()
        and positions.ndim == 1
        and positions.stride(0) == 1
    )


def split_qkv_rmsnorm_rope_impl(
    input: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _can_use_fast_split(
        input,
        cos_sin_cache,
        positions,
        q_weight,
        k_weight,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        q_bias,
        k_bias,
    ):
        if not input.is_contiguous():
            input = input.contiguous()
        num_vectorcore = _get_safe_vec_cores()
        batch_size = input.shape[0]
        q_output = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
        k_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
        v_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
        feps = float(eps)
        total_hidden = q_hidden_size + 2 * kv_hidden_size
        q_heads = q_hidden_size // head_dim
        kv_heads = kv_hidden_size // head_dim
        rope_dim = head_dim
        half_rope = head_dim // 2
        if _get_raw_stream is None or torch.npu.is_current_stream_capturing():
            fast_split_qkv_rope_kernel[(num_vectorcore,)](
                input,
                q_output,
                k_output,
                v_output,
                cos_sin_cache,
                positions,
                q_weight,
                k_weight,
                batch_size,
                feps,
                q_hidden_size,
                kv_hidden_size,
                total_hidden,
                q_heads,
                kv_heads,
                head_dim,
                rope_dim,
                half_rope,
                num_vectorcore,
            )
        else:
            dev_idx = input.device.index if input.device.index is not None else 0
            key = (dev_idx, feps, num_vectorcore, positions.dtype, q_hidden_size, kv_hidden_size, head_dim)
            k = _FAST_SPLIT_KERNELS.get(key)
            if k is None:
                k = fast_split_qkv_rope_kernel[(num_vectorcore,)](
                    input,
                    q_output,
                    k_output,
                    v_output,
                    cos_sin_cache,
                    positions,
                    q_weight,
                    k_weight,
                    batch_size,
                    feps,
                    q_hidden_size,
                    kv_hidden_size,
                    total_hidden,
                    q_heads,
                    kv_heads,
                    head_dim,
                    rope_dim,
                    half_rope,
                    num_vectorcore,
                )
                _FAST_SPLIT_KERNELS[key] = k
            else:
                st = _get_raw_stream(dev_idx)
                k.run(  # type: ignore[attr-defined]
                    num_vectorcore,
                    1,
                    1,
                    st,
                    k.function,  # type: ignore[attr-defined]
                    k.packed_metadata,  # type: ignore[attr-defined]
                    None,
                    None,
                    None,
                    input,
                    q_output,
                    k_output,
                    v_output,
                    cos_sin_cache,
                    positions,
                    q_weight,
                    k_weight,
                    batch_size,
                    feps,
                    q_hidden_size,
                    kv_hidden_size,
                    total_hidden,
                    q_heads,
                    kv_heads,
                    head_dim,
                    rope_dim,
                    half_rope,
                    num_vectorcore,
                )
        return q_output, k_output, v_output

    # get available vector core
    num_vectorcore = get_vectorcore_num()
    rope_dim = cos_sin_cache.shape[-1]
    batch_size = input.shape[0]
    BIAS = q_bias is not None
    IS_PARTIAL_ROPE = rope_dim != head_dim
    # Q + K + V
    total_hidden_size = q_hidden_size + kv_hidden_size * 2

    q_output = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
    k_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    v_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)

    q_head_num = q_hidden_size // head_dim
    kv_head_num = kv_hidden_size // head_dim

    # set number of line loading from GM data is x
    # x*(q_head_num + kv_head_num)*HEAD_DIM: values_tmp
    # 2x*(q_head_num + kv_head_num)*HEAD_DIM: normalized_values(float32)
    # x*ROPE_DIM*2 : cos/sin
    # x*q_head_num*HEAD_DIM*2： normalized_values_tmp
    # x*q_head_num*ROPE_DIM*(0.5) (not IS_PARTIAL_ROPE) x*q_head_num*ROPE_DIM*(0.5): y
    UB_SIZE = 87040  # 85K = 85 * 1024
    # the factor is the sum of elements number
    if IS_PARTIAL_ROPE:
        factor = 5 * q_hidden_size + 3 * kv_hidden_size + rope_dim * 4 + q_head_num * rope_dim
        batch_size_per_iter_per_vec = int(UB_SIZE / input.element_size()) // factor
    else:
        factor = 5 * q_hidden_size + 3 * kv_hidden_size + rope_dim * 2 + q_head_num * rope_dim // 2
        batch_size_per_iter_per_vec = int(UB_SIZE / input.element_size()) // factor
    batch_size_per_iter_per_vec = max(1, batch_size_per_iter_per_vec)
    qk_head_num_sum = int(q_head_num + kv_head_num)
    qk_head_nums_per_iter_per_vec = batch_size_per_iter_per_vec * qk_head_num_sum

    grid = (num_vectorcore, 1, 1)
    # v tiling
    v_batch_size_per_iter_per_vec = UB_SIZE / torch.bfloat16.itemsize // (kv_hidden_size + 1)

    split_qkv_rmsnorm_rope_kernel[grid](
        input,
        q_output,
        k_output,
        v_output,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        batch_size,
        q_hidden_size,
        kv_hidden_size,
        total_hidden_size,
        eps,
        BIAS,
        head_dim,
        rope_dim,
        rope_dim // 2,
        IS_PARTIAL_ROPE,
        num_vectorcore,
        int(batch_size_per_iter_per_vec),
        int(qk_head_nums_per_iter_per_vec),
        q_head_num,
        kv_head_num,
        qk_head_num_sum,
        int(v_batch_size_per_iter_per_vec),
        positions,
        cos_sin_cache,
    )
    return q_output, k_output, v_output


def split_qkv_rmsnorm_rope_impl_fake(
    input: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Fake implementation for shape inference during Dynamo/AOT tracing.
    # Note: sin and cos are not used in shape computation, but must be present in signature.
    batch_size = input.shape[0]
    q_output = torch.empty(
        batch_size,
        int(q_hidden_size),
        device=input.device,
        dtype=input.dtype,
    )
    k_output = torch.empty(
        batch_size,
        int(kv_hidden_size),
        device=input.device,
        dtype=input.dtype,
    )
    v_output = torch.empty(
        batch_size,
        int(kv_hidden_size),
        device=input.device,
        dtype=input.dtype,
    )
    return q_output, k_output, v_output


direct_register_custom_op(
    op_name="qkv_rmsnorm_rope",
    op_func=split_qkv_rmsnorm_rope_impl,
    fake_impl=split_qkv_rmsnorm_rope_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


def qkv_rmsnorm_rope_cache_impl(
    input: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    layer_name: LayerNameType,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = input.shape[0]
    if batch_size == 0:
        return (
            torch.empty(0, q_hidden_size, device=input.device, dtype=input.dtype),
            torch.empty(0, kv_hidden_size, device=input.device, dtype=input.dtype),
            torch.empty(0, kv_hidden_size, device=input.device, dtype=input.dtype),
        )

    if not input.is_contiguous():
        input = input.contiguous()

    resolved_name = _resolve_layer_name(layer_name)
    is_capturing = torch.npu.is_current_stream_capturing()
    has_fc = is_forward_context_available()
    is_full_capture = bool(_EXTRA_CTX.capturing) if has_fc else False
    if (is_full_capture or not is_capturing) and _can_use_fast_split(
        input,
        cos_sin_cache,
        positions,
        q_weight,
        k_weight,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        None,
        None,
    ):
        attn_metadata, attn_layer, kv_cache, fc_slot_mapping = _resolve_qkv_cache_ctx(resolved_name)
        impl = getattr(attn_layer, "impl", None) if attn_layer is not None else None
        attn_state_cls = _get_attn_state_cls()
        attn_type_str = str(getattr(impl, "attn_type", "")) if impl is not None else ""
        if (
            attn_metadata is not None
            and impl is not None
            and "ENCODER_DECODER" not in attn_type_str
            and not getattr(impl, "use_bnsd_kv_cache", False)
            and getattr(impl, "kv_sharing_target_layer_name", None) is None
            and kv_cache is not None
            and len(kv_cache) >= 2
            and not getattr(attn_metadata, "uses_pcp_sharded_layout", False)
            and getattr(attn_metadata, "model_runner_type", "") != "pooling"
        ):
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            slot_mapping = getattr(attn_metadata, "slot_mapping", None)
            if slot_mapping is None:
                slot_mapping = fc_slot_mapping
            q_heads = q_hidden_size // head_dim
            kv_heads = kv_hidden_size // head_dim
            total_hidden = q_hidden_size + 2 * kv_hidden_size
            rope_dim = head_dim
            half_rope = head_dim // 2
            block_size = int(key_cache.shape[1]) if key_cache.ndim == 4 else 0
            if (
                key_cache.ndim == 4
                and key_cache.shape == value_cache.shape
                and key_cache.shape[0] > 0
                and block_size > 0
                and key_cache.shape[2] == kv_heads
                and key_cache.shape[3] == head_dim
                and key_cache.stride()[1:] == (kv_hidden_size, head_dim, 1)
                and value_cache.stride()[1:] == (kv_hidden_size, head_dim, 1)
                and key_cache.dtype == torch.bfloat16
                and value_cache.dtype == torch.bfloat16
                and slot_mapping is not None
                and slot_mapping.ndim == 1
                and slot_mapping.stride(0) == 1
                and slot_mapping.numel() > 0
            ):
                if getattr(impl, "key_cache", None) is None:
                    impl.key_cache, impl.value_cache = key_cache, value_cache
                write_kv_tensors = False
                if not is_full_capture:
                    write_kv_tensors = (
                        attn_metadata.attn_state == attn_state_cls.PrefillNoCache
                        or getattr(attn_metadata, "block_tables", None) is None
                        or _is_pure_chunked_pfa_step(attn_metadata, attn_state_cls, batch_size)
                    )
                num_vectorcore = _get_safe_vec_cores()
                q_output = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
                k_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
                v_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
                total_slots = int(key_cache.shape[0]) * block_size
                n_slots = min(int(slot_mapping.shape[0]), batch_size)
                if not is_full_capture:
                    n_actual = getattr(attn_metadata, "num_actual_tokens", None)
                    if n_actual is not None:
                        n_slots = min(n_slots, int(n_actual))
                kc_bs = int(key_cache.stride(0))
                vc_bs = int(value_cache.stride(0))
                feps = float(eps)
                if _get_raw_stream is None or is_full_capture or is_capturing:
                    fused_qkv_cache_bs1_kernel[(num_vectorcore,)](
                        input,
                        q_output,
                        k_output,
                        v_output,
                        key_cache,
                        value_cache,
                        slot_mapping,
                        cos_sin_cache,
                        positions,
                        q_weight,
                        k_weight,
                        batch_size,
                        n_slots,
                        total_slots,
                        kc_bs,
                        vc_bs,
                        feps,
                        q_hidden_size,
                        kv_hidden_size,
                        total_hidden,
                        q_heads,
                        kv_heads,
                        head_dim,
                        rope_dim,
                        half_rope,
                        block_size,
                        num_vectorcore,
                        write_kv_tensors,
                    )
                else:
                    dev_idx = input.device.index if input.device.index is not None else 0
                    key = (
                        dev_idx,
                        feps,
                        num_vectorcore,
                        positions.dtype,
                        slot_mapping.dtype,
                        write_kv_tensors,
                        q_hidden_size,
                        kv_hidden_size,
                        head_dim,
                        block_size,
                    )
                    k = _FUSED_QKV_CACHE_KERNELS.get(key)
                    if k is None:
                        k = fused_qkv_cache_bs1_kernel[(num_vectorcore,)](
                            input,
                            q_output,
                            k_output,
                            v_output,
                            key_cache,
                            value_cache,
                            slot_mapping,
                            cos_sin_cache,
                            positions,
                            q_weight,
                            k_weight,
                            batch_size,
                            n_slots,
                            total_slots,
                            kc_bs,
                            vc_bs,
                            feps,
                            q_hidden_size,
                            kv_hidden_size,
                            total_hidden,
                            q_heads,
                            kv_heads,
                            head_dim,
                            rope_dim,
                            half_rope,
                            block_size,
                            num_vectorcore,
                            write_kv_tensors,
                        )
                        _FUSED_QKV_CACHE_KERNELS[key] = k
                    else:
                        st = _get_raw_stream(dev_idx)
                        k.run(  # type: ignore[attr-defined]
                            num_vectorcore,
                            1,
                            1,
                            st,
                            k.function,  # type: ignore[attr-defined]
                            k.packed_metadata,  # type: ignore[attr-defined]
                            None,
                            None,
                            None,
                            input,
                            q_output,
                            k_output,
                            v_output,
                            key_cache,
                            value_cache,
                            slot_mapping,
                            cos_sin_cache,
                            positions,
                            q_weight,
                            k_weight,
                            batch_size,
                            n_slots,
                            total_slots,
                            kc_bs,
                            vc_bs,
                            feps,
                            q_hidden_size,
                            kv_hidden_size,
                            total_hidden,
                            q_heads,
                            kv_heads,
                            head_dim,
                            rope_dim,
                            half_rope,
                            block_size,
                            num_vectorcore,
                            write_kv_tensors,
                        )
                done_layers = getattr(attn_metadata, "_m19_kv_done_layers", None)
                if done_layers is None:
                    done_layers = set()
                    attn_metadata._m19_kv_done_layers = done_layers
                done_layers.add(resolved_name)
                return q_output, k_output, v_output

    return split_qkv_rmsnorm_rope_impl(
        input=input,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
        q_weight=q_weight,
        k_weight=k_weight,
        q_hidden_size=q_hidden_size,
        kv_hidden_size=kv_hidden_size,
        head_dim=head_dim,
        eps=eps,
        q_bias=None,
        k_bias=None,
    )


def qkv_rmsnorm_rope_cache_fake(
    input: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    layer_name: LayerNameType,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return split_qkv_rmsnorm_rope_impl_fake(
        input=input,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
        q_weight=q_weight,
        k_weight=k_weight,
        q_hidden_size=q_hidden_size,
        kv_hidden_size=kv_hidden_size,
        head_dim=head_dim,
        eps=eps,
    )


direct_register_custom_op(
    op_name="qkv_rmsnorm_rope_cache",
    op_func=qkv_rmsnorm_rope_cache_impl,
    fake_impl=qkv_rmsnorm_rope_cache_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
