import math

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.triton_utils import extract_slice, get_vectorcore_num, insert_slice


@triton.jit
def precompute_rope_cos_sin_kernel(
    positions_gm_ptr,
    cos_sin_cache_gm_ptr,
    out_cos_sin_gm_ptr,
    batch_size,
    N,
    batch_size_per_vec: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    num_vectorcore: tl.constexpr,
):
    row_pid = tl.program_id(0)
    input_batch_offset = row_pid * batch_size_per_vec
    input_batch_offset_end = min(input_batch_offset + batch_size_per_vec, batch_size)

    positions_off = tl.arange(0, batch_size_per_vec)
    if input_batch_offset >= batch_size:
        return

    x = tl.load(
        positions_gm_ptr + positions_off + input_batch_offset,
        mask=(positions_off + input_batch_offset) < input_batch_offset_end,
    )

    sin_cos_range = tl.arange(0, ROPE_DIM)
    offset = x[:, None] * ROPE_DIM + sin_cos_range[None, :]
    sin_cos_val = tl.load(cos_sin_cache_gm_ptr + offset).to(tl.float32)
    output_offset = (positions_off + input_batch_offset)[:, None] * ROPE_DIM + sin_cos_range[None, :]
    tl.store(out_cos_sin_gm_ptr + output_offset, sin_cos_val, mask=output_offset < N)


@triton.jit
def split_qkv_rmsnorm_rope_simt_kernel(
    input_gm_ptr,
    q_gm_ptr,
    k_gm_ptr,
    v_gm_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    k_bias_ptr,
    cos_sin_precomputed_ptr,
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
    batch_size_per_vec: tl.constexpr,
):
    row_pid = tl.program_id(0)

    q_weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    k_weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)

    iter_num_per_vec = tl.cdiv(batch_size_per_vec, batch_size_per_iter_per_vec)
    v_iter_num_per_vec = tl.cdiv(batch_size_per_vec, v_batch_size_per_iter_per_vec)
    input_batch_offset = row_pid * batch_size_per_vec
    mblk_idx = tl.arange(0, batch_size_per_iter_per_vec) + input_batch_offset
    nblk_idx = tl.arange(0, q_hidden_size + kv_hidden_size)
    nmask = nblk_idx < total_hidden_size

    input_batch_offset_end = min(input_batch_offset + batch_size_per_vec, batch_size)

    output_q_nblk_idx = tl.arange(0, q_hidden_size)
    output_q_nmask = output_q_nblk_idx < q_hidden_size
    output_kv_nblk_idx = tl.arange(0, kv_hidden_size)
    output_kv_nmask = output_kv_nblk_idx < kv_hidden_size

    if input_batch_offset >= batch_size:
        return

    for iter in tl.range(iter_num_per_vec):
        pos_offset = iter * batch_size_per_iter_per_vec
        mmask = (mblk_idx + pos_offset) < input_batch_offset_end
        mask = (mmask[:, None]) & (nmask[None, :])
        idx = (mblk_idx + pos_offset)[:, None] * total_hidden_size + nblk_idx[None, :]
        values_tmp1 = (
            tl.load(input_gm_ptr + idx, mask=mask).reshape(qk_head_nums_per_iter_per_vec, HEAD_DIM).to(tl.float32)
        )
        if BIAS:
            q_bias_values = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
            k_bias_values = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)

        base = (input_batch_offset + pos_offset) * ROPE_DIM
        cos_sin_offset = base + tl.arange(0, batch_size_per_iter_per_vec * ROPE_DIM)
        cos_sin_value = tl.load(
            cos_sin_precomputed_ptr + cos_sin_offset, cos_sin_offset < (input_batch_offset_end * ROPE_DIM)
        ).reshape(batch_size_per_iter_per_vec, 1, ROPE_DIM)
        cos = extract_slice(
            cos_sin_value, offsets=(0, 0, 0), sizes=(batch_size_per_iter_per_vec, 1, HALF_ROPE_DIM), strides=(1, 1, 1)
        )
        sin = extract_slice(
            cos_sin_value,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, 1, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )

        normalized_values = values_tmp1
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
            normalized_values_tmp = normalized_values_tmp * q_weight_values + q_bias_values
        else:
            normalized_values_tmp = normalized_values_tmp * q_weight_values

        values_tmp = tl.zeros((batch_size_per_iter_per_vec, q_head_num, ROPE_DIM), dtype=tl.float32)
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

        normalized_values_tmp1 = extract_slice(
            normalized_values.reshape(batch_size_per_iter_per_vec, qk_head_num_sum, HEAD_DIM),
            offsets=(0, q_head_num, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HEAD_DIM),
            strides=(1, 1, 1),
        )

        if BIAS:
            normalized_values_tmp1 = normalized_values_tmp1 * k_weight_values + k_bias_values
        else:
            normalized_values_tmp1 = normalized_values_tmp1 * k_weight_values

        values_tmp2 = tl.zeros((batch_size_per_iter_per_vec, kv_head_num, ROPE_DIM), dtype=tl.float32)

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


def split_qkv_rmsnorm_rope_simt_impl(
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
    num_vectorcore = get_vectorcore_num()
    rope_dim = cos_sin_cache.shape[-1]
    batch_size = input.shape[0]
    BIAS = q_bias is not None
    IS_PARTIAL_ROPE = rope_dim != head_dim
    total_hidden_size = q_hidden_size + kv_hidden_size * 2

    q_output = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
    k_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    v_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)

    q_head_num = q_hidden_size // head_dim
    kv_head_num = kv_hidden_size // head_dim
    UB_SIZE = 87040

    if IS_PARTIAL_ROPE:
        factor = 5 * q_hidden_size + 3 * kv_hidden_size + rope_dim * 4 + q_head_num * rope_dim
        batch_size_per_iter_per_vec = int(UB_SIZE / input.element_size()) // factor
    else:
        factor = 5 * q_hidden_size + 3 * kv_hidden_size + rope_dim * 2 + q_head_num * rope_dim // 2
        batch_size_per_iter_per_vec = int(UB_SIZE / input.element_size()) // factor
    batch_size_per_iter_per_vec = max(1, batch_size_per_iter_per_vec)
    qk_head_num_sum = int(q_head_num + kv_head_num)
    qk_head_nums_per_iter_per_vec = batch_size_per_iter_per_vec * qk_head_num_sum

    batch_size_per_vec = math.ceil(batch_size / num_vectorcore)
    iter_num_per_vec = math.ceil(batch_size_per_vec / batch_size_per_iter_per_vec)
    batch_size_per_vec = iter_num_per_vec * batch_size_per_iter_per_vec

    v_batch_size_per_iter_per_vec = UB_SIZE / torch.float32.itemsize // (kv_hidden_size + 1)

    cos_sin_precomputed = torch.empty(batch_size, rope_dim, dtype=torch.float32, device=input.device)
    N = cos_sin_precomputed.numel()
    grid = (num_vectorcore, 1, 1)
    batch_size_per_vec_cos_sin = pow(2, math.ceil(math.log2(batch_size_per_vec)))
    precompute_rope_cos_sin_kernel[grid](
        positions,
        cos_sin_cache,
        cos_sin_precomputed,
        batch_size,
        N,
        batch_size_per_vec_cos_sin,
        rope_dim,
        num_vectorcore,
        force_simt_only=True,
    )

    grid = (num_vectorcore, 1, 1)
    split_qkv_rmsnorm_rope_simt_kernel[grid](
        input,
        q_output,
        k_output,
        v_output,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        cos_sin_precomputed,
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
        batch_size_per_vec,
    )
    return q_output, k_output, v_output


def split_qkv_rmsnorm_rope_simt_impl_fake(
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
    op_name="qkv_rmsnorm_rope_simt",
    op_func=split_qkv_rmsnorm_rope_simt_impl,
    fake_impl=split_qkv_rmsnorm_rope_simt_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
