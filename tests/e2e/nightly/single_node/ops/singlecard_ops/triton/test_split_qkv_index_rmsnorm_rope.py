import gc

import numpy as np
import pytest
import torch

import vllm_ascend.ops  # noqa: F401  ensures torch.ops.vllm.qkv_index_rmsnorm_rope is registered

MAX_POSITION_EMBEDDINGS = [262144]
NUM_TOKENS = [1, 8, 32, 1024]
NUM_QKV_HEADS = [(12, 1), (64, 4)]
HEAD_DIMS = [128]
NUM_IDX_HEADS = [8, 16]
IDX_HEAD_DIMS = [48, 64]
ROPE_DIMS = [64, 128]
EPS = [1e-6]
DTYPES = [torch.bfloat16]
SEEDS = [0]
DEVICES = [f"npu:{0}"]
HAS_BIAS = [False, True]
DEFAULT_ATOL = 5e-2
DEFAULT_RTOL = 5e-3


def _build_cos_sin_cache(max_position_embeddings, rope_dim, dtype, device):
    cache = torch.from_numpy(
        np.random.uniform(0, 1, [max_position_embeddings, rope_dim])
    ).to(dtype).to(device)
    return cache.contiguous()


def _apply_rope_neox(x, cos, sin, rotary_dim):
    """Apply NeoX RoPE to ``x`` of shape ``[num_tokens, num_heads, head_dim]``."""
    half = rotary_dim // 2
    x_f32 = x.to(torch.float32)
    cos = cos.to(torch.float32).unsqueeze(1)
    sin = sin.to(torch.float32).unsqueeze(1)

    x1 = x_f32[..., :half]
    x2 = x_f32[..., half:rotary_dim]
    x_rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    x_out = torch.cat([x_rot, x_f32[..., rotary_dim:]], dim=-1)
    return x_out


def _rms_norm(x, weight, eps):
    """RMSNorm over the last dimension. ``x`` is already float32."""
    rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * rstd * weight.to(torch.float32)


def _reference_impl(
    qkv,
    cos_sin_cache,
    positions,
    q_weight,
    k_weight,
    index_q_weight,
    index_k_weight,
    q_hidden_size,
    kv_hidden_size,
    index_q_size,
    head_dim,
    idx_head_dim,
    eps,
    q_bias=None,
    k_bias=None,
):
    # Move everything to CPU for the reference implementation
    qkv = qkv.cpu()
    cos_sin_cache = cos_sin_cache.cpu()
    positions = positions.cpu()
    q_weight = q_weight.cpu()
    k_weight = k_weight.cpu()
    index_q_weight = index_q_weight.cpu()
    index_k_weight = index_k_weight.cpu()
    if q_bias is not None:
        q_bias = q_bias.cpu()
    if k_bias is not None:
        k_bias = k_bias.cpu()

    num_tokens = qkv.shape[0]
    cache_dim = cos_sin_cache.shape[-1]
    half_cache = cache_dim // 2

    # Split concat [q | k | v | index_q | index_k]
    index_offset = q_hidden_size + 2 * kv_hidden_size
    q = qkv[:, :q_hidden_size]
    k = qkv[:, q_hidden_size:q_hidden_size + kv_hidden_size]
    v = qkv[:, q_hidden_size + kv_hidden_size:index_offset]
    index_q = qkv[:, index_offset:index_offset + index_q_size]
    index_k = qkv[:, index_offset + index_q_size:]

    # Gather cos/sin rows by position
    cache_rows = cos_sin_cache[positions]

    # --- main Q/K: RMSNorm + optional bias + NeoX RoPE ---
    attn_rotary_dim = min(cache_dim, head_dim)
    attn_half = attn_rotary_dim // 2
    cos = cache_rows[:, :attn_half]
    sin = cache_rows[:, half_cache:half_cache + attn_half]

    num_q_heads = q_hidden_size // head_dim
    num_kv_heads = kv_hidden_size // head_dim

    q_3d = q.to(torch.float32).reshape(num_tokens, num_q_heads, head_dim)
    q_normed = _rms_norm(q_3d, q_weight, eps)
    if q_bias is not None:
        q_normed = q_normed + q_bias.to(torch.float32)
    q_out = _apply_rope_neox(q_normed, cos, sin, attn_rotary_dim)

    k_3d = k.to(torch.float32).reshape(num_tokens, num_kv_heads, head_dim)
    k_normed = _rms_norm(k_3d, k_weight, eps)
    if k_bias is not None:
        k_normed = k_normed + k_bias.to(torch.float32)
    k_out = _apply_rope_neox(k_normed, cos, sin, attn_rotary_dim)

    # V: copy only
    v_out = v.to(torch.float32)

    # --- indexer Q/K: RMSNorm + NeoX RoPE ---
    idx_rotary_dim = min(cache_dim, idx_head_dim)
    idx_half = idx_rotary_dim // 2
    cos_idx = cache_rows[:, :idx_half]
    sin_idx = cache_rows[:, half_cache:half_cache + idx_half]

    num_idx_heads = index_q_size // idx_head_dim

    iq_3d = index_q.to(torch.float32).reshape(num_tokens, num_idx_heads, idx_head_dim)
    iq_normed = _rms_norm(iq_3d, index_q_weight, eps)
    iq_out = _apply_rope_neox(iq_normed, cos_idx, sin_idx, idx_rotary_dim)

    ik_3d = index_k.to(torch.float32).reshape(num_tokens, 1, idx_head_dim)
    ik_normed = _rms_norm(ik_3d, index_k_weight, eps)
    ik_out = _apply_rope_neox(ik_normed, cos_idx, sin_idx, idx_rotary_dim)

    # Reshape back to 2-D
    q_out = q_out.reshape(num_tokens, q_hidden_size)
    k_out = k_out.reshape(num_tokens, kv_hidden_size)
    v_out = v_out.reshape(num_tokens, kv_hidden_size)
    iq_out = iq_out.reshape(num_tokens, index_q_size)
    ik_out = ik_out.reshape(num_tokens, idx_head_dim)

    return q_out, k_out, v_out, iq_out, ik_out


@pytest.mark.parametrize("max_position_embeddings", MAX_POSITION_EMBEDDINGS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_q_heads, num_kv_heads", NUM_QKV_HEADS)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("num_idx_heads", NUM_IDX_HEADS)
@pytest.mark.parametrize("idx_head_dim", IDX_HEAD_DIMS)
@pytest.mark.parametrize("rope_dim", ROPE_DIMS)
@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("has_bias", HAS_BIAS)
@torch.inference_mode()
def test_split_qkv_index_rmsnorm_rope(
    max_position_embeddings,
    num_tokens,
    num_q_heads,
    num_kv_heads,
    head_dim,
    num_idx_heads,
    idx_head_dim,
    rope_dim,
    eps,
    dtype,
    seed,
    device,
    has_bias,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_default_device(device)

    q_hidden_size = num_q_heads * head_dim
    kv_hidden_size = num_kv_heads * head_dim
    index_q_size = num_idx_heads * idx_head_dim
    # concat layout: [q | k | v | index_q | index_k]
    total_hidden = q_hidden_size + 2 * kv_hidden_size + index_q_size + idx_head_dim

    qkv = torch.randn(num_tokens, total_hidden, dtype=dtype, device=device)
    q_weight = torch.randn(head_dim, dtype=torch.float32, device=device) * 0.1 + 1.0
    k_weight = torch.randn(head_dim, dtype=torch.float32, device=device) * 0.1 + 1.0
    index_q_weight = torch.randn(idx_head_dim, dtype=torch.float32, device=device) * 0.1 + 1.0
    index_k_weight = torch.randn(idx_head_dim, dtype=torch.float32, device=device) * 0.1 + 1.0
    q_bias = (
        torch.randn(head_dim, dtype=torch.float32, device=device) * 0.1
        if has_bias
        else None
    )
    k_bias = (
        torch.randn(head_dim, dtype=torch.float32, device=device) * 0.1
        if has_bias
        else None
    )
    cos_sin_cache = _build_cos_sin_cache(max_position_embeddings, rope_dim, dtype, device)
    positions = torch.randint(
        low=0, high=max_position_embeddings, size=(num_tokens,), dtype=torch.int64, device=device
    )

    # fused kernel
    q_fused, k_fused, v_fused, iq_fused, ik_fused = torch.ops.vllm.qkv_index_rmsnorm_rope(
        input=qkv.clone(),
        cos_sin_cache=cos_sin_cache,
        positions=positions,
        q_weight=q_weight,
        k_weight=k_weight,
        index_q_weight=index_q_weight,
        index_k_weight=index_k_weight,
        q_hidden_size=q_hidden_size,
        kv_hidden_size=kv_hidden_size,
        index_q_size=index_q_size,
        head_dim=head_dim,
        idx_head_dim=idx_head_dim,
        eps=eps,
        attn_out_fp8=False,
        indexer_out_fp8=False,
        q_bias=q_bias,
        k_bias=k_bias,
    )
    # reference
    q_ref, k_ref, v_ref, iq_ref, ik_ref = _reference_impl(
        qkv=qkv.clone(),
        cos_sin_cache=cos_sin_cache,
        positions=positions,
        q_weight=q_weight,
        k_weight=k_weight,
        index_q_weight=index_q_weight,
        index_k_weight=index_k_weight,
        q_hidden_size=q_hidden_size,
        kv_hidden_size=kv_hidden_size,
        index_q_size=index_q_size,
        head_dim=head_dim,
        idx_head_dim=idx_head_dim,
        eps=eps,
        q_bias=q_bias,
        k_bias=k_bias,
    )

    torch.testing.assert_close(q_fused.to(torch.float32).cpu(), q_ref, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    torch.testing.assert_close(k_fused.to(torch.float32).cpu(), k_ref, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    torch.testing.assert_close(v_fused.to(torch.float32).cpu(), v_ref, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    torch.testing.assert_close(iq_fused.to(torch.float32).cpu(), iq_ref, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    torch.testing.assert_close(ik_fused.to(torch.float32).cpu(), ik_ref, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
