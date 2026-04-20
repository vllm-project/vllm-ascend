import gc

import numpy as np
import pytest
import torch

import vllm_ascend.ops.register_custom_ops  # noqa
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

NUM_TOKENS = [1, 8, 32]
NUM_QKV_HEADS = [(6, 1), (8, 2)]
HEAD_DIMS = [128]
ROTARY_DIMS = [64, 128]
TP_WORLDS = [1]
EPS = [1e-6]
DTYPES = [torch.bfloat16]
SEEDS = [0]
DEVICES = [f"npu:{0}"]
DEFAULT_ATOL = 5e-2
DEFAULT_RTOL = 5e-3


def _build_rope(num_tokens, rotary_dim, dtype, device):
    cos = torch.from_numpy(np.random.uniform(0, 1, [num_tokens, rotary_dim // 2])).to(dtype).to(device)
    sin = torch.from_numpy(np.random.uniform(0, 1, [num_tokens, rotary_dim // 2])).to(dtype).to(device)
    return cos.contiguous(), sin.contiguous()


def _apply_rope_neox(q, k, cos, sin, rotary_dim):
    half = rotary_dim // 2
    cos = cos.to(torch.float32).unsqueeze(1)
    sin = sin.to(torch.float32).unsqueeze(1)

    q_f32 = q.to(torch.float32)
    k_f32 = k.to(torch.float32)

    q1 = q_f32[..., :half]
    q2 = q_f32[..., half:rotary_dim]
    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    q_out = torch.cat([q_rot, q_f32[..., rotary_dim:]], dim=-1).to(q.dtype)

    k1 = k_f32[..., :half]
    k2 = k_f32[..., half:rotary_dim]
    k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    k_out = torch.cat([k_rot, k_f32[..., rotary_dim:]], dim=-1).to(k.dtype)
    return q_out.contiguous(), k_out.contiguous()


def _fused_impl(
    qkv,
    q_weight,
    k_weight,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    rotary_dim,
    eps,
    tp_world,
    cos,
    sin,
):
    return torch.ops.vllm.split_qkv_tp_rmsnorm_rope(
        input=qkv,
        q_weight=q_weight,
        k_weight=k_weight,
        q_hidden_size=q_hidden_size,
        kv_hidden_size=kv_hidden_size,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        eps=eps,
        tp_world=tp_world,
        cos=cos,
        sin=sin,
    )


def _reference_impl(
    qkv,
    q_weight,
    k_weight,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    rotary_dim,
    eps,
    tp_world,
    cos,
    sin,
):
    q, k, v = qkv.split([q_hidden_size, kv_hidden_size, kv_hidden_size], dim=-1)
    orig_dtype = q.dtype

    q_f32 = q.to(torch.float32)
    k_f32 = k.to(torch.float32)
    q_var = q_f32.pow(2).mean(dim=-1, keepdim=True)
    k_var = k_f32.pow(2).mean(dim=-1, keepdim=True)

    q_out = (q_f32 * torch.rsqrt(q_var + eps) * q_weight.to(torch.float32)).to(orig_dtype)
    k_out = (k_f32 * torch.rsqrt(k_var + eps) * k_weight.to(torch.float32)).to(orig_dtype)

    q_3d = q_out.view(q.shape[0], -1, head_dim).contiguous()
    k_3d = k_out.view(k.shape[0], -1, head_dim).contiguous()
    q_3d, k_3d = _apply_rope_neox(q_3d, k_3d, cos.contiguous(), sin.contiguous(), rotary_dim)

    return (
        q_3d.view(q.shape[0], q_hidden_size).contiguous(),
        k_3d.view(k.shape[0], kv_hidden_size).contiguous(),
        v.contiguous(),
    )


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_q_heads, num_kv_heads", NUM_QKV_HEADS)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("tp_world", TP_WORLDS)
@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_split_qkv_tp_rmsnorm_rope(
    num_tokens, num_q_heads, num_kv_heads, head_dim, rotary_dim, tp_world, eps, dtype, seed, device
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_default_device(device)
    init_device_properties_triton()

    q_hidden_size = num_q_heads * head_dim
    kv_hidden_size = num_kv_heads * head_dim

    qkv = torch.randn(num_tokens, q_hidden_size + kv_hidden_size * 2, dtype=dtype, device=device)
    q_weight = torch.randn(q_hidden_size, dtype=torch.float32, device=device) * 0.1 + 1.0
    k_weight = torch.randn(kv_hidden_size, dtype=torch.float32, device=device) * 0.1 + 1.0
    cos, sin = _build_rope(num_tokens, rotary_dim, dtype, device)

    q_fused, k_fused, v_fused = _fused_impl(
        qkv=qkv.clone(),
        q_weight=q_weight.clone(),
        k_weight=k_weight.clone(),
        q_hidden_size=q_hidden_size,
        kv_hidden_size=kv_hidden_size,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        eps=eps,
        tp_world=tp_world,
        cos=cos,
        sin=sin,
    )
    q_ref, k_ref, v_ref = _reference_impl(
        qkv=qkv.clone(),
        q_weight=q_weight.clone(),
        k_weight=k_weight.clone(),
        q_hidden_size=q_hidden_size,
        kv_hidden_size=kv_hidden_size,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        eps=eps,
        tp_world=tp_world,
        cos=cos,
        sin=sin,
    )

    torch.testing.assert_close(q_fused.to(torch.float32), q_ref.to(torch.float32), atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    torch.testing.assert_close(k_fused.to(torch.float32), k_ref.to(torch.float32), atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    torch.testing.assert_close(v_fused.to(torch.float32), v_ref.to(torch.float32), atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
