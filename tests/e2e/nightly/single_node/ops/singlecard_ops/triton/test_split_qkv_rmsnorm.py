import gc

import numpy as np
import pytest
import torch

import vllm_ascend.ops.register_custom_ops  # noqa
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

NUM_TOKENS = [1, 4, 8, 16, 1024]
NUM_QKV_HEADS = [(12, 1), (16, 1), (32, 4), (64, 4)]
HEAD_SIZES = [128]
EPS = [1e-6]
DTYPES = [torch.bfloat16]
SEEDS = [0]
DEVICES = [f"npu:{0}"]
DEFAULT_ATOL = 5e-2
DEFAULT_RTOL = 5e-3


def rms_norm(
    input,
    norm_weight,
    eps,
    norm_bias=None,
):
    input = input.to(torch.float32)
    norm_weight = norm_weight.to(torch.float32)
    reciprocal_std = 1 / torch.sqrt(
        torch.mean(input**2, axis=-1, keepdims=True) + eps)
    out = input * reciprocal_std * norm_weight
    if norm_bias is not None:
        norm_bias = norm_bias.to(torch.float32)
        out = out + norm_bias
    return out


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_q_heads, num_kv_heads", NUM_QKV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_split_qkv_rmsnorm(num_tokens, num_q_heads, num_kv_heads,
                                head_size, eps, dtype, seed, device):
    torch.manual_seed(seed)
    torch.set_default_device(device)
    init_device_properties_triton()

    q_hidden_size = num_q_heads * head_size
    kv_hidden_size = num_kv_heads * head_size
    qkv = torch.randn(num_tokens,
                      q_hidden_size + kv_hidden_size * 2,
                      dtype=dtype,
                      device=device)
    q_weight = torch.randn(head_size, dtype=dtype, device=device)
    k_weight = torch.randn(head_size, dtype=dtype, device=device)
    q, k, v = torch.ops.vllm.qkv_rmsnorm(input=qkv,
                                              q_weight=q_weight,
                                              k_weight=k_weight,
                                              q_hidden_size=q_hidden_size,
                                              kv_hidden_size=kv_hidden_size,
                                              head_dim=head_size,
                                              eps=eps)

    # split
    _q, _k, v_gold = qkv.cpu().split(
        [q_hidden_size, kv_hidden_size, kv_hidden_size], dim=-1)
    # norm
    q_gold = rms_norm(_q.reshape(-1, head_size), q_weight.cpu(), eps)
    k_gold = rms_norm(_k.reshape(-1, head_size), k_weight.cpu(), eps)
    q_gold = q_gold.reshape(num_tokens, -1)
    k_gold = k_gold.reshape(num_tokens, -1)

    # Compare the results.
    torch.testing.assert_close(q.to(torch.float32).cpu(),
                               q_gold,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)

    torch.testing.assert_close(k.to(torch.float32).cpu(),
                               k_gold,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)

    torch.testing.assert_close(v.to(torch.float32).cpu(),
                               v_gold.to(torch.float32),
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_q_heads, num_kv_heads", NUM_QKV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_split_qkv_rmsnorm_with_bias(num_tokens, num_q_heads,
                                          num_kv_heads, head_size, eps, dtype,
                                          seed, device):
    torch.manual_seed(seed)
    torch.set_default_device(device)
    init_device_properties_triton()

    q_hidden_size = num_q_heads * head_size
    kv_hidden_size = num_kv_heads * head_size
    qkv = torch.randn(num_tokens,
                      q_hidden_size + kv_hidden_size * 2,
                      dtype=dtype,
                      device=device)
    q_weight = torch.randn(head_size, dtype=dtype, device=device)
    k_weight = torch.randn(head_size, dtype=dtype, device=device)
    q_bias = torch.randn(head_size, dtype=dtype, device=device)
    k_bias = torch.randn(head_size, dtype=dtype, device=device)

    # fused kernel
    q, k, v = torch.ops.vllm.qkv_rmsnorm(input=qkv,
                                              q_weight=q_weight,
                                              k_weight=k_weight,
                                              q_hidden_size=q_hidden_size,
                                              kv_hidden_size=kv_hidden_size,
                                              head_dim=head_size,
                                              eps=eps,
                                              q_bias=q_bias,
                                              k_bias=k_bias)

    # split
    _q, _k, v_gold = qkv.cpu().split(
        [q_hidden_size, kv_hidden_size, kv_hidden_size], dim=-1)
    # norm
    q_gold = rms_norm(_q.reshape(-1, head_size),
                  q_weight.cpu(),
                  eps,
                  norm_bias=q_bias.cpu())
    k_gold = rms_norm(_k.reshape(-1, head_size),
                  k_weight.cpu(),
                  eps,
                  norm_bias=k_bias.cpu())

    q_gold = q_gold.reshape(num_tokens, -1)
    k_gold = k_gold.reshape(num_tokens, -1)

    # Compare the results.
    torch.testing.assert_close(q.to(torch.float32).cpu(),
                               q_gold,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)

    torch.testing.assert_close(k.to(torch.float32).cpu(),
                               k_gold,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)

    torch.testing.assert_close(v.to(torch.float32).cpu(),
                               v_gold.to(torch.float32),
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()