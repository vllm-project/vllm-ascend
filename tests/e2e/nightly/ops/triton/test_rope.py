import gc

import pytest
import torch

from vllm_ascend.ops.triton.rope import rope_forward_triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

IS_NEOX_STYLE = [True, False]
DTYPES = [torch.bfloat16, torch.float16]
HEAD_SIZES = [64, 128]
ROTARY_DIMS = [32, 64]
NUM_Q_HEADS = [64]
NUM_K_HEADS = [1]
NUM_TOKENS = [1, 4, 8, 16, 1024]
SEEDS = [0]
DEVICES = [f"npu:{0}"]
DEFAULT_ATOL = 1e-3
DEFAULT_RTOL = 1e-3


def rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def _rope_pytorch_native(
        query, key, cos, sin, rope_dim,
        is_neox_style) -> tuple[torch.Tensor, torch.Tensor | None]:
    """PyTorch-native implementation equivalent to forward()."""
    assert key is not None
    orig_dtype = query.dtype
    query_rot = query[..., :rope_dim].to(torch.float32)
    key_rot = key[..., :rope_dim].to(torch.float32)
    head_size = query.shape[-1]
    if rope_dim < head_size:
        query_pass = query[..., rope_dim:]
        key_pass = key[..., rope_dim:]

    if is_neox_style:
        cos = cos.repeat(1, 2).unsqueeze(-2).to(torch.float32)
        sin = sin.repeat(1, 2).unsqueeze(-2).to(torch.float32)
    else:
        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2).to(torch.float32)
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2).to(torch.float32)

    rotate_fn = rotate_neox if is_neox_style else rotate_gptj
    query_rot = query_rot * cos + rotate_fn(query_rot) * sin
    key_rot = key_rot * cos + rotate_fn(key_rot) * sin

    if rope_dim < head_size:
        query = torch.cat((query_rot.to(orig_dtype), query_pass), dim=-1)
        key = torch.cat((key_rot.to(orig_dtype), key_pass), dim=-1)
    else:
        query = query_rot.to(orig_dtype)
        key = key_rot.to(orig_dtype)
    return query, key


@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_q_heads", NUM_Q_HEADS)
@pytest.mark.parametrize("num_k_heads", NUM_K_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_rotary_embedding_triton_kernel(
    is_neox_style: bool,
    num_tokens: int,
    num_q_heads: int,
    num_k_heads: int,
    head_size: int,
    rotary_dim: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device(device)
    init_device_properties_triton()
    if rotary_dim == -1:
        rotary_dim = head_size
    sin = torch.randn(num_tokens, rotary_dim // 2, dtype=dtype, device=device)
    cos = torch.randn(num_tokens, rotary_dim // 2, dtype=dtype, device=device)
    q_trt = torch.randn(num_tokens,
                        num_q_heads,
                        head_size,
                        dtype=dtype,
                        device=device)
    k_trt = torch.randn(num_tokens,
                        num_k_heads,
                        head_size,
                        dtype=dtype,
                        device=device)
    q_gold = torch.randn(num_tokens,
                         num_q_heads,
                         head_size,
                         dtype=dtype,
                         device=device)
    k_gold = torch.randn(num_tokens,
                         num_k_heads,
                         head_size,
                         dtype=dtype,
                         device=device)
    q_trt.copy_(q_gold)
    k_trt.copy_(k_gold)
    q_trt, k_trt = rope_forward_triton(q_trt,
                                       k_trt,
                                       cos,
                                       sin,
                                       rope_dim=rotary_dim,
                                       is_neox_style=is_neox_style)
    q_gold, k_gold = _rope_pytorch_native(q_gold,
                                          k_gold,
                                          cos,
                                          sin,
                                          rope_dim=rotary_dim,
                                          is_neox_style=is_neox_style)
    # Compare the results.
    torch.testing.assert_close(q_trt.view(q_gold.size()),
                               q_gold,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)
    torch.testing.assert_close(k_trt.view(k_gold.size()),
                               k_gold,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
