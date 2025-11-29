import gc
from typing import Optional

import pytest
import torch

from vllm_ascend.ops.triton.rope import rope_forward_triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

IS_NEOX_STYLE = [True, False]
DTYPES = [torch.bfloat16, torch.float16]
HEAD_SIZES = [64, 128]
ROTARY_DIMS = [32, 64]  # None means rotary dim == head size
NUM_Q_HEADS = [64]
NUM_K_HEADS = [1]
NUM_TOKENS = [1, 4, 8, 16, 1024]
SEEDS = [0]
DEVICES = [f"npu:{0}"]
# Set tolerance to 1 for quant ops
DEFAULT_ATOL = 1e-3
DEFAULT_RTOL = 1e-3


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


# test with leading dimension and merge seqlen and batch_size as num_tokens
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
    rotary_dim: Optional[int],
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

    q_trt, k_trt = rope_forward_triton(q_trt,
                                       k_trt,
                                       cos,
                                       sin,
                                       rope_dim=rotary_dim,
                                       is_neox_style=is_neox_style)
    q_gold = _apply_rotary_emb(q_gold, cos, sin, is_neox=is_neox_style)
    k_gold = _apply_rotary_emb(k_gold, cos, sin, is_neox=is_neox_style)

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
