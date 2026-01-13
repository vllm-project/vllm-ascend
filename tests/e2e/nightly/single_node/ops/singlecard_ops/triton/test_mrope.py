import gc
from typing import List

import pytest
import torch
from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

MROPE_SECTION = [[32, 32, 32]]
DTYPES = [torch.bfloat16, torch.float16]
HEAD_SIZES = [128]
ROTARY_DIMS = [128]
NUM_Q_HEADS = [64]
NUM_K_HEADS = [1]
NUM_TOKENS = [1, 4, 8, 16]
SEEDS = [0]
DEVICES = [f"npu:{0}"]
DEFAULT_ATOL = 1e-3
DEFAULT_RTOL = 1e-3


def pytorch_forward_native(q, k, cos, sin, mrope_section, head_size,
                           rotary_dim, mrope_interleaved):
    """PyTorch-native implementation equivalent to forward().
    """

    num_tokens = q.shape[0]
    n_q_head = q.shape[1] // head_size
    n_kv_head = k.shape[1] // head_size

    q_reshaped = q.view(num_tokens, n_q_head, head_size)
    k_reshaped = k.view(num_tokens, n_kv_head, head_size)

    cos_reshaped = cos.permute(1, 2, 0)
    sin_reshaped = sin.permute(1, 2, 0)

    half_rd = rotary_dim // 2

    for token_idx in range(num_tokens):
        token_cos = cos_reshaped[token_idx]
        token_sin = sin_reshaped[token_idx]

        cos_row = torch.zeros(head_size // 2, device=q.device, dtype=q.dtype)
        sin_row = torch.zeros(head_size // 2, device=q.device, dtype=q.dtype)

        if mrope_interleaved:
            cos_offsets = torch.arange(0, head_size // 2, device=q.device)
            h_mask = (
                (cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section[1])
            w_mask = (
                (cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section[2])
            t_mask = ~(h_mask | w_mask)

            cos_row[t_mask] = token_cos[t_mask, 0]
            cos_row[h_mask] = token_cos[h_mask, 1]
            cos_row[w_mask] = token_cos[w_mask, 2]

            sin_row[t_mask] = token_sin[t_mask, 0]
            sin_row[h_mask] = token_sin[h_mask, 1]
            sin_row[w_mask] = token_sin[w_mask, 2]
        else:
            t_end = mrope_section[0]
            h_end = t_end + mrope_section[1]

            if t_end > 0:
                cos_row[:t_end] = token_cos[:t_end, 0]
                sin_row[:t_end] = token_sin[:t_end, 0]

            if mrope_section[1] > 0:
                cos_row[t_end:h_end] = token_cos[t_end:h_end, 1]
                sin_row[t_end:h_end] = token_sin[t_end:h_end, 1]

            if mrope_section[2] > 0:
                w_start = h_end
                cos_row[w_start:half_rd] = token_cos[w_start:half_rd, 2]
                sin_row[w_start:half_rd] = token_sin[w_start:half_rd, 2]

        q_token = q_reshaped[token_idx]
        k_token = k_reshaped[token_idx]

        q1 = q_token[:, :half_rd]
        q2 = q_token[:, half_rd:]
        k1 = k_token[:, :half_rd]
        k2 = k_token[:, half_rd:]

        cos_half = cos_row.unsqueeze(0)
        sin_half = sin_row.unsqueeze(0)

        new_q1 = q1 * cos_half - q2 * sin_half
        new_q2 = q2 * cos_half + q1 * sin_half

        new_k1 = k1 * cos_half - k2 * sin_half
        new_k2 = k2 * cos_half + k1 * sin_half

        q_reshaped[token_idx] = torch.cat([new_q1, new_q2], dim=1)
        k_reshaped[token_idx] = torch.cat([new_k1, new_k2], dim=1)

    q_result = q_reshaped.view(num_tokens, -1)
    k_result = k_reshaped.view(num_tokens, -1)

    return q_result, k_result


def create_test_data(num_tokens, n_q_head, n_kv_head, rotary_dim, head_size,
                     device, dtype):
    q = torch.randn(num_tokens,
                    n_q_head * head_size,
                    dtype=dtype,
                    device=device)
    k = torch.randn(num_tokens,
                    n_kv_head * head_size,
                    dtype=dtype,
                    device=device)

    sin = torch.randn(3,
                      num_tokens,
                      rotary_dim // 2,
                      dtype=dtype,
                      device=device)
    cos = torch.randn(3,
                      num_tokens,
                      rotary_dim // 2,
                      dtype=dtype,
                      device=device)

    norm = torch.sqrt(cos**2 + sin**2)
    cos = cos / norm
    sin = sin / norm

    return q, k, cos, sin


@pytest.mark.parametrize("mrope_section", MROPE_SECTION)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_q_heads", NUM_Q_HEADS)
@pytest.mark.parametrize("num_k_heads", NUM_K_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_mrotary_embedding_triton_kernel(
    mrope_section: List[int],
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

    q_trt, k_trt, cos, sin = create_test_data(num_tokens=num_tokens,
                                              n_q_head=num_q_heads,
                                              n_kv_head=num_k_heads,
                                              head_size=head_size,
                                              rotary_dim=rotary_dim,
                                              device=device,
                                              dtype=dtype)

    q_gold, k_gold = q_trt.clone(), k_trt.clone()

    q_trt, k_trt = triton_mrope(q_trt, k_trt, cos, sin, mrope_section,
                                head_size, rotary_dim, True)

    q_gold, k_gold = pytorch_forward_native(q_gold, k_gold, cos, sin,
                                            mrope_section, head_size,
                                            rotary_dim, True)
    atol = DEFAULT_ATOL
    rtol = DEFAULT_RTOL
    if dtype == torch.bfloat16:
        atol = 1e-02
        rtol = 1e-02
    # Compare the results.
    torch.testing.assert_close(q_trt.view(q_gold.size()),
                               q_gold,
                               atol=atol,
                               rtol=rtol)
    torch.testing.assert_close(k_trt.view(k_gold.size()),
                               k_gold,
                               atol=atol,
                               rtol=rtol)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
