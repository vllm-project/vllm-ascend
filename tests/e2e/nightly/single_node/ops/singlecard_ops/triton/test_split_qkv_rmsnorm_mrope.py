import gc

import pytest
import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

NUM_TOKENS = [1, 4, 8, 16, 1024, 4096]
NUM_QKV_HEADS = [(8, 2), (2, 1), (16, 2)]
HEAD_SIZES = [128, 256]
EPS = [1e-6]
MROPE_SECTION = [[11, 11, 10], [24, 20, 20]]
IS_INTERLEAVED = [True, False]
HAS_GATE = [True, False]
DTYPES = [torch.bfloat16, torch.float16]
DEVICES = [f"npu:{0}"]
DEFAULT_ATOL = 1e-2
DEFAULT_RTOL = 1e-2


def apply_interleaved_rope(x: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
    """Apply interleaved MRoPE to 3D rotary embeddings.
    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THTHWHTHW...TT], preserving frequency continuity.
    """
    x_t = x[0].clone()
    x_t[..., 1 : mrope_section[1] * 3 : 3] = x[1, ..., 1 : mrope_section[1] * 3 : 3]
    x_t[..., 2 : mrope_section[2] * 3 : 3] = x[2, ..., 2 : mrope_section[2] * 3 : 3]
    return x_t


def rms_norm(x: torch.Tensor,
            norm_weight: torch.Tensor,
            eps,
            norm_bias=None,
):
    x = x.cpu()
    norm_weight = norm_weight.cpu()

    x = x.to(torch.float32)
    norm_weight = norm_weight.to(torch.float32).cpu()
    reciprocal_std = 1 / torch.sqrt(
        torch.mean(x ** 2, axis=-1, keepdims=True) + eps)
    out = x * reciprocal_std * norm_weight

    if norm_bias is not None:
        norm_bias = norm_bias.cpu().to(torch.float32)
        out = out + norm_bias

    return out


def naive_split_qkv_rmsnorm_mrope(
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        q_bias: torch.Tensor,
        k_weight: torch.Tensor,
        k_bias: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        head_size: int,
        eps: float,
        mrope_section: list[int],
        rope_dim: int,
):
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size

    # split
    qkv = qkv.cpu()
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    # norm
    q = rms_norm(q.reshape(-1, head_size), q_weight, eps, norm_bias=q_bias)
    k = rms_norm(k.reshape(-1, head_size), k_weight, eps, norm_bias=k_bias)

    # mrope
    rotary_dim = rope_dim
    num_tokens = qkv.shape[0]
    n_q_head = num_q_heads
    n_kv_head = num_kv_heads
    q_reshaped = q.view(num_tokens, n_q_head, head_size)
    k_reshaped = k.view(num_tokens, n_kv_head, head_size)
    cos_reshaped = cos.permute(1, 2, 0)
    sin_reshaped = sin.permute(1, 2, 0)
    half_rd = rotary_dim // 2

    for token_idx in range(num_tokens):
        token_cos = cos_reshaped[token_idx]
        token_sin = sin_reshaped[token_idx]

        cos_row = torch.zeros(half_rd, device=q.device, dtype=q.dtype)
        sin_row = torch.zeros(half_rd, device=q.device, dtype=q.dtype)

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
        q2 = q_token[:, half_rd:rotary_dim]
        k1 = k_token[:, :half_rd]
        k2 = k_token[:, half_rd:rotary_dim]

        cos_half = cos_row.unsqueeze(0)
        sin_half = sin_row.unsqueeze(0)

        new_q1 = q1 * cos_half - q2 * sin_half
        new_q2 = q2 * cos_half + q1 * sin_half

        new_k1 = k1 * cos_half - k2 * sin_half
        new_k2 = k2 * cos_half + k1 * sin_half

        q_reshaped[token_idx, :, :rotary_dim] = torch.cat([new_q1, new_q2], dim=1)
        k_reshaped[token_idx, :, :rotary_dim] = torch.cat([new_k1, new_k2], dim=1)

    q_result = q_reshaped.view(num_tokens, -1)
    k_result = k_reshaped.view(num_tokens, -1)

    q = q_result.to(qkv.dtype)
    k = k_result.to(qkv.dtype)
    v = v.to(qkv.dtype)

    return q, k, v


def naive_split_qkv_rmsnorm_mrope_interleaved(
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        q_bias: torch.Tensor,
        k_weight: torch.Tensor,
        k_bias: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        head_size: int,
        eps: float,
        mrope_section: list[int],
        rope_dim: int,
):
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size

    # split
    qkv = qkv.cpu()
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    # norm
    q = rms_norm(q.reshape(-1, head_size), q_weight, eps, norm_bias=q_bias)
    k = rms_norm(k.reshape(-1, head_size), k_weight, eps, norm_bias=k_bias)

    # mrope
    rotary_dim = rope_dim
    num_tokens = qkv.shape[0]
    n_q_head = num_q_heads
    n_kv_head = num_kv_heads
    q_reshaped = q.view(num_tokens, n_q_head, head_size)
    k_reshaped = k.view(num_tokens, n_kv_head, head_size)
    cos_reshaped = apply_interleaved_rope(cos, mrope_section)
    sin_reshaped = apply_interleaved_rope(sin, mrope_section)
    half_rd = rotary_dim // 2

    for token_idx in range(num_tokens):
        cos_row = cos_reshaped[token_idx]
        sin_row = sin_reshaped[token_idx]

        q_token = q_reshaped[token_idx]
        k_token = k_reshaped[token_idx]
        
        q1 = q_token[:, :half_rd]
        q2 = q_token[:, half_rd:rotary_dim]
        k1 = k_token[:, :half_rd]
        k2 = k_token[:, half_rd:rotary_dim]

        cos_half = cos_row.unsqueeze(0)
        sin_half = sin_row.unsqueeze(0)

        new_q1 = q1 * cos_half - q2 * sin_half
        new_q2 = q2 * cos_half + q1 * sin_half

        new_k1 = k1 * cos_half - k2 * sin_half
        new_k2 = k2 * cos_half + k1 * sin_half

        q_reshaped[token_idx, :, :rotary_dim] = torch.cat([new_q1, new_q2], dim=1)
        k_reshaped[token_idx, :, :rotary_dim] = torch.cat([new_k1, new_k2], dim=1)

    q_result = q_reshaped.view(num_tokens, -1)
    k_result = k_reshaped.view(num_tokens, -1)

    q = q_result.to(qkv.dtype)
    k = k_result.to(qkv.dtype)
    v = v.to(qkv.dtype)

    return q, k, v


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_q_heads, num_kv_heads", NUM_QKV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("mrope_section", MROPE_SECTION)
@pytest.mark.parametrize("is_interleaved", IS_INTERLEAVED)
@pytest.mark.parametrize("has_gate", HAS_GATE)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_split_qkv_rmsnorm_mrope(
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    mrope_section: list[int],
    eps: float,
    dtype: torch.dtype,
    device: str,
    is_interleaved: bool,
    has_gate: bool,
):

    torch.set_default_device(device)
    init_device_properties_triton()
    rope_dim = 2 * sum(mrope_section)
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size

    # input tensor
    if has_gate:
        qkv = torch.randn(num_tokens,
                          2 * q_size + kv_size * 2,
                          dtype=dtype,
                          device=device)
    else:
        qkv = torch.randn(num_tokens,
                          q_size + kv_size * 2,
                          dtype=dtype,
                          device=device)
    q_weight = torch.randn(head_size, dtype=dtype, device=device)
    k_weight = torch.randn(head_size, dtype=dtype, device=device)
    q_bias = None
    k_bias = None

    cos_sin = torch.randn(3, num_tokens, rope_dim, dtype=dtype,
                          device=device)
    cos, sin = cos_sin.chunk(2, dim=-1)
    
    cos = cos.contiguous()
    sin = sin.contiguous()

    if has_gate:
        q_gate_data = qkv[:, :q_size * 2].view(-1, num_q_heads, head_size * 2)
        q_data, golden_gate = torch.chunk(q_gate_data, 2, dim=-1)
        golden_gate = golden_gate.reshape(-1, q_size)
        q_data = q_data.reshape(-1, q_size)
        k_data = qkv[:, 2 * q_size:2 * q_size + kv_size]
        v_data = qkv[:, 2 * q_size + kv_size:]
        qkv_for_ref = torch.cat([q_data, k_data, v_data], dim=-1)
    else:
        qkv_for_ref = qkv

    if is_interleaved:
        golden_q, golden_k, golden_v = naive_split_qkv_rmsnorm_mrope_interleaved(qkv_for_ref.cpu(),
                                                                 q_weight.cpu(),
                                                                 q_bias,
                                                                 k_weight.cpu(),
                                                                 k_bias,
                                                                 cos.cpu(),
                                                                 sin.cpu(),
                                                                 num_q_heads,
                                                                 num_kv_heads,
                                                                 head_size,
                                                                 eps,
                                                                 mrope_section,
                                                                 rope_dim)
    else:
        golden_q, golden_k, golden_v = naive_split_qkv_rmsnorm_mrope(qkv_for_ref.cpu(),
                                                                    q_weight.cpu(),
                                                                    q_bias,
                                                                    k_weight.cpu(),
                                                                    k_bias,
                                                                    cos.cpu(),
                                                                    sin.cpu(),
                                                                    num_q_heads,
                                                                    num_kv_heads,
                                                                    head_size,
                                                                    eps,
                                                                    mrope_section,
                                                                    rope_dim)

    real_q, real_k, real_v, real_gate = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
            qkv=qkv,
            q_weight=q_weight,
            k_weight=k_weight,
            cos_sin=cos_sin,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            eps=eps,
            mrope_section=mrope_section,
            is_interleaved=is_interleaved,
            rope_dim=rope_dim,
            has_gate=has_gate,
    )

    torch.testing.assert_close(real_q.cpu(),
                               golden_q.cpu(),
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)

    torch.testing.assert_close(real_k.cpu(),
                               golden_k.cpu(),
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)

    torch.testing.assert_close(real_v.cpu(),
                               golden_v.cpu(),
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)
    if has_gate:
        torch.testing.assert_close(real_gate.cpu(),
                                golden_gate.cpu(),
                                atol=DEFAULT_ATOL,
                                rtol=DEFAULT_RTOL)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()