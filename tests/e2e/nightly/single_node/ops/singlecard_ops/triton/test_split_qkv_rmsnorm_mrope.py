import gc

import pytest
import torch

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

NUM_TOKENS = [1, 4096]
NUM_QKV_HEADS = [(2, 1), (16, 2)]
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


def rms_norm(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    eps,
    norm_bias=None,
):
    x = x.cpu()
    norm_weight = norm_weight.cpu()

    x = x.to(torch.float32)
    norm_weight = norm_weight.to(torch.float32).cpu()
    reciprocal_std = 1 / torch.sqrt(torch.mean(x**2, axis=-1, keepdims=True) + eps)
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
    rope_dim = 2 * sum(mrope_section)
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size

    # input tensor
    if has_gate:
        qkv = torch.randn(num_tokens, 2 * q_size + kv_size * 2, dtype=dtype, device=device)
    else:
        qkv = torch.randn(num_tokens, q_size + kv_size * 2, dtype=dtype, device=device)
    q_weight = torch.randn(head_size, dtype=dtype, device=device)
    k_weight = torch.randn(head_size, dtype=dtype, device=device)
    q_bias = None
    k_bias = None

    cos_sin = torch.randn(3, num_tokens, rope_dim, dtype=dtype, device=device)
    cos, sin = cos_sin.chunk(2, dim=-1)

    cos = cos.contiguous()
    sin = sin.contiguous()

    if has_gate:
        q_gate_data = qkv[:, : q_size * 2].view(-1, num_q_heads, head_size * 2)
        q_data, golden_gate = torch.chunk(q_gate_data, 2, dim=-1)
        golden_gate = golden_gate.reshape(-1, q_size)
        q_data = q_data.reshape(-1, q_size)
        k_data = qkv[:, 2 * q_size : 2 * q_size + kv_size]
        v_data = qkv[:, 2 * q_size + kv_size :]
        qkv_for_ref = torch.cat([q_data, k_data, v_data], dim=-1)
    else:
        qkv_for_ref = qkv

    if is_interleaved:
        golden_q, golden_k, golden_v = naive_split_qkv_rmsnorm_mrope_interleaved(
            qkv_for_ref.cpu(),
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
            rope_dim,
        )
    else:
        golden_q, golden_k, golden_v = naive_split_qkv_rmsnorm_mrope(
            qkv_for_ref.cpu(),
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
            rope_dim,
        )

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

    torch.testing.assert_close(real_q.cpu(), golden_q.cpu(), atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)

    torch.testing.assert_close(real_k.cpu(), golden_k.cpu(), atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)

    torch.testing.assert_close(real_v.cpu(), golden_v.cpu(), atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    if has_gate:
        torch.testing.assert_close(real_gate.cpu(), golden_gate.cpu(), atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Inline cos/sin (positions + inv_freq) tests
# ---------------------------------------------------------------------------

# Model-like configs mirroring the real inline MRoPE call sites:
# - qwen35: Qwen3.5-35B-A3B text attention, 16q/2kv, head_size 256, gate
#   output, interleaved sections [11,11,10], partial rope 64, zero-centered
#   weights (rms_weight_offset=1.0).
# - qwen3vl: Qwen3-VL-32B text attention, 64q/8kv, head_size 128, full rope
#   128, interleaved sections [24,20,20], plain weights.
# - partial_small: small partial-rope shape exercising kv_head_tile == 1.
INLINE_CONFIGS = [
    ("qwen35", 16, 2, 256, [11, 11, 10], True, 1.0, 64),
    ("qwen3vl", 64, 8, 128, [24, 20, 20], False, 0.0, 128),
    ("partial_small", 2, 1, 256, [24, 20, 20], False, 0.0, 128),
]
INLINE_NUM_TOKENS = [1, 40, 512, 4096]
ROPE_THETA = 10000000.0
MAX_POSITION_ID = 4096


def make_mrope_inv_freq(rope_dim: int, device: str) -> torch.Tensor:
    """fp32 contiguous inv_freq of length rope_dim // 2, as registered by the model."""
    return (
        1.0 / (ROPE_THETA ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim))
    ).contiguous()


def make_inline_cos_sin(
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin planes [3, num_tokens, rope_dim // 2] computed from positions."""
    freqs = positions.to(torch.float32)[:, :, None] * inv_freq[None, None, :]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _run_inline_case(
    cfg: tuple,
    num_tokens: int,
    dtype: torch.dtype,
    device: str,
    has_bias: bool,
    is_interleaved: bool,
    positions: torch.Tensor | None = None,
) -> dict:
    """Build inputs, run the inline and cached paths, and compute the golden.

    The golden cos/sin are derived from positions/inv_freq exactly like the
    model does (outer product, rounded to the qkv dtype), and the golden
    weights carry rms_weight_offset pre-applied in the weight dtype, matching
    the cached cos/sin call sites of the old baseline.
    """
    torch.set_default_device(device)
    torch.manual_seed(0)
    _, num_q_heads, num_kv_heads, head_size, mrope_section, has_gate, rms_weight_offset, rope_dim = cfg
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    gate_size = q_size if has_gate else 0
    eps = EPS[0]

    qkv = torch.randn(num_tokens, q_size + gate_size + 2 * kv_size, dtype=dtype, device=device)
    q_weight = torch.randn(head_size, dtype=dtype, device=device)
    k_weight = torch.randn(head_size, dtype=dtype, device=device)
    q_bias = 0.1 * torch.randn(head_size, dtype=dtype, device=device) if has_bias else None
    k_bias = 0.1 * torch.randn(head_size, dtype=dtype, device=device) if has_bias else None
    if positions is None:
        positions = torch.randint(0, MAX_POSITION_ID, (3, num_tokens), dtype=torch.int64, device=device)
    positions_ref = positions.contiguous()
    inv_freq = make_mrope_inv_freq(rope_dim, device)
    cos, sin = make_inline_cos_sin(positions_ref, inv_freq, dtype)
    cos_sin_cache = torch.cat([cos, sin], dim=-1)

    q_weight_ref = rms_weight_offset + q_weight if rms_weight_offset != 0.0 else q_weight
    k_weight_ref = rms_weight_offset + k_weight if rms_weight_offset != 0.0 else k_weight

    if has_gate:
        q_gate_data = qkv[:, : q_size * 2].view(-1, num_q_heads, head_size * 2)
        q_data, golden_gate = torch.chunk(q_gate_data, 2, dim=-1)
        golden_gate = golden_gate.reshape(-1, q_size)
        q_data = q_data.reshape(-1, q_size)
        k_data = qkv[:, 2 * q_size : 2 * q_size + kv_size]
        v_data = qkv[:, 2 * q_size + kv_size :]
        qkv_for_ref = torch.cat([q_data, k_data, v_data], dim=-1)
    else:
        golden_gate = None
        qkv_for_ref = qkv

    naive_fn = naive_split_qkv_rmsnorm_mrope_interleaved if is_interleaved else naive_split_qkv_rmsnorm_mrope
    golden_q, golden_k, golden_v = naive_fn(
        qkv_for_ref.cpu(),
        q_weight_ref.cpu(),
        q_bias.cpu() if has_bias else None,
        k_weight_ref.cpu(),
        k_bias.cpu() if has_bias else None,
        cos.cpu(),
        sin.cpu(),
        num_q_heads,
        num_kv_heads,
        head_size,
        eps,
        mrope_section,
        rope_dim,
    )

    op = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope
    common = dict(
        qkv=qkv,
        q_weight=q_weight,
        k_weight=k_weight,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        eps=eps,
        mrope_section=mrope_section,
        is_interleaved=is_interleaved,
        rope_dim=rope_dim,
        q_bias=q_bias,
        k_bias=k_bias,
        has_gate=has_gate,
    )
    inline_out = op(positions=positions, inv_freq=inv_freq, rms_weight_offset=rms_weight_offset, **common)
    cache_out = op(cos_sin=cos_sin_cache, rms_weight_offset=rms_weight_offset, **common)

    return dict(
        golden=(golden_q, golden_k, golden_v, golden_gate),
        inline=inline_out,
        cache=cache_out,
    )


def _assert_inline_case(result: dict, has_gate: bool) -> None:
    golden_q, golden_k, golden_v, golden_gate = result["golden"]
    real_q, real_k, real_v, real_gate = result["inline"]
    torch.testing.assert_close(real_q.cpu(), golden_q, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    torch.testing.assert_close(real_k.cpu(), golden_k, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    torch.testing.assert_close(real_v.cpu(), golden_v, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    if has_gate:
        torch.testing.assert_close(real_gate.cpu(), golden_gate.cpu(), atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    # P0 numerics: the inline path must be bit-exact with the cached cos/sin path
    cache_q, cache_k, cache_v, cache_gate = result["cache"]
    assert torch.equal(real_q, cache_q)
    assert torch.equal(real_k, cache_k)
    assert torch.equal(real_v, cache_v)
    if has_gate:
        assert torch.equal(real_gate, cache_gate)


@pytest.mark.parametrize("num_tokens", INLINE_NUM_TOKENS)
@pytest.mark.parametrize("inline_cfg", INLINE_CONFIGS, ids=["qwen35", "qwen3vl", "partial_small"])
@pytest.mark.parametrize("is_interleaved", [True, False])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_split_qkv_rmsnorm_mrope_inline_cos_sin(
    num_tokens: int,
    inline_cfg: tuple,
    is_interleaved: bool,
    has_bias: bool,
    dtype: torch.dtype,
    device: str,
):
    result = _run_inline_case(inline_cfg, num_tokens, dtype, device, has_bias, is_interleaved)
    _assert_inline_case(result, has_gate=inline_cfg[5])
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize(
    "case",
    ["strided_tokens", "strided_planes", "int32"],
)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_split_qkv_rmsnorm_mrope_inline_cos_sin_positions_layout(case: str, device: str):
    """Non-contiguous and int32 positions must match the contiguous int64 path."""
    torch.set_default_device(device)
    torch.manual_seed(0)
    cfg = INLINE_CONFIGS[0]
    num_tokens = 64
    positions_ref = torch.randint(0, MAX_POSITION_ID, (3, num_tokens), dtype=torch.int64, device=device)
    if case == "strided_tokens":
        # [3, 2T] contiguous tensor viewed at even columns: strides (2T, 2)
        wide = torch.zeros(3, 2 * num_tokens, dtype=torch.int64, device=device)
        wide[:, ::2] = positions_ref
        positions = wide[:, ::2]
    elif case == "strided_planes":
        # transposed storage: [3, T] view with strides (1, 3)
        planes = torch.zeros(num_tokens, 3, dtype=torch.int64, device=device)
        planes.copy_(positions_ref.t())
        positions = planes.t()
    else:
        positions = positions_ref.to(torch.int32)
    result = _run_inline_case(
        cfg,
        num_tokens,
        torch.bfloat16,
        device,
        has_bias=False,
        is_interleaved=True,
        positions=positions,
    )
    _assert_inline_case(result, has_gate=cfg[5])
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_split_qkv_rmsnorm_mrope_inline_cos_sin_core_boundary(device: str):
    """Token counts around the per-core token budget (uneven core loads)."""
    cfg = INLINE_CONFIGS[0]
    core_num = get_vectorcore_num()
    for num_tokens in (core_num - 1, core_num, core_num + 1):
        result = _run_inline_case(cfg, num_tokens, torch.bfloat16, device, has_bias=False, is_interleaved=True)
        _assert_inline_case(result, has_gate=cfg[5])
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_split_qkv_rmsnorm_mrope_inline_cos_sin_empty(device: str):
    """num_tokens == 0 early-returns empty outputs on both cos/sin paths."""
    torch.set_default_device(device)
    _, num_q_heads, num_kv_heads, head_size, mrope_section, has_gate, rms_weight_offset, rope_dim = INLINE_CONFIGS[0]
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    gate_size = q_size if has_gate else 0
    width = q_size + gate_size + 2 * kv_size
    qkv = torch.randn(0, width, dtype=torch.bfloat16, device=device)
    q_weight = torch.randn(head_size, dtype=torch.bfloat16, device=device)
    k_weight = torch.randn(head_size, dtype=torch.bfloat16, device=device)
    positions = torch.zeros(3, 0, dtype=torch.int64, device=device)
    inv_freq = make_mrope_inv_freq(rope_dim, device)
    cos_sin = torch.randn(3, 0, rope_dim, dtype=torch.bfloat16, device=device)
    op = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope
    common = dict(
        qkv=qkv,
        q_weight=q_weight,
        k_weight=k_weight,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        eps=EPS[0],
        mrope_section=mrope_section,
        is_interleaved=True,
        rope_dim=rope_dim,
        has_gate=has_gate,
    )
    q_i, k_i, v_i, g_i = op(positions=positions, inv_freq=inv_freq, rms_weight_offset=rms_weight_offset, **common)
    q_c, k_c, v_c, g_c = op(cos_sin=cos_sin, rms_weight_offset=rms_weight_offset, **common)
    assert q_i.shape == q_c.shape == (0, q_size)
    assert k_i.shape == k_c.shape == (0, kv_size)
    assert v_i.shape == v_c.shape == (0, kv_size)
    assert g_i.shape == g_c.shape == (0, gate_size)


@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_split_qkv_rmsnorm_mrope_inline_cos_sin_invalid_args(device: str):
    """Host-side validation of the inline arguments (no kernel launch)."""
    torch.set_default_device(device)
    _, num_q_heads, num_kv_heads, head_size, mrope_section, _, _, rope_dim = INLINE_CONFIGS[2]
    num_tokens = 4
    qkv = torch.randn(num_tokens, num_q_heads * head_size + num_kv_heads * head_size * 2, dtype=torch.bfloat16)
    q_weight = torch.randn(head_size, dtype=torch.bfloat16)
    k_weight = torch.randn(head_size, dtype=torch.bfloat16)
    op = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope
    base = dict(
        qkv=qkv,
        q_weight=q_weight,
        k_weight=k_weight,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        eps=EPS[0],
        mrope_section=mrope_section,
        is_interleaved=False,
        rope_dim=rope_dim,
        has_gate=False,
    )
    good_pos = torch.randint(0, 16, (3, num_tokens), dtype=torch.int64)
    good_inv = make_mrope_inv_freq(rope_dim, device)
    cos_sin = torch.randn(3, num_tokens, rope_dim, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        op(cos_sin=cos_sin, positions=good_pos, inv_freq=good_inv, **base)
    with pytest.raises(ValueError):
        op(positions=good_pos, **base)
    with pytest.raises(ValueError):
        op(inv_freq=good_inv, **base)
    with pytest.raises(ValueError):
        op(positions=good_pos[0], inv_freq=good_inv, **base)
    with pytest.raises(ValueError):
        op(positions=good_pos[:, : num_tokens - 1], inv_freq=good_inv, **base)
    with pytest.raises(ValueError):
        op(positions=good_pos.to(torch.float32), inv_freq=good_inv, **base)
    with pytest.raises(ValueError):
        op(positions=good_pos[:, ::-1], inv_freq=good_inv, **base)
    with pytest.raises(ValueError):
        op(positions=good_pos, inv_freq=good_inv[:-1], **base)
    with pytest.raises(ValueError):
        # build on CPU: the NPU silently casts fp64 back to fp32
        op(positions=good_pos, inv_freq=make_mrope_inv_freq(rope_dim, "cpu").to(torch.float64), **base)
    with pytest.raises(ValueError):
        op(positions=good_pos, inv_freq=good_inv.repeat(2)[::2], **base)

