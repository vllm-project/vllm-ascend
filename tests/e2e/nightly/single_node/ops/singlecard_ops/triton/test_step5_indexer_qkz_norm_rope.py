# SPDX-License-Identifier: Apache-2.0
"""Single-operator accuracy test for the Step5 fused indexer QKZ norm+RoPE op.

Compares the fused kernel against a pure-torch reference replicating the
module's eager path (``AscendStep5DSAAttention._finish_indexer``):
- index_q: per-head RMSNorm (zero_centered -> weight + 1), no mean subtract
- index_k: per-head LayerNorm (subtract mean, with bias)
- NeoX RoPE on the first ``rotary_dim`` dims = rotary_dim//2 pairs
  (pair layout: dims i and i + rotary_dim//2), angles from the caller's
  cos/sin cache whose row width is rotary_dim//2.
- index_z: passthrough copy.

The inputs are the strided ``split`` views of one packed row -- the
production layout (the ``split_indexer`` slices of the fused qkv+indexer
GEMM output), so the kernel's in-place strided reads are covered.

Accuracy follows the repository operator criterion: index_k (bf16) uses
rtol = atol = 2^-6 with max abs error <= 1; index_q lands in e4m3
(quantized at the kernel store) and is compared against the reference cast
to the same dtype under rtol = 2^-3, atol = 2^-6, max abs error 4.0 (the
criterion's max(fixed_limit, 32 * ULP(1.0)) rule with the 3-bit e4m3
mantissa); both require at least 99% of elements within tolerance.
index_z is a bit-exact passthrough.

Regression coverage for the layout bug where the kernel treated
``rotary_dim`` as a pair count: with sparse_indexer_rope_dim=32 it read
dims 32-63 as the imaginary half and ran cache lookups past the 16-wide row
into the next position's angles (the pos-0 identity test fails loudly on
that bug).
"""

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.step5.indexer_qkz_norm_rope import (
    fused_indexer_qkz_norm_rope,
)

HEAD_DIM = 256  # proxy_dim
ROTARY_DIM = 32  # sparse_indexer_rope_dim: 32 dims = 16 pairs
EPS = 1e-6
THETA = 10000000.0  # indexer rope_theta (DSA layers)

# BF16 output criterion (repository operator standard).
BF16_RTOL = 2.0**-6
BF16_ATOL = 2.0**-6
BF16_MAX_ABS_ERR = 1.0
MIN_WITHIN_FRACTION = 0.99

# index_q lands in e4m3 (quantized at the store). The repository criterion's
# maximum-error rule is max(fixed_limit, 32 * ULP(1.0)); e4m3 has a 3-bit
# mantissa, so ULP(1.0) = 2^-3 and the two quantizations may differ by one
# code, bounded by rtol = 2^-3, atol = 2^-6 (near-zero slack), max abs 4.0.
FP8_RTOL = 2.0**-3
FP8_ATOL = 2.0**-6
FP8_MAX_ABS_ERR = 4.0


def _assert_close(
    actual: torch.Tensor,
    ref: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    max_abs_err: float,
    name: str,
) -> None:
    a = actual.float()
    r = ref.float()
    diff = (a - r).abs()
    within = diff <= atol + rtol * r.abs()
    frac = within.float().mean().item()
    assert frac >= MIN_WITHIN_FRACTION, (
        f"{name}: only {frac:.4%} elements within tolerance (max abs err {diff.max().item():.6f})"
    )
    assert diff.max().item() <= max_abs_err, f"{name}: max abs err {diff.max().item():.6f} exceeds {max_abs_err}"


def _build_cos_sin_cache(max_pos: int, pairs: int, theta: float, device):
    """Production [cos | sin] cache; callers may chunk it for references."""
    idx = torch.arange(0, 2 * pairs, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (idx / (2 * pairs)))
    pos = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.bfloat16)
    return cache


def _reference_indexer_norm_rope(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int,
    num_q_heads: int,
    num_k_heads: int,
    rotary_dim: int,
    eps: float,
    q_norm_weight_bias: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = index_q.shape[0]
    dtype = index_q.dtype

    def rms_norm(x_flat: torch.Tensor) -> torch.Tensor:
        x = x_flat.view(tokens, -1, head_dim).float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + eps)
        return (normed * (q_weight.float() + q_norm_weight_bias)).to(dtype)

    def layer_norm(x_flat: torch.Tensor) -> torch.Tensor:
        x = x_flat.view(tokens, -1, head_dim).float()
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, unbiased=False, keepdim=True)
        normed = (x - mean) * torch.rsqrt(variance + eps)
        return (normed * k_weight.float() + k_bias.float()).to(dtype)

    q = rms_norm(index_q)
    k = layer_norm(index_k)

    half = rotary_dim // 2

    def apply_rope(x_flat: torch.Tensor) -> torch.Tensor:
        x = x_flat.view(tokens, -1, head_dim).float()
        x1 = x[..., :half]
        x2 = x[..., half:rotary_dim]
        cos_t = cos[positions.long()].float()[:, None, :]
        sin_t = sin[positions.long()].float()[:, None, :]
        out = x.clone()
        out[..., :half] = x1 * cos_t - x2 * sin_t
        out[..., half:rotary_dim] = x1 * sin_t + x2 * cos_t
        return out.to(dtype).view(x_flat.shape)

    return apply_rope(q).view(tokens, -1), apply_rope(k).view(tokens, -1), index_z.clone()


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Requires Ascend A5",
)
@pytest.mark.parametrize("num_tokens", [1, 333, 4096])
@pytest.mark.parametrize("num_q_heads,num_k_heads", [(1, 1), (4, 1), (16, 1)])
def test_fused_indexer_norm_rope_accuracy(num_tokens: int, num_q_heads: int, num_k_heads: int):
    """Real Step5 geometry: 4 local Q heads (16 / index_tp_size=4), 1 K head;
    16 is the unsharded (index TP=1) case and 1 the sharded boundary."""
    device = "npu"

    max_pos = 4096
    torch.manual_seed(42)
    q_len = num_q_heads * HEAD_DIM
    k_len = num_k_heads * HEAD_DIM
    x_cat = torch.randn(num_tokens, q_len + 2 * k_len, device=device, dtype=torch.bfloat16)
    q_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_bias = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    positions = torch.randint(0, max_pos, (num_tokens,), device=device, dtype=torch.int32)
    cos_sin_cache = _build_cos_sin_cache(max_pos, ROTARY_DIM // 2, THETA, device)
    cos, sin = cos_sin_cache.chunk(2, dim=-1)

    index_q, index_k, index_z = x_cat.split([q_len, k_len, k_len], dim=-1)
    # Verify the inputs are views from a single tensor, not contiguous copies.
    # index_q is the offset-0 slice (its data_ptr equals x_cat's); index_k and
    # index_z sit at positive offsets.
    assert index_q.storage().data_ptr() == x_cat.storage().data_ptr()
    assert index_k.storage().data_ptr() == x_cat.storage().data_ptr()
    assert index_z.storage().data_ptr() == x_cat.storage().data_ptr()
    assert index_k.data_ptr() != x_cat.data_ptr()
    assert index_z.data_ptr() != x_cat.data_ptr()

    q_ref, k_ref, z_ref = _reference_indexer_norm_rope(
        index_q,
        index_k,
        index_z,
        q_weight,
        k_weight,
        k_bias,
        cos,
        sin,
        positions,
        head_dim=HEAD_DIM,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        rotary_dim=ROTARY_DIM,
        eps=EPS,
        q_norm_weight_bias=1.0,
    )

    q_out, k_out, z_out = fused_indexer_qkz_norm_rope(
        index_q,
        index_k,
        index_z,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        k_norm_bias=k_bias,
        head_dim=HEAD_DIM,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        rotary_dim=ROTARY_DIM,
        eps=EPS,
        q_norm_weight_bias=1.0,
    )

    # index_q is e4m3 (quantized at the kernel store): compare against the
    # reference cast to the same dtype under the e4m3 criterion.
    q_ref8 = q_ref.to(torch.float8_e4m3fn)
    assert q_out.dtype == torch.float8_e4m3fn
    _assert_close(q_out, q_ref8, rtol=FP8_RTOL, atol=FP8_ATOL, max_abs_err=FP8_MAX_ABS_ERR, name="Q")
    _assert_close(k_out, k_ref, rtol=BF16_RTOL, atol=BF16_ATOL, max_abs_err=BF16_MAX_ABS_ERR, name="K")
    assert torch.equal(z_ref, z_out), "Z passthrough mismatch"


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Requires Ascend A5",
)
def test_fused_indexer_norm_rope_pos0_is_identity():
    """At position 0 the rope angles are 0: rotated dims must equal the
    normalized values. The buggy pair/dims layout already failed here (the
    imaginary half landed outside the rotated window and cache reads ran
    into the next position's angles)."""
    device = "npu"

    num_tokens, num_q_heads, num_k_heads = 4, 4, 1
    torch.manual_seed(7)
    q_len = num_q_heads * HEAD_DIM
    k_len = num_k_heads * HEAD_DIM
    x_cat = torch.randn(num_tokens, q_len + 2 * k_len, device=device, dtype=torch.bfloat16)
    q_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_bias = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    positions = torch.zeros(num_tokens, device=device, dtype=torch.int32)
    cos_sin_cache = _build_cos_sin_cache(8, ROTARY_DIM // 2, THETA, device)
    cos, sin = cos_sin_cache.chunk(2, dim=-1)

    index_q, index_k, index_z = x_cat.split([q_len, k_len, k_len], dim=-1)

    q_ref, k_ref, _ = _reference_indexer_norm_rope(
        index_q,
        index_k,
        index_z,
        q_weight,
        k_weight,
        k_bias,
        cos,
        sin,
        positions,
        head_dim=HEAD_DIM,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        rotary_dim=ROTARY_DIM,
        eps=EPS,
        q_norm_weight_bias=1.0,
    )
    q_out, k_out, _ = fused_indexer_qkz_norm_rope(
        index_q,
        index_k,
        index_z,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        k_norm_bias=k_bias,
        head_dim=HEAD_DIM,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        rotary_dim=ROTARY_DIM,
        eps=EPS,
        q_norm_weight_bias=1.0,
    )
    # index_q is e4m3; one-ULP tolerance for the quantized comparison.
    q_ref8 = q_ref.to(torch.float8_e4m3fn)
    assert q_out.dtype == torch.float8_e4m3fn
    _assert_close(q_out, q_ref8, rtol=FP8_RTOL, atol=FP8_ATOL, max_abs_err=FP8_MAX_ABS_ERR, name="Q")
    _assert_close(k_out, k_ref, rtol=BF16_RTOL, atol=BF16_ATOL, max_abs_err=BF16_MAX_ABS_ERR, name="K")


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Requires Ascend A5",
)
def test_fused_indexer_norm_rope_empty_input():
    """Zero tokens must return correctly typed empty outputs without a zero-sized grid."""
    device = "npu"
    num_q_heads, num_k_heads = 4, 1
    q_len = num_q_heads * HEAD_DIM
    k_len = num_k_heads * HEAD_DIM
    x_cat = torch.empty((0, q_len + 2 * k_len), device=device, dtype=torch.bfloat16)
    index_q, index_k, index_z = x_cat.split([q_len, k_len, k_len], dim=-1)
    q_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_bias = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    cos_sin_cache = _build_cos_sin_cache(8, ROTARY_DIM // 2, THETA, device)
    cos, sin = cos_sin_cache.chunk(2, dim=-1)
    positions = torch.empty((0,), device=device, dtype=torch.int32)

    q_out, k_out, z_out = fused_indexer_qkz_norm_rope(
        index_q,
        index_k,
        index_z,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        k_norm_bias=k_bias,
        head_dim=HEAD_DIM,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        rotary_dim=ROTARY_DIM,
        eps=EPS,
        q_norm_weight_bias=1.0,
    )

    assert q_out.shape == (0, q_len) and q_out.dtype == torch.float8_e4m3fn
    assert k_out.shape == (0, k_len) and k_out.dtype == torch.bfloat16
    assert z_out.shape == (0, k_len) and z_out.dtype == torch.bfloat16


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Requires Ascend A5",
)
def test_fused_indexer_norm_rope_full_rope():
    """Full-RoPE geometry (rotary_dim == head_dim) stays on the fused path."""
    device = "npu"
    num_tokens, num_q_heads, num_k_heads = 96, 4, 1
    q_len = num_q_heads * HEAD_DIM
    k_len = num_k_heads * HEAD_DIM
    torch.manual_seed(11)
    x_cat = torch.randn(num_tokens, q_len + 2 * k_len, device=device, dtype=torch.bfloat16)
    index_q, index_k, index_z = x_cat.split([q_len, k_len, k_len], dim=-1)
    q_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_bias = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    positions = torch.randint(0, 4096, (num_tokens,), device=device, dtype=torch.int32)
    cos_sin_cache = _build_cos_sin_cache(4096, HEAD_DIM // 2, THETA, device)
    cos, sin = cos_sin_cache.chunk(2, dim=-1)

    q_ref, k_ref, z_ref = _reference_indexer_norm_rope(
        index_q,
        index_k,
        index_z,
        q_weight,
        k_weight,
        k_bias,
        cos,
        sin,
        positions,
        head_dim=HEAD_DIM,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        rotary_dim=HEAD_DIM,
        eps=EPS,
        q_norm_weight_bias=1.0,
    )
    q_out, k_out, z_out = fused_indexer_qkz_norm_rope(
        index_q,
        index_k,
        index_z,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        k_bias,
        head_dim=HEAD_DIM,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        rotary_dim=HEAD_DIM,
        eps=EPS,
        q_norm_weight_bias=1.0,
    )

    q_ref8 = q_ref.to(torch.float8_e4m3fn)
    assert q_out.dtype == torch.float8_e4m3fn
    _assert_close(q_out, q_ref8, rtol=FP8_RTOL, atol=FP8_ATOL, max_abs_err=FP8_MAX_ABS_ERR, name="Q")
    _assert_close(k_out, k_ref, rtol=BF16_RTOL, atol=BF16_ATOL, max_abs_err=BF16_MAX_ABS_ERR, name="K")
    assert torch.equal(z_ref, z_out), "Z passthrough mismatch"


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Requires Ascend A5",
)
def test_fused_indexer_norm_rope_k_layernorm_large_mean():
    """K LayerNorm must use centered variance for a large-mean, small-spread row."""
    device = "npu"
    num_tokens, num_q_heads, num_k_heads = 8, 4, 1
    q_len = num_q_heads * HEAD_DIM
    k_len = num_k_heads * HEAD_DIM
    torch.manual_seed(19)
    x_cat = torch.randn(num_tokens, q_len + 2 * k_len, device=device, dtype=torch.bfloat16)
    index_q, index_k, index_z = x_cat.split([q_len, k_len, k_len], dim=-1)
    index_k.copy_(128.0 + 0.1 * torch.randn_like(index_k))
    q_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_bias = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    positions = torch.randint(0, 4096, (num_tokens,), device=device, dtype=torch.int32)
    cos_sin_cache = _build_cos_sin_cache(4096, ROTARY_DIM // 2, THETA, device)
    cos, sin = cos_sin_cache.chunk(2, dim=-1)

    q_ref, k_ref, z_ref = _reference_indexer_norm_rope(
        index_q,
        index_k,
        index_z,
        q_weight,
        k_weight,
        k_bias,
        cos,
        sin,
        positions,
        head_dim=HEAD_DIM,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        rotary_dim=ROTARY_DIM,
        eps=EPS,
        q_norm_weight_bias=1.0,
    )
    q_out, k_out, z_out = fused_indexer_qkz_norm_rope(
        index_q,
        index_k,
        index_z,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        k_bias,
        head_dim=HEAD_DIM,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        rotary_dim=ROTARY_DIM,
        eps=EPS,
        q_norm_weight_bias=1.0,
    )

    q_ref8 = q_ref.to(torch.float8_e4m3fn)
    _assert_close(q_out, q_ref8, rtol=FP8_RTOL, atol=FP8_ATOL, max_abs_err=FP8_MAX_ABS_ERR, name="Q")
    _assert_close(k_out, k_ref, rtol=BF16_RTOL, atol=BF16_ATOL, max_abs_err=BF16_MAX_ABS_ERR, name="K")
    assert torch.equal(z_ref, z_out), "Z passthrough mismatch"
