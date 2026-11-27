# SPDX-License-Identifier: Apache-2.0
"""Single-operator accuracy test for the Step5 gated_scale Triton op.

Compares the fused kernel output against the eager path it replaces:
``attn_output.view(t, H, D) * gate.unsqueeze(-1).sigmoid()``. The gate is
exercised in every layout the model feeds the single base+offset
interface: a strided column view of a wider packed buffer with the packed
buffer itself as base (the sliding-window qkvg layout), the same view with
the buffer's front row-slice as base (the DSA qkv layout, where the gate
columns live past the base's logical width), and a standalone contiguous
gate with offset zero (the sliding-window g_proj fallback layout), so the
kernel's addressing is covered end to end.

Accuracy follows the repository operator criterion for the bf16 output:
rtol = atol = 2^-6, at least 99% of elements within the combined tolerance,
and the maximum absolute error <= 1. Shape coverage uses the real Step5
attention geometry (head_dim 192 and the other supported 128; per-rank head
counts 4/8/16 for TP16/TP8/TP4 on the 64-head checkpoint) with token counts
spanning single-token decode, an odd boundary and a prefill chunk.
"""

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.step5.gated_scale import gated_scale

RTOL = 2.0**-6
ATOL = 2.0**-6
MAX_ABS_ERR = 1.0
MIN_WITHIN_FRACTION = 0.99


def _reference(attn_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    num_tokens = attn_output.shape[0]
    num_heads = gate.shape[-1]
    head_dim = attn_output.shape[-1] // num_heads
    return (attn_output.view(num_tokens, num_heads, head_dim) * gate.unsqueeze(-1).sigmoid()).view(num_tokens, -1)


def _assert_bf16_close(actual: torch.Tensor, ref: torch.Tensor, name: str = "out") -> None:
    a = actual.float()
    r = ref.float()
    diff = (a - r).abs()
    within = diff <= ATOL + RTOL * r.abs()
    frac = within.float().mean().item()
    assert frac >= MIN_WITHIN_FRACTION, (
        f"{name}: only {frac:.4%} elements within BF16 tolerance (max abs err {diff.max().item():.6f})"
    )
    assert diff.max().item() <= MAX_ABS_ERR, f"{name}: max abs err {diff.max().item():.6f} exceeds {MAX_ABS_ERR}"


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Requires Ascend A5",
)
@pytest.mark.parametrize("num_tokens", [1, 333, 4096])
@pytest.mark.parametrize("num_heads", [4, 8, 16])
@pytest.mark.parametrize("head_dim", [128, 192])
def test_gated_scale_standalone_gate(num_tokens: int, num_heads: int, head_dim: int):
    """g_proj layout: contiguous gate as its own base, offset 0."""
    device = "npu"

    torch.manual_seed(123)
    attn_output = torch.randn(num_tokens, num_heads * head_dim, device=device, dtype=torch.bfloat16)
    gate = torch.randn(num_tokens, num_heads, device=device, dtype=torch.bfloat16)

    out = gated_scale(attn_output, gate, 0, num_heads)
    ref = _reference(attn_output.clone(), gate.clone())

    assert out.shape == attn_output.shape and out.dtype == torch.bfloat16
    _assert_bf16_close(out, ref)


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Requires Ascend A5",
)
@pytest.mark.parametrize("num_tokens", [1, 4096])
def test_gated_scale_packed_base_gate(num_tokens: int):
    """qkvg layout: gate as a strided column view, packed buffer as base."""
    device = "npu"
    num_heads, head_dim = 8, 192

    torch.manual_seed(123)
    kv_width = 2 * head_dim
    row_width = num_heads * head_dim + kv_width + num_heads
    packed = torch.randn(num_tokens, row_width, device=device, dtype=torch.bfloat16)
    attn_output = packed[..., : num_heads * head_dim].contiguous()
    gate = packed[..., -num_heads:]
    assert gate.storage_offset() != 0

    out = gated_scale(attn_output, packed, gate.storage_offset(), num_heads)
    ref = _reference(attn_output.clone(), gate.contiguous())

    _assert_bf16_close(out, ref)


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Requires Ascend A5",
)
@pytest.mark.parametrize("num_tokens", [1, 333, 4096])
def test_gated_scale_front_slice_base_gate(num_tokens: int):
    """DSA layout: front row-slice as base, gate beyond its logical width.

    The gate columns live past the base's shape but inside its storage
    row span, exactly how the DSA layer reads the gate tail of the fused
    qkv+indexer row through the offset-0 qkv view.
    """
    device = "npu"
    num_heads, head_dim = 8, 192

    torch.manual_seed(123)
    indexer_width = 4 * 256
    qkv_width = num_heads * head_dim
    combined = torch.randn(
        num_tokens,
        qkv_width + indexer_width + num_heads,
        device=device,
        dtype=torch.bfloat16,
    )
    gate_view = combined[..., -num_heads:]
    qkv = combined[..., :qkv_width]
    assert qkv.storage_offset() == 0
    assert gate_view.storage_offset() >= qkv_width

    attn_output = torch.randn(num_tokens, qkv_width, device=device, dtype=torch.bfloat16)
    out = gated_scale(attn_output, qkv, gate_view.storage_offset(), num_heads)
    ref = _reference(attn_output.clone(), gate_view.contiguous())

    _assert_bf16_close(out, ref)


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Requires Ascend A5",
)
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("head_dim", [128, 192])
def test_gated_scale_eager_equivalence(num_heads: int, head_dim: int):
    """The fused path must match the eager math per input element."""
    device = "npu"

    torch.manual_seed(123)
    num_tokens = 333
    row_width = num_heads * head_dim + num_heads
    packed = torch.randn(num_tokens, row_width, device=device, dtype=torch.bfloat16)
    attn_output = torch.randn(num_tokens, num_heads * head_dim, device=device, dtype=torch.bfloat16)
    gate = packed[..., -num_heads:]

    out = gated_scale(attn_output, packed, row_width - num_heads, num_heads)
    ref = _reference(attn_output.clone(), gate.contiguous())

    _assert_bf16_close(out, ref)


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Requires Ascend A5",
)
def test_gated_scale_bounds_check():
    """Out-of-range offsets must fail loudly instead of reading garbage."""
    device = "npu"

    num_heads, head_dim = 8, 192
    packed = torch.randn(8, num_heads * head_dim + 4, device=device, dtype=torch.bfloat16)
    attn_output = torch.randn(8, num_heads * head_dim, device=device, dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        # offset + num_heads exceeds the base's row span.
        gated_scale(attn_output, packed, packed.shape[1] - 4, num_heads)


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Requires Ascend A5",
)
def test_gated_scale_empty_input():
    """Zero tokens must return an empty output without launching a zero-sized grid."""
    device = "npu"
    num_heads, head_dim = 8, 192
    attn_output = torch.empty((0, num_heads * head_dim), device=device, dtype=torch.bfloat16)
    gate = torch.empty((0, num_heads), device=device, dtype=torch.bfloat16)

    out = gated_scale(attn_output, gate, 0, num_heads)

    assert out.shape == attn_output.shape
    assert out.dtype == attn_output.dtype
    assert out.numel() == 0
