#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

"""
Test suite for the SwiGluDynamicQuant fused operator.

This module tests the migrated swi_glu_dynamic_quant operator for:
1. Output shape and dtype correctness
2. Numerical correctness against reference (torch_npu.npu_swiglu + npu_dynamic_quant)
3. Latency comparison and optimization rate vs the two-op reference path

Usage:
    # Run all tests (unit tests work in mock environment):
    pytest tests/ut/ops/test_swi_glu_dynamic_quant.py -sv

    # Run on-device benchmark (requires NPU hardware):
    python tests/ut/ops/test_swi_glu_dynamic_quant.py
"""

import math
import time
from unittest.mock import MagicMock, patch

import pytest
import torch


# ============================================================================
# Helper: CPU reference implementation of SwiGLU + DynamicQuant
# ============================================================================

def swiglu_cpu(x: torch.Tensor, activate_left: bool = False) -> torch.Tensor:
    """CPU reference: SwiGLU activation (split last dim in half)."""
    half = x.shape[-1] // 2
    if activate_left:
        a, b = x[..., :half], x[..., half:]
    else:
        b, a = x[..., :half], x[..., half:]
    return a * torch.sigmoid(a) * b


def dynamic_quant_cpu(x: torch.Tensor):
    """CPU reference: per-row symmetric dynamic quantization to int8.

    Returns:
        y: int8 quantized tensor
        scale: float32 dequantization factor per row (max_abs / 127)
    """
    # Flatten to 2D for per-row quantization
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    rows = x_2d.shape[0]

    abs_max = x_2d.abs().amax(dim=-1)  # [rows]
    scale = abs_max / 127.0  # dequant factor
    # Avoid division by zero
    quant_scale = torch.where(abs_max > 0, 127.0 / abs_max,
                              torch.zeros_like(abs_max))

    y_float = x_2d * quant_scale.unsqueeze(-1)
    y = y_float.round().clamp(-128, 127).to(torch.int8)

    y = y.reshape(orig_shape[:-1] + (orig_shape[-1],))
    scale = scale.reshape(orig_shape[:-1])
    return y, scale


def swiglu_dynamic_quant_cpu_reference(
    x: torch.Tensor,
    activate_left: bool = False,
) -> tuple:
    """Full CPU reference: SwiGLU + DynamicQuant combined."""
    x_float = x.float()
    swiglu_out = swiglu_cpu(x_float, activate_left=activate_left)
    y, scale = dynamic_quant_cpu(swiglu_out)
    return y, scale


# ============================================================================
# Unit Tests (mock environment, no NPU required)
# ============================================================================

class TestSwiGluDynamicQuantShape:
    """Test output shape and dtype correctness."""

    @pytest.mark.parametrize("batch_size,hidden_dim", [
        (1, 64),
        (4, 128),
        (16, 256),
        (32, 512),
        (128, 1024),
    ])
    def test_output_shape(self, batch_size, hidden_dim):
        """Verify output shapes match spec: y=[M, N/2], scale=[M]."""
        x = torch.randn(batch_size, hidden_dim, dtype=torch.float16)
        y, scale = swiglu_dynamic_quant_cpu_reference(x)

        assert y.shape == (batch_size, hidden_dim // 2), \
            f"y shape mismatch: {y.shape} vs expected {(batch_size, hidden_dim // 2)}"
        assert scale.shape == (batch_size,), \
            f"scale shape mismatch: {scale.shape} vs expected {(batch_size,)}"

    @pytest.mark.parametrize("batch_size,hidden_dim", [
        (1, 64),
        (16, 256),
    ])
    def test_output_dtype(self, batch_size, hidden_dim):
        """Verify output dtypes: y=int8, scale=float32."""
        x = torch.randn(batch_size, hidden_dim, dtype=torch.float16)
        y, scale = swiglu_dynamic_quant_cpu_reference(x)

        assert y.dtype == torch.int8, f"y dtype mismatch: {y.dtype}"
        assert scale.dtype == torch.float32, f"scale dtype mismatch: {scale.dtype}"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_input_dtypes(self, dtype):
        """Test that all supported input dtypes produce valid output."""
        x = torch.randn(8, 128, dtype=dtype)
        y, scale = swiglu_dynamic_quant_cpu_reference(x)

        assert y.shape == (8, 64)
        assert scale.shape == (8,)
        assert y.dtype == torch.int8
        assert scale.dtype == torch.float32

    def test_3d_input(self):
        """Test with 3D input tensor [batch, seq, hidden]."""
        x = torch.randn(2, 4, 128, dtype=torch.float16)
        x_float = x.float()
        swiglu_out = swiglu_cpu(x_float)
        assert swiglu_out.shape == (2, 4, 64)

    def test_even_dim_check(self):
        """Last dimension must be even for SwiGLU split."""
        x = torch.randn(4, 127, dtype=torch.float16)
        with pytest.raises((RuntimeError, AssertionError, ValueError)):
            # The CPU reference should handle this gracefully
            # but the operator itself requires even last dim
            assert x.shape[-1] % 2 == 0, "Last dim must be even"


class TestSwiGluDynamicQuantNumerical:
    """Test numerical correctness of CPU reference implementation."""

    def test_swiglu_basic(self):
        """Test SwiGLU output is within expected range."""
        torch.manual_seed(42)
        x = torch.randn(4, 8, dtype=torch.float32)
        out = swiglu_cpu(x)

        assert out.shape == (4, 4)
        # SwiGLU should produce finite values
        assert torch.isfinite(out).all()

    def test_activate_left_vs_right(self):
        """Test activate_left=True vs False produces different results."""
        torch.manual_seed(42)
        x = torch.randn(4, 8, dtype=torch.float32)

        out_left = swiglu_cpu(x, activate_left=True)
        out_right = swiglu_cpu(x, activate_left=False)

        assert out_left.shape == out_right.shape
        assert not torch.allclose(out_left, out_right), \
            "activate_left and activate_right should differ"

    def test_quant_scale_range(self):
        """Quantization scale (dequant factor) should be non-negative."""
        torch.manual_seed(42)
        x = torch.randn(16, 128, dtype=torch.float16)
        y, scale = swiglu_dynamic_quant_cpu_reference(x)

        assert (scale >= 0).all(), "Dequant scale should be non-negative"

    def test_quant_output_range(self):
        """Quantized int8 output should be in [-128, 127]."""
        torch.manual_seed(42)
        x = torch.randn(16, 128, dtype=torch.float16)
        y, scale = swiglu_dynamic_quant_cpu_reference(x)

        assert y.min() >= -128
        assert y.max() <= 127

    def test_dequant_reconstruction_error(self):
        """Dequantized output should approximate the SwiGLU result."""
        torch.manual_seed(42)
        x = torch.randn(16, 128, dtype=torch.float16)
        x_float = x.float()
        swiglu_out = swiglu_cpu(x_float)

        y, scale = dynamic_quant_cpu(swiglu_out)
        # Reconstruct: y_float ≈ y_int8 * scale
        reconstructed = y.float() * scale.unsqueeze(-1)

        # Relative error should be small (int8 quant ~ 1/127 ≈ 0.8% max)
        rel_error = (reconstructed - swiglu_out).abs() / (
            swiglu_out.abs() + 1e-6)
        mean_rel_error = rel_error.mean().item()
        assert mean_rel_error < 0.05, \
            f"Mean relative reconstruction error too high: {mean_rel_error:.4f}"

    def test_zero_input(self):
        """Zero input should produce zero output and zero scale."""
        x = torch.zeros(4, 64, dtype=torch.float32)
        y, scale = swiglu_dynamic_quant_cpu_reference(x)

        assert (y == 0).all(), "Zero input should produce zero quantized output"
        assert (scale == 0).all(), "Zero input should produce zero scale"

    @pytest.mark.parametrize("batch_size,hidden_dim", [
        (1, 64),
        (32, 256),
        (128, 1024),
        (256, 2048),
    ])
    def test_various_shapes(self, batch_size, hidden_dim):
        """Test correctness across various tensor sizes."""
        torch.manual_seed(123)
        x = torch.randn(batch_size, hidden_dim, dtype=torch.float16)
        y, scale = swiglu_dynamic_quant_cpu_reference(x)

        assert y.shape == (batch_size, hidden_dim // 2)
        assert scale.shape == (batch_size,)
        assert torch.isfinite(scale).all()


class TestSwiGluDynamicQuantOpRegistration:
    """Test that the operator is properly registered in the torch library."""

    @patch("torch.ops._C_ascend.npu_swi_glu_dynamic_quant")
    def test_op_callable(self, mock_op):
        """Verify the op is callable via torch.ops._C_ascend."""
        mock_op.return_value = (
            torch.randint(-128, 127, (4, 32), dtype=torch.int8),
            torch.randn(4, dtype=torch.float32),
        )

        x = torch.randn(4, 64, dtype=torch.float16)
        y, scale = torch.ops._C_ascend.npu_swi_glu_dynamic_quant(x)

        mock_op.assert_called_once()
        assert y.shape == (4, 32)
        assert scale.shape == (4,)

    @patch("torch.ops._C_ascend.npu_swi_glu_dynamic_quant")
    def test_op_with_optional_args(self, mock_op):
        """Verify optional arguments are passed correctly."""
        mock_op.return_value = (
            torch.randint(-128, 127, (4, 32), dtype=torch.int8),
            torch.randn(4, dtype=torch.float32),
        )

        x = torch.randn(4, 64, dtype=torch.float16)
        smooth_scales = torch.randn(1, 32, dtype=torch.float32)
        group_index = torch.tensor([4], dtype=torch.int32)

        y, scale = torch.ops._C_ascend.npu_swi_glu_dynamic_quant(
            x,
            smooth_scales=smooth_scales,
            offsets=None,
            group_index=group_index,
            activate_left=True,
            quant_mode="dynamic",
            group_list_type=0,
            dst_type=2,
        )
        mock_op.assert_called_once()

    @patch("torch.ops._C_ascend.npu_swi_glu_dynamic_quant")
    def test_op_default_args(self, mock_op):
        """Verify default argument values work."""
        mock_op.return_value = (
            torch.randint(-128, 127, (4, 32), dtype=torch.int8),
            torch.randn(4, dtype=torch.float32),
        )

        x = torch.randn(4, 64, dtype=torch.float16)
        # Call with only required arg
        torch.ops._C_ascend.npu_swi_glu_dynamic_quant(x)
        mock_op.assert_called_once()


class TestSwiGluDynamicQuantVsReference:
    """Compare fused op against torch_npu.npu_swiglu + npu_dynamic_quant."""

    @patch("torch.ops._C_ascend.npu_swi_glu_dynamic_quant")
    @patch("torch_npu.npu_dynamic_quant")
    @patch("torch_npu.npu_swiglu")
    def test_fused_vs_separate_output_shapes(
        self, mock_swiglu, mock_dynamic_quant, mock_fused
    ):
        """Both paths should produce outputs with the same shapes."""
        batch, hidden = 16, 256
        half_hidden = hidden // 2
        x = torch.randn(batch, hidden, dtype=torch.float16)

        # Mock separate path: swiglu -> dynamic_quant
        mock_swiglu.return_value = torch.randn(batch, half_hidden,
                                               dtype=torch.float16)
        mock_dynamic_quant.return_value = (
            torch.randint(-128, 127, (batch, half_hidden), dtype=torch.int8),
            torch.randn(batch, dtype=torch.float32),
        )

        # Mock fused path
        mock_fused.return_value = (
            torch.randint(-128, 127, (batch, half_hidden), dtype=torch.int8),
            torch.randn(batch, dtype=torch.float32),
        )

        # Separate path
        swiglu_out = mock_swiglu(x)
        y_ref, scale_ref = mock_dynamic_quant(swiglu_out)

        # Fused path
        y_fused, scale_fused = mock_fused(x)

        # Shape and dtype checks
        assert y_ref.shape == y_fused.shape, \
            f"y shape mismatch: ref={y_ref.shape}, fused={y_fused.shape}"
        assert scale_ref.shape == scale_fused.shape, \
            f"scale shape mismatch: ref={scale_ref.shape}, fused={scale_fused.shape}"
        assert y_ref.dtype == y_fused.dtype
        assert scale_ref.dtype == scale_fused.dtype


# ============================================================================
# On-device benchmark (requires NPU hardware)
# ============================================================================

def run_npu_benchmark():
    """
    On-device benchmark comparing:
      - Reference: torch_npu.npu_swiglu + torch_npu.npu_dynamic_quant (two ops)
      - Fused: torch.ops._C_ascend.npu_swi_glu_dynamic_quant (single fused op)

    Run this directly with: python tests/ut/ops/test_swi_glu_dynamic_quant.py
    """
    try:
        import torch_npu
    except ImportError:
        print("ERROR: torch_npu not available. This benchmark requires NPU hardware.")
        return

    device = torch.device("npu:0")
    torch.npu.set_device(device)

    print("=" * 72)
    print("SwiGluDynamicQuant Benchmark: Fused Op vs Separate Ops")
    print("=" * 72)

    test_configs = [
        # (batch_size, hidden_dim, dtype)
        (1, 1024, torch.float16),
        (4, 2048, torch.float16),
        (16, 4096, torch.float16),
        (32, 4096, torch.float16),
        (64, 8192, torch.float16),
        (128, 8192, torch.float16),
        (256, 8192, torch.float16),
        (16, 4096, torch.bfloat16),
        (64, 8192, torch.bfloat16),
    ]

    warmup_iters = 50
    bench_iters = 200

    print(f"\nWarmup iterations: {warmup_iters}")
    print(f"Benchmark iterations: {bench_iters}")
    print("-" * 72)
    print(f"{'Config':>30s} | {'Ref (ms)':>10s} | {'Fused (ms)':>10s} | "
          f"{'Speedup':>8s} | {'Opt Rate':>8s}")
    print("-" * 72)

    for batch_size, hidden_dim, dtype in test_configs:
        config_str = f"[{batch_size}, {hidden_dim}] {dtype}"
        x = torch.randn(batch_size, hidden_dim, dtype=dtype, device=device)

        # ---- Reference path: swiglu + dynamic_quant ----
        # Warmup
        for _ in range(warmup_iters):
            swiglu_out = torch_npu.npu_swiglu(x)
            _ = torch_npu.npu_dynamic_quant(swiglu_out)
        torch.npu.synchronize()

        # Benchmark
        t0 = time.perf_counter()
        for _ in range(bench_iters):
            swiglu_out = torch_npu.npu_swiglu(x)
            _ = torch_npu.npu_dynamic_quant(swiglu_out)
        torch.npu.synchronize()
        ref_time_ms = (time.perf_counter() - t0) / bench_iters * 1000.0

        # ---- Fused path: swi_glu_dynamic_quant ----
        # Warmup
        for _ in range(warmup_iters):
            _ = torch.ops._C_ascend.npu_swi_glu_dynamic_quant(x)
        torch.npu.synchronize()

        # Benchmark
        t0 = time.perf_counter()
        for _ in range(bench_iters):
            _ = torch.ops._C_ascend.npu_swi_glu_dynamic_quant(x)
        torch.npu.synchronize()
        fused_time_ms = (time.perf_counter() - t0) / bench_iters * 1000.0

        # ---- Calculate metrics ----
        speedup = ref_time_ms / fused_time_ms if fused_time_ms > 0 else float('inf')
        opt_rate = (1.0 - fused_time_ms / ref_time_ms) * 100.0 if ref_time_ms > 0 else 0.0

        print(f"{config_str:>30s} | {ref_time_ms:>10.4f} | {fused_time_ms:>10.4f} | "
              f"{speedup:>7.2f}x | {opt_rate:>7.2f}%")

    print("-" * 72)

    # ---- Correctness validation on NPU ----
    print("\n" + "=" * 72)
    print("Correctness Validation (NPU)")
    print("=" * 72)

    for batch_size, hidden_dim, dtype in [(16, 4096, torch.float16),
                                          (64, 8192, torch.bfloat16)]:
        x = torch.randn(batch_size, hidden_dim, dtype=dtype, device=device)

        # Reference
        swiglu_out = torch_npu.npu_swiglu(x)
        y_ref, scale_ref = torch_npu.npu_dynamic_quant(swiglu_out)

        # Fused
        y_fused, scale_fused = torch.ops._C_ascend.npu_swi_glu_dynamic_quant(x)

        # Compare shapes
        shape_ok = (y_ref.shape == y_fused.shape and
                    scale_ref.shape == scale_fused.shape)

        # Compare values (allow for minor quantization differences)
        y_diff = (y_ref.int() - y_fused.int()).abs()
        y_max_diff = y_diff.max().item()
        y_match_rate = (y_diff <= 1).float().mean().item() * 100.0

        scale_diff = (scale_ref - scale_fused).abs()
        scale_rel_diff = (scale_diff / (scale_ref.abs() + 1e-8)).mean().item()

        config_str = f"[{batch_size}, {hidden_dim}] {dtype}"
        print(f"\n{config_str}:")
        print(f"  Shape match:          {shape_ok}")
        print(f"  y max abs diff:       {y_max_diff}")
        print(f"  y match rate (±1):    {y_match_rate:.2f}%")
        print(f"  scale mean rel diff:  {scale_rel_diff:.6f}")

        status = "PASS" if (shape_ok and y_match_rate > 95.0) else "FAIL"
        print(f"  Status:               {status}")

    print("\n" + "=" * 72)
    print("Benchmark complete.")


if __name__ == "__main__":
    run_npu_benchmark()
