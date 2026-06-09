# SPDX-License-Identifier: Apache-2.0
"""E2E correctness test for AscendC fused_gdn_gating kernel.

Validates torch.ops._C_ascend.npu_fused_gdn_gating against a CPU golden
reference across num_heads / batch / dtype combinations.

Prerequisite: the AscendC kernel must be compiled and installed via
  bash csrc/build_aclnn.sh <ROOT_DIR> <SOC_VERSION>

Run:
  pytest tests/e2e/schedule/ops/one_card/test_fused_gdn_gating.py -v
"""

import gc

import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

SEED = 42
NUM_HEADS_VALUES = [4, 6, 8, 12, 16, 24, 32, 48, 64, 128]
BATCH_SIZES = [1, 7, 37, 128, 512, 4096, 16384]
DTYPES = [torch.bfloat16, torch.float16]


# ---------------------------------------------------------------------------
# Golden reference (CPU, pure PyTorch)
# ---------------------------------------------------------------------------


def _golden_fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU golden reference for fused_gdn_gating.

    Uses the same numerically stable softplus form as the AscendC kernel:
        softplus_o = max(x, 0) + log(1 + exp(-|beta*x|)) / beta

    Returns:
        g:           [1, batch, num_heads], fp32.
        beta_output: [1, batch, num_heads], original dtype.
    """
    batch, num_heads = a.shape
    compute_dtype = torch.float32

    A_log_f = A_log.to(compute_dtype)
    a_f = a.to(compute_dtype)
    b_f = b.to(compute_dtype)
    dt_bias_f = dt_bias.to(compute_dtype)

    A_log_expanded = A_log_f.unsqueeze(0).expand(batch, -1)
    dt_bias_expanded = dt_bias_f.unsqueeze(0).expand(batch, -1)

    x = a_f + dt_bias_expanded
    beta_x = beta * x
    softplus_o = torch.maximum(x, torch.tensor(0.0)) + torch.log(1.0 + torch.exp(-torch.abs(beta_x))) / beta

    g = -torch.exp(A_log_expanded) * softplus_o
    g = g.unsqueeze(0)

    beta_output = torch.sigmoid(b_f).to(b.dtype)
    beta_output = beta_output.unsqueeze(0)

    return g, beta_output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(
    num_heads: int,
    batch: int,
    dtype: torch.dtype,
    seed: int = SEED,
):
    """Build random tensors on CPU for both golden and NPU execution."""
    torch.manual_seed(seed)
    A_log = torch.randn(num_heads, dtype=torch.float32)
    dt_bias = torch.randn(num_heads, dtype=torch.float32)
    a = torch.randn(batch, num_heads, dtype=dtype)
    b = torch.randn(batch, num_heads, dtype=dtype)
    return A_log, a, b, dt_bias


def _npu_op_exec(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the AscendC operator on NPU and return CPU tensors."""
    # Ensure correct dtypes and contiguity for the NPU operator.
    if A_log.dtype != torch.float32:
        A_log = A_log.to(torch.float32)
    if dt_bias.dtype != torch.float32:
        dt_bias = dt_bias.to(torch.float32)
    if not A_log.is_contiguous():
        A_log = A_log.contiguous()
    if not dt_bias.is_contiguous():
        dt_bias = dt_bias.contiguous()

    g, beta_output = torch.ops._C_ascend.npu_fused_gdn_gating(
        A_log.npu(),
        a.npu(),
        b.npu(),
        dt_bias.npu(),
        float(beta),
    )
    return g.cpu(), beta_output.cpu()


def _assert_close(actual_g, actual_beta, ref_g, ref_beta, rtol=3e-3, atol=1e-2):
    torch.testing.assert_close(
        actual_g.to(torch.float32),
        ref_g.to(torch.float32),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    torch.testing.assert_close(
        actual_beta.to(torch.float32),
        ref_beta.to(torch.float32),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
# Tests: core correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_heads", NUM_HEADS_VALUES)
@pytest.mark.parametrize("batch", BATCH_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_gdn_gating_vs_reference(num_heads, batch, dtype):
    A_log, a, b, dt_bias = _make_inputs(num_heads, batch, dtype)

    ref_g, ref_beta = _golden_fused_gdn_gating(A_log, a, b, dt_bias)
    npu_g, npu_beta = _npu_op_exec(A_log, a, b, dt_bias)

    _assert_close(npu_g, npu_beta, ref_g, ref_beta)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("num_heads", [16, 32, 64])
@pytest.mark.parametrize("batch", [1, 37])
def test_fused_gdn_gating_non_default_params(num_heads, batch):
    A_log, a, b, dt_bias = _make_inputs(num_heads, batch, torch.bfloat16)

    ref_g, ref_beta = _golden_fused_gdn_gating(
        A_log,
        a,
        b,
        dt_bias,
        beta=0.5,
    )
    npu_g, npu_beta = _npu_op_exec(
        A_log,
        a,
        b,
        dt_bias,
        beta=0.5,
    )

    _assert_close(npu_g, npu_beta, ref_g, ref_beta)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def test_fused_gdn_gating_output_shapes():
    A_log, a, b, dt_bias = _make_inputs(32, 17, torch.bfloat16)

    npu_g, npu_beta = _npu_op_exec(A_log, a, b, dt_bias)

    assert npu_g.shape == (1, 17, 32), f"unexpected g shape: {npu_g.shape}"
    assert npu_g.dtype == torch.float32, f"unexpected g dtype: {npu_g.dtype}"
    assert npu_beta.shape == (1, 17, 32), f"unexpected beta shape: {npu_beta.shape}"
    assert npu_beta.dtype == torch.bfloat16, f"unexpected beta dtype: {npu_beta.dtype}"

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Tests: multi-row processing and Bulk DMA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_heads", [8, 16, 32, 64])
@pytest.mark.parametrize("batch", [64, 256, 1024, 4096])
def test_fused_gdn_gating_large_batch_multi_row(num_heads, batch):
    A_log, a, b, dt_bias = _make_inputs(num_heads, batch, torch.bfloat16)

    ref_g, ref_beta = _golden_fused_gdn_gating(A_log, a, b, dt_bias)
    npu_g, npu_beta = _npu_op_exec(A_log, a, b, dt_bias)

    _assert_close(npu_g, npu_beta, ref_g, ref_beta)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("num_heads", [16, 32, 48])
def test_fused_gdn_gating_bulk_dma_alignment(num_heads):
    """Bulk DMA fast path for nh % 16 == 0."""
    batch = 512
    A_log, a, b, dt_bias = _make_inputs(num_heads, batch, torch.bfloat16)

    ref_g, ref_beta = _golden_fused_gdn_gating(A_log, a, b, dt_bias)
    npu_g, npu_beta = _npu_op_exec(A_log, a, b, dt_bias)

    _assert_close(npu_g, npu_beta, ref_g, ref_beta)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("num_heads", [6, 12, 24])
def test_fused_gdn_gating_non_bulk_dma_fallback(num_heads):
    """Per-row fallback for head counts that don't satisfy Bulk DMA alignment."""
    batch = 256
    A_log, a, b, dt_bias = _make_inputs(num_heads, batch, torch.bfloat16)

    ref_g, ref_beta = _golden_fused_gdn_gating(A_log, a, b, dt_bias)
    npu_g, npu_beta = _npu_op_exec(A_log, a, b, dt_bias)

    _assert_close(npu_g, npu_beta, ref_g, ref_beta)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def test_fused_gdn_gating_small_batch_optimization():
    """Small batch < rows_per_iter; adaptive UB budgeting."""
    num_heads = 32
    batch = 8
    A_log, a, b, dt_bias = _make_inputs(num_heads, batch, torch.bfloat16)

    ref_g, ref_beta = _golden_fused_gdn_gating(A_log, a, b, dt_bias)
    npu_g, npu_beta = _npu_op_exec(A_log, a, b, dt_bias)

    _assert_close(npu_g, npu_beta, ref_g, ref_beta)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def test_fused_gdn_gating_extreme_large_batch():
    """Extreme large batch stress test."""
    num_heads = 32
    batch = 65536
    A_log, a, b, dt_bias = _make_inputs(num_heads, batch, torch.bfloat16)

    ref_g, ref_beta = _golden_fused_gdn_gating(A_log, a, b, dt_bias)
    npu_g, npu_beta = _npu_op_exec(A_log, a, b, dt_bias)

    _assert_close(npu_g, npu_beta, ref_g, ref_beta)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
