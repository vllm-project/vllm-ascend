# SPDX-License-Identifier: Apache-2.0
# Kernel source: vllm_ascend/ops/triton/v2/metrics/num_nans.py
# Coverage: _num_nans_kernel
"""
Precision test for the Ascend-specific _num_nans_kernel.

Kernel signature:
    _num_nans_kernel(
        logits_ptr,               # fp32 logits [num_reqs, vocab_size]
        logits_stride,            # stride(0) of logits
        num_nans_ptr,             # int32 output [num_reqs]
        vocab_size,               # vocab size
        BLOCK_SIZE: tl.constexpr, # block size for iteration
    )

Counts NaN values in logits per request. Uses the CANN libdevice.isnan to
detect NaNs and sums them per row, validated against a CPU reference.

The upstream kernel in ``vllm/v1/worker/gpu/metrics/logits.py`` imports its
libdevice from ``torch._inductor.runtime.triton_helpers`` (CUDA-oriented), and
its launch path pulls in ``triton.experimental.gluon.nvidia``, which
triton-ascend does not provide. vllm-ascend therefore ships this Ascend kernel
(CANN libdevice) and swaps it into the sampler and rejection sampler via
``vllm_ascend/patch/worker/patch_v2/patch_triton.py``. Testing the Ascend
implementation directly matches what actually runs in production.
"""

import os

import pytest
import torch

# Importing ``vllm_ascend.ops`` runs ``vllm_version_is("0.27.1")`` in
# ``vllm_ascend/ops/fused_moe/fused_moe.py`` at import time. When ``vllm`` is
# resolved as a PEP 420 namespace package (no top-level ``__init__.py``),
# ``vllm.__version__`` is unavailable and that call raises AttributeError.
# Setting VLLM_VERSION makes ``vllm_version_is`` use it instead of
# ``vllm.__version__``. This must happen before the first ``vllm_ascend``
# import below, and it must not override an explicitly exported value.
os.environ.setdefault("VLLM_VERSION", "0.27.1")

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton  # noqa: E402
from vllm_ascend.ops.triton.v2.metrics.num_nans import _num_nans_kernel  # noqa: E402


def _num_nans_ref(logits: torch.Tensor) -> torch.Tensor:
    """CPU reference: count NaNs row-wise."""
    return torch.isnan(logits).sum(dim=-1).to(torch.int32)


class TestNumNansKernel:
    @pytest.mark.parametrize("num_reqs", [1, 2, 4, 8])
    @pytest.mark.parametrize("vocab_size", [128, 1024, 8192, 16384])
    @pytest.mark.parametrize("frac_nan", [0.0, 0.1, 0.5, 1.0])
    def test_num_nans(self, num_reqs, vocab_size, frac_nan):
        """Compare kernel NaN count with the CPU reference."""
        init_device_properties_triton()
        torch.manual_seed(42)
        device = "npu"

        logits = torch.randn(num_reqs, vocab_size, dtype=torch.float32, device=device)
        # Inject NaNs at the requested fraction.
        num_nan = int(vocab_size * frac_nan)
        if num_nan > 0:
            logits[:, :num_nan] = float("nan")

        num_nans = torch.empty(num_reqs, dtype=torch.int32, device=device)
        _num_nans_kernel[(num_reqs,)](
            logits,
            logits.stride(0),
            num_nans,
            vocab_size,
            BLOCK_SIZE=8192,
        )
        torch.npu.synchronize()

        expected = _num_nans_ref(logits.cpu())
        torch.testing.assert_close(num_nans.cpu(), expected, rtol=0, atol=0)

    def test_no_nans(self):
        """When there are no NaNs, all counts should be zero."""
        init_device_properties_triton()
        torch.manual_seed(42)
        device = "npu"

        num_reqs, vocab_size = 4, 4096
        logits = torch.ones(num_reqs, vocab_size, dtype=torch.float32, device=device)

        num_nans = torch.empty(num_reqs, dtype=torch.int32, device=device)
        _num_nans_kernel[(num_reqs,)](
            logits,
            logits.stride(0),
            num_nans,
            vocab_size,
            BLOCK_SIZE=8192,
        )
        torch.npu.synchronize()

        expected = torch.zeros(num_reqs, dtype=torch.int32)
        torch.testing.assert_close(num_nans.cpu(), expected, rtol=0, atol=0)

    def test_all_nans(self):
        """When all values are NaN, each request should report vocab_size NaN."""
        init_device_properties_triton()
        torch.manual_seed(42)
        device = "npu"

        num_reqs, vocab_size = 3, 512
        logits = torch.full((num_reqs, vocab_size), float("nan"), dtype=torch.float32, device=device)

        num_nans = torch.empty(num_reqs, dtype=torch.int32, device=device)
        _num_nans_kernel[(num_reqs,)](
            logits,
            logits.stride(0),
            num_nans,
            vocab_size,
            BLOCK_SIZE=8192,
        )
        torch.npu.synchronize()

        expected = torch.full((num_reqs,), vocab_size, dtype=torch.int32)
        torch.testing.assert_close(num_nans.cpu(), expected, rtol=0, atol=0)
