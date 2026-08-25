# SPDX-License-Identifier: Apache-2.0
# Kernel source: vllm/v1/worker/gpu/metrics/logits.py
# Coverage: _num_nans_kernel
"""
Precision test for _num_nans_kernel.

Kernel signature:
    _num_nans_kernel(
        logits_ptr,               # fp32 logits [num_reqs, vocab_size]
        logits_stride,            # stride(0) of logits
        num_nans_ptr,             # int32 output [num_reqs]
        vocab_size,               # vocab size
        BLOCK_SIZE: tl.constexpr, # block size for iteration
    )

Counts NaN values in logits per request. Uses libdevice.isnan to detect NaNs
and sums them per row. The upstream kernel has no Ascend-specific
implementation, so it is validated against a CPU reference.
"""

import pytest
import torch

from vllm.triton_utils import triton
from vllm.v1.worker.gpu.metrics import logits as metrics_logits
from vllm.v1.worker.gpu.metrics.logits import _num_nans_kernel

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

# The upstream kernel imports its libdevice from
# ``torch._inductor.runtime.triton_helpers``, which resolves CUDA symbols and
# fails to compile on Ascend. Rebind the module-level libdevice to the CANN
# libdevice before the kernel is compiled (mirrors
# ``vllm_ascend/patch/worker/patch_v2/patch_triton.py``; needed when the patch
# is not loaded, e.g. when running this file standalone with --noconftest).
metrics_logits.libdevice = triton.language.extra.cann.libdevice


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
