# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Unit tests for the triton-ascend ScatterNdUpdateV2 integration.

Three layers of behaviour are covered, none of which needs an NPU:

  * `can_use_triton_scatter`  -- the pure feasibility gate. Its reject branches
    exercise only tensor metadata (shape/dtype/layout), so they hold on a
    device-free host. Eligible branches are asserted against `HAS_TRITON`
    rather than a hard-coded ``True`` so the suite stays valid where the kernel
    is unavailable.
  * `_trailing_block_is_contiguous` -- the helper backing the layout check.
  * `triton_scatter_nd_update` -- correctness of the *kernel path itself*. The
    kernel launch is mocked (host can't target a device), but the assert that
    matters -- "no AscendC op is reached when triton is eligible" -- is
    enforced by mocking the kernel and asserting it is the one called.
"""

import unittest
from unittest.mock import patch

import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.triton.scatter_nd_update import (
    _trailing_block_is_contiguous,
    can_use_triton_scatter,
    triton_scatter_nd_update,
)


def _ref(var: torch.Tensor, idx: torch.Tensor, upd: torch.Tensor) -> torch.Tensor:
    """PyTorch eager in-place scatter, bit-exact against the kernel's contract."""
    out = var.clone()
    out[tuple(idx.to(torch.long)[:, d] for d in range(idx.shape[-1]))] = upd
    return out


class TestCanUseTritonScatter(unittest.TestCase):
    """Feasibility gate: eligible shapes resolve to HAS_TRITON, the rest reject."""

    def test_canonical_bf16_case_eligible(self):
        var = torch.zeros((47442, 32, 1, 512), dtype=torch.bfloat16)
        idx = torch.zeros((8160, 2), dtype=torch.int32)
        self.assertEqual(can_use_triton_scatter(var, idx), HAS_TRITON)

    def test_int8_indexer_case_eligible(self):
        # [32490,32,1,128] int8 = the captured INT8 indexer path.
        var = torch.zeros((32490, 32, 1, 128), dtype=torch.int8)
        idx = torch.zeros((1013, 2), dtype=torch.int32)
        self.assertEqual(can_use_triton_scatter(var, idx), HAS_TRITON)

    def test_multi_head_indexer_h_gt_1_eligible(self):
        # var.shape[2] (=num_kv_heads) can be > 1; kernel scatters H*D contiguous.
        var = torch.zeros((100, 32, 4, 128), dtype=torch.bfloat16)
        idx = torch.zeros((5, 2), dtype=torch.int32)
        self.assertEqual(can_use_triton_scatter(var, idx), HAS_TRITON)

    def test_k3_eligible(self):
        # indices with depth 3 -> indexes 3 leading dims of a 4D var.
        var = torch.zeros((8, 4, 2, 16), dtype=torch.bfloat16)
        idx = torch.zeros((5, 3), dtype=torch.int32)
        self.assertEqual(can_use_triton_scatter(var, idx), HAS_TRITON)

    def test_k1_eligible(self):
        var = torch.zeros((64, 16), dtype=torch.bfloat16)
        idx = torch.zeros((5, 1), dtype=torch.int32)
        self.assertEqual(can_use_triton_scatter(var, idx), HAS_TRITON)

    # ---- reject branches (pure feasibility, no NPU needed) ----
    def test_rejects_int64_indices(self):
        var = torch.zeros((47442, 32, 1, 512), dtype=torch.bfloat16)
        idx = torch.zeros((8, 2), dtype=torch.int64)
        self.assertFalse(can_use_triton_scatter(var, idx))

    def test_rejects_1d_indices(self):
        idx = torch.zeros(8, dtype=torch.int32)
        self.assertFalse(can_use_triton_scatter(torch.zeros(8, 8), idx))

    def test_rejects_k_gt_4(self):
        var = torch.zeros((2, 2, 2, 2, 2), dtype=torch.bfloat16)
        idx = torch.zeros((4, 5), dtype=torch.int32)
        self.assertFalse(can_use_triton_scatter(var, idx))

    def test_rejects_non_contiguous_trailing_block(self):
        base = torch.zeros((4, 32, 2, 128), dtype=torch.bfloat16)
        var = base[:, :, ::2, :]  # stride(2) == 256 != shape[-1]*1 (128)
        self.assertEqual(var.shape, (4, 32, 1, 128))
        idx = torch.zeros((8, 2), dtype=torch.int32)
        self.assertFalse(can_use_triton_scatter(var, idx))

    def test_rejects_d_dtype_unsupported(self):
        var = torch.zeros((4, 32, 1, 128), dtype=torch.float64)
        idx = torch.zeros((8, 2), dtype=torch.int32)
        self.assertFalse(can_use_triton_scatter(var, idx))

    def test_rejects_i32_offset_overflow(self):
        # stride(0) = (1<<23)*512 = 1<<32 > int32 -> falls back.
        big = torch.zeros((2, (1 << 23), 1, 512), dtype=torch.bfloat16)
        self.assertTrue(big.stride(0) >= (1 << 31))
        idx = torch.zeros((8, 2), dtype=torch.int32)
        self.assertFalse(can_use_triton_scatter(big, idx))

    def test_rejects_d_overflows_ub_budget(self):
        # The kernel materialises a [BLOCK_D] tile; power-of-two padding must
        # fit half the 192KB UB. A 64K-element bf16 row = 128KB > 96KB budget.
        var = torch.zeros((4, 8, 1, 65536), dtype=torch.bfloat16)
        idx = torch.zeros((4, 2), dtype=torch.int32)
        self.assertFalse(can_use_triton_scatter(var, idx))

    def test_rejects_num_tokens_over_grid_limit(self):
        # triton-ascend grid upper bound along each axis is 65535.
        var = torch.zeros((8, 8, 1, 4), dtype=torch.bfloat16)
        idx = torch.zeros((65536, 2), dtype=torch.int32)
        self.assertFalse(can_use_triton_scatter(var, idx))
        # and a hair under is still feasible when triton is available
        idx2 = torch.zeros((65535, 2), dtype=torch.int32)
        self.assertEqual(can_use_triton_scatter(var, idx2), HAS_TRITON)


class TestTrailingBlockIsContiguous(unittest.TestCase):
    """Helper behind the layout check."""

    def test_canonical_contiguous(self):
        cont = torch.zeros((4, 32, 1, 128), dtype=torch.bfloat16)
        self.assertTrue(_trailing_block_is_contiguous(cont, 2))

    def test_stride_gap_rejected(self):
        base = torch.zeros((4, 32, 2, 128), dtype=torch.bfloat16)
        stride_gapped = base[:, :, ::2, :]
        self.assertFalse(_trailing_block_is_contiguous(stride_gapped, 2))

    def test_k1_flat_2d_contiguous(self):
        # K=1 of a 2D contiguous tensor (last dim stride == 1) is contiguous.
        flat = torch.zeros((100, 16), dtype=torch.bfloat16)
        self.assertTrue(_trailing_block_is_contiguous(flat, 1))


class TestScatterNdUpdateKernelDispatch(unittest.TestCase):
    """When triton is eligible, the kernel -- and not AscendC -- is what runs.

    Host has no device, so the kernel launch is mocked; we assert the mock is
    the thing called (i.e. the dispatcher routed through triton, never through
    ``torch.ops._C_ascend.npu_scatter_nd_update_v2``). ``triton_scatter_nd_update``
    owns no fallback of its own -- ``device_op._scatter_nd_update`` holds the
    PyTorch-eager path, so the kernel function must never silently write back via
    eager; mocking the kernel must leave ``var`` untouched (no-op semantics).
    """

    def test_eligible_routes_to_triton_kernel(self):
        if not HAS_TRITON:
            self.skipTest("triton unavailable; kernel path not reachable on host")
        var = torch.zeros((8, 8, 1, 4), dtype=torch.bfloat16)
        idx = torch.zeros((2, 2), dtype=torch.int32)
        idx[0] = torch.tensor([1, 2], dtype=torch.int32)
        idx[1] = torch.tensor([3, 4], dtype=torch.int32)
        upd = torch.randn((2, 1, 4), dtype=torch.bfloat16)
        self.assertTrue(can_use_triton_scatter(var, idx))
        # Eligible input must reach the triton kernel and never AscendC. The
        # torch op namespace can't be intercepted cheaply, so the proxy is the
        # kernel mock being called exactly once: that only happens when the
        # dispatcher routes through triton rather than the AscendC op. The
        # launch syntax is ``kernel[grid](...)``, so the recorded call lands on
        # the mock returned by ``__getitem__``, not on the mock itself.
        with patch(
            "vllm_ascend.ops.triton.scatter_nd_update._scatter_nd_update_kernel",
        ) as mocked_kernel:
            triton_scatter_nd_update(var, idx, upd)
            self.assertEqual(mocked_kernel.__getitem__.call_count, 1)
            self.assertEqual(mocked_kernel.__getitem__.return_value.call_count, 1)

    @unittest.skipUnless(HAS_TRITON, "triton unavailable on host")
    def test_kernel_function_has_no_eager_writeback(self):
        # The kernel function must NOT carry an eager fallback: with the launch
        # mocked to a no-op, `var` stays byte-for-byte unchanged. If a fallback
        # were added here it would shadow the dispatcher's gate and write on
        # ineligible shapes too -- this guard pins that responsibility boundary.
        var = torch.zeros((8, 8, 1, 4), dtype=torch.bfloat16)
        idx = torch.zeros((2, 2), dtype=torch.int32)
        idx[0] = torch.tensor([1, 2], dtype=torch.int32)
        idx[1] = torch.tensor([3, 4], dtype=torch.int32)
        upd = torch.randn((2, 1, 4), dtype=torch.bfloat16)
        before = var.clone()
        self.assertTrue(can_use_triton_scatter(var, idx))

        # The launch syntax is ``kernel[grid](...)``, so a no-op replacement
        # must be subscriptable and return a callable.
        class _NoOpKernel:
            def __getitem__(self, grid):
                return lambda *args, **kwargs: None

        with patch(
            "vllm_ascend.ops.triton.scatter_nd_update._scatter_nd_update_kernel",
            _NoOpKernel(),
        ):
            triton_scatter_nd_update(var, idx, upd)
        self.assertTrue(torch.equal(var, before))


if __name__ == "__main__":
    unittest.main()
