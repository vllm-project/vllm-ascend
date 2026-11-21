# Copyright (c) 2026 Huawei Technologies Co., Ltd.

"""Device-residency guards for the dspark SWA indices triton wrapper.

Launching the kernel with host (CPU) tensors dereferences host pointers on
the device and faults it with AI Core Error 95 instead of raising (#21270);
the wrapper must reject such calls loudly, and the builder's fast-path
eligibility must fall back to the eager torch chain for CPU-device builders.
"""

from __future__ import annotations

import pytest
import torch

from tests.ut.base import TestBase
from vllm_ascend.ops.triton.dspark_swa_indices import (
    build_dspark_swa_slot_indices_triton,
)


class TestDsparkSwaIndicesDeviceGuard(TestBase):
    @staticmethod
    def _cpu_args():
        block_table = torch.tensor([[2, 3], [5, 6]], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 3, 6], dtype=torch.int32)
        seq_lens = torch.tensor([10, 14], dtype=torch.int32)
        return block_table, query_start_loc, seq_lens

    def test_rejects_cpu_tensors_with_clear_error(self):
        block_table, query_start_loc, seq_lens = self._cpu_args()
        with pytest.raises(ValueError, match="requires an accelerator device"):
            build_dspark_swa_slot_indices_triton(
                block_table,
                num_speculative_tokens=3,
                window_size=4096,
                block_size=128,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
            )

    def test_rejects_mixed_devices_naming_the_argument(self):
        block_table, query_start_loc, seq_lens = self._cpu_args()
        # meta-device tensors need no accelerator, so this test also runs on
        # the CPU-only CI lane.
        device_slots = torch.empty((16, 1, 4224), dtype=torch.int32, device="meta")
        with pytest.raises(ValueError, match="indices_output on meta"):
            build_dspark_swa_slot_indices_triton(
                block_table,
                num_speculative_tokens=3,
                window_size=4096,
                block_size=128,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                indices_output=device_slots,
            )
