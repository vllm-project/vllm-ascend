# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import torch

from vllm_ascend.patch.worker.patch_v2.patch_uva import UvaBufferWrapper


def test_uva_method_copies_dirty_prefix_and_returns_requested_view():
    # Use CPU storage to check the copy/view contract without requiring an NPU.
    buffer = UvaBufferWrapper.__new__(UvaBufferWrapper)
    buffer._cpu = torch.tensor([10, 20, 30, 40])
    buffer._uva = torch.zeros(4, dtype=torch.int64)
    buffer._modified_indices = {0, 1}

    with patch("vllm_ascend.patch.worker.patch_v2.patch_uva.is_uva_available", return_value=False):
        view = buffer.uva(2)
        assert view.tolist() == [10, 20]
        assert view.data_ptr() == buffer._uva.data_ptr()
        assert buffer._modified_indices == set()
        assert buffer.uva().tolist() == [10, 20, 0, 0]


def test_uva_method_supports_host_visible_storage_and_empty_view():
    buffer = UvaBufferWrapper.__new__(UvaBufferWrapper)
    buffer._cpu = torch.tensor([7, 8])
    buffer._uva = buffer._cpu
    with patch("vllm_ascend.patch.worker.patch_v2.patch_uva.is_uva_available", return_value=True):
        assert buffer.uva(0).numel() == 0
        assert buffer.uva(1).tolist() == [7]
        assert buffer.uva() is buffer._cpu
