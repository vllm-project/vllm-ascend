# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm_ascend.ops.triton.spec_decode import suffix_local_scan as ops


def test_local_scan_uses_active_mapping_and_tensor_strides(monkeypatch):
    kernel = MagicMock()
    monkeypatch.setattr(ops, "_suffix_local_scan_kernel", kernel)
    token_ids = SimpleNamespace(shape=(128, 2048), stride=lambda axis: 2056)
    scratch = SimpleNamespace(shape=(128, 8), stride=lambda axis: 12)
    mapping = SimpleNamespace(shape=(7,))
    total_len, num_sampled = object(), object()

    ops.launch_suffix_local_scan(
        token_ids, 2048, mapping, total_len, num_sampled, scratch, 8, 2049, MAX_DEPTH=3, BLOCK=256
    )

    kernel.__getitem__.assert_called_once_with((7, 8))
    kernel.__getitem__.return_value.assert_called_once_with(
        token_ids, 2056, mapping, total_len, num_sampled, scratch, 12, 2049, MAX_DEPTH=3, BLOCK=256
    )
