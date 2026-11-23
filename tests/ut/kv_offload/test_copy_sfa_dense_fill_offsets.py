# SPDX-License-Identifier: Apache-2.0
"""Large physical block IDs require int64 copy descriptors, not large buffers."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import SparseKVOffloadManager


@pytest.mark.parametrize("block_table_kind", ["numpy-int32", "torch-int32", "torch-int64"])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_dense_fill_uses_exact_int64_offsets_above_2gib_and_4gib(monkeypatch, block_table_kind, num_layers):
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    # K uses 4 bytes/token and V uses 12: independently exercise both strides.
    manager.topk_buffers_k = [torch.empty((2, 12, 4), dtype=torch.int8) for _ in range(num_layers)]
    manager.topk_buffers_v = [torch.empty((2, 12, 6), dtype=torch.bfloat16) for _ in range(num_layers)]
    host_bases = [(2**32 + layer * 2**24, 2**33 + layer * 2**24) for layer in range(num_layers)]
    device_bases = [(3 * 2**32 + layer * 2**24, 4 * 2**32 + layer * 2**24) for layer in range(num_layers)]
    manager.copy_sfa_host_bases = [torch.tensor(bases, dtype=torch.int64).view(2, 1) for bases in host_bases]
    manager.copy_sfa_device_bases = [torch.tensor(bases, dtype=torch.int64).view(2, 1) for bases in device_bases]
    copy = Mock()
    monkeypatch.setattr(manager, "copy_sfa_kv", copy)
    values = [[5, 7, 9], [150_000_003, 320_000_007, 11]]
    if block_table_kind == "numpy-int32":
        block_table = np.array(values, dtype=np.int32)
        original = block_table.copy()
    else:
        dtype = torch.int32 if block_table_kind == "torch-int32" else torch.int64
        block_table = torch.tensor(values, dtype=dtype)
        original = block_table.clone()

    # Restore 6 valid tokens into slot 1: one full 4-token block plus a 2-token tail.
    manager.dense_fill_copy_sfa_rows({1: (1, 6)}, block_size=4, block_table=block_table)

    assert copy.call_count == num_layers
    token_bytes = (4, 12)
    assert values[1][0] * 4 * token_bytes[0] > 2**31
    assert values[1][1] * 4 * token_bytes[0] > 2**32
    assert values[1][0] * 4 * token_bytes[1] > 2**32
    for layer, call in enumerate(copy.call_args_list):
        sources, destinations, lengths, count = call.args
        for tensor in (sources, destinations, lengths):
            assert tensor.dtype == torch.int64
            assert tensor.device.type == "cpu"
        assert sources.tolist() == [
            host_bases[layer][component] + block_id * 4 * stride
            for component, stride in enumerate(token_bytes)
            for block_id in values[1][:2]
        ]
        assert destinations.tolist() == [
            device_bases[layer][component] + (12 + block_index * 4) * stride
            for component, stride in enumerate(token_bytes)
            for block_index in range(2)
        ]
        assert lengths.tolist() == [16, 8, 48, 24]
        assert count.dtype == torch.int32
        assert count.tolist() == [4]
    if isinstance(block_table, np.ndarray):
        np.testing.assert_array_equal(block_table, original)
        assert block_table.dtype == np.int32
    else:
        assert torch.equal(block_table, original)
        assert block_table.dtype == (torch.int32 if block_table_kind == "torch-int32" else torch.int64)
