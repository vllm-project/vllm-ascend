# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple import npu_mem_ops as ops


@pytest.mark.parametrize("direction", [ops.DIRECTION_H2D, ops.DIRECTION_D2H])
def test_build_params_preserves_cache_order_offsets_and_block_bytes(direction):
    storage = torch.arange(48, dtype=torch.int16).reshape(6, 8)
    src = {"v": storage[1:], "k": torch.zeros(3, 4)}
    dst = {"v": torch.zeros(8, 8, dtype=torch.int16), "k": torch.zeros(7, 4)}
    params = ops.build_params(src, dst, direction)
    assert params.src_bases.tolist() == [storage.data_ptr() + 16, src["k"].data_ptr()]
    assert params.dst_bases.tolist() == [dst["v"].data_ptr(), dst["k"].data_ptr()]
    assert params.bpb.tolist() == [16, 16]
    assert params.src_bases.dtype == np.int64
    assert params.num_sub_tensors == 2
    assert params.direction == direction


def test_empty_cache_descriptors():
    params = ops.build_params({}, {}, ops.DIRECTION_H2D)
    assert params.num_sub_tensors == 0
    assert params.src_bases.size == params.dst_bases.size == params.bpb.size == 0


def test_descriptor_validation_rejects_key_order_and_payload_size():
    tensor = torch.zeros(2, 4)
    with pytest.raises(AssertionError, match="key order"):
        ops.build_params({"a": tensor, "b": tensor}, {"b": tensor, "a": tensor}, 0)
    with pytest.raises(AssertionError, match="per-block bytes mismatch"):
        ops.build_params({"a": tensor}, {"a": torch.zeros(2, 5)}, 0)


@pytest.mark.parametrize("direction", [ops.DIRECTION_H2D, ops.DIRECTION_D2H])
def test_copy_submits_tensor_major_pointer_descriptors(monkeypatch, direction):
    swap = MagicMock()
    monkeypatch.setattr(torch.ops._C_ascend, "swap_blocks_batch", swap, raising=False)
    params = ops.BatchMemcpyParams(np.array([100, 200]), np.array([300, 500]), np.array([8, 16]), 2, direction)
    ops.copy_blocks([2, 0], [1, 3], params)
    src, dst, sizes, actual_direction = swap.call_args.args
    assert src.tolist() == [116, 100, 232, 200]
    assert dst.tolist() == [308, 324, 516, 548]
    assert sizes.tolist() == [8, 8, 16, 16]
    assert all(t.dtype == torch.int64 and t.device.type == "cpu" for t in (src, dst, sizes))
    assert actual_direction == direction
    assert swap.call_count == 1


def test_empty_copy_and_mismatched_ids_never_submit(monkeypatch):
    swap = MagicMock()
    monkeypatch.setattr(torch.ops._C_ascend, "swap_blocks_batch", swap, raising=False)
    params = ops.build_params({}, {}, 0)
    ops.copy_blocks([], [], params)
    with pytest.raises(AssertionError, match="block counts"):
        ops.copy_blocks([0], [], params)
    swap.assert_not_called()
