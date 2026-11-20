# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.current_kv_scatter import (
    try_paired_current_scatter,
)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)
@pytest.mark.parametrize(("key_width", "rope_width"), [(512, 64), (5, 3)])
@pytest.mark.parametrize("slot_pattern", ["valid", "mixed", "invalid", "padding"])
@torch.inference_mode()
def test_paired_current_scatter_masks_invalid_slots(key_width: int, rope_width: int, slot_pattern: str) -> None:
    capacity = 8
    limits = torch.iinfo(torch.int32)
    slot_values = {
        "valid": [0, capacity - 1],
        "mixed": [limits.min, -1, 0, capacity - 1, capacity, limits.max],
        "invalid": [limits.min, -1, capacity, limits.max],
        "padding": [-1],
    }[slot_pattern]
    rows = len(slot_values)
    keys = torch.arange(rows * key_width, dtype=torch.float32).reshape(rows, key_width).to(torch.bfloat16)
    ropes = torch.arange(rows * rope_width, dtype=torch.float32).reshape(rows, rope_width).to(torch.bfloat16)
    cache_keys = torch.full((capacity, key_width), -1, dtype=torch.bfloat16, device="npu")
    cache_ropes = torch.full((capacity, rope_width), -2, dtype=torch.bfloat16, device="npu")
    expected_keys = cache_keys.cpu()
    expected_ropes = cache_ropes.cpu()
    for row, slot in enumerate(slot_values):
        if 0 <= slot < capacity:
            expected_keys[slot].copy_(keys[row])
            expected_ropes[slot].copy_(ropes[row])

    assert try_paired_current_scatter(
        torch.tensor(slot_values, dtype=torch.int32, device="npu"),
        keys.to("npu"),
        ropes.to("npu"),
        cache_keys,
        cache_ropes,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(cache_keys.cpu(), expected_keys, rtol=0, atol=0)
    torch.testing.assert_close(cache_ropes.cpu(), expected_ropes, rtol=0, atol=0)
