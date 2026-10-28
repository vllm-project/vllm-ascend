# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""NPU KV-cache write path used by the DFlash context prefix (PAD-slot check).

Context rows whose slot is ``PAD_SLOT_ID`` must not write K/V into the cache.
The Ascend context path calls ``DeviceOperator.reshape_and_cache`` once per
layer with that layer's own physical slot table, so this drives the same
wrapper (not a test-local copy) and checks negative slots are skipped.

Graph replay and full-model context behaviour are validated by the GSM8K run,
not here: this file only guards the write path.
"""

import pytest
import torch
import torch_npu
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.device.device_op import DeviceOperator

_INITIAL_CACHE_VALUE = 2
_FIRST_REAL_SLOT = 1
_CACHE_SHAPE = (16, 8, 2, 128)


def _new_cache():
    return torch.full(_CACHE_SHAPE, _INITIAL_CACHE_VALUE, device="npu", dtype=torch.bfloat16)


@pytest.mark.parametrize("num_slots", [2, 4, 8])
def test_context_padding_slots_skip_cache_writes(num_slots):
    torch.manual_seed(57632)
    actual = num_slots - 1

    # One row per slot; the trailing row is padding (slot == PAD_SLOT_ID).
    # ``reshape_and_cache`` requires slot_mapping and key rows to match, so the
    # padded row must be present in both.
    key = torch.randn(num_slots, 2, 128, device="npu", dtype=torch.bfloat16)
    value = torch.randn(num_slots, 2, 128, device="npu", dtype=torch.bfloat16)

    layer_slots = []
    for layer in range(3):
        first = _FIRST_REAL_SLOT + layer * num_slots
        slots = torch.full((num_slots,), PAD_SLOT_ID, dtype=torch.int32, device="npu")
        slots[:actual] = torch.arange(first, first + actual, device="npu", dtype=torch.int32)
        layer_slots.append(slots)

    cached = []
    for slots in layer_slots:
        key_cache, value_cache = _new_cache(), _new_cache()
        DeviceOperator.reshape_and_cache(key, value, key_cache, value_cache, slots)
        cached.append((key_cache, value_cache))
    torch.npu.synchronize()

    for (key_cache, value_cache), slots in zip(cached, layer_slots):
        # Reference: same real rows only, into a fresh cache.
        ref_key, ref_value = _new_cache(), _new_cache()
        DeviceOperator.reshape_and_cache(
            key[:actual], value[:actual], ref_key, ref_value, slots[:actual]
        )
        torch.npu.synchronize()
        # Padding wrote nothing: the padded pass is identical to real-only.
        torch.testing.assert_close(key_cache, ref_key, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(value_cache, ref_value, rtol=1e-2, atol=1e-2)

        # Sanity: the real rows actually landed (a no-op must not pass).
        flat = key_cache.reshape(-1, *key_cache.shape[2:])
        written = flat[slots[:actual].long()]
        assert not torch.equal(written, torch.full_like(written, _INITIAL_CACHE_VALUE))
