# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""NPU cache-write integration; full-model accuracy/performance is separate.

Context rows whose slot is ``PAD_SLOT_ID`` must not write K/V into the cache.
Native vLLM produces those slots; this test checks the NPU write path honours
them, which is the accuracy-critical half of the context prefix.
"""

import pytest
import torch
import torch.nn as nn
import torch_npu
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

_INITIAL_CACHE_VALUE = 2
_FIRST_REAL_SLOT = 1


class ContextKV(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(512, 64, device="npu", dtype=torch.bfloat16) * 0.02
        self.gamma = torch.ones(64, device="npu", dtype=torch.bfloat16)
        self.caches = self.new_caches()

    @staticmethod
    def new_caches():
        return [
            (
                torch.full((16, 8, 2, 128), _INITIAL_CACHE_VALUE, device="npu", dtype=torch.bfloat16),
                torch.full((16, 8, 2, 128), _INITIAL_CACHE_VALUE, device="npu", dtype=torch.bfloat16),
            )
            for _ in range(3)
        ]

    def precompute_and_store_context_kv(self, states, positions, slots):
        normalized = torch_npu.npu_rms_norm(states, self.gamma, 1e-6)[0]
        projected = torch.nn.functional.linear(normalized, self.weight)
        key, value = projected.chunk(2, dim=-1)
        key = key.reshape(-1, 2, 128).contiguous()
        value = value.reshape(-1, 2, 128).contiguous()
        for (key_cache, value_cache), layer_slots in zip(self.caches, slots):
            torch_npu.npu_scatter_pa_kv_cache(
                key, value, key_cache, value_cache, layer_slots.contiguous(), cache_mode="Norm"
            )


@pytest.mark.parametrize("num_slots", [2, 4, 8])
def test_context_padding_slots_skip_cache_writes(num_slots):
    torch.manual_seed(57632)
    model = ContextKV()
    actual = num_slots - 1

    # One state row per slot; the trailing row is padding (slot == PAD_SLOT_ID).
    # ``npu_scatter_pa_kv_cache`` requires slot_mapping and key rows to match,
    # so the padded row must be present in both.
    states = torch.randn(num_slots, 64, device="npu", dtype=torch.bfloat16)
    positions = torch.arange(num_slots, device="npu", dtype=torch.int64)
    layer_slots = []
    for layer in range(3):
        first = _FIRST_REAL_SLOT + layer * num_slots
        slots = torch.full((num_slots,), PAD_SLOT_ID, dtype=torch.int32, device="npu")
        slots[:actual] = torch.arange(first, first + actual, device="npu", dtype=torch.int32)
        layer_slots.append(slots)

    model.precompute_and_store_context_kv(states, positions, layer_slots)
    torch.npu.synchronize()
    with_padding = [(key.clone(), value.clone()) for key, value in model.caches]

    # Reference: same weights, fresh cache, only the real rows written.
    model.caches = ContextKV.new_caches()
    model.precompute_and_store_context_kv(
        states[:actual], positions[:actual], [slots[:actual] for slots in layer_slots]
    )
    torch.npu.synchronize()

    for (key_cache, value_cache), (ref_key, ref_value) in zip(with_padding, model.caches):
        # Padding wrote nothing: the padded pass is byte-identical to real-only.
        torch.testing.assert_close(key_cache, ref_key, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(value_cache, ref_value, rtol=1e-2, atol=1e-2)

    # Sanity: the real rows actually landed (a no-op scatter must not pass).
    for (key_cache, _), slots in zip(model.caches, layer_slots):
        flat = key_cache.reshape(-1, *key_cache.shape[2:])
        written = flat[slots[:actual].long()]
        initial = torch.full_like(written, _INITIAL_CACHE_VALUE)
        assert not torch.equal(written, initial), "real slots were not written"
