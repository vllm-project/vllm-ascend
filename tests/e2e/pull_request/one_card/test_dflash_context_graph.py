# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""NPU cache-write integration; full-model accuracy/performance is separate."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch_npu
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.worker.v2.spec_decode.dflash.context import DFlashContextGraphMixin


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
                torch.full((16, 8, 2, 128), 2, device="npu", dtype=torch.bfloat16),
                torch.full((16, 8, 2, 128), 2, device="npu", dtype=torch.bfloat16),
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


class ContextBuffers(DFlashContextGraphMixin):
    def __init__(self):
        self.num_speculative_steps = 7
        self.max_num_tokens = 128
        self.hidden_states = torch.zeros(128, 64, device="npu", dtype=torch.bfloat16)
        self.context_positions = torch.zeros(128, device="npu", dtype=torch.int64)
        self._context_slot_mappings = torch.full((2, 128), -1, device="npu", dtype=torch.int32)
        self._layer_group_idx = [1, 0, 1]
        self.model = ContextKV()


@pytest.mark.parametrize("num_reqs_padded", [1, 2, 4, 8])
def test_context_prefix_replay_keeps_padding_and_current_layer_slots(num_reqs_padded):
    torch.manual_seed(57632)
    spec = ContextBuffers()
    num_context = spec._compat_num_graph_context_tokens(num_reqs_padded)
    captured_caches = spec.model.caches
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            spec._compat_precompute_context_kv(0, num_context)
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        spec._compat_precompute_context_kv(0, num_context)
    pointers = (spec.hidden_states.data_ptr(), spec._context_slot_mappings.data_ptr())

    # Shrink, refill, and overflow the same captured prefix. Rejected rows
    # remain PAD; each layer uses its current physical cache group's slots.
    for actual in [num_context, 1, num_context - 3, num_context + 7]:
        for key_cache, value_cache in captured_caches:
            key_cache.fill_(2)
            value_cache.fill_(2)
        states = torch.randn(actual, 64, device="npu", dtype=torch.bfloat16)
        positions = torch.arange(actual, device="npu", dtype=torch.int64)
        ids = torch.arange(actual, device="npu", dtype=torch.int32) + 11
        ids[::5] = -1
        group_slots = torch.stack([ids, torch.where(ids < 0, ids, ids + 17)])
        spec.hidden_states[:actual].copy_(states)
        spec.context_positions[:actual].copy_(positions)
        spec._context_slot_mappings[:, :actual].copy_(group_slots)
        desc = SimpleNamespace(cg_mode=CUDAGraphMode.FULL, num_reqs=num_reqs_padded)
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            spec._prepare_context_kv_for_graph(desc, actual)
            graph.replay()
        stream.synchronize()

        reference_caches = spec.model.new_caches()
        spec.model.caches = reference_caches
        spec.model.precompute_and_store_context_kv(states, positions, [group_slots[index] for index in [1, 0, 1]])
        torch.npu.synchronize()
        for captured, reference in zip(captured_caches, reference_caches):
            for produced, expected in zip(captured, reference):
                torch.testing.assert_close(produced, expected, rtol=1e-2, atol=1e-2)
        spec.model.caches = captured_caches
        assert pointers == (spec.hidden_states.data_ptr(), spec._context_slot_mappings.data_ptr())
