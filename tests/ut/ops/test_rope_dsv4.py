# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.ops import rope_dsv4


@pytest.fixture
def rope_state(monkeypatch):
    state = rope_dsv4.RopeGlobalState()
    monkeypatch.setattr(rope_dsv4, "_ROPE_STATE", state)
    for config, width in (("small", 4), ("large", 8)):
        angles = torch.arange(32 * width, dtype=torch.float32).reshape(32, 1, 1, width)
        state.full_rope_cache[config] = (angles.cos(), angles.sin())
        state.registry_summary[config] = {"default", "compressed"}
        state.runtime_buffer[config] = {}
        state.spec_runtime_buffer[config] = {}
        for group in state.registry_summary[config]:
            state.layer_info[f"{config}.{group}"] = (config, [group])
            state.runtime_buffer[config][group] = (torch.empty(16, 1, 1, width), torch.empty(16, 1, 1, width))
            state.spec_runtime_buffer[config][group] = (
                [torch.empty(16, 1, 1, width)],
                [torch.empty(16, 1, 1, width)],
            )
    return state


def _assert_rope(state, result, positions):
    for config, cache in state.full_rope_cache.items():
        for group, pos in positions.items():
            for proxy, full in zip(result, cache):
                torch.testing.assert_close(proxy[f"{config}.{group}"], full[pos], rtol=0, atol=0)


def test_global_local_rope_buffers_survive_interleaved_cache_groups(rope_state):
    # Each PCP cache-group builder owns its global buffers, while local
    # metadata reuses the first group's result for the rest of the step.
    global_buffers: list[dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]]] = [{}, {}, {}]
    addresses: dict[tuple[int, str, str, int], int] = {}
    for count in (2, 16, 5, 16, 2):
        global_pos = {"default": torch.arange(count), "compressed": torch.arange(count) + 1}
        local_pos = {group: pos.flip(0)[: max(1, count - 1)] + 3 for group, pos in global_pos.items()}
        global_results = []
        for group_index, buffers in enumerate(global_buffers):
            global_result = rope_dsv4.get_cos_and_sin_dsa(global_pos, use_cache=True, runtime_buffer=buffers)
            global_results.append(global_result)
            if group_index == 0:
                local_result = rope_dsv4.get_cos_and_sin_dsa(local_pos, use_cache=True)
            # A later global build must preserve the already-created local view.
            _assert_rope(rope_state, local_result, local_pos)
            for result in global_results:
                _assert_rope(rope_state, result, global_pos)
            for config, groups in buffers.items():
                for group, pair in groups.items():
                    for index, buf in enumerate(pair):
                        default = rope_state.runtime_buffer[config][group][index]
                        assert buf.shape == default.shape
                        assert buf.data_ptr() != default.data_ptr()
                        key = (group_index, config, group, index)
                        assert buf.data_ptr() == addresses.setdefault(key, buf.data_ptr())
        assert len(set(addresses.values())) == len(addresses)


@pytest.mark.parametrize("draft_index", [None, 1])
def test_uncached_rope_does_not_allocate_caller_buffers(rope_state, draft_index):
    buffers: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    positions = {"default": torch.tensor([3, 1, 7])}
    result = rope_dsv4.get_cos_and_sin_dsa(positions, draft_index=draft_index, runtime_buffer=buffers)
    _assert_rope(rope_state, result, positions)
    assert buffers == {}


def test_draft_rope_keeps_existing_speculative_buffers(rope_state):
    buffers: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    positions = {"default": torch.tensor([3, 1, 7])}
    result = rope_dsv4.get_cos_and_sin_dsa(positions, use_cache=True, draft_index=1, runtime_buffer=buffers)
    _assert_rope(rope_state, result, positions)
    for config in rope_state.full_rope_cache:
        for index, proxy in enumerate(result):
            assert proxy[f"{config}.default"].data_ptr() == (
                rope_state.spec_runtime_buffer[config]["default"][index][0].data_ptr()
            )
    assert buffers == {}
