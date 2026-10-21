# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

from types import SimpleNamespace

import pytest

pytest.importorskip(
    "vllm.transformers_utils.configs.deepseek_v41",
    reason="DeepSeek V4.1 is unavailable on this vLLM release",
)

import torch
from vllm.v1.core.kv_cache_utils import _project_kv_cache_groups_to_worker

from tests.deepseek_v41_utils import allocate_cache_views, make_cache_config
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.models.deepseek_v41.cache_config import (
    get_deepseek_v41_kv_cache_config,
    get_deepseek_v41_pool_bytes_per_block,
)


@pytest.mark.parametrize("partition", [(20, 20), (8, 12, 20), (8, 6, 6, 20)])
def test_pp_projection_preserves_three_draft_caches_and_isolates_live_group_pages(partition):
    """A last PP stage with one KV source still needs all three draft SWAs."""
    global_config = make_cache_config(17, head_size=8, index_size=4, draft_layers=3)
    global_groups = global_config.kv_cache_groups
    global_specs = {name: spec for group in global_groups for name, spec in group.kv_cache_spec.kv_cache_specs.items()}
    draft_names = {f"mtp.{index}.self_attn.swa_cache" for index in range(3)}
    runtime = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None, prefix_cache_retention_interval=None)
    )
    allocated_names = set()
    start = 0
    for rank, count in enumerate(partition):
        end = start + count
        is_last_rank = rank == len(partition) - 1
        local_specs = {
            name: spec
            for name, spec in global_specs.items()
            if (name in draft_names and is_last_rank)
            or (name not in draft_names and start <= int(name.split(".layers.")[1].split(".")[0]) < end)
        }
        projected = _project_kv_cache_groups_to_worker(global_groups, local_specs)
        bytes_per_block = get_deepseek_v41_pool_bytes_per_block(projected)
        config = get_deepseek_v41_kv_cache_config(runtime, projected, bytes_per_block * 17)
        slots = {name: index for index, allocation in enumerate(config.kv_cache_tensors) for name in allocation.layers}
        assert sum(len(allocation.layers) for allocation in config.kv_cache_tensors) == len(local_specs)
        assert set(slots) == set(local_specs)
        assert config.num_blocks == 17
        assert sum(allocation.size for allocation in config.kv_cache_tensors) == bytes_per_block * 17
        assert draft_names.intersection(slots) == (draft_names if is_last_rank else set())
        if is_last_rank:
            assert len({slots[name] for name in draft_names}) == 3

        _, caches = allocate_cache_views(config)
        assert set(caches) == set(local_specs)
        expected = []
        for group_index, group in enumerate(projected):
            aliases = [name for name in group.layer_names if not isinstance(local_specs[name], AscendMLAAttentionSpec)]
            assert len({slots[name] for name in aliases}) == len(aliases)
            # Every scheduler group gets a distinct live block ID. Members of
            # that group must remain independent at the same physical page.
            for member_index, name in enumerate(group.layer_names):
                cache = caches[name]
                for plane_index, plane in enumerate(cache if isinstance(cache, tuple) else (cache,)):
                    value = group_index * 8 + member_index * 2 + plane_index + 1
                    page = plane[group_index + 1]
                    page.fill_(value)
                    expected.append((page, value))
        for page, value in expected:
            torch.testing.assert_close(page, torch.full_like(page, value))
        for cache in caches.values():
            for plane in cache if isinstance(cache, tuple) else (cache,):
                assert torch.count_nonzero(plane[0]) == 0

        assert allocated_names.isdisjoint(slots)
        allocated_names.update(slots)
        start = end

    assert allocated_names == set(global_specs)
