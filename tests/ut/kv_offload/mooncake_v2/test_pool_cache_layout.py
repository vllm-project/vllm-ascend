# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_layout import (
    bind_token_page,
    describe_cache_bindings,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.restore_planner import RestorePlanner, SourceCacheSlice


def binding(tensor, spec):
    config = SimpleNamespace(
        num_blocks=tensor.shape[0],
        kv_cache_groups=[
            SimpleNamespace(layer_names=["layer.0"], kv_cache_spec=spec),
        ],
    )
    return describe_cache_bindings(config, {"layer.0": tensor})["layer.0"][0]


def test_real_hnd_nhd_views_produce_compatible_restoration_ranges():
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16)
    hnd = binding(torch.empty((2, 2, 16, 8), dtype=torch.float16), spec)
    nhd = binding(torch.empty((2, 16, 2, 8), dtype=torch.float16), spec)
    source = bind_token_page(hnd, spec, 0, token_start=0, head_start=2)
    target = bind_token_page(nhd, spec, 1, token_start=0, head_start=2)
    assert source.cache.format_id == target.cache.format_id
    plan = RestorePlanner().build([SourceCacheSlice(source.cache, "source", hnd.page_span, 0)], [target])
    assert sum(item.size for item in plan) == 2 * 16 * 8 * 2
    assert all(target.address <= item.destination < target.address + nhd.page_span for item in plan)


def test_mla_component_major_and_token_fused_views_have_equivalent_semantics():
    spec = MLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16)
    component_major = torch.empty((2, 16, 1, 8), dtype=torch.float16)
    token_fused = torch.empty((2, 16, 1, 12), dtype=torch.float16)[..., :8]
    source_binding = binding(component_major, spec)
    target_binding = binding(token_fused, spec)
    source = bind_token_page(source_binding, spec, 0, token_start=0)
    target = bind_token_page(target_binding, spec, 1, token_start=0)
    assert source.cache.format_id == target.cache.format_id
    assert source.cache.vector_bytes == target.cache.vector_bytes == 16
    plan = RestorePlanner().build([SourceCacheSlice(source.cache, "latent", source_binding.page_span, 0)], [target])
    assert sum(item.size for item in plan) == 16 * 8 * 2
    assert len(plan) == 16  # Destination row padding cannot be overwritten.


def test_mla_without_rope_restores_only_the_nonempty_latent_component():
    spec = AscendMLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16)
    config = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[SimpleNamespace(layer_names=["layer.0", "draft.0"], kv_cache_spec=spec)],
    )
    latent = torch.arange(2 * 16 * 8, dtype=torch.float16).reshape(2, 16, 1, 8)
    destination = torch.full_like(latent, -1)
    empty_rope = torch.empty((2, 16, 1, 0), dtype=torch.float16)
    bindings = describe_cache_bindings(config, {"layer.0": (latent, empty_rope), "draft.0": (destination, empty_rope)})
    assert [item.component for item in bindings["layer.0"]] == [0]
    assert [item.component for item in bindings["draft.0"]] == [0]
    source = bind_token_page(bindings["layer.0"][0], spec, 0, token_start=0)
    # P/D use the same layer name on the wire, including draft layers.
    target_binding = describe_cache_bindings(
        SimpleNamespace(num_blocks=2, kv_cache_groups=config.kv_cache_groups[:1]),
        {"layer.0": (destination, empty_rope), "draft.0": (latent, empty_rope)},
    )["layer.0"][0]
    target = bind_token_page(target_binding, spec, 1, token_start=0)
    plan = RestorePlanner().build(
        [SourceCacheSlice(source.cache, "latent", source.cache.vector_bytes * 16, 0)], [target]
    )
    for item in plan:
        ctypes.memmove(item.destination, source.address + item.source_offset, item.size)
    assert torch.equal(destination[1], latent[0])
    assert torch.all(destination[0] == -1)


@pytest.mark.parametrize("empty_shape", [(2, 16, 0, 8), (2, 8, 1, 0)])
def test_mla_rejects_empty_components_that_are_not_matching_rope_views(empty_shape):
    spec = AscendMLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16)
    config = SimpleNamespace(
        num_blocks=2, kv_cache_groups=[SimpleNamespace(layer_names=["layer.0"], kv_cache_spec=spec)]
    )
    with pytest.raises(ValueError, match="invalid dimensions"):
        describe_cache_bindings(config, {"layer.0": (torch.empty((2, 16, 1, 8)), torch.empty(empty_shape))})


def test_layout_ambiguity_and_non_attention_state_fail_explicitly():
    spec = FullAttentionSpec(block_size=16, num_kv_heads=16, head_size=8, dtype=torch.float16)
    component = binding(torch.empty((2, 16, 16, 8), dtype=torch.float16), spec)
    with pytest.raises(ValueError, match="Ambiguous"):
        bind_token_page(component, spec, 0, token_start=0)
    with pytest.raises(ValueError, match="No token-cache"):
        bind_token_page(component, SimpleNamespace(block_size=16), 0, token_start=0)


@pytest.mark.parametrize("quant_flag", ["cache_sparse_sfa_c8", "cache_sparse_sfa_turboquant"])
@pytest.mark.parametrize("token_view", [False, True])
def test_packed_mla_uses_v2_raw_kernel_pages_with_different_logical_block_sizes(quant_flag, token_view):
    source_spec = AscendMLAAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=8, dtype=torch.uint8, **{quant_flag: True}
    )
    target_spec = AscendMLAAttentionSpec(
        block_size=8, num_kv_heads=1, head_size=8, dtype=torch.uint8, **{quant_flag: True}
    )
    source_storage = torch.arange(4 * 40, dtype=torch.uint8).reshape(4, 40)
    target_storage = torch.full((4, 48), 255, dtype=torch.uint8)
    source_view, target_view = source_storage[:, :32], target_storage[:, :32]
    if token_view:
        # MRV2 represents a packed page with this token-shaped view. Its
        # underlying packet still needs V2's whole-page copy semantics.
        source_view, target_view = source_view.view(4, 4, 1, 8), target_view.view(4, 4, 1, 8)
    source = binding(source_view, source_spec)
    target_config = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[SimpleNamespace(layer_names=["layer.0"], kv_cache_spec=target_spec)],
    )
    target = describe_cache_bindings(target_config, {"layer.0": target_view})["layer.0"][0]
    assert target.block_size_scale == 2
    p = bind_token_page(source, source_spec, 1, token_start=4, storage_rows=4)
    d = bind_token_page(target, target_spec, 3, token_start=4, storage_rows=4)
    assert p.cache.format_id == d.cache.format_id
    assert (p.cache.token_start, p.cache.token_end) == (1, 2)  # Kernel-page coordinates.
    plan = RestorePlanner().build([SourceCacheSlice(p.cache, "packed", p.cache.span_bytes, 0)], [d])
    assert len(plan) == 1 and plan[0].size == 32
    ctypes.memmove(plan[0].destination, p.address, plan[0].size)
    torch.testing.assert_close(target_storage[3, :32], source_storage[1, :32])
    assert torch.all(target_storage[3, 32:] == 255)
    assert torch.all(target_storage[:3] == 255)


def test_physical_rows_are_explicit_when_scheduler_blocks_contain_multiple_kernel_pages():
    spec = FullAttentionSpec(block_size=8, num_kv_heads=1, head_size=2, dtype=torch.float16)
    config = SimpleNamespace(
        num_blocks=2, kv_cache_groups=[SimpleNamespace(layer_names=["layer.0"], kv_cache_spec=spec)]
    )
    component = describe_cache_bindings(config, {"layer.0": torch.empty((4, 1, 4, 2), dtype=torch.float16)})["layer.0"][
        0
    ]
    page = bind_token_page(component, spec, 3, token_start=12, storage_rows=4)
    assert (page.cache.token_start, page.cache.token_end) == (12, 16)
