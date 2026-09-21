# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch
from vllm.model_executor.layers.attention import MLAAttention
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from tests.ut.kvpp_utils import indexer_name, layer_name, make_dspark_kvpp_case, make_kvpp_config, make_kvpp_specs
from vllm_ascend.ascend_config import KVPPConfig
from vllm_ascend.core import kv_cache_placement as placement
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec


@pytest.mark.parametrize("tp,pcp,expected", [(2, 2, 4), (4, 2, 8)])
def test_kvpp_size_covers_pcp_replicas(tp, pcp, expected):
    config = make_kvpp_config(tp)
    config.parallel_config.prefill_context_parallel_size = pcp
    config.parallel_config.pipeline_parallel_size = 2
    config.parallel_config.data_parallel_size = 2
    assert KVPPConfig.from_vllm_config(config).size == expected


@pytest.mark.parametrize("tp,owners", [(3, [0, 0, 0, 1, 1, 1, 2, 2]), (10, list(range(8)))])
@pytest.mark.parametrize("with_mtp", [False, True])
def test_pp_local_owners_and_bundles(tp, owners, with_mtp):
    specs = make_kvpp_specs()
    if not with_mtp:
        del specs[layer_name(17)]
    config = make_kvpp_config(tp)
    plan = placement.create_kvpp_cache_allocation_plan(config, specs, kvpp_rank=1)
    reverse = placement.create_kvpp_cache_allocation_plan(config, dict(reversed(list(specs.items()))), kvpp_rank=1)
    expected = {layer_name(i): owner for i, owner in zip(range(9, 17), owners)}
    expected[indexer_name(11)] = owners[2]
    assert plan.layer_owner_ranks == expected
    assert list(plan.layer_owner_ranks.items()) == list(reverse.layer_owner_ranks.items())
    assert list(plan.layer_bundles.items()) == list(reverse.layer_bundles.items())
    assert list(plan.layer_bundles) == [layer_name(i) for i in range(9, 18 if with_mtp else 17)]
    assert plan.layer_bundles[layer_name(11)] == (layer_name(11), indexer_name(11))
    assert plan.logical_cache_spec == reverse.logical_cache_spec == specs
    assert plan.tensor_sizes == reverse.tensor_sizes
    assert layer_name(17) not in plan.layer_owner_ranks


@pytest.mark.parametrize(
    "packed,scale_dtype,expected_sizes,expected_layout,total",
    [
        (False, torch.float16, ((32, 16), (8, 4)), (((0, 96), (96, 48)), ((144, 24), (168, 12))), 180),
        (True, torch.float16, ((32,), (8, 4)), (((0, 96),), ((96, 24), (120, 12))), 132),
        (True, torch.float32, ((32,), (8, 8)), (((0, 96),), ((96, 24), (120, 24))), 144),
    ],
)
def test_component_sizes_and_layout(monkeypatch, packed, scale_dtype, expected_sizes, expected_layout, total):
    main, indexer = layer_name(11), indexer_name(11)
    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.kv_lora_rank, layer.qk_rope_head_dim = 8, 4
    config = make_kvpp_config()
    specs = {
        main: AscendMLAAttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=16 if packed else 12,
            dtype=torch.int8 if packed else torch.bfloat16,
            cache_sparse_sfa_c8=packed,
        ),
        indexer: AscendSFAIndexerCacheSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.int8,
            scale_dim=1,
            scale_dtype=scale_dtype,
            cache_sparse_li_c8=True,
        ),
    }
    monkeypatch.setattr(placement, "get_layers_from_vllm_config", lambda *_args: {main: layer})
    monkeypatch.setattr(placement, "enable_sfa", lambda _: True)
    sizes = placement.build_kvpp_buffer_sizes(config, specs)
    assert sizes == {main: expected_sizes[0], indexer: expected_sizes[1]}
    layout, size = placement.build_kvpp_layer_layout((main, indexer), sizes, 3)
    assert layout == {main: expected_layout[0], indexer: expected_layout[1]}
    assert size == total


def test_unquantized_indexer_and_quantized_mla_sizes(monkeypatch):
    main, indexer = layer_name(11), indexer_name(11)
    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.kv_lora_rank, layer.qk_rope_head_dim = 8, 4
    config = make_kvpp_config()
    config.quant_config = Mock()
    config.quant_config.get_kv_quant_split_factor.return_value = (2, 2)
    specs = {
        main: AscendMLAAttentionSpec(block_size=2, num_kv_heads=1, head_size=16, dtype=torch.int8),
        indexer: replace(make_kvpp_specs()[indexer], dtype=torch.bfloat16, scale_dim=0, cache_sparse_li_c8=False),
    }
    monkeypatch.setattr(placement, "get_layers_from_vllm_config", lambda *_args: {main: layer})
    monkeypatch.setattr(placement, "enable_sfa", lambda _: False)
    monkeypatch.setattr(placement, "enable_fa_quant", lambda _: True)
    assert placement.build_kvpp_buffer_sizes(config, specs) == {main: (16, 16), indexer: (16,)}
    config.quant_config.get_kv_quant_split_factor.assert_called_once_with(main, [8, 4])


@pytest.mark.parametrize("tp,rank,cost", [(3, 0, 404), (3, 1, 392), (3, 2, 328), (10, 9, 248)])
def test_physical_cost_per_rank(tp, rank, cost):
    plan = placement.create_kvpp_cache_allocation_plan(make_kvpp_config(tp), make_kvpp_specs(), rank)
    assert plan.get_num_blocks(cost - 1) == 0
    assert plan.get_num_blocks(cost) == 1


@pytest.mark.parametrize("available,blocks", [(0, 0), (391, 0), (392, 1), (1175, 2), (1176, 3)])
def test_budget_floors_complete_blocks(available, blocks):
    plan = placement.create_kvpp_cache_allocation_plan(make_kvpp_config(), make_kvpp_specs(), 1)
    assert plan.get_num_blocks(available) == blocks


@pytest.mark.parametrize("with_mtp,expected", [(True, 3), (False, 0)])
def test_stage_without_target_has_no_scratch_cost(with_mtp, expected):
    specs = make_kvpp_specs()
    specs = {layer_name(17): specs[layer_name(17)]} if with_mtp else {}
    plan = placement.create_kvpp_cache_allocation_plan(make_kvpp_config(), specs, 1)
    assert plan.get_num_blocks(288) == expected


@pytest.mark.parametrize("rank,target_cost", [(0, 308), (1, 296), (2, 232)])
def test_dspark_keeps_draft_caches_in_every_rank_budget(rank, target_cost):
    config, specs, drafts = make_dspark_kvpp_case()
    target_specs = {name: spec for name, spec in specs.items() if name not in drafts}
    target_plan = placement.create_kvpp_cache_allocation_plan(config, target_specs, rank)
    plan = placement.create_kvpp_cache_allocation_plan(config, specs, rank)
    assert plan.layer_owner_ranks == target_plan.layer_owner_ranks
    for name in drafts:
        assert plan.layer_bundles[name] == (name,)
        assert plan.tensor_sizes[name] == (64, 64)
    cost = target_cost + 3 * 128
    assert plan.get_num_blocks(cost - 1) == 0
    assert plan.get_num_blocks(3 * cost) == 3
    reverse = placement.create_kvpp_cache_allocation_plan(config, dict(reversed(list(specs.items()))), rank)
    assert list(plan.layer_owner_ranks.items()) == list(reverse.layer_owner_ranks.items())
    assert list(plan.layer_bundles.items()) == list(reverse.layer_bundles.items())


def test_dspark_draft_only_stage_has_no_scratch_cost():
    config, specs, drafts = make_dspark_kvpp_case()
    plan = placement.create_kvpp_cache_allocation_plan(config, {name: specs[name] for name in drafts}, 1)
    assert plan.layer_owner_ranks == {}
    assert plan.get_num_blocks(3 * 384) == 3


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("draft_block_size", [1, 2])
def test_dspark_sliding_window_draft_uses_full_allocation_budget(rank, draft_block_size):
    config, specs, drafts = make_dspark_kvpp_case()
    full_plan = placement.create_kvpp_cache_allocation_plan(config, specs, rank)
    for name in drafts:
        specs[name] = SlidingWindowSpec(
            block_size=draft_block_size, num_kv_heads=2, head_size=8, dtype=torch.float16, sliding_window=4
        )
    plan = placement.create_kvpp_cache_allocation_plan(config, specs, rank)
    assert plan.layer_owner_ranks == full_plan.layer_owner_ranks
    assert plan.tensor_sizes == full_plan.tensor_sizes
    assert plan.get_num_blocks(8192) == full_plan.get_num_blocks(8192)
    for name in drafts:
        assert isinstance(specs[name], SlidingWindowSpec)
        assert specs[name].sliding_window == 4
        assert specs[name].block_size == draft_block_size
        assert isinstance(plan.logical_cache_spec[name], FullAttentionSpec)
        assert plan.logical_cache_spec[name].block_size == 2
        assert name not in plan.layer_owner_ranks


def test_mtp_still_rejects_sliding_window_cache_specs():
    config, specs, drafts = make_dspark_kvpp_case()
    config.speculative_config.method = "mtp"
    specs[drafts[0]] = SlidingWindowSpec(
        block_size=2, num_kv_heads=2, head_size=8, dtype=torch.float16, sliding_window=4
    )
    with pytest.raises(ValueError, match="KVPP requires one full-attention cache group"):
        placement.create_kvpp_cache_allocation_plan(config, specs, 0)


def test_dspark_draft_range_uses_local_layers_and_draft_model_depth():
    config, specs, drafts = make_dspark_kvpp_case()
    del config.model_config.hf_config.num_nextn_predict_layers
    assert placement.find_draft_layers(config, specs) == set(drafts)
    assert placement.find_draft_layers(config, [layer_name(16), drafts[-1], layer_name(20)]) == {drafts[-1]}
