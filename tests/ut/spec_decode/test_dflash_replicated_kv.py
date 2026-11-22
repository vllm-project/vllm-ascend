# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch
from vllm.config import CacheConfig
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    KVQuantMode,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.dflash_kv_cache import apply_replicated_dflash_specs
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendReplicatedDraftCacheSpec,
    AscendSFAIndexerCacheSpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.core import kv_cache_config_builder as planner
from vllm_ascend.platform import _validate_draft_decode_context_parallel_config
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.spec_decode.dflash.kv_cache import ReplicatedDFlashBlockTables, draft_block_tables


@pytest.fixture
def case():
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    cache_config = CacheConfig()
    cache_config.kv_cache_layout = "LBHNC"
    config = NS(
        cache_config=cache_config,
        speculative_config=NS(method="dflash"),
        parallel_config=NS(decode_context_parallel_size=4, prefill_context_parallel_size=1),
        model_config=NS(
            hf_config=NS(model_type="glm_moe_dsa"), max_model_len=2048, supports_kv_cache_grouping=False
        ),
        scheduler_config=NS(disable_hybrid_kv_cache_manager=False),
        attention_config=NS(hisparse_config=None),
        kv_transfer_config=None,
    )
    specs = {
        "mla": AscendMLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16),
        "indexer": AscendSFAIndexerCacheSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.int8),
        "draft": FullAttentionSpec(
            block_size=128, num_kv_heads=2, head_size=128, dtype=torch.bfloat16, dcp_sharded=False, is_draft=True
        ),
    }
    return config, specs, NS(dcp_size=1, draft_attn_layer_names={"draft"})


def test_glm_target_and_replicated_draft_form_one_group(case):
    config, original, speculator = case
    assert UniformTypeKVCacheSpecs.from_specs(original) is None
    specs = apply_replicated_dflash_specs(original, config, speculator)
    draft = specs["draft"]
    assert isinstance(draft, AscendReplicatedDraftCacheSpec)
    assert draft.block_size == original["draft"].block_size
    assert draft.page_size_bytes == 4 * original["draft"].page_size_bytes
    assert original["draft"].dcp_sharded is False
    assert speculator.dcp_size == 1
    groups = planner.AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs)
    assert len(groups) == 1
    grouped = groups[0].kv_cache_spec
    assert isinstance(grouped, UniformTypeKVCacheSpecs)
    assert grouped.max_num_blocks_per_req(config, 513) == 2
    assert grouped.page_size_bytes == sum(spec.page_size_bytes for spec in specs.values())
    budget = grouped.page_size_bytes * 16
    builder = planner.AscendKVCacheConfigBuilder()
    cache = builder.get_kv_cache_config_from_groups(config, groups, num_blocks=16)
    assert cache.num_blocks == 16
    # Each descriptor names a view into the same backing, not a new allocation.
    assert {tensor.size for tensor in cache.kv_cache_tensors} == {budget}
    ranges = sorted(
        (tensor.offset, tensor.offset + 16 * specs[tensor.layers[0]].page_size_bytes)
        for tensor in cache.kv_cache_tensors
    )
    assert ranges[0][0] == 0 and ranges[-1][1] == budget
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))


def test_dense_target_does_not_merge_away_draft_replication(case):
    config, specs, speculator = case
    specs = {"target": replace(specs["draft"], is_draft=False, dcp_sharded=True), "draft": specs["draft"]}
    specs = apply_replicated_dflash_specs(specs, config, speculator)
    groups = planner.AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs)
    assert len(groups) == 1
    grouped = groups[0].kv_cache_spec
    assert isinstance(grouped, UniformTypeKVCacheSpecs)
    assert grouped.kv_cache_specs["draft"].replication_size == 4
    assert grouped.page_size_bytes == 5 * specs["target"].page_size_bytes


@pytest.mark.parametrize(
    "fault", ["dcp1", "draft_sharded", "pcp", "other_method", "block_layout", "next", "padding", "quantized"]
)
def test_replication_is_scoped_to_compatible_drafts(case, fault):
    config, specs, speculator = case
    if fault == "dcp1":
        config.parallel_config.decode_context_parallel_size = 1
    elif fault == "draft_sharded":
        speculator.dcp_size = 4
    elif fault == "pcp":
        config.parallel_config.prefill_context_parallel_size = 2
    elif fault == "other_method":
        config.speculative_config.method = "eagle3"
    elif fault == "block_layout":
        config.cache_config.kv_cache_layout = "BLHNC"
    elif fault == "next":
        specs["mla"] = replace(specs["mla"], model_version="glm5_next")
    elif fault == "padding":
        specs["draft"] = replace(specs["draft"], page_size_padded=specs["draft"].page_size_bytes)
    else:
        specs["draft"] = replace(specs["draft"], kv_quant_mode=KVQuantMode.FP8_PER_TENSOR)
    assert apply_replicated_dflash_specs(specs, config, speculator) == specs


def test_tp_only_dflash_bypasses_draft_dcp_head_constraint(case):
    config, _, _ = case
    config.use_v2_model_runner = True
    config.speculative_config.use_dspark = lambda: False
    config.speculative_config.draft_model_config = NS(use_mla=False)
    config.speculative_config.num_speculative_tokens_per_batch_size = None
    _validate_draft_decode_context_parallel_config(config)
    config.speculative_config.num_speculative_tokens_per_batch_size = [[1, 8, 4]]
    with pytest.raises(ValueError, match="Dynamic speculative"):
        _validate_draft_decode_context_parallel_config(config)


@pytest.mark.parametrize("factor", [2, 4, 8, 16])
@pytest.mark.parametrize("block_size", [128, 256, 384])
def test_expanded_table_addresses_and_reuse(factor, block_size):
    kernel_size = 128
    splits = block_size // kernel_size
    ids = torch.tensor([[7, 2, 0], [5, 11, 0]], dtype=torch.int32)
    local = (ids.unsqueeze(-1) * splits + torch.arange(splits)).flatten(1)
    source = NS(
        input_block_tables=[local],
        slot_mappings=torch.full((1, 32), -1, dtype=torch.int32),
        block_sizes=[block_size],
        kernel_block_sizes=[kernel_size],
    )
    before = local.clone()
    tables = ReplicatedDFlashBlockTables(source, {0: factor})
    address = tables.input_block_tables[0].data_ptr()
    for manager_ids in (ids, ids.flip(0)):
        source.input_block_tables[0].copy_((manager_ids.unsqueeze(-1) * splits + torch.arange(splits)).flatten(1))
        tables.refresh()
        assert tables.input_block_tables[0].data_ptr() == address
        positions = torch.arange(2 * block_size * factor)
        actual = tables.input_block_tables[0][:, positions // kernel_size] * kernel_size + positions % kernel_size
        expected = manager_ids[:, positions // (block_size * factor)] * block_size * factor + positions % (
            block_size * factor
        )
        torch.testing.assert_close(actual, expected, check_dtype=False)
        assert torch.count_nonzero(tables.input_block_tables[0][:, 2 * splits * factor :]) == 0
    tables.slot_mappings.fill_(42)
    assert torch.all(source.slot_mappings == -1)
    torch.testing.assert_close(local, before.flip(0))


def test_block_table_adapter_only_replaces_replicated_draft_groups(case):
    config, specs, speculator = case
    source = NS(
        input_block_tables=[torch.tensor([[7, 2]], dtype=torch.int32)],
        slot_mappings=torch.zeros((1, 4), dtype=torch.int32),
        block_sizes=[128],
        kernel_block_sizes=[128],
    )
    plain = NS(kv_cache_groups=[KVCacheGroupSpec(["draft"], specs["draft"])])
    assert draft_block_tables(source, plain, {"draft"}) is source
    replicated = apply_replicated_dflash_specs(specs, config, speculator)
    cache = NS(kv_cache_groups=[KVCacheGroupSpec(list(specs), UniformTypeKVCacheSpecs.from_specs(replicated))])
    assert isinstance(draft_block_tables(source, cache, {"draft"}), ReplicatedDFlashBlockTables)


def test_different_block_sizes_keep_independent_lb_pools(case):
    config, specs, speculator = case
    specs["draft"] = replace(specs["draft"], block_size=256)
    specs = apply_replicated_dflash_specs(specs, config, speculator)
    groups = planner.AscendKVCacheConfigBuilder().get_kv_cache_groups(config, specs)
    assert len(groups) == 2
    assert specs["draft"].block_size == 256
    page_bytes = sum(group.kv_cache_spec.page_size_bytes for group in groups)
    cache = planner.get_ascend_kv_cache_config_from_groups(config, groups, page_bytes * 16)
    assert cache.num_blocks == 16
    assert planner.AscendKVCacheConfigBuilder()._get_pool_bytes_per_block(config, groups) == page_bytes


@pytest.mark.parametrize("hnd", [False, True])
def test_real_allocator_and_reshape_cover_all_replicated_pages(case, monkeypatch, hnd):
    config, specs, speculator = case
    spec = apply_replicated_dflash_specs(specs, config, speculator)["draft"]
    config.additional_config = {}
    config.quant_config = None
    config.compilation_config = NS(static_forward_context={})
    num_blocks = 3
    size = num_blocks * spec.page_size_bytes
    cache = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_groups=[KVCacheGroupSpec(["draft"], spec)],
        kv_cache_tensors=[
            KVCacheTensor(size=size, layers=["draft"], layer_stride=size, block_stride=spec.page_size_bytes)
        ],
    )
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda *args: True)
    monkeypatch.setattr(attn_utils, "enable_fa_quant", lambda *args: False)
    monkeypatch.setattr(attn_utils, "_is_dsv4_model", lambda *args: False)
    raw = attn_utils._allocate_kv_cache(cache, {}, torch.device("cpu"))
    assert sum(tensor.numel() for tensor in raw["draft"]) == size
    backend = NS(
        get_kv_cache_shape=lambda blocks, tokens, heads, dim, *_: (
            (2, blocks, heads, tokens, dim) if hnd else (2, blocks, tokens, heads, dim)
        )
    )
    group = NS(kv_cache_group_id=0, kv_cache_spec=spec, backend=backend, layer_names=["draft"])
    key, value = attn_utils._reshape_kv_cache_v2([group], raw, "auto", [128], {}, cache)["draft"]
    expected_shape = (12, 2, 128, 128) if hnd else (12, 128, 2, 128)
    assert key.shape == value.shape == expected_shape
    assert key.numel() * key.element_size() + value.numel() * value.element_size() == size
    key.zero_()
    value.zero_()
    key[4:8].fill_(3)
    value[4:8].fill_(5)
    assert torch.all(key[4:8] == 3) and torch.all(value[4:8] == 5)
    assert not torch.any(key[:4]) and not torch.any(key[8:])
