# SPDX-License-Identifier: Apache-2.0
"""Allocation descriptors and ownership for uncompressed paired SFA caches."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.core.paired_sfa_cache import (
    is_paired_sfa_cache,
    paired_sfa_cache_config,
    paired_sfa_cache_groups,
    paired_sfa_pool_bytes,
)


def make_specs():
    specs = {}
    sparse = [*range(3, 92, 4), 92]
    for layer in range(93):
        name = f"model.layers.{layer}"
        if layer in sparse:
            specs[name + ".attn"] = AscendMLAAttentionSpec(
                block_size=1536,
                num_kv_heads=1,
                head_size=656,
                dtype=torch.int8,
                cache_sparse_sfa_c8=True,
            )
            specs[name + ".indexer.k_cache"] = AscendSFAIndexerCacheSpec(
                block_size=1536,
                num_kv_heads=1,
                head_size=128,
                dtype=torch.int8,
                scale_dim=1,
                scale_dtype=torch.float32,
                cache_sparse_li_c8=True,
            )
        else:
            specs[name + ".linear_attn"] = MambaSpec(
                block_size=1536,
                shapes=((3, 4608), (12, 128, 128)),
                dtypes=(torch.bfloat16, torch.float32),
                mamba_cache_mode="align",
            )
    return specs


def make_config():
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None, prefix_cache_retention_interval=None),
    )


def test_token_blocks_unchanged_and_components_share_ids():
    specs, config = make_specs(), make_config()
    groups = paired_sfa_cache_groups(config, specs)
    assert len(groups) == 4
    assert [len(g.layer_names) for g in groups] == [48, 23, 23, 23]
    assert {s.block_size for s in specs.values()} == {1536}
    assert {s.page_size_bytes for s in specs.values()} == {1007616, 202752}
    assert paired_sfa_pool_bytes(groups) == 29048832
    plan = paired_sfa_cache_config(config, groups, 17556701184)
    assert plan.num_blocks == 604
    assert sum(t.size for t in plan.kv_cache_tensors) <= 17556701184
    assert len(plan.kv_cache_tensors) == 48
    assert len(groups[0].kv_cache_spec.kv_cache_specs) == 48
    assert all(t.offset == t.layer_stride == 0 for t in plan.kv_cache_tensors)


@pytest.mark.parametrize("interval", [None, 5])
def test_paired_config_propagates_prefix_cache_retention_interval(interval):
    config = make_config()
    config.cache_config.prefix_cache_retention_interval = interval
    groups = paired_sfa_cache_groups(config, make_specs())
    plan = paired_sfa_cache_config(config, groups, 17556701184)
    assert plan.prefix_cache_retention_interval == interval


def test_pipeline_projection_retains_only_local_slot_cost():
    groups = paired_sfa_cache_groups(make_config(), make_specs())
    projected = [
        KVCacheGroupSpec([n for n in g.layer_names if int(n.split(".")[2]) < 24], g.kv_cache_spec) for g in groups
    ]
    assert paired_sfa_pool_bytes(projected) == 6 * (1007616 + 202752)


def test_mismatched_component_blocks_rejected():
    specs = make_specs()
    name = next(n for n in specs if n.endswith(".k_cache"))
    specs[name] = replace(specs[name], block_size=3072)
    with pytest.raises(ValueError, match="same local token block"):
        paired_sfa_cache_groups(make_config(), specs)


def test_unrelated_pure_dsa_layout_keeps_existing_planner():
    specs = {n: s for n, s in make_specs().items() if not isinstance(s, MambaSpec)}
    assert not is_paired_sfa_cache(specs)


def test_alignment_is_idempotent_and_discards_obsolete_state_padding():
    specs = {
        n: replace(s, page_size_padded=66502656) if isinstance(s, MambaSpec) else s for n, s in make_specs().items()
    }
    first = paired_sfa_cache_groups(make_config(), specs)
    second = paired_sfa_cache_groups(make_config(), specs)
    assert paired_sfa_pool_bytes(first) == paired_sfa_pool_bytes(second) == 29048832


def test_a3_int8_fp16_pages_preserve_tp16_recurrent_ownership():
    specs = {}
    for name, spec in make_specs().items():
        if isinstance(spec, MambaSpec):
            spec = replace(spec, shapes=((3, 2304), (6, 128, 128)))
        elif isinstance(spec, AscendSFAIndexerCacheSpec):
            spec = replace(spec, scale_dtype=torch.float16)
        specs[name] = replace(spec, block_size=768)
    config = make_config()
    groups = paired_sfa_cache_groups(config, specs)
    # 24 packed-MLA pages (768*656 bytes) plus 24 INT8/FP16 indexer
    # pages (768*(128+2) bytes); recurrent checkpoints alias main slots.
    assert {spec.page_size_bytes for spec in specs.values()} == {503808, 99840}
    assert {spec.block_size for spec in specs.values()} == {768}
    assert paired_sfa_pool_bytes(groups) == 14487552
    plan = paired_sfa_cache_config(config, groups, 115900539)
    assert plan.num_blocks == 8
    assert sum(tensor.size for tensor in plan.kv_cache_tensors) == 115900416
    assert all(
        spec.sfa_dcp_replicated_indexer_size == 1
        for spec in specs.values()
        if isinstance(spec, AscendSFAIndexerCacheSpec)
    )
