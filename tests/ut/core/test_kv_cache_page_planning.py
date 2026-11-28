# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Page planning aligns shared Attention/Mamba pages without mutating owner specs."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.core.kv_cache_planning import (
    _align_hybrid_attention_page_sizes,
    align_hybrid_kv_cache_specs,
)


def _attention():
    return FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=4, dtype=torch.float16)


def _state(elements):
    return MambaSpec(block_size=4, shapes=((elements,),), dtypes=(torch.float32,))


@pytest.mark.parametrize("include_attention", [False, True])
def test_shared_pages_keep_existing_padding_and_owner_specs(include_attention):
    state = _state(32)
    specs = {"large_state": state, "small_state": _state(4)}
    if include_attention:
        specs["attention"] = _attention()
    original = specs.copy()
    planned = align_hybrid_kv_cache_specs(specs)

    assert all(spec.page_size_bytes == 128 for spec in planned.values())
    assert specs == original
    assert specs["small_state"].page_size_padded is None
    assert planned["large_state"] is state
    assert align_hybrid_kv_cache_specs(planned) == planned


@pytest.mark.parametrize("specs", [{}, {"attention": _attention()}])
def test_nonhybrid_specs_are_unchanged(specs):
    assert align_hybrid_kv_cache_specs(specs) is specs


def test_contiguous_kv_segments_grow_before_page_padding():
    full = _attention()  # 64 bytes: 32 K + 32 V
    sliding = SlidingWindowSpec(
        block_size=4, num_kv_heads=1, head_size=2, dtype=torch.float16, sliding_window=16
    )  # 32 bytes: 16 K + 16 V
    state = replace(_state(16), page_size_padded=80)
    planned = align_hybrid_kv_cache_specs({"full": full, "sliding": sliding, "state": state})

    assert planned["sliding"].block_size == 8
    assert planned["sliding"].real_page_size_bytes == full.real_page_size_bytes
    assert planned["sliding"].sliding_window == 16
    assert all(spec.page_size_bytes == 80 for spec in planned.values())
    assert sliding.block_size == 4 and sliding.page_size_padded is None


def test_descriptor_layout_does_not_resize_scheduler_blocks_or_state_pages():
    full = _attention()
    state = _state(4)
    planned = align_hybrid_kv_cache_specs({"full": full, "state": state}, descriptor_native=True)
    assert planned == {"full": full, "state": state}

    larger_state = _state(32)
    planned = align_hybrid_kv_cache_specs({"full": full, "state": larger_state}, descriptor_native=True)
    assert planned["full"].block_size == full.block_size
    assert planned["full"].page_size_bytes == 128
    assert planned["state"] is larger_state


@pytest.mark.parametrize("head_size,head_size_v", [(3, 3), (2, 6)])
def test_planning_rejects_incompatible_contiguous_kv_pages(head_size, head_size_v):
    sliding = SlidingWindowSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=head_size,
        head_size_v=head_size_v,
        dtype=torch.float16,
        sliding_window=16,
    )
    specs = {"full": _attention(), "sliding": sliding, "state": _state(32)}
    with pytest.raises(ValueError, match="Cannot align hybrid attention K/V pages"):
        align_hybrid_kv_cache_specs(specs)


def test_planning_pads_mla_without_resizing_token_blocks():
    mla = AscendMLAAttentionSpec(block_size=4, num_kv_heads=1, head_size=3, dtype=torch.float16)
    planned = align_hybrid_kv_cache_specs({"full": _attention(), "mla": mla, "state": _state(32)})
    assert planned["mla"].block_size == mla.block_size
    assert planned["mla"].head_size == mla.head_size
    assert planned["mla"].page_size_bytes == 128
    assert mla.page_size_padded is None


def test_paired_sfa_indexer_skips_generic_hybrid_alignment():
    indexer = AscendSFAIndexerCacheSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
        cache_sparse_li_c8=True,
    )
    specs = {"full": _attention(), "state": _state(32), "indexer": indexer}
    planned = align_hybrid_kv_cache_specs(specs)
    assert planned is specs
    assert indexer.page_size_bytes == 40
    assert specs["full"].page_size_padded is None


def test_sparse_summary_cache_is_excluded_from_dense_alignment():
    full = FullAttentionSpec(block_size=128, num_kv_heads=2, head_size=64, dtype=torch.bfloat16)
    summary = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=15, dtype=torch.float8_e4m3fn)
    specs = {"model.layers.0.self_attn": full, "model.layers.0.self_attn.sparse_summary_cache": summary}
    _align_hybrid_attention_page_sizes(specs)
    assert specs["model.layers.0.self_attn.sparse_summary_cache"] is summary


def test_builder_aligns_aggregated_pp_stage_specs_before_default_grouping(monkeypatch):
    planning = pytest.importorskip("vllm.v1.core.kv_cache_planning")
    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder

    # Simulate PP stage 0 (attention-only) + PP stage 1 (mamba-only) merged by Core:
    stage0_specs = {"model.layers.0.self_attn": _attention()}
    stage1_specs = {"model.layers.1.linear_attn": _state(32)}
    merged_specs = {**stage0_specs, **stage1_specs}

    upstream = Mock(return_value=[])
    monkeypatch.setattr(planning.DefaultKVCacheConfigBuilder, "get_kv_cache_groups", upstream)
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_next")),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        attention_config=SimpleNamespace(hisparse_config=None),
    )
    assert AscendKVCacheConfigBuilder().get_kv_cache_groups(config, merged_specs) == []

    planned = upstream.call_args.args[1]
    assert planned["model.layers.0.self_attn"].page_size_bytes == 128
    assert planned["model.layers.1.linear_attn"].page_size_bytes == 128
    assert merged_specs["model.layers.0.self_attn"].page_size_padded is None


def _planning_vllm_config(*, pp_size: int = 1, dcp_size: int = 1, kv_cache_layout: str | None = None):
    from vllm.config import CacheConfig

    cache_config = CacheConfig(mamba_cache_mode="none")
    cache_config.kv_cache_layout = kv_cache_layout
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_next"),
            supports_kv_cache_grouping=False,
            original_max_model_len=128,
            max_model_len=128,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp_size,
            tensor_parallel_size=1,
            decode_context_parallel_size=dcp_size,
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        cache_config=cache_config,
        kv_transfer_config=None,
        speculative_config=None,
        max_in_flight_tokens=16,
        num_prefill_lookahead_tokens=0,
    )


def test_pp_separate_stages_allow_distinct_kv_ratios_while_shared_backing_rejects(monkeypatch):
    planning = pytest.importorskip("vllm.v1.core.kv_cache_planning")
    builder_mod = pytest.importorskip("vllm.v1.core.kv_cache_config_builder")
    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder

    monkeypatch.setattr(builder_mod, "get_kv_cache_config_builder", lambda _cfg: AscendKVCacheConfigBuilder())

    # Both attention specs and Mamba state have 32-byte pages, with 2/2 vs 1/3 K/V widths.
    full_2_2 = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=2, head_size_v=2, dtype=torch.float16)
    sliding_1_3 = SlidingWindowSpec(
        block_size=4, num_kv_heads=1, head_size=1, head_size_v=3, dtype=torch.float16, sliding_window=16
    )
    mamba_32b = MambaSpec(block_size=4, shapes=((8,),), dtypes=(torch.float32,))
    assert full_2_2.page_size_bytes == sliding_1_3.page_size_bytes == mamba_32b.page_size_bytes == 32

    # Separate PP=2 stages never share a physical backing allocation -> accepted.
    pp2_config = _planning_vllm_config(pp_size=2, kv_cache_layout="LBNHC")
    separate_stage_specs = [
        {"model.layers.0.self_attn": full_2_2, "model.layers.1.linear_attn": mamba_32b},
        {"model.layers.2.self_attn": sliding_1_3, "model.layers.3.linear_attn": mamba_32b},
    ]
    global_groups = planning.get_global_kv_cache_groups(pp2_config, separate_stage_specs)
    configs = planning.get_kv_cache_configs(pp2_config, separate_stage_specs, [4096, 4096], global_groups)
    assert len(configs) == 2
    assert configs[0].num_blocks == configs[1].num_blocks > 0

    # Same stage sharing a physical backing -> rejected under both PP=2 and PP=1.
    for dcp_sharded in (False, True):
        shared_stage_specs = [
            {
                "model.layers.0.self_attn": replace(full_2_2, dcp_sharded=dcp_sharded),
                "model.layers.1.self_attn": replace(sliding_1_3, dcp_sharded=dcp_sharded),
                "model.layers.2.linear_attn": mamba_32b,
            },
            {"model.layers.3.linear_attn": mamba_32b},
        ]
        stage_pp2_cfg = _planning_vllm_config(pp_size=2, dcp_size=1, kv_cache_layout="LBNHC")
        shared_groups = planning.get_global_kv_cache_groups(stage_pp2_cfg, shared_stage_specs)
        with pytest.raises(ValueError, match="matching K/V size ratios"):
            planning.get_kv_cache_configs(stage_pp2_cfg, shared_stage_specs, [4096, 4096], shared_groups)

        stage_pp1_cfg = _planning_vllm_config(pp_size=1, dcp_size=1, kv_cache_layout="LBNHC")
        with pytest.raises(ValueError, match="matching K/V size ratios"):
            planning.get_global_kv_cache_groups(stage_pp1_cfg, [shared_stage_specs[0]])

    # DCP-sharded target + replicated draft live in disjoint pools and never share a backing -> accepted.
    disjoint_dcp_specs = [
        {
            "target.attn": replace(full_2_2, dcp_sharded=True),
            "draft.attn": FullAttentionSpec(
                block_size=4,
                num_kv_heads=1,
                head_size=1,
                head_size_v=3,
                dtype=torch.float16,
                dcp_sharded=False,
                is_draft=True,
            ),
            "target.mamba": mamba_32b,
        }
    ]
    dcp_cfg = _planning_vllm_config(pp_size=1, dcp_size=2, kv_cache_layout="LBNHC")
    dcp_groups = planning.get_global_kv_cache_groups(dcp_cfg, disjoint_dcp_specs)
    dcp_configs = planning.get_kv_cache_configs(dcp_cfg, disjoint_dcp_specs, [4096], dcp_groups)
    assert dcp_configs[0].num_blocks > 0


@pytest.mark.parametrize(
    ("requested_layout", "expected_layout"),
    [(None, "LBNHC"), ("LBNHC", "LBNHC"), ("BLNHC", "BLNHC")],
)
def test_layout_resolution_and_grouping_agree_on_effective_pages_without_mutating_owners(
    monkeypatch, requested_layout, expected_layout
):
    import vllm.envs as envs
    import vllm.v1.engine.core as engine_core

    planning = pytest.importorskip("vllm.v1.core.kv_cache_planning")
    builder_mod = pytest.importorskip("vllm.v1.core.kv_cache_config_builder")
    import vllm_ascend.patch.platform.patch_kv_cache_layout  # noqa: F401
    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder

    monkeypatch.setattr(builder_mod, "get_kv_cache_config_builder", lambda _cfg: AscendKVCacheConfigBuilder())
    monkeypatch.setattr(envs, "VLLM_KV_CACHE_LAYOUT", requested_layout)

    target = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float16, dcp_sharded=True)
    draft = FullAttentionSpec(
        block_size=4, num_kv_heads=2, head_size=1, dtype=torch.float16, dcp_sharded=False, is_draft=True
    )
    mamba = MambaSpec(block_size=4, shapes=((8,),), dtypes=(torch.float32,))
    worker_specs = [{"target.attn": target, "draft.attn": draft, "target.mamba": mamba}]
    vllm_config = _planning_vllm_config(pp_size=1, dcp_size=2, kv_cache_layout=None)

    # Follow EngineCore.__init__ call order: resolve_kv_cache_layout -> grouping -> per-worker configs.
    resolved = engine_core.resolve_kv_cache_layout(
        vllm_config,
        [["LBNHC", "BLNHC"]],
        [spec for specs in worker_specs for spec in specs.values()],
    )
    assert resolved.name == expected_layout
    assert vllm_config.cache_config.kv_cache_layout == expected_layout

    # Owner specs remain completely untouched.
    assert target.block_size == 4 and target.page_size_padded is None and target.page_size_bytes == 16
    assert draft.block_size == 4 and draft.page_size_padded is None and draft.page_size_bytes == 32
    assert mamba.page_size_padded is None and mamba.page_size_bytes == 32

    global_groups = planning.get_global_kv_cache_groups(vllm_config, worker_specs)
    assert {group.kv_cache_spec.page_size_bytes for group in global_groups} == {32}
    configs = planning.get_kv_cache_configs(vllm_config, worker_specs, [4096], global_groups)
    assert configs[0].num_blocks > 0
