# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.models.kimi_k3 import packed_sfa_cache
from vllm_ascend.models.kimi_k3.packed_sfa_cache import (
    map_packed_sfa_metadata,
    packed_sfa_cache_config,
    packed_sfa_cache_groups,
    packed_sfa_cache_views,
    packed_sfa_row_bytes,
    uses_block_outermost_sfa,
)


def make_packed_specs(block_size=10752, draft_layers=5, speculative_steps=3, state_heads=96):
    specs = {}
    for layer in range(24):
        specs[f"model.layers.{layer}.attn"] = AscendMLAAttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=656, dtype=torch.int8, cache_sparse_sfa_c8=True
        )
        specs[f"model.layers.{layer}.indexer.k_cache"] = AscendSFAIndexerCacheSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.int8,
            scale_dim=1,
            scale_dtype=torch.float32,
            cache_sparse_li_c8=True,
        )
    for layer in range(24, 93):
        specs[f"model.layers.{layer}.linear_attn"] = MambaSpec(
            block_size=block_size,
            shapes=((3 + speculative_steps, state_heads * 384), (state_heads, 128, 128)),
            dtypes=(torch.bfloat16, torch.float32),
            mamba_cache_mode="align",
            num_speculative_blocks=speculative_steps,
        )
    for layer in range(93, 93 + draft_layers):
        specs[f"model.layers.{layer}.attn"] = AscendMLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            is_draft=True,
            non_causal_multi_token_decode=True,
        )
    return specs


def packed_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="kimi_k31"), max_model_len=1048576),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False, max_num_seqs=4, max_num_batched_tokens=256
        ),
        cache_config=SimpleNamespace(prefix_cache_retention_interval=None, kv_cache_layout=None),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[]),
        speculative_config=None,
    )


@pytest.mark.parametrize(
    "resolved,override,expected",
    [(None, None, False), (None, "BLNHC", True), ("LBNHC", "BLNHC", False), ("BLHNC", None, True)],
)
def test_layout_selection_before_engine_resolution(monkeypatch, resolved, override, expected):
    config = packed_config()
    config.cache_config.kv_cache_layout = resolved
    monkeypatch.setattr(packed_sfa_cache.envs_vllm, "VLLM_KV_CACHE_LAYOUT", override)
    assert uses_block_outermost_sfa(config) == expected


@pytest.mark.parametrize("draft_layers", [0, 3, 5])
def test_row_is_derived_from_actual_draft_specs(draft_layers):
    specs = make_packed_specs(draft_layers=draft_layers)
    groups = packed_sfa_cache_groups(packed_config(), specs)
    row = packed_sfa_row_bytes(groups)
    payload = 10752 * (24 * (656 + 132) + draft_layers * 1152)
    alignment = 66502656 if draft_layers else 2770944
    assert row == -(-payload // alignment) * alignment
    assert len(groups[0].layer_names) == 48 + draft_layers
    assert sum(len(g.layer_names) for g in groups[1:]) == 69
    assert all(g.kv_cache_spec.page_size_padded is None for g in groups[1:])
    assert all(len(g.layer_names) * g.kv_cache_spec.page_size_bytes <= row for g in groups[1:])
    plan = packed_sfa_cache_config(packed_config(), groups, row * 2, lambda _, n: n)
    assert plan.num_blocks == 2
    assert {t.size for t in plan.kv_cache_tensors} == {row * 2}
    assert {t.block_stride for t in plan.kv_cache_tensors} == {row}
    assert (
        sum(t.layer_stride for t in plan.kv_cache_tensors if not isinstance(specs[t.layers[0]], MambaSpec)) == payload
    )


def test_a5_capacity_regression_with_five_draft_layers():
    groups = packed_sfa_cache_groups(packed_config(), make_packed_specs())
    assert [len(g.layer_names) for g in groups[1:]] == [35, 34]
    assert packed_sfa_row_bytes(groups) == 266010624
    assert (98 + 10 + 1) * packed_sfa_row_bytes(groups) == 28995158016


def test_no_draft_and_no_speculation():
    groups = packed_sfa_cache_groups(packed_config(), make_packed_specs(8192, 0, 0))
    assert [len(g.layer_names) for g in groups[1:]] == [23, 23, 23]
    assert packed_sfa_row_bytes(groups) == 155172864
    assert (128 + 6 + 1) * packed_sfa_row_bytes(groups) == 20948336640


def test_three_layer_mla_draft_capacity():
    groups = packed_sfa_cache_groups(packed_config(), make_packed_specs(11776, 3, 3))
    assert [len(g.layer_names) for g in groups[1:]] == [35, 34]
    assert packed_sfa_row_bytes(groups) == 266010624
    assert (90 + 10 + 1) * packed_sfa_row_bytes(groups) == 26867073024


def test_pattern_floor_uses_actual_counts_and_upstream_repeat_selection():
    specs = {
        name: spec
        for name, spec in make_packed_specs(65536, 0).items()
        if isinstance(spec, MambaSpec) or name.startswith(("model.layers.0.", "model.layers.1."))
    }
    groups = packed_sfa_cache_groups(packed_config(), specs)
    # Upstream's approximate GCD selects three repeats for counts [2, 69]:
    # ties in total padding prefer the larger repeat size.
    assert len(groups[1:]) == 23
    assert {len(group.layer_names) for group in groups[1:]} == {3}


@dataclass
class PackedCommonMetadata:
    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor


@pytest.mark.parametrize("draft_layers", [0, 3, 5])
def test_views_and_metadata_address_the_same_bytes(draft_layers):
    # Small real pages permit exhaustive writes without allocating A5-sized
    # pools. Keep different dtypes and a manager block larger than the kernel.
    config = packed_config()
    specs = make_packed_specs(256, draft_layers, 0, state_heads=1)
    groups = packed_sfa_cache_groups(config, specs)
    row = packed_sfa_row_bytes(groups)
    plan = packed_sfa_cache_config(config, groups, 3 * row, lambda _, n: n)
    raw = torch.zeros(3 * row, dtype=torch.int8)
    inputs = PackedCommonMetadata(
        torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int32),
        torch.tensor([0, 255, 256, 511, 512, 767, -1], dtype=torch.int64),
    )
    for desc in plan.kv_cache_tensors:
        for i, name in enumerate(desc.layers):
            spec = specs[name]
            offset = desc.offset + i * desc.layer_stride
            span = 2 * row + spec.page_size_bytes
            layer_raw = raw[offset : offset + span]
            dims = (512, 64) if spec.is_draft else None
            views = packed_sfa_cache_views(layer_raw, spec, 3, row, dims)
            if isinstance(spec, MambaSpec):
                assert all(v.shape[0] == 3 for v in views)
                assert all(v.stride(0) * v.element_size() == row for v in views)
                continue
            builder = SimpleNamespace(vllm_config=config)
            mapped = map_packed_sfa_metadata(builder, inputs, spec, row)
            pointer = mapped.block_table_tensor.data_ptr()
            assert map_packed_sfa_metadata(builder, inputs, spec, row).block_table_tensor.data_ptr() == pointer
            assert mapped.slot_mapping[-1] == -1
            page = spec.page_size_bytes // 2
            assert mapped.block_table_tensor.tolist() == [
                [0, 1, row // page, row // page + 1, 2 * row // page, 2 * row // page + 1]
            ]
            for token, slot in zip(inputs.slot_mapping[:-1].tolist(), mapped.slot_mapping[:-1].tolist()):
                bid, within = divmod(token, 256)
                kid, inner = divmod(slot, 128)
                address = (
                    (views[0].data_ptr() - raw.data_ptr())
                    + kid * views[0].stride(0) * views[0].element_size()
                    + inner * views[0].stride(1) * views[0].element_size()
                )
                width = views[0].shape[-1] * views[0].element_size()
                assert address == offset + bid * row + (within // 128) * page + (within % 128) * width
            if isinstance(spec, AscendMLAAttentionSpec) and spec.uses_packed_sfa_main_cache:
                assert (
                    views[0].view(-1, spec.head_size).untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
                )


def test_too_small_row_rejected():
    with pytest.raises(ValueError, match="increase --block-size"):
        packed_sfa_cache_groups(packed_config(), make_packed_specs(128, 0, 3))


@pytest.mark.parametrize("parallel_axis", ["prefill_context_parallel_size", "decode_context_parallel_size"])
def test_unsupported_cp_fails_before_changing_specs(parallel_axis):
    config = packed_config()
    setattr(config.parallel_config, parallel_axis, 2)
    specs = make_packed_specs(draft_layers=3)
    before = specs.copy()
    with pytest.raises(ValueError, match="PCP=1 and DCP=1"):
        packed_sfa_cache_groups(config, specs)
    assert specs == before
