# SPDX-License-Identifier: Apache-2.0
"""Unit regressions for Qwen4Exp MTP hybrid KV-cache groups."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.worker.v1.spec_decode.qwen4_exp import AscendQwen4ExpMTPProposer


def _attention_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
    )


def test_map_draft_layer_accepts_direct_group_spec() -> None:
    layer_name = "model.mtp.0.self_attn"
    spec = _attention_spec()
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=spec)],
    )
    proposer = SimpleNamespace(_draft_attn_layer_names={layer_name})

    layer_to_gid, layer_to_spec = AscendQwen4ExpMTPProposer._map_draft_layers_to_groups(proposer, config)

    assert layer_to_gid == {layer_name: 0}
    assert layer_to_spec == {layer_name: spec}


def test_map_draft_layer_accepts_uniform_group_spec() -> None:
    layer_name = "model.mtp.0.self_attn"
    spec = _attention_spec()
    uniform = UniformTypeKVCacheSpecs.from_specs({layer_name: spec})
    assert uniform is not None
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=uniform)],
    )
    proposer = SimpleNamespace(_draft_attn_layer_names={layer_name})

    layer_to_gid, layer_to_spec = AscendQwen4ExpMTPProposer._map_draft_layers_to_groups(proposer, config)

    assert layer_to_gid == {layer_name: 0}
    assert layer_to_spec == {layer_name: spec}


def _draft_group(gid: int, layer_name: str, metadata: object):
    builder = MagicMock()
    builder.build_for_drafting.return_value = metadata
    group = SimpleNamespace(
        kv_cache_group_id=gid,
        layer_names=[layer_name],
        get_metadata_builder=MagicMock(return_value=builder),
    )
    return group, builder


def test_ascend_step_zero_uses_qwen_per_group_block_tables() -> None:
    proposer = AscendQwen4ExpMTPProposer.__new__(AscendQwen4ExpMTPProposer)
    main_metadata = object()
    raw_metadata = object()
    main_group, main_builder = _draft_group(0, "draft.main", main_metadata)
    raw_group, raw_builder = _draft_group(1, "draft.raw", raw_metadata)
    proposer.draft_attn_groups = [main_group, raw_group]
    proposer.kv_cache_gid = 0
    raw_table = torch.tensor([[23, 24]], dtype=torch.int32)
    raw_slots = torch.tensor([23 * 128 + 7, -1], dtype=torch.int32)
    proposer._per_group_block_tables = {1: raw_table}
    proposer._per_group_slot_mappings = {1: raw_slots}
    proposer._get_primary_draft_attn_group = MagicMock(return_value=main_group)
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=1,
        block_table_tensor=torch.tensor([[3, 4]], dtype=torch.int32),
        slot_mapping=torch.tensor([3 * 128 + 7], dtype=torch.int32),
    )

    multi_step, primary = proposer.build_draft_attn_metadata(
        common,
        num_input_tokens=4,
        num_actual_tokens=2,
    )

    main_common = main_builder.build_for_drafting.call_args.kwargs["common_attn_metadata"]
    raw_common = raw_builder.build_for_drafting.call_args.kwargs["common_attn_metadata"]
    assert main_common is common
    assert main_common.block_table_tensor.tolist() == [[3, 4]]
    assert raw_common is not common
    assert raw_common.block_table_tensor.data_ptr() == raw_table.data_ptr()
    assert raw_common.slot_mapping.data_ptr() == raw_slots.data_ptr()
    assert multi_step == [{"draft.main": main_metadata, "draft.raw": raw_metadata}]
    assert primary is main_metadata


def test_ascend_later_step_uses_qwen_cache_only_group_block_table() -> None:
    proposer = AscendQwen4ExpMTPProposer.__new__(AscendQwen4ExpMTPProposer)
    proposer.kv_cache_gid = 0
    raw_table = torch.tensor([[41, 42]], dtype=torch.int32)
    proposer._per_group_block_tables = {1: raw_table}
    proposer._per_group_kernel_block_sizes = {0: 16, 1: 128}
    proposer._per_group_next_step_slot_mappings = {}
    proposer.slot_mapping_group = [torch.full((8,), -1, dtype=torch.int32) for _ in range(3)]
    primary_metadata = object()
    raw_metadata = object()
    primary_group, _ = _draft_group(0, "draft.main", primary_metadata)
    raw_group, raw_builder = _draft_group(1, "draft.raw", raw_metadata)
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=1,
        positions=torch.tensor([129], dtype=torch.int64),
        block_table_tensor=torch.tensor([[7, 8]], dtype=torch.int32),
    )

    per_layer = proposer._build_cache_only_group_next_step_attn_metadata(
        common,
        draft_index=2,
        num_input_tokens=8,
        primary_group=primary_group,
        primary_metadata=primary_metadata,
        cache_only_groups=[raw_group],
    )

    raw_common, draft_index = raw_builder.build_for_drafting.call_args.args
    assert draft_index == 2
    assert raw_common is not common
    assert raw_common.block_table_tensor.data_ptr() == raw_table.data_ptr()
    assert raw_common.slot_mapping.tolist() == [42 * 128 + 1] + [-1] * 7
    assert per_layer == {
        "draft.main": primary_metadata,
        "draft.raw": raw_metadata,
    }


def test_qwen_mtp_uses_primary_group_kernel_block_size_for_later_steps() -> None:
    proposer = AscendQwen4ExpMTPProposer.__new__(AscendQwen4ExpMTPProposer)
    proposer.has_gdn = True
    proposer.kernel_block_size = 16
    proposer.block_size = 128

    # Qwen is a hybrid GDN/QSA model, but its draft layer writes the QSA main
    # cache. The target-wide GDN flag must not select the GDN kernel size.
    assert proposer._get_draft_slot_block_size() == 128
