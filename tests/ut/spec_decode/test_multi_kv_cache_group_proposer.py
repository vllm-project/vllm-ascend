import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

import vllm_ascend.spec_decode as spec_decode
import vllm_ascend.spec_decode.multi_kv_cache_group_proposer as multi_group_proposer
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.spec_decode.multi_kv_cache_group_proposer import (
    AscendMultiKVCacheGroupMTPProposer,
    is_multi_kv_cache_group_mtp,
)


def test_multi_group_proposer_directly_inherits_ascend_eagle():
    assert AscendMultiKVCacheGroupMTPProposer.__bases__ == (AscendEagleProposer,)
    assert inspect.signature(AscendMultiKVCacheGroupMTPProposer.__init__) == inspect.signature(
        AscendEagleProposer.__init__
    )


def test_draft_step_update_matches_parent_signature():
    assert inspect.signature(AscendMultiKVCacheGroupMTPProposer.attn_update_stack_num_spec_norm) == inspect.signature(
        AscendEagleProposer.attn_update_stack_num_spec_norm
    )


@pytest.mark.parametrize("use_keyword_arguments", [False, True])
def test_draft_step_update_forwards_optional_parent_arguments(use_keyword_arguments):
    proposer = AscendMultiKVCacheGroupMTPProposer.__new__(AscendMultiKVCacheGroupMTPProposer)
    proposer._uses_multi_group_kv_cache = False
    args = (1, object(), 2, 4, object(), CUDAGraphMode.NONE)
    optional_args = dict(
        ori_seq_len=object(),
        ori_seq_len_cpu=object(),
        slot_indices=object(),
        mtp_slot_mapping=object(),
        attn_group=object(),
    )
    expected = object()
    with patch.object(AscendEagleProposer, "attn_update_stack_num_spec_norm", return_value=expected) as parent_update:
        if use_keyword_arguments:
            result = proposer.attn_update_stack_num_spec_norm(*args, **optional_args)
        else:
            result = proposer.attn_update_stack_num_spec_norm(*args, *optional_args.values())

    assert result is expected
    parent_update.assert_called_once_with(*args, **optional_args)


def test_copy_cache_only_metadata_clones_dataclass_instance_tensors():
    @dataclass
    class Metadata:
        slot_mapping: torch.Tensor
        num_reqs: int

    metadata = Metadata(torch.tensor([3, 7]), num_reqs=2)
    copied = AscendMultiKVCacheGroupMTPProposer._copy_cache_only_draft_metadata(metadata)
    assert isinstance(copied, Metadata)
    assert copied is not metadata
    assert copied.num_reqs == 2
    assert copied.slot_mapping.data_ptr() != metadata.slot_mapping.data_ptr()
    metadata.slot_mapping.fill_(-1)
    assert copied.slot_mapping.tolist() == [3, 7]


def test_copy_cache_only_metadata_passes_through_classes_and_non_dataclasses():
    @dataclass
    class Metadata:
        num_reqs: int

    for metadata in (Metadata, None, object(), SimpleNamespace(num_reqs=2)):
        assert AscendMultiKVCacheGroupMTPProposer._copy_cache_only_draft_metadata(metadata) is metadata


def test_glm5next_mtp_is_selected_by_draft_model_type():
    speculative_config = MagicMock()
    speculative_config.use_gemma4_mtp.return_value = False
    speculative_config.use_step3p5_mtp.return_value = False
    speculative_config.draft_model_config.hf_config = SimpleNamespace(
        model_type="glm5_next_mtp",
        architectures=["Glm5NextMTPModel"],
    )
    vllm_config = SimpleNamespace(speculative_config=speculative_config)
    proposer = object()

    assert is_multi_kv_cache_group_mtp(vllm_config)
    with patch.object(spec_decode, "AscendMultiKVCacheGroupMTPProposer", return_value=proposer) as proposer_cls:
        assert spec_decode.get_spec_decode_method("mtp", vllm_config, "npu", "runner") is proposer

    proposer_cls.assert_called_once_with(vllm_config, "npu", "runner")


def test_initialize_attn_backend_delegates_single_kv_cache_group():
    proposer = AscendMultiKVCacheGroupMTPProposer.__new__(AscendMultiKVCacheGroupMTPProposer)
    proposer._draft_attn_layer_names = {"draft.attn"}
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["draft.attn"], kv_cache_spec=MagicMock())]
    )

    with patch.object(AscendEagleProposer, "initialize_attn_backend") as parent_init:
        proposer.initialize_attn_backend(kv_cache_config, kernel_block_sizes=[128])

    parent_init.assert_called_once_with(kv_cache_config, [128])
    assert proposer._uses_multi_group_kv_cache is False


def test_initialize_attn_backend_splits_glm_physical_cache_groups():
    proposer = AscendMultiKVCacheGroupMTPProposer.__new__(AscendMultiKVCacheGroupMTPProposer)
    proposer._draft_attn_layer_names = {"draft.attn", "draft.indexer.k_cache"}
    proposer.vllm_config = MagicMock()
    proposer.device = torch.device("cpu")

    main_backend = MagicMock()
    main_backend.full_cls_name.return_value = "main.backend"
    main_backend.get_impl_cls.return_value = object
    indexer_backend = MagicMock()
    indexer_backend.full_cls_name.return_value = "indexer.backend"
    indexer_backend.get_impl_cls.return_value = None
    main_layer = MagicMock()
    main_layer.get_attn_backend.return_value = main_backend
    indexer_layer = MagicMock()
    indexer_layer.get_attn_backend.return_value = indexer_backend
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["draft.attn"], kv_cache_spec=MagicMock()),
            SimpleNamespace(layer_names=["draft.indexer.k_cache"], kv_cache_spec=MagicMock()),
        ]
    )

    with (
        patch.object(multi_group_proposer.AttentionGroup, "create_metadata_builders") as create_builders,
        patch.object(
            multi_group_proposer,
            "get_layers_from_vllm_config",
            return_value={"draft.attn": main_layer, "draft.indexer.k_cache": indexer_layer},
        ),
    ):
        proposer.initialize_attn_backend(kv_cache_config, kernel_block_sizes=[128, 32])

    assert proposer._uses_multi_group_kv_cache is True
    assert proposer.kv_cache_gid == 0
    assert proposer.block_size == 128
    assert [group.kv_cache_group_id for group in proposer.draft_attn_groups] == [0, 1]
    assert [call.kwargs["kernel_block_size"] for call in create_builders.call_args_list] == [128, None]


def test_secondary_group_recomputes_slot_mapping_from_its_block_table():
    proposer = AscendMultiKVCacheGroupMTPProposer.__new__(AscendMultiKVCacheGroupMTPProposer)
    proposer._uses_multi_group_kv_cache = True
    proposer.kv_cache_gid = 0
    proposer._draft_block_table_width = MagicMock(return_value=2)

    secondary_block_table = MagicMock()
    secondary_block_table.get_device_tensor.return_value = torch.arange(8, dtype=torch.int32).reshape(2, 4)
    secondary_block_table.slot_mapping.gpu = torch.zeros(4, dtype=torch.int32)
    proposer.runner = SimpleNamespace(input_batch=SimpleNamespace(block_table=[MagicMock(), secondary_block_table]))
    attn_group = SimpleNamespace(kv_cache_group_id=1)
    common_attn_metadata = SimpleNamespace(
        _seq_lens_cpu=torch.tensor([5, 8], dtype=torch.int32),
        seq_lens_cpu=None,
        num_reqs=2,
        num_actual_tokens=2,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        positions=torch.tensor([4, 7], dtype=torch.int32),
        block_table_tensor=torch.full((2, 4), 99, dtype=torch.int32),
        slot_mapping=torch.full((4,), 77, dtype=torch.int32),
    )

    group_metadata = proposer._common_attn_metadata_for_draft_group(
        common_attn_metadata,
        attn_group,
        num_input_tokens=4,
    )

    num_reqs, query_start_loc, positions = secondary_block_table.compute_slot_mapping.call_args.args
    assert num_reqs == 2
    assert query_start_loc.tolist() == [0, 1, 2]
    assert positions.tolist() == [4, 7]
    assert group_metadata is not common_attn_metadata
    assert group_metadata.block_table_tensor.shape == (2, 2)
    assert group_metadata.slot_mapping.tolist() == [0, 0, -1, -1]
    assert common_attn_metadata.slot_mapping.tolist() == [77, 77, 77, 77]


def test_primary_group_crops_block_table_to_builder_width():
    proposer = AscendMultiKVCacheGroupMTPProposer.__new__(AscendMultiKVCacheGroupMTPProposer)
    proposer._uses_multi_group_kv_cache = True
    proposer.kv_cache_gid = 0
    proposer._draft_block_table_width = MagicMock(return_value=2)
    proposer.runner = SimpleNamespace(input_batch=SimpleNamespace(block_table=[MagicMock()]))
    attn_group = SimpleNamespace(kv_cache_group_id=0)
    common_attn_metadata = SimpleNamespace(
        _seq_lens_cpu=torch.tensor([5, 8], dtype=torch.int32),
        seq_lens_cpu=None,
        num_reqs=2,
        num_actual_tokens=2,
        block_table_tensor=torch.arange(8, dtype=torch.int32).reshape(2, 4),
        slot_mapping=torch.arange(4, dtype=torch.int32),
    )

    group_metadata = proposer._common_attn_metadata_for_draft_group(
        common_attn_metadata,
        attn_group,
        num_input_tokens=4,
    )

    assert group_metadata is not common_attn_metadata
    assert group_metadata.block_table_tensor.shape == (2, 2)
    assert group_metadata.slot_mapping.data_ptr() == common_attn_metadata.slot_mapping.data_ptr()


def test_cache_only_next_step_uses_group_metadata_without_advancing_state():
    proposer = AscendMultiKVCacheGroupMTPProposer.__new__(AscendMultiKVCacheGroupMTPProposer)
    proposer._uses_multi_group_kv_cache = True
    proposer.use_compress = False
    common_attn_metadata = MagicMock()
    group_common_attn_metadata = MagicMock()
    primary_metadata = object()
    cache_only_metadata = object()
    primary_group = SimpleNamespace(layer_names=["draft.attn"])
    builder = MagicMock()
    builder.build_for_drafting.return_value = cache_only_metadata
    cache_only_group = SimpleNamespace(
        layer_names=["draft.indexer.k_cache"],
        get_metadata_builder=MagicMock(return_value=builder),
    )
    proposer._common_attn_metadata_for_draft_group = MagicMock(return_value=group_common_attn_metadata)

    per_layer_metadata = proposer._build_cache_only_group_next_step_attn_metadata(
        common_attn_metadata,
        draft_index=1,
        num_input_tokens=2,
        primary_group=primary_group,
        primary_metadata=primary_metadata,
        cache_only_groups=[cache_only_group],
    )

    proposer._common_attn_metadata_for_draft_group.assert_called_once_with(
        common_attn_metadata,
        cache_only_group,
        2,
    )
    builder.build_for_drafting.assert_called_once_with(group_common_attn_metadata, 1)
    assert per_layer_metadata == {
        "draft.attn": primary_metadata,
        "draft.indexer.k_cache": cache_only_metadata,
    }


def test_secondary_tail_slots_cover_prefill_and_keep_each_step():
    proposer = AscendMultiKVCacheGroupMTPProposer.__new__(AscendMultiKVCacheGroupMTPProposer)
    proposer._uses_multi_group_kv_cache = True
    proposer.kv_cache_gid = 0
    proposer._draft_block_table_width = MagicMock(return_value=1)
    slots = torch.full((8,), -1, dtype=torch.int32)
    table = torch.tensor([[2], [7]], dtype=torch.int32)
    block_table = MagicMock()
    block_table.get_device_tensor.return_value = table
    block_table.slot_mapping.gpu = slots

    def compute(num_reqs, query_start_loc, positions):
        slots.fill_(-1)
        for req in range(num_reqs):
            start, end = query_start_loc[req : req + 2].tolist()
            slots[start:end] = table[req, 0] * 9 + positions[start:end] % 9

    block_table.compute_slot_mapping.side_effect = compute
    proposer.runner = SimpleNamespace(input_batch=SimpleNamespace(block_table=[None, block_table]))
    group = SimpleNamespace(kv_cache_group_id=1)
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=5,
        positions=torch.tensor([5, 6, 7, 10, 11, 0, 0, 0]),
        query_start_loc=torch.tensor([0, 3, 5]),
        _seq_lens_cpu=None,
        seq_lens_cpu=None,
    )
    first = proposer._common_attn_metadata_for_draft_group(common, group, 8)
    assert first.slot_mapping.tolist() == [23, 24, 25, 64, 65, -1, -1, -1]
    common.positions = torch.tensor([8, 12, 0, 0, 0, 0, 0, 0])
    common.query_start_loc = torch.tensor([0, 1, 2])
    common.num_actual_tokens = 2
    second = proposer._common_attn_metadata_for_draft_group(common, group, 8)
    assert second.slot_mapping.tolist() == [26, 66, -1, -1, -1, -1, -1, -1]
    assert first.slot_mapping.tolist() == [23, 24, 25, 64, 65, -1, -1, -1]


def test_later_draft_sequence_lengths_exclude_rejected_tokens():
    proposer = AscendMultiKVCacheGroupMTPProposer.__new__(AscendMultiKVCacheGroupMTPProposer)
    proposer._uses_multi_group_kv_cache = True
    proposer.kv_cache_gid = 0
    proposer.uses_mrope = False
    proposer.method = "mtp"
    proposer.max_model_len = 4096
    proposer.block_size = 128
    proposer.has_gdn = False
    proposer.use_compress = False
    proposer.sliding_window = None
    proposer.arange = torch.arange(8, dtype=torch.int32)
    proposer.token_arange_np = np.arange(8, dtype=np.int32)
    proposer.slot_mapping_group = [torch.zeros(8, dtype=torch.int32) for _ in range(5)]
    proposer.seq_lens_group = [torch.zeros(8, dtype=torch.int32) for _ in range(5)]
    proposer.query_start_loc_group = [torch.zeros(8, dtype=torch.int32) for _ in range(5)]
    proposer.runner = SimpleNamespace(dcp_manager=None)
    proposer._draft_block_table_width = MagicMock(return_value=1)
    builder = MagicMock()
    group = SimpleNamespace(kv_cache_group_id=0, get_metadata_builder=lambda: builder)
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=12,
        seq_lens=torch.tensor([15, 24], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([15, 24], dtype=torch.int32),
        seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        positions=torch.arange(12, dtype=torch.int32),
        block_table_tensor=torch.tensor([[2], [7]], dtype=torch.int32),
    )
    positions = torch.tensor([10, 20], dtype=torch.int32)
    for step in range(1, 5):
        common, _ = proposer.attn_update_stack_num_spec_norm(
            step, common, 2, 12, positions, CUDAGraphMode.NONE, attn_group=group
        )
        assert common.seq_lens.tolist() == [11 + step, 21 + step]
        assert common.positions[:2].tolist() == [10 + step, 20 + step]
        assert common.slot_mapping[:2].tolist() == [256 + 10 + step, 896 + 20 + step]
        assert common._seq_lens_cpu is None
