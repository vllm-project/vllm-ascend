import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

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
        block_table_tensor=torch.full((2, 4), 99, dtype=torch.int32),
        slot_mapping=torch.full((4,), 77, dtype=torch.int32),
    )

    group_metadata = proposer._common_attn_metadata_for_draft_group(
        common_attn_metadata,
        attn_group,
        num_input_tokens=4,
    )

    req_indices, positions = secondary_block_table.compute_slot_mapping_draft.call_args.args
    np.testing.assert_array_equal(req_indices, np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(positions, np.array([4, 7], dtype=np.int32))
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
