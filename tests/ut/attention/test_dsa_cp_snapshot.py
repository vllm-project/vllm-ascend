from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.attention.context_parallel.dsa_cp import (
    AscendDSACPImpl,
    AscendDSACPMetadataBuilder,
)
from vllm_ascend.attention.context_parallel.sfa_cp import (
    AscendSFADSACPImpl,
    AscendSFAPCPImpl,
)
from vllm_ascend.attention.sfa_v1 import SMLA_DEFAULT_SINK_VALUE, AscendSFAImpl, AscendSFAMetadataBuilder


def test_smla_sinks_restored_in_place():
    builder = AscendSFAMetadataBuilder.__new__(AscendSFAMetadataBuilder)
    state = SimpleNamespace(
        use_smla=True,
        block_table_buffer=torch.ones(2),
        metadata_buffer=torch.ones(2),
        length_buffer=torch.ones(2),
        sinks=torch.zeros(4),
    )
    builder.nope_states = {0: state, 1: SimpleNamespace(use_smla=False, block_table_buffer=torch.ones(2))}
    address = state.sinks.data_ptr()
    for _ in range(2):
        state.sinks.zero_()
        builder.reset_runtime_state_after_snapshot_restore()
        torch.testing.assert_close(state.sinks, torch.full((4,), SMLA_DEFAULT_SINK_VALUE))
        assert state.sinks.data_ptr() == address


def test_metadata_builder_reset_clears_requests_and_preserves_configuration():
    builder = AscendDSACPMetadataBuilder.__new__(AscendDSACPMetadataBuilder)
    builder.num_decodes = 4
    builder.num_prefills = 3
    builder.num_decode_tokens = 8
    builder.num_prefill_tokens = 12
    builder.num_actual_tokens = 20
    builder.block_table = torch.ones((2, 2), dtype=torch.int32)
    builder.seq_lens = torch.full((4,), 7, dtype=torch.int32)
    builder.seq_lens_cpu = torch.full((4,), 11, dtype=torch.int32)

    builder.start_pos_prefill = torch.full((4,), 13, dtype=torch.int32)
    builder.req_sas_metadata = torch.full((8,), 17, dtype=torch.int32)
    builder.req_qli_metadata = torch.full((8,), 19, dtype=torch.int32)
    builder.qli_seqused_k = torch.full((4,), 21, dtype=torch.int32)
    builder.qli_cmp_residual_k = torch.full((4,), 22, dtype=torch.int32)
    builder.cu_seqlens_ori_kv = torch.full((4,), 23, dtype=torch.int32)
    builder.cu_seqlens_cmp_kv = torch.full((4,), 29, dtype=torch.int32)
    builder.seqused_q = torch.full((4,), 31, dtype=torch.int32)
    builder._zero_i32 = torch.full((1,), 37, dtype=torch.int32)
    builder.local_query_start_loc = torch.full((5,), 41, dtype=torch.int32)
    builder.local_seq_lens = torch.full((4,), 43, dtype=torch.int32)
    builder.slot_mapping = torch.full((4, 2), 47, dtype=torch.int32)
    builder.spec_slot_mapping = [torch.full((4, 2), 53, dtype=torch.int32)]
    builder.spec_local_query_start_loc = [torch.full((5,), 59, dtype=torch.int32)]
    builder.spec_local_seq_lens = [torch.full((4,), 61, dtype=torch.int32)]
    builder.spec_sas_metadata = [torch.full((8,), 67, dtype=torch.int32)]
    builder.spec_start_pos = [torch.full((4,), 71, dtype=torch.int32)]
    builder.common_ratio_to_sas_metadata = {
        "input_positions": torch.ones(1),
        "cp_sas_c4": torch.ones(1),
    }
    builder._device_metadata_enabled = True
    builder._device_metadata_tasks = (object(),)
    compressor_metadata_buffers = (
        torch.ones(4),
        torch.ones(4),
        builder.slot_mapping,
    )
    builder.compressor_metadata_buffers = compressor_metadata_buffers
    builder.hadamard = None

    builder.reset_runtime_state_after_snapshot_restore()

    assert builder.num_decodes == 0
    assert builder.num_prefills == 0
    assert builder.num_decode_tokens == 0
    assert builder.num_prefill_tokens == 0
    assert builder.num_actual_tokens is None
    assert builder.block_table is None
    assert builder.seq_lens is None
    assert builder.seq_lens_cpu is None
    assert builder.common_ratio_to_sas_metadata == {}
    assert builder._device_metadata_enabled is True
    assert builder._device_metadata_tasks == ()
    assert builder.compressor_metadata_buffers is compressor_metadata_buffers
    assert torch.count_nonzero(compressor_metadata_buffers[0]) == 0
    assert torch.count_nonzero(compressor_metadata_buffers[1]) == 0

    buffers = [
        builder.start_pos_prefill,
        builder.req_sas_metadata,
        builder.req_qli_metadata,
        builder.qli_seqused_k,
        builder.qli_cmp_residual_k,
        builder.cu_seqlens_ori_kv,
        builder.cu_seqlens_cmp_kv,
        builder.seqused_q,
        builder._zero_i32,
        builder.local_query_start_loc,
        builder.local_seq_lens,
        builder.slot_mapping,
        *builder.spec_slot_mapping,
        *builder.spec_local_query_start_loc,
        *builder.spec_local_seq_lens,
        *builder.spec_sas_metadata,
        *builder.spec_start_pos,
    ]
    assert all(torch.count_nonzero(buffer) == 0 for buffer in buffers)


def test_dsa_cp_impl_refreshes_tp_group_after_restore():
    impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
    impl._o_proj_weight_switch_enabled = True
    impl.wo_a_weight_state = Mock()
    impl.wo_b_weight_state = Mock()
    group = SimpleNamespace(world_size=8, rank_in_group=3)

    with patch(
        "vllm_ascend.attention.context_parallel.dsa_cp.get_tp_group",
        return_value=group,
    ):
        impl.reset_runtime_state_after_snapshot_restore()

    assert impl.tp_group is group
    assert impl.tp_size == 8
    assert impl.tp_rank == 3
    assert impl.o_proj_weight_switch_config.group is group
    impl.wo_a_weight_state.rebuild_after_snapshot_restore.assert_called_once_with(impl.o_proj_weight_switch_config)
    impl.wo_b_weight_state.rebuild_after_snapshot_restore.assert_called_once_with(impl.o_proj_weight_switch_config)


@pytest.mark.parametrize("enabled", [False, True])
def test_sfa_cp_impls_refresh_weight_switch_group_after_restore(enabled):
    pcp_impl = AscendSFAPCPImpl.__new__(AscendSFAPCPImpl)
    dsa_cp_impl = AscendSFADSACPImpl.__new__(AscendSFADSACPImpl)
    pcp_impl._o_proj_weight_switch_enabled = enabled
    pcp_impl.o_proj_weight_state = Mock()
    dsa_cp_impl._o_proj_weight_switch_enabled = enabled
    dsa_cp_impl.o_proj_weight_state = Mock()
    pcp_group = SimpleNamespace(world_size=2, rank_in_group=1)
    tp_group = SimpleNamespace(world_size=8, rank_in_group=3)

    with (
        patch(
            "vllm_ascend.attention.context_parallel.sfa_cp.get_pcp_group",
            return_value=pcp_group,
        ),
        patch(
            "vllm_ascend.attention.context_parallel.sfa_cp.get_tp_group",
            return_value=tp_group,
        ),
        patch.object(
            AscendSFAImpl,
            "reset_runtime_state_after_snapshot_restore",
        ) as reset_sfa,
    ):
        pcp_impl.reset_runtime_state_after_snapshot_restore()
        dsa_cp_impl.reset_runtime_state_after_snapshot_restore()

    assert pcp_impl.o_proj_weight_switch_config.group is pcp_group
    assert pcp_impl.o_proj_weight_switch_config.shard_axis == "input"
    assert dsa_cp_impl.o_proj_weight_switch_config.group is tp_group
    assert reset_sfa.call_count == 2
    for impl in (pcp_impl, dsa_cp_impl):
        if enabled:
            impl.o_proj_weight_state.rebuild_after_snapshot_restore.assert_called_once_with(
                impl.o_proj_weight_switch_config
            )
        else:
            impl.o_proj_weight_state.rebuild_after_snapshot_restore.assert_not_called()
