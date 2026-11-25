import torch

from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.attention.dsa_v41 import AscendDSAV41MetadataBuilder


def test_metadata_builder_reset_clears_requests_and_preserves_configuration():
    builder = AscendDSAMetadataBuilder.__new__(AscendDSAMetadataBuilder)
    builder.num_decodes = 4
    builder.num_prefills = 3
    builder.num_decode_tokens = 8
    builder.num_prefill_tokens = 12
    builder.num_actual_tokens = 20
    builder.block_table = torch.ones((2, 2), dtype=torch.int32)
    builder.seq_lens = torch.ones(2, dtype=torch.int32)
    builder.common_ratio_to_sas_metadata = {"stale": object()}
    builder._device_metadata_enabled = True
    builder._device_metadata_tasks = (object(),)
    builder.hadamard = None
    builder.dspark_swa_indices_buffer = torch.ones(2, dtype=torch.int32)

    tensor_names = (
        "start_pos_prefill",
        "sas_metadata_buffer",
        "qli_metadata_buffer",
        "qli_seqused_k",
        "qli_cmp_residual_k",
        "cu_seqlens_ori_kv",
        "cu_seqlens_cmp_kv",
        "seqused_q",
        "_zero_i32",
        "slot_mapping",
    )
    for name in tensor_names:
        setattr(builder, name, torch.ones(2, dtype=torch.int32))
    compressor_metadata_buffers = (
        torch.ones(2),
        torch.ones(2),
        builder.slot_mapping,
    )
    builder.compressor_metadata_buffers = compressor_metadata_buffers
    builder.spec_slot_mapping = [torch.ones(2, dtype=torch.int32)]
    builder.spec_sas_metadata = [torch.ones(2, dtype=torch.int32)]

    builder.reset_runtime_state_after_snapshot_restore()

    assert builder.num_decodes == 0
    assert builder.num_prefills == 0
    assert builder.num_decode_tokens == 0
    assert builder.num_prefill_tokens == 0
    assert builder.num_actual_tokens is None
    assert builder.block_table is None
    assert builder.seq_lens is None
    assert builder.common_ratio_to_sas_metadata == {}
    assert builder._device_metadata_enabled is True
    assert builder._device_metadata_tasks == ()
    assert builder.compressor_metadata_buffers is compressor_metadata_buffers
    assert torch.count_nonzero(compressor_metadata_buffers[0]) == 0
    assert torch.count_nonzero(compressor_metadata_buffers[1]) == 0
    assert all(torch.count_nonzero(getattr(builder, name)) == 0 for name in tensor_names)
    assert torch.count_nonzero(builder.spec_slot_mapping[0]) == 0
    assert torch.count_nonzero(builder.spec_sas_metadata[0]) == 0
    assert torch.count_nonzero(builder.dspark_swa_indices_buffer) == 0


def test_v41_metadata_builder_reset_clears_requests_and_preserves_configuration():
    builder = AscendDSAV41MetadataBuilder.__new__(AscendDSAV41MetadataBuilder)
    builder._slot_mapping = torch.ones(4, dtype=torch.int64)
    builder._slot_mapping_2d = torch.ones((4, 2), dtype=torch.int32)
    builder._flat_slot_mapping = torch.ones(4, dtype=torch.int64)
    builder._dspark_swa_indices = builder._a5_causal_swa_indices = None
    builder._dspark_swa_lengths = builder._a5_causal_swa_lengths = None
    builder._a5_smla_metadata = builder._a5_smla_length_rows = None
    builder._build_compressor_metadata = False
    zeroed_names = (
        "_seq_lens",
        "_cache_seq_lens",
        "_cmp_residual",
        "_smla_metadata",
        "_qli_metadata",
        "_c2_ring_metadata",
        "_c2_complete_mask",
        "_c2_source_positions",
        "_c2_source_sin",
    )
    for name in zeroed_names:
        setattr(builder, name, torch.ones(4, dtype=torch.int32))
    builder._c2_source_cos = torch.zeros(4)
    c2_full_source_rope = (torch.ones(1), torch.ones(1))
    builder._c2_full_source_rope = c2_full_source_rope
    builder._device_metadata_enabled = True
    builder._device_metadata_tasks = (object(),)

    builder.reset_runtime_state_after_snapshot_restore()

    assert torch.all(builder._slot_mapping == -1)
    assert torch.all(builder._slot_mapping_2d == -1)
    assert all(torch.count_nonzero(getattr(builder, name)) == 0 for name in zeroed_names)
    assert torch.all(builder._c2_source_cos == 1)
    assert builder._c2_full_source_rope is c2_full_source_rope
    assert builder._device_metadata_enabled is True
    assert builder._device_metadata_tasks == ()
