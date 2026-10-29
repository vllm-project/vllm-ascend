import torch

from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder


def test_metadata_builder_reset_restores_cold_start_state():
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
    builder.common_ratio_to_sas_metadata = {
        "input_positions": torch.ones(1),
        "cp_sas_c4": torch.ones(1),
    }

    builder.reset_runtime_cache()

    assert builder.num_decodes == 0
    assert builder.num_prefills == 0
    assert builder.num_decode_tokens == 0
    assert builder.num_prefill_tokens == 0
    assert builder.num_actual_tokens is None
    assert builder.block_table is None
    assert builder.seq_lens is None
    assert builder.seq_lens_cpu is None
    assert builder.common_ratio_to_sas_metadata == {}

    buffers = [
        builder.start_pos_prefill,
        builder.req_sas_metadata,
        builder.req_qli_metadata,
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
    ]
    assert all(torch.count_nonzero(buffer) == 0 for buffer in buffers)
