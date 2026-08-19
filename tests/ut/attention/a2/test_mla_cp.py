# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.mla_cp import (
    AscendMLADCPDecodeMetadata,
    AscendMlaDCPImpl,
    AscendMlaDCPMetadataBuilder,
    AscendMlaPCPDCPImpl,
    AscendMlaPCPDCPMetadataBuilder,
    DCPChunkedContextMetadata,
)
from vllm_ascend.attention.mla_v1 import (
    AscendMLADecodeMetadata,
    AscendMLAImpl,
    AscendMLAMetadata,
    AscendMLAMetadataBuilder,
    AscendMLAPCPImpl,
    AscendMLAPCPMetadataBuilder,
    AscendMLAPrefillMetadata,
)


def test_mla_dcp_extends_v1_backend() -> None:
    assert issubclass(AscendMlaDCPImpl, AscendMLAImpl)
    assert AscendMlaDCPImpl.can_return_lse_for_decode
    assert issubclass(
        AscendMlaDCPMetadataBuilder,
        AscendMLAMetadataBuilder,
    )
    assert AscendMlaDCPMetadataBuilder.decode_metadata_cls is (AscendMLADCPDecodeMetadata)
    base_fields = {field.name for field in fields(AscendMLADecodeMetadata)}
    dcp_fields = {field.name for field in fields(AscendMLADCPDecodeMetadata)}
    assert {"cp_seq_len", "dcp_mtp_attn_mask"}.isdisjoint(base_fields)
    assert {"cp_seq_len", "dcp_mtp_attn_mask"} <= dcp_fields
    assert issubclass(AscendMlaPCPDCPImpl, AscendMLAPCPImpl)
    assert issubclass(AscendMlaPCPDCPImpl, AscendMlaDCPImpl)
    assert issubclass(AscendMlaPCPDCPMetadataBuilder, AscendMLAPCPMetadataBuilder)
    assert issubclass(AscendMlaPCPDCPMetadataBuilder, AscendMlaDCPMetadataBuilder)


def test_mla_pcp_dcp_equal_axis_keeps_decode_query_local() -> None:
    impl = AscendMlaPCPDCPImpl.__new__(AscendMlaPCPDCPImpl)
    impl.dcp_size = 2
    impl.pcp_size = 2
    impl.tp_group = SimpleNamespace(all_gather=MagicMock())
    q_nope = torch.randn(1, 2, 3)
    q_pe = torch.randn(1, 2, 2)

    actual_nope, actual_pe = impl.reorg_decode_q(q_nope, q_pe)

    assert actual_nope is q_nope
    assert actual_pe is q_pe


def test_mla_pcp_dcp_full_axis_gathers_decode_query_on_tp_only() -> None:
    impl = AscendMlaPCPDCPImpl.__new__(AscendMlaPCPDCPImpl)
    impl.dcp_size = 4
    impl.pcp_size = 2
    q_nope = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    q_pe = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)
    tp_group = SimpleNamespace(
        all_gather=lambda tensor, dim: torch.cat((tensor, tensor + 100), dim=dim),
    )
    impl.tp_group = tp_group
    impl.dcp_group = SimpleNamespace(all_gather=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError))

    gathered_nope, gathered_pe = impl.reorg_decode_q(q_nope, q_pe)

    assert gathered_nope.shape == (1, 4, 3)
    assert gathered_pe.shape == (1, 4, 2)
    torch.testing.assert_close(gathered_nope[:, 2:], q_nope + 100)
    torch.testing.assert_close(gathered_pe[:, 2:], q_pe + 100)


def test_mla_pcp_dcp_full_axis_selects_tp_local_heads() -> None:
    impl = AscendMlaPCPDCPImpl.__new__(AscendMlaPCPDCPImpl)
    impl.dcp_size = 4
    impl.pcp_size = 2
    impl.num_heads = 2
    impl.tp_group = SimpleNamespace(rank_in_group=1)
    merged = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    impl._merge_dcp_replicated_attention_output = MagicMock(return_value=merged)

    actual = impl._merge_dcp_attention_output(
        torch.empty(1, 4, 4),
        torch.empty(1, 4, 1),
        4,
    )

    torch.testing.assert_close(actual, merged[:, 2:4])


def test_mla_dcp_derives_decode_lengths_from_cpu_without_device_sync() -> None:
    builder = AscendMlaDCPMetadataBuilder.__new__(AscendMlaDCPMetadataBuilder)
    builder.seq_lens = torch.tensor([9, 4], dtype=torch.int32)
    builder.num_decodes = 2
    builder.dcp_size = 2
    builder.dcp_rank = 1
    builder.cp_local_block_size = 2

    assert builder._get_mrv2_dcp_rank_seq_lens() == [4, 2]


def test_mla_dcp_reorg_decode_query_gathers_fused_query() -> None:
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.dcp_size = 2
    impl.kv_lora_rank = 3
    impl.qk_rope_head_dim = 2
    q_nope = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    q_pe = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)

    group = SimpleNamespace(all_gather=lambda tensor, dim: torch.cat([tensor, tensor + 100], dim=dim))
    impl.dcp_group = group
    gathered_nope, gathered_pe = impl.reorg_decode_q(q_nope, q_pe)

    assert gathered_nope.shape == (1, 4, 3)
    assert gathered_pe.shape == (1, 4, 2)
    torch.testing.assert_close(gathered_nope[:, :2], q_nope)
    torch.testing.assert_close(gathered_pe[:, :2], q_pe)
    torch.testing.assert_close(gathered_nope[:, 2:], q_nope + 100)
    torch.testing.assert_close(gathered_pe[:, 2:], q_pe + 100)


def test_mla_dcp_uses_padded_local_chunk_lengths() -> None:
    padded_lengths = torch.tensor([[4, 2], [1, 0]], dtype=torch.int32)
    chunked = DCPChunkedContextMetadata(
        cu_seq_lens=torch.tensor([0, 2]),
        starts=torch.zeros(1, dtype=torch.int32),
        seq_tot=[6, 1],
        max_seq_lens=[4, 1],
        workspace=torch.empty(0),
        chunk_seq_lens=torch.empty(0, dtype=torch.int32),
        chunk_seq_lens_npu=torch.empty(0, dtype=torch.int32),
        chunk_actual_seq_lengths_kv_list=[[4, 6], [1, 1]],
        padded_chunk_seq_lens_npu=padded_lengths,
    )
    metadata = AscendMLAMetadata(
        num_actual_tokens=2,
        slot_mapping=torch.arange(2),
        query_start_loc=torch.tensor([0, 2]),
        seq_lens=torch.tensor([2]),
        seq_lens_cpu=torch.tensor([2]),
        block_tables=torch.zeros(1, 1, dtype=torch.int32),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        prefill=AscendMLAPrefillMetadata(
            attn_mask=None,
            query_lens=torch.tensor([2]),
            seq_lens=[2],
            context_lens=torch.tensor([0]),
            input_positions=torch.arange(2),
            query_start_loc=torch.tensor([0, 2]),
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            max_query_len=2,
            max_seq_lens=2,
            chunked_context=chunked,
        ),
    )
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)

    torch.testing.assert_close(impl.get_context_seq_len_npu(1, metadata), padded_lengths[1])


@patch(
    "vllm_ascend.attention.context_parallel.mla_cp._EXTRA_CTX",
    SimpleNamespace(is_draft_model=False, capturing=False),
)
@patch("vllm_ascend.attention.context_parallel.mla_cp.torch_npu.npu_fused_infer_attention_score")
def test_mla_dcp_mixed_cache_hit_batch_uses_decode_bsnd_metadata(mock_fia) -> None:
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.dcp_size = 1
    impl.num_heads = 2
    impl.num_kv_heads = 1
    impl.kv_lora_rank = 3
    impl.qk_rope_head_dim = 2
    impl.scale = 1.0
    impl.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    impl._merge_dcp_attention_output = lambda output, _lse, _rank: output
    impl._v_up_proj_batch_major = lambda output: output

    decode = AscendMLADCPDecodeMetadata(
        input_positions=torch.arange(4),
        block_table=torch.ones((1, 2), dtype=torch.int32),
        seq_lens=torch.tensor([20]),
        max_seq_lens=20,
        seq_lens_list=[20],
        cp_seq_len=torch.tensor([10], dtype=torch.int32),
        dcp_mtp_attn_mask=torch.zeros((1, 1, 4, 4)),
    )
    metadata = AscendMLAMetadata(
        num_actual_tokens=102,
        slot_mapping=torch.arange(102),
        query_start_loc=torch.tensor([0, 4, 18, 32, 46, 60, 74, 88, 102]),
        seq_lens=torch.tensor([20, 14, 14, 14, 14, 14, 14, 14]),
        seq_lens_cpu=torch.tensor([20, 14, 14, 14, 14, 14, 14, 14]),
        block_tables=torch.ones((8, 2), dtype=torch.int32),
        num_decodes=1,
        num_decode_tokens=4,
        num_prefills=7,
        query_lens=[4, 14, 14, 14, 14, 14, 14, 14],
        attn_state=AscendAttentionState.PrefillCacheHit,
        decode=decode,
    )

    q_nope = torch.randn(4, 2, 3)
    q_pe = torch.randn(4, 2, 2)
    k_nope = torch.randn(2, 1, 2, 3)
    k_pe = torch.randn(2, 1, 2, 2)
    mock_fia.return_value = (
        torch.randn(1, 4, 2, 3),
        torch.randn(1, 2, 4, 1),
    )

    impl._forward_decode(q_nope, q_pe, k_nope, k_pe, 2, metadata)

    call_args = mock_fia.call_args.args
    call_kwargs = mock_fia.call_args.kwargs
    assert call_args[0].shape == (1, 4, 2, 3)
    assert call_kwargs["input_layout"] == "BSND"
    assert call_kwargs["actual_seq_lengths"] == [4]
    assert call_kwargs["block_table"].shape[0] == 1
    assert call_kwargs["actual_seq_lengths_kv"].tolist() == [10]
