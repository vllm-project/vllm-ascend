# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.fia_mla_heads import (
    pad_fia_mla_query_heads,
    trim_fia_mla_query_heads,
)
from vllm_ascend.attention.context_parallel.mla_cp import (
    AscendMLADCPDecodeMetadata,
    AscendMlaDCPImpl,
    AscendMlaDCPMetadataBuilder,
    DCPChunkedContextMetadata,
)
from vllm_ascend.attention.mla_v1 import (
    AscendMLADecodeMetadata,
    AscendMLAImpl,
    AscendMLAMetadata,
    AscendMLAMetadataBuilder,
    AscendMLAPrefillMetadata,
)


def test_mla_dcp_extends_v1_backend() -> None:
    assert issubclass(AscendMlaDCPImpl, AscendMLAImpl)
    assert issubclass(
        AscendMlaDCPMetadataBuilder,
        AscendMLAMetadataBuilder,
    )
    assert AscendMlaDCPMetadataBuilder.decode_metadata_cls is (AscendMLADCPDecodeMetadata)
    base_fields = {field.name for field in fields(AscendMLADecodeMetadata)}
    dcp_fields = {field.name for field in fields(AscendMLADCPDecodeMetadata)}
    assert {"cp_seq_len", "cp_seq_len_tensor", "dcp_mtp_attn_mask"}.isdisjoint(base_fields)
    assert {"cp_seq_len", "cp_seq_len_tensor", "dcp_mtp_attn_mask"} <= dcp_fields


def test_kimi_k3_fia_query_heads_pad_and_trim() -> None:
    q_nope = torch.arange(2 * 12 * 512, dtype=torch.float32).view(2, 12, 1, 512)
    q_pe = torch.arange(2 * 12 * 64, dtype=torch.float32).view(2, 12, 1, 64)

    padded_nope, padded_pe, padded_heads = pad_fia_mla_query_heads(q_nope, q_pe, "BNSD", 12)
    output, lse = trim_fia_mla_query_heads(
        padded_nope,
        torch.arange(2 * 16, dtype=torch.float32).view(2, 16, 1, 1),
        "BNSD",
        12,
    )

    assert padded_heads == 16
    torch.testing.assert_close(output, q_nope)
    torch.testing.assert_close(lse, torch.arange(2 * 16, dtype=torch.float32).view(2, 16, 1, 1)[:, :12])
    torch.testing.assert_close(padded_nope[:, 12:], torch.zeros_like(padded_nope[:, 12:]))
    torch.testing.assert_close(padded_pe[:, 12:], torch.zeros_like(padded_pe[:, 12:]))


def test_kimi_k3_fia_query_heads_pad_and_trim_bsnd() -> None:
    q_nope = torch.randn(2, 1, 12, 512)
    q_pe = torch.randn(2, 1, 12, 64)

    padded_nope, padded_pe, padded_heads = pad_fia_mla_query_heads(q_nope, q_pe, "BSND", 12)
    lse = torch.randn(2, 16, 1, 1)
    output, trimmed_lse = trim_fia_mla_query_heads(padded_nope, lse, "BSND", 12)

    assert padded_heads == 16
    torch.testing.assert_close(output, q_nope)
    torch.testing.assert_close(trimmed_lse, lse[:, :12])
    torch.testing.assert_close(padded_pe[:, :, 12:], torch.zeros_like(padded_pe[:, :, 12:]))


def test_kimi_k3_fia_query_head_padding_graph_replay() -> None:
    compiled = torch.compile(pad_fia_mla_query_heads, backend="eager", fullgraph=True)
    first = torch.ones(1, 12, 1, 512)
    second = torch.full((1, 12, 1, 512), 2.0)
    rope = torch.ones(1, 12, 1, 64)

    first_padded, _, first_heads = compiled(first, rope, "BNSD", 12)
    second_padded, _, second_heads = compiled(second, rope, "BNSD", 12)

    assert first_heads == second_heads == 16
    torch.testing.assert_close(first_padded[:, :12], first)
    torch.testing.assert_close(second_padded[:, :12], second)
    torch.testing.assert_close(second_padded[:, 12:], torch.zeros_like(second_padded[:, 12:]))


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


def test_mla_dcp_decode_metadata_keeps_graph_stable_local_lengths() -> None:
    builder = AscendMlaDCPMetadataBuilder.__new__(AscendMlaDCPMetadataBuilder)
    builder.num_decodes = 2
    builder.graph_pad_size = 4
    builder.cp_seq_len_tensor = torch.empty(8, dtype=torch.int32)
    builder._require_dcp_metadata = lambda _metadata: SimpleNamespace(
        draft_cp_seq_len=torch.tensor([8, 0], dtype=torch.int32),
        dcp_mtp_attn_mask=None,
    )
    decode = AscendMLADCPDecodeMetadata(
        input_positions=torch.arange(2),
        block_table=torch.ones((2, 2), dtype=torch.int32),
        seq_lens=torch.tensor([8, 0]),
        max_seq_lens=8,
        seq_lens_list=[8, 0],
    )

    with patch.object(AscendMLAMetadataBuilder, "build_decode_metadata", return_value=decode):
        result = AscendMlaDCPMetadataBuilder.build_decode_metadata(builder, 0, SimpleNamespace())

    assert result.cp_seq_len_tensor.data_ptr() == builder.cp_seq_len_tensor.data_ptr()
    torch.testing.assert_close(result.cp_seq_len_tensor, torch.tensor([8, 0, 0, 0], dtype=torch.int32))


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
        cp_seq_len_tensor=torch.tensor([10], dtype=torch.int32),
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


@patch(
    "vllm_ascend.attention.context_parallel.mla_cp._EXTRA_CTX",
    SimpleNamespace(is_draft_model=False, capturing=False),
)
@patch("vllm_ascend.attention.context_parallel.mla_cp.torch_npu.npu_fused_infer_attention_score")
def test_kimi_k3_dcp_decode_handles_graph_padded_local_lengths(mock_fia) -> None:
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.dcp_size = 1
    impl.num_heads = 12
    impl.num_kv_heads = 1
    impl.kv_lora_rank = 3
    impl.qk_rope_head_dim = 2
    impl.scale = 1.0
    impl.speculative_config = None
    merged = {}

    def merge(output, lse, _head_size):
        merged["output"] = output
        merged["lse"] = lse
        return output

    impl._merge_dcp_attention_output = merge
    impl._v_up_proj_batch_major = lambda output: output
    metadata = AscendMLAMetadata(
        num_actual_tokens=2,
        slot_mapping=torch.arange(2),
        query_start_loc=torch.tensor([0, 1, 2]),
        seq_lens=torch.tensor([0, 8]),
        seq_lens_cpu=torch.tensor([0, 8]),
        block_tables=torch.ones((2, 2), dtype=torch.int32),
        num_decodes=2,
        num_decode_tokens=2,
        num_prefills=0,
        query_lens=[1, 1],
        attn_state=AscendAttentionState.DecodeOnly,
        decode=AscendMLADCPDecodeMetadata(
            input_positions=torch.arange(2),
            block_table=torch.ones((2, 2), dtype=torch.int32),
            seq_lens=torch.tensor([0, 8]),
            max_seq_lens=8,
            seq_lens_list=[0, 8],
            cp_seq_len=[0, 8],
            cp_seq_len_tensor=torch.tensor([0, 8, 0, 0], dtype=torch.int32),
        ),
    )
    mock_fia.return_value = (
        torch.ones(2, 16, 1, 3),
        torch.full((2, 16, 1, 1), 4.0),
    )

    impl._forward_decode(
        torch.randn(2, 12, 3),
        torch.randn(2, 12, 2),
        torch.randn(2, 1, 2, 3),
        torch.randn(2, 1, 2, 2),
        2,
        metadata,
    )

    query = mock_fia.call_args.args[0]
    assert query.shape == (2, 16, 1, 3)
    assert mock_fia.call_args.kwargs["num_heads"] == 16
    torch.testing.assert_close(query[:, 12:], torch.zeros_like(query[:, 12:]))
    assert merged["output"].shape == (2, 12, 3)
    torch.testing.assert_close(merged["output"][0], torch.zeros_like(merged["output"][0]))
    assert torch.isneginf(merged["lse"][0]).all()
    torch.testing.assert_close(merged["output"][1], torch.ones_like(merged["output"][1]))
