# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for MiniMax M3 sparse attention layer wiring in ``msa_m3``."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.attention.msa_m3 import (
    AscendMiniMaxM3IndexerLinear,
    AscendMiniMaxM3IndexerMetadataBuilder,
    AscendMiniMaxM3SparseBackend,
    AscendMiniMaxM3SparseDecodeMetadata,
    AscendMiniMaxM3SparseImpl,
    AscendMiniMaxM3SparseMetadata,
    AscendMiniMaxM3SparseMetadataBuilder,
    AscendMiniMaxM3SparsePrefillMetadata,
    MiniMaxM3SparseAttention,
    _register_m3_sparse_packed_modules,
    _sparse_proj_quant_type,
    _use_fused_qkv_indexer,
    minimax_m3_sparse_forward,
)


@dataclass
class BatchSpec:
    seq_lens: list[int]
    query_lens: list[int]
    name: str = "unnamed"

    @property
    def batch_size(self) -> int:
        return len(self.seq_lens)


def _create_common_attn_metadata(
    batch_spec: BatchSpec,
    block_size: int,
    device: torch.device,
) -> CommonAttentionMetadata:
    query_start_loc = torch.zeros(
        batch_spec.batch_size + 1,
        dtype=torch.int32,
        device=device,
    )
    query_start_loc[1:] = torch.tensor(
        batch_spec.query_lens,
        dtype=torch.int32,
        device=device,
    ).cumsum(0)
    query_start_loc_cpu = query_start_loc.cpu()
    num_tokens = sum(batch_spec.query_lens)

    seq_lens = torch.tensor(batch_spec.seq_lens, dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens.cpu()
    max_seq_len = int(seq_lens_cpu.max())
    context_lens = [batch_spec.seq_lens[i] - batch_spec.query_lens[i] for i in range(batch_spec.batch_size)]
    num_computed_tokens_cpu = torch.tensor(context_lens, dtype=torch.int32)
    max_blocks = (max(batch_spec.seq_lens) + block_size - 1) // block_size
    block_table_tensor = torch.arange(
        batch_spec.batch_size * max_blocks,
        dtype=torch.int32,
        device=device,
    ).view(batch_spec.batch_size, max_blocks)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=batch_spec.batch_size,
        num_actual_tokens=num_tokens,
        max_query_len=max(batch_spec.query_lens),
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
    )


def _make_vllm_config(
    *,
    max_num_batched_tokens: int = 8192,
) -> SimpleNamespace:
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            tensor_parallel_size=1,
        ),
    )


def _make_sparse_builder(device: torch.device) -> AscendMiniMaxM3SparseMetadataBuilder:
    vllm_config = _make_vllm_config()
    spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=128,
        head_size_v=128,
        dtype=torch.bfloat16,
    )
    return AscendMiniMaxM3SparseMetadataBuilder(
        spec,
        ["layer0.attn"],
        vllm_config,
        device,
    )


def _make_indexer_builder(device: torch.device) -> AscendMiniMaxM3IndexerMetadataBuilder:
    vllm_config = _make_vllm_config()
    spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        head_size_v=128,
        dtype=torch.bfloat16,
    )
    return AscendMiniMaxM3IndexerMetadataBuilder(
        spec,
        ["layer0.attn.index_cache"],
        vllm_config,
        device,
    )


def test_minimax_m3_sparse_custom_op_registered() -> None:
    assert hasattr(torch.ops.vllm, "minimax_m3_sparse_forward")


def test_sparse_backend_get_name() -> None:
    assert AscendMiniMaxM3SparseBackend.get_name() == "ASCEND_MINIMAX_M3_SPARSE"
    assert AscendMiniMaxM3SparseBackend.is_sparse() is True


@patch("vllm_ascend.attention.msa_m3.get_forward_context")
def test_minimax_m3_sparse_forward_dispatches_to_layer(
    mock_get_forward_context: MagicMock,
) -> None:
    layer = MagicMock()
    layer._run_sparse_attention = MagicMock()
    mock_get_forward_context.return_value = MagicMock(
        attn_metadata={"layer.attn": object()},
        no_compile_layers={"layer.attn": layer},
    )

    q = torch.randn(2, 32, 128)
    k = torch.randn(2, 2, 128)
    v = torch.randn(2, 2, 128)
    index_q = torch.randn(2, 2, 128)
    index_k = torch.randn(2, 128)
    out = torch.empty(2, 32 * 128)

    minimax_m3_sparse_forward(q, k, v, index_q, index_k, out, "layer.attn")

    layer._run_sparse_attention.assert_called_once_with(q, k, v, index_q, index_k, out)


@patch("vllm_ascend.attention.msa_m3.get_forward_context")
def test_minimax_m3_sparse_forward_zeros_output_without_dict_metadata(
    mock_get_forward_context: MagicMock,
) -> None:
    mock_get_forward_context.return_value = MagicMock(attn_metadata=None)
    out = torch.ones(2, 32 * 128)
    minimax_m3_sparse_forward(
        torch.randn(2, 32, 128),
        torch.randn(2, 2, 128),
        torch.randn(2, 2, 128),
        torch.randn(2, 2, 128),
        torch.randn(2, 128),
        out,
        "layer.attn",
    )
    assert torch.all(out == 0)


@pytest.mark.parametrize(
    "batch_spec",
    [
        BatchSpec(seq_lens=[129, 257], query_lens=[129, 257], name="prefill_only"),
        BatchSpec(seq_lens=[130, 131], query_lens=[1, 1], name="decode_only"),
    ],
    ids=lambda case: case.name,
)
def test_sparse_metadata_builder(batch_spec: BatchSpec) -> None:
    device = torch.device("cpu")
    builder = _make_sparse_builder(device)
    common = _create_common_attn_metadata(batch_spec, block_size=128, device=device)
    metadata = builder.build(0, common)

    assert metadata.num_actual_tokens == sum(batch_spec.query_lens)
    assert metadata.num_decodes + metadata.num_prefills == batch_spec.batch_size
    if batch_spec.name == "decode_only":
        assert metadata.num_decodes == batch_spec.batch_size
        assert metadata.prefill is None
        assert metadata.decode is not None
        assert metadata.decode.decode_query_len == 1
    else:
        assert metadata.num_prefills == batch_spec.batch_size
        assert metadata.decode is None
        assert metadata.prefill is not None
        assert metadata.prefill.cu_seqlens_k.shape[0] == batch_spec.batch_size + 1


@pytest.mark.parametrize(
    "batch_spec",
    [
        BatchSpec(seq_lens=[129, 257], query_lens=[129, 257], name="prefill_only"),
        BatchSpec(seq_lens=[130, 131], query_lens=[1, 1], name="decode_only"),
    ],
    ids=lambda case: case.name,
)
def test_indexer_metadata_builder(batch_spec: BatchSpec) -> None:
    device = torch.device("cpu")
    builder = _make_indexer_builder(device)
    common = _create_common_attn_metadata(batch_spec, block_size=128, device=device)
    metadata = builder.build(0, common)

    assert metadata.num_actual_tokens == sum(batch_spec.query_lens)
    assert metadata.num_decodes + metadata.num_prefills == batch_spec.batch_size
    if batch_spec.name == "decode_only":
        assert metadata.num_decodes == batch_spec.batch_size
        assert metadata.prefill is None
        assert metadata.decode is not None
    else:
        assert metadata.num_prefills == batch_spec.batch_size
        assert metadata.decode is None
        assert metadata.prefill is not None


def test_sparse_metadata_builder_fia_padded_dummy_request() -> None:
    """FIA mixed-batch padding can append a dummy request beyond max_num_seqs."""
    device = torch.device("cpu")
    batch_size = 16
    query_len = 128
    batch_spec = BatchSpec(
        seq_lens=[1024] * batch_size,
        query_lens=[query_len] * batch_size,
        name="prefill_only",
    )
    common = _create_common_attn_metadata(batch_spec, block_size=128, device=device)

    padded_query_start_loc = torch.zeros(batch_size + 2, dtype=torch.int32, device=device)
    padded_query_start_loc[: batch_size + 1] = common.query_start_loc
    padded_query_start_loc[batch_size + 1] = common.query_start_loc[batch_size]
    padded_query_start_loc_cpu = padded_query_start_loc.cpu()

    padded_common = CommonAttentionMetadata(
        query_start_loc=padded_query_start_loc,
        query_start_loc_cpu=padded_query_start_loc_cpu,
        seq_lens=common.seq_lens,
        _seq_lens_cpu=common._seq_lens_cpu,
        _num_computed_tokens_cpu=common._num_computed_tokens_cpu,
        num_reqs=batch_size + 1,
        num_actual_tokens=common.num_actual_tokens,
        max_query_len=common.max_query_len,
        max_seq_len=common.max_seq_len,
        block_table_tensor=common.block_table_tensor,
        slot_mapping=common.slot_mapping,
        causal=True,
    )

    sparse_builder = _make_sparse_builder(device)
    sparse_metadata = sparse_builder.build(0, padded_common)
    assert sparse_metadata.num_prefills == batch_size
    assert sparse_metadata.num_decodes == 0
    assert sparse_metadata.prefill is not None
    assert sparse_metadata.prefill.seq_lens.shape[0] == batch_size
    assert sparse_metadata.prefill.context_lens.shape[0] == batch_size

    indexer_builder = _make_indexer_builder(device)
    indexer_metadata = indexer_builder.build(0, padded_common)
    assert indexer_metadata.num_prefills == batch_size
    assert indexer_metadata.prefill is not None
    assert indexer_metadata.prefill.context_lens.shape[0] == batch_size


def test_sparse_proj_quant_type_falls_back_to_language_model_prefix() -> None:
    quant_config = SimpleNamespace(quant_description={"language_model.model.layers.0.self_attn.q_proj.weight": "w8a8"})

    assert _sparse_proj_quant_type(quant_config, "model.layers.0.self_attn", "q_proj") == "w8a8"


def test_use_fused_qkv_indexer_returns_false_for_mixed_qkv_and_indexer_quant() -> None:
    quant_config = SimpleNamespace(
        quant_description={
            "model.layers.0.self_attn.q_proj.weight": "w8a8",
            "model.layers.0.self_attn.k_proj.weight": "w8a8",
            "model.layers.0.self_attn.v_proj.weight": "w8a8",
            "model.layers.0.self_attn.index_q_proj.weight": "int8",
            "model.layers.0.self_attn.index_k_proj.weight": "int8",
        }
    )

    assert _use_fused_qkv_indexer(quant_config, "model.layers.0.self_attn") is False


def test_use_fused_qkv_indexer_rejects_mismatched_index_quant_types() -> None:
    quant_config = SimpleNamespace(
        quant_description={
            "model.layers.0.self_attn.q_proj.weight": "w8a8",
            "model.layers.0.self_attn.k_proj.weight": "w8a8",
            "model.layers.0.self_attn.v_proj.weight": "w8a8",
            "model.layers.0.self_attn.index_q_proj.weight": "int8",
            "model.layers.0.self_attn.index_k_proj.weight": "fp16",
        }
    )

    with pytest.raises(ValueError, match="index_q/index_k quantization types differ"):
        _use_fused_qkv_indexer(quant_config, "model.layers.0.self_attn")


def test_register_m3_sparse_packed_modules_adds_split_indexer_mapping() -> None:
    quant_config = SimpleNamespace(packed_modules_mapping={})

    _register_m3_sparse_packed_modules(quant_config, fused_qkv_indexer=False)

    assert quant_config.packed_modules_mapping == {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "indexer_proj": ["index_q_proj", "index_k_proj"],
    }


@patch("vllm_ascend.attention.msa_m3.logger.warning")
@patch("vllm_ascend.attention.msa_m3.AscendMiniMaxM3Indexer")
@patch("vllm_ascend.attention.msa_m3.AscendMiniMaxM3IndexerLinear")
@patch("vllm_ascend.attention.msa_m3.AscendMiniMaxM3SparseImpl")
@patch("vllm_ascend.attention.msa_m3.kv_cache_dtype_str_to_dtype")
@patch("vllm_ascend.attention.msa_m3.get_current_vllm_config")
@patch("vllm_ascend.attention.msa_m3.GemmaRMSNorm")
@patch("vllm_ascend.attention.msa_m3.get_rope")
@patch("vllm_ascend.attention.msa_m3.RowParallelLinear")
@patch("vllm_ascend.attention.msa_m3.QKVParallelLinear")
@patch("vllm_ascend.attention.msa_m3.AscendMiniMaxM3QKVParallelLinearWithIndexer")
@patch("vllm_ascend.attention.msa_m3.get_tensor_model_parallel_world_size", return_value=1)
def test_sparse_attention_uses_split_indexer_projection_when_quant_types_differ(
    _mock_tp_size: MagicMock,
    mock_fused_qkv_linear: MagicMock,
    mock_qkv_linear: MagicMock,
    mock_row_linear: MagicMock,
    mock_get_rope: MagicMock,
    mock_rms_norm: MagicMock,
    mock_get_vllm_config: MagicMock,
    mock_kv_dtype: MagicMock,
    mock_sparse_impl: MagicMock,
    mock_indexer_linear: MagicMock,
    mock_indexer: MagicMock,
    _mock_logger_warning: MagicMock,
) -> None:
    mock_qkv_linear.return_value = SimpleNamespace(name="split_qkv")
    mock_fused_qkv_linear.return_value = SimpleNamespace(name="fused_qkv")
    mock_row_linear.return_value = SimpleNamespace(name="o_proj")
    mock_get_rope.return_value = SimpleNamespace(is_neox_style=True)
    mock_rms_norm.side_effect = lambda *args, **kwargs: SimpleNamespace(
        weight_plus_one=torch.ones(1),
        variance_epsilon=kwargs.get("eps", 1e-6),
    )
    mock_get_vllm_config.return_value = SimpleNamespace(
        model_config=SimpleNamespace(),
        compilation_config=SimpleNamespace(static_forward_context={}),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
    )
    mock_kv_dtype.return_value = torch.bfloat16
    mock_sparse_impl.return_value = SimpleNamespace(name="impl")
    split_indexer_proj = SimpleNamespace(name="indexer_proj")
    runtime_indexer = SimpleNamespace(name="runtime_indexer")
    mock_indexer_linear.return_value = split_indexer_proj
    mock_indexer.return_value = runtime_indexer
    quant_config = SimpleNamespace(
        quant_description={
            "model.layers.0.self_attn.q_proj.weight": "w8a8",
            "model.layers.0.self_attn.k_proj.weight": "w8a8",
            "model.layers.0.self_attn.v_proj.weight": "w8a8",
            "model.layers.0.self_attn.index_q_proj.weight": "int8",
            "model.layers.0.self_attn.index_k_proj.weight": "int8",
        },
        packed_modules_mapping={},
    )

    layer = MiniMaxM3SparseAttention(
        hidden_size=128,
        num_heads=8,
        num_kv_heads=2,
        rotary_dim=128,
        head_dim=16,
        cache_config=SimpleNamespace(cache_dtype="auto"),
        quant_config=quant_config,
        prefix="model.layers.0.self_attn",
        sparse_cfg={
            "sparse_num_index_heads": 2,
            "sparse_index_dim": 16,
            "sparse_topk_blocks": 8,
            "sparse_block_size": 128,
        },
    )

    assert layer._use_fused_qkv_indexer is False
    assert layer.qkv_proj.name == "split_qkv"
    assert layer.indexer_proj is split_indexer_proj
    assert layer.indexer is runtime_indexer
    assert quant_config.packed_modules_mapping["qkv_proj"] == [
        "q_proj",
        "k_proj",
        "v_proj",
    ]
    assert quant_config.packed_modules_mapping["indexer_proj"] == [
        "index_q_proj",
        "index_k_proj",
    ]


def test_indexer_linear_weight_loader_uses_first_index_k_shard_for_all_ranks() -> None:
    layer = object.__new__(AscendMiniMaxM3IndexerLinear)
    layer.index_q_size = 2
    layer.index_k_size = 1
    layer.tp_rank = 3
    layer.num_index_head_replicas = 2

    param = torch.nn.Parameter(torch.zeros(3, 4))
    param.output_dim = 0
    loaded_weight = torch.arange(12, dtype=torch.float32).view(3, 4)

    layer.weight_loader(param, loaded_weight, "index_k")

    assert torch.equal(param.data[2:3], loaded_weight[:1])


@patch("vllm_ascend.attention.msa_m3.minimax_m3_sparse_attn")
@patch("vllm_ascend.attention.msa_m3.minimax_m3_sparse_attn_decode")
@patch("vllm_ascend.attention.msa_m3.get_forward_context")
def test_sparse_impl_forward_dispatches_decode_and_prefill_paths(
    mock_get_forward_context: MagicMock,
    mock_sparse_attn_decode: MagicMock,
    mock_sparse_attn_prefill: MagicMock,
) -> None:
    impl = AscendMiniMaxM3SparseImpl(
        num_heads=2,
        head_size=4,
        scale=0.5,
        num_kv_heads=2,
        topk_blocks=8,
        sparse_block_size=128,
    )
    metadata = AscendMiniMaxM3SparseMetadata(
        seq_lens=torch.tensor([5, 7], dtype=torch.int32),
        max_seq_len=7,
        slot_mapping=torch.arange(3, dtype=torch.int64),
        num_actual_tokens=3,
        num_decodes=1,
        num_decode_tokens=1,
        num_prefills=1,
        num_prefill_tokens=2,
        decode=AscendMiniMaxM3SparseDecodeMetadata(
            seq_lens=torch.tensor([5], dtype=torch.int32),
            block_table=torch.tensor([[0, 1]], dtype=torch.int32),
            max_seq_len=5,
            decode_query_len=1,
        ),
        prefill=AscendMiniMaxM3SparsePrefillMetadata(
            cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 7], dtype=torch.int32),
            seq_lens=torch.tensor([7], dtype=torch.int32),
            context_lens=torch.tensor([5], dtype=torch.int32),
            block_table=torch.tensor([[2, 3]], dtype=torch.int32),
            max_query_len=2,
            max_seq_len=7,
        ),
    )
    mock_get_forward_context.return_value = SimpleNamespace(attn_metadata={"layer.attn": metadata})
    layer = SimpleNamespace(layer_name="layer.attn")
    query = torch.arange(24, dtype=torch.float32).view(3, 8)
    kv_cache = torch.zeros(2, 4, 128, 2, 4)
    output = torch.zeros_like(query)
    decode_topk = torch.tensor([[0, 1]], dtype=torch.int32)
    prefill_topk = torch.tensor([[2, 3]], dtype=torch.int32)

    result = impl.forward(
        layer,
        query,
        kv_cache,
        (decode_topk, prefill_topk),
        output,
    )

    assert result is output
    mock_sparse_attn_decode.assert_called_once()
    mock_sparse_attn_prefill.assert_called_once()
    assert mock_sparse_attn_decode.call_args.args[0].shape == (1, 2, 4)
    assert mock_sparse_attn_prefill.call_args.args[0].shape == (2, 2, 4)
