# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for MiniMax M3 sparse attention layer wiring in ``msa_m3``."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.models.minimax_m3 import MiniMaxM3SparseAttention
from vllm_ascend.models.minimax_m3 import msa_m3 as msa_m3_module
from vllm_ascend.models.minimax_m3.minimax_m3 import _scatter_index_cache
from vllm_ascend.models.minimax_m3.msa_m3 import (
    AscendMiniMaxM3IndexerBackend,
    AscendMiniMaxM3IndexerCache,
    AscendMiniMaxM3IndexerDecodeMetadata,
    AscendMiniMaxM3IndexerImpl,
    AscendMiniMaxM3IndexerLinear,
    AscendMiniMaxM3IndexerMetadata,
    AscendMiniMaxM3IndexerMetadataBuilder,
    AscendMiniMaxM3IndexerPrefillMetadata,
    AscendMiniMaxM3SparseBackend,
    AscendMiniMaxM3SparseDecodeMetadata,
    AscendMiniMaxM3SparseImpl,
    AscendMiniMaxM3SparseMetadata,
    AscendMiniMaxM3SparseMetadataBuilder,
    AscendMiniMaxM3SparsePrefillMetadata,
    _register_m3_sparse_packed_modules,
    _should_use_tp_sharded_index_decode,
    _sparse_proj_quant_type,
    _use_fused_qkv_indexer,
    minimax_m3_sparse_forward,
)
from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import (
    MiniMaxM3TPDecodeScoreMetadata,
    _as_ascendc_index_kv_cache,
    _index_score_topk_candidates,
    _minimax_m3_index_decode,
    _minimax_m3_index_score,
    minimax_m3_index_decode,
    minimax_m3_index_prefill,
    minimax_m3_index_tp_block_parallel_decode,
)
from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import (
    minimax_m3_sparse_attn_decode as minimax_m3_sparse_attn_decode_npu,
)
from vllm_ascend.utils import AscendDeviceType


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


def _make_indexer_builder(
    device: torch.device,
    *,
    tp_size: int = 1,
) -> AscendMiniMaxM3IndexerMetadataBuilder:
    vllm_config = _make_vllm_config()
    vllm_config.parallel_config.tensor_parallel_size = tp_size
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


@patch("vllm_ascend.models.minimax_m3.msa_m3.get_current_vllm_config")
def test_indexer_cache_uses_sfa_indexer_spec(mock_get_vllm_config: MagicMock) -> None:
    mock_get_vllm_config.return_value = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=128))

    index_cache = AscendMiniMaxM3IndexerCache(
        head_dim=128,
        prefix="layer0.attn.index_cache",
    )
    spec = index_cache.get_kv_cache_spec(vllm_config)
    old_full_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        head_size_v=128,
        dtype=torch.bfloat16,
    )

    assert type(spec) is AscendSFAIndexerCacheSpec
    assert spec.page_size_bytes * 2 == old_full_spec.page_size_bytes
    assert AscendMiniMaxM3IndexerBackend.get_kv_cache_shape(
        num_blocks=4,
        block_size=128,
        num_kv_heads=1,
        head_size=128,
    ) == (4, 128, 128)


@patch("vllm_ascend.models.minimax_m3.msa_m3.get_forward_context")
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


@patch("vllm_ascend.models.minimax_m3.msa_m3.get_forward_context")
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
    assert metadata.causal_mask.shape == (2048, 2048)
    if batch_spec.name == "decode_only":
        assert metadata.num_decodes == batch_spec.batch_size
        assert metadata.prefill is None
        assert metadata.decode is not None
        assert metadata.decode.start_loc is None
    else:
        assert metadata.num_prefills == batch_spec.batch_size
        assert metadata.decode is None
        assert metadata.prefill is not None
        assert torch.equal(
            metadata.prefill.start_loc,
            metadata.prefill.context_lens // 128,
        )


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


def test_indexer_metadata_builder_trims_graph_padded_spec_decode() -> None:
    device = torch.device("cpu")
    common = _create_common_attn_metadata(
        BatchSpec(
            seq_lens=[132] * 16,
            query_lens=[4] * 16,
            name="padded_spec_decode",
        ),
        block_size=128,
        device=device,
    )
    common.num_actual_tokens = 60
    builder = _make_indexer_builder(device, tp_size=4)

    with (
        patch(
            "vllm_ascend.models.minimax_m3.msa_m3.get_tp_group",
            return_value=SimpleNamespace(rank_in_group=0),
        ),
        patch(
            "vllm_ascend.models.minimax_m3.msa_m3.split_decodes_and_prefills",
            return_value=(16, 0, 60, 0),
        ),
    ):
        first = builder.build(0, common)
        second = builder.build(0, common)

    assert first.num_decodes == 15
    assert first.num_decode_tokens == 60
    assert first.decode is not None
    assert first.decode.tp_score is not None
    assert first.decode.cu_seqlens_q.shape == (16,)
    assert first.decode.cu_seqlens_q[-1] == 60
    assert first.decode.block_table.shape[0] == 15
    assert first.decode.tp_score.context_lens.shape == (15,)
    assert second.decode is not None
    assert second.decode.tp_score is not None
    assert first.decode.tp_score.block_table.data_ptr() == second.decode.tp_score.block_table.data_ptr()
    assert first.decode.tp_score.context_lens.data_ptr() == second.decode.tp_score.context_lens.data_ptr()
    assert first.decode.tp_score.cu_seqlens_q.data_ptr() == second.decode.tp_score.cu_seqlens_q.data_ptr()


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


def test_sparse_prepare_bypasses_fused_qkv_norm_rope_on_a5() -> None:
    source = inspect.getsource(MiniMaxM3SparseAttention._sparse_prepare)

    assert "qkv_rmsnorm_rope" in source
    assert "get_ascend_device_type() == AscendDeviceType.A5" in source
    assert 'main_qkv.device.type != "npu"' in source
    assert "1.0 + self.q_norm.weight" in source


def test_a5_index_decode_uses_a5_triton_without_tp_block_sharding() -> None:
    module_source = inspect.getsource(msa_m3_module)
    a5_branch_start = module_source.index("if not _USE_ASCENDC_INDEX_SCORE:")
    a5_branch_end = module_source.index("\n\ndef _should_use_tp_sharded_index_decode", a5_branch_start)
    import_branches = module_source[a5_branch_start:a5_branch_end]

    assert import_branches.count("minimax_m3_index_decode") == 1
    assert "msa_m3_triton_a5" in import_branches
    assert "msa_m3_triton" not in import_branches.replace("msa_m3_triton_a5", "")
    assert "get_ascend_device_type() != AscendDeviceType.A5" in module_source
    with patch(
        "vllm_ascend.models.minimax_m3.msa_m3.get_ascend_device_type",
        return_value=AscendDeviceType.A5,
    ):
        assert not _should_use_tp_sharded_index_decode(tp_size=4, num_prefills=0)


def test_non_a5_decode_keeps_tp_block_sharding() -> None:
    with patch(
        "vllm_ascend.models.minimax_m3.msa_m3.get_ascend_device_type",
        return_value=AscendDeviceType.A3,
    ):
        assert _should_use_tp_sharded_index_decode(tp_size=4, num_prefills=0)
        assert not _should_use_tp_sharded_index_decode(tp_size=1, num_prefills=0)
        assert not _should_use_tp_sharded_index_decode(tp_size=4, num_prefills=1)


def test_a5_indexer_forward_keeps_original_decode_path() -> None:
    impl = object.__new__(AscendMiniMaxM3IndexerImpl)
    torch.nn.Module.__init__(impl)
    impl.num_index_heads = 1
    impl.num_kv_heads = 1
    impl.index_head_dim = 4
    impl.topk_blocks = 2
    impl.init_blocks = 1
    impl.local_blocks = 1
    impl.scale = 0.5
    impl.index_cache = SimpleNamespace(
        prefix="layer.attn.index_cache",
        kv_cache=torch.zeros(4, 128, 4),
    )
    decode = AscendMiniMaxM3IndexerDecodeMetadata(
        seq_lens=torch.tensor([129], dtype=torch.int32),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        max_seq_len=129,
        decode_query_len=1,
    )
    metadata = AscendMiniMaxM3IndexerMetadata(
        seq_lens=decode.seq_lens,
        max_seq_len=decode.max_seq_len,
        slot_mapping=torch.zeros(1, dtype=torch.int64),
        num_actual_tokens=1,
        num_decodes=1,
        num_decode_tokens=1,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=decode,
    )
    expected = torch.zeros(1, 1, 2, dtype=torch.int32)
    tp_group = SimpleNamespace(world_size=4, rank_in_group=0)

    with (
        patch.object(msa_m3_module, "_USE_ASCENDC_INDEX_SCORE", False),
        patch.object(
            msa_m3_module,
            "get_forward_context",
            return_value=SimpleNamespace(
                attn_metadata={impl.index_cache.prefix: metadata},
            ),
        ),
        patch.object(msa_m3_module, "get_tp_group", return_value=tp_group),
        patch.object(
            msa_m3_module,
            "_should_use_tp_sharded_index_decode",
            return_value=False,
        ),
        patch.object(
            msa_m3_module,
            "minimax_m3_index_decode",
            return_value=expected,
            create=True,
        ) as mock_decode,
    ):
        actual, prefill = impl.forward(torch.zeros(1, 4))

    assert actual is expected
    assert prefill is None
    mock_decode.assert_called_once()
    decode_args = mock_decode.call_args.args
    assert torch.equal(decode_args[0], torch.zeros(1, 1, 4))
    assert decode_args[1] is impl.index_cache.kv_cache
    assert decode_args[2] is decode.block_table
    assert decode_args[3] is decode.seq_lens
    assert decode_args[4:] == (
        decode.max_seq_len,
        impl.topk_blocks,
        impl.init_blocks,
        impl.local_blocks,
        impl.num_kv_heads,
        decode.decode_query_len,
    )
    assert mock_decode.call_args.kwargs == {"sm_scale": impl.scale}


def test_a5_indexer_forward_keeps_original_prefill_path() -> None:
    impl = object.__new__(AscendMiniMaxM3IndexerImpl)
    torch.nn.Module.__init__(impl)
    impl.num_index_heads = 1
    impl.num_kv_heads = 1
    impl.index_head_dim = 4
    impl.topk_blocks = 2
    impl.init_blocks = 1
    impl.local_blocks = 1
    impl.scale = 0.5
    impl.index_cache = SimpleNamespace(
        prefix="layer.attn.index_cache",
        kv_cache=torch.zeros(4, 128, 4),
    )
    prefill_metadata = AscendMiniMaxM3IndexerPrefillMetadata(
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([129], dtype=torch.int32),
        context_lens=torch.tensor([128], dtype=torch.int32),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        max_query_len=1,
        max_seq_len=129,
    )
    metadata = AscendMiniMaxM3IndexerMetadata(
        seq_lens=prefill_metadata.seq_lens,
        max_seq_len=prefill_metadata.max_seq_len,
        slot_mapping=torch.zeros(1, dtype=torch.int64),
        num_actual_tokens=1,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        num_prefill_tokens=1,
        prefill=prefill_metadata,
    )
    score = torch.zeros(1, 1, 2)
    expected = torch.zeros(1, 1, 2, dtype=torch.int32)

    with (
        patch.object(msa_m3_module, "_USE_ASCENDC_INDEX_SCORE", False),
        patch.object(
            msa_m3_module,
            "get_forward_context",
            return_value=SimpleNamespace(
                attn_metadata={impl.index_cache.prefix: metadata},
            ),
        ),
        patch.object(
            msa_m3_module,
            "minimax_m3_index_score",
            return_value=score,
            create=True,
        ) as mock_score,
        patch.object(
            msa_m3_module,
            "minimax_m3_index_topk",
            return_value=expected,
            create=True,
        ) as mock_topk,
    ):
        decode, actual = impl.forward(torch.zeros(1, 4))

    assert decode is None
    assert actual is expected
    mock_score.assert_called_once()
    score_args = mock_score.call_args.args
    assert torch.equal(score_args[0], torch.zeros(1, 1, 4))
    assert score_args[1] is impl.index_cache.kv_cache
    assert score_args[2] is prefill_metadata.block_table
    assert score_args[3] is prefill_metadata.cu_seqlens_q
    assert score_args[4] is prefill_metadata.seq_lens
    assert score_args[5] is prefill_metadata.context_lens
    assert score_args[6:] == (
        prefill_metadata.max_query_len,
        prefill_metadata.max_seq_len,
        impl.num_kv_heads,
        impl.scale,
    )
    mock_topk.assert_called_once_with(
        score,
        prefill_metadata.cu_seqlens_q,
        prefill_metadata.context_lens,
        prefill_metadata.max_query_len,
        impl.topk_blocks,
        impl.init_blocks,
        impl.local_blocks,
    )


@patch(
    "vllm_ascend.models.minimax_m3.minimax_m3.get_ascend_device_type",
    return_value=AscendDeviceType.A5,
)
@patch(
    "vllm_ascend.models.minimax_m3.minimax_m3.torch_npu.npu_scatter_pa_cache",
    create=True,
)
def test_scatter_index_cache_uses_pa_cache_on_a5(
    mock_scatter_pa_cache: MagicMock,
    _mock_device_type: MagicMock,
) -> None:
    cache = torch.zeros(2, 128, 4, dtype=torch.bfloat16)
    updates = torch.randn(3, 4, dtype=torch.bfloat16)
    slots = torch.tensor([0, 129, -1], dtype=torch.int64)

    _scatter_index_cache(cache, updates, slots)

    mock_scatter_pa_cache.assert_called_once()
    key, actual_slots = mock_scatter_pa_cache.call_args.args
    key_cache = mock_scatter_pa_cache.call_args.kwargs["key_cache"]
    assert key.shape == (3, 1, 4)
    assert key.dtype == torch.bfloat16
    assert key.is_contiguous()
    assert torch.equal(actual_slots, slots)
    assert actual_slots.is_contiguous()
    assert key_cache.shape == (2, 128, 1, 4)
    assert key_cache.data_ptr() == cache.data_ptr()


@patch(
    "vllm_ascend.models.minimax_m3.minimax_m3.get_ascend_device_type",
    return_value=AscendDeviceType.A2,
)
def test_scatter_index_cache_keeps_legacy_op_off_a5(_mock_device_type: MagicMock) -> None:
    cache = torch.zeros(2, 128, 4, dtype=torch.bfloat16)
    updates = torch.randn(3, 4, dtype=torch.bfloat16)
    slots = torch.tensor([0, 1, 2], dtype=torch.int64)

    with patch.object(
        torch.ops._C_ascend,
        "npu_scatter_nd_update_v2",
        create=True,
    ) as mock_scatter_nd_update:
        _scatter_index_cache(cache, updates, slots)

    mock_scatter_nd_update.assert_called_once()
    flat_cache, actual_slots, actual_updates = mock_scatter_nd_update.call_args.args
    assert flat_cache.shape == (256, 4)
    assert actual_slots.shape == (3, 1)
    assert torch.equal(actual_updates, updates)


@patch("vllm_ascend.models.minimax_m3.minimax_m3.logger.warning")
@patch("vllm_ascend.models.minimax_m3.minimax_m3.AscendMiniMaxM3Indexer")
@patch("vllm_ascend.models.minimax_m3.minimax_m3.AscendMiniMaxM3IndexerLinear")
@patch("vllm_ascend.models.minimax_m3.minimax_m3.AscendMiniMaxM3SparseImpl")
@patch("vllm_ascend.models.minimax_m3.minimax_m3.kv_cache_dtype_str_to_dtype")
@patch("vllm_ascend.models.minimax_m3.minimax_m3.get_current_vllm_config")
@patch("vllm_ascend.models.minimax_m3.minimax_m3.GemmaRMSNorm")
@patch("vllm_ascend.models.minimax_m3.minimax_m3.get_rope")
@patch("vllm_ascend.models.minimax_m3.minimax_m3.RowParallelLinear")
@patch("vllm_ascend.models.minimax_m3.minimax_m3.QKVParallelLinear")
@patch("vllm_ascend.models.minimax_m3.minimax_m3.AscendMiniMaxM3QKVParallelLinearWithIndexer")
@patch("vllm_ascend.models.minimax_m3.minimax_m3.get_tensor_model_parallel_world_size", return_value=1)
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
        weight=torch.zeros(1),
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


def test_ascendc_index_cache_converts_runtime_shape_to_bbnd() -> None:
    index_key_cache = torch.zeros(4, 128, 128)

    actual = _as_ascendc_index_kv_cache(index_key_cache)

    assert actual.shape == (4, 128, 1, 128)
    assert actual.data_ptr() == index_key_cache.data_ptr()


def test_ascendc_index_cache_unwraps_runtime_tuple() -> None:
    index_key_cache = torch.zeros(4, 128, 128)

    actual = _as_ascendc_index_kv_cache((index_key_cache,))

    assert actual.shape == (4, 128, 1, 128)
    assert actual.data_ptr() == index_key_cache.data_ptr()


def test_ascendc_index_score_forwards_metadata_operands() -> None:
    idx_q = torch.zeros(1, 2, 128)
    index_key_cache = torch.zeros(4, 128, 128)
    block_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32)
    seq_lens = torch.tensor([129], dtype=torch.int32)
    start_loc = torch.tensor([1], dtype=torch.int32)
    causal_mask = torch.zeros(2048, 2048, dtype=torch.int8)
    expected = torch.zeros(2, 1, 4)

    with patch(
        "vllm_ascend.models.minimax_m3.ops.msa_m3_npu.torch.ops._C_ascend.npu_msa_index_score",
        return_value=expected,
        create=True,
    ) as mock_index_score:
        actual = _minimax_m3_index_score(
            idx_q,
            index_key_cache,
            block_table,
            cu_seqlens_q,
            seq_lens,
            start_loc,
            causal_mask,
            init_blocks=2,
            local_blocks=3,
        )

    assert actual is expected
    args, kwargs = mock_index_score.call_args
    assert args[1].shape == (4, 128, 1, 128)
    assert args[3] is start_loc
    assert kwargs["atten_mask"] is causal_mask
    assert "init_blocks" not in kwargs
    assert "local_blocks" not in kwargs


def test_ascendc_index_score_uses_dense_mode_without_mask() -> None:
    idx_q = torch.zeros(1, 2, 128)
    index_key_cache = torch.zeros(4, 128, 128)
    block_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32)
    seq_lens = torch.tensor([128], dtype=torch.int32)
    start_loc = torch.tensor([1], dtype=torch.int32)

    with patch(
        "vllm_ascend.models.minimax_m3.ops.msa_m3_npu.torch.ops._C_ascend.npu_msa_index_score",
        return_value=torch.zeros(2, 1, 4),
        create=True,
    ) as mock_index_score:
        _minimax_m3_index_score(
            idx_q,
            index_key_cache,
            block_table,
            cu_seqlens_q,
            seq_lens,
            start_loc,
            None,
        )

    kwargs = mock_index_score.call_args.kwargs
    assert kwargs["atten_mask"] is None
    assert kwargs["sparse_mode"] == 0


@pytest.mark.parametrize(
    ("max_seq_len", "expected_score_width"),
    [
        (1, 16),
        (128, 16),
        (129, 16),
        (2048, 16),
        (2049, 32),
        (133000, 1040),
    ],
)
def test_ascendc_index_prefill_limits_score_width_for_topk(
    max_seq_len: int,
    expected_score_width: int,
) -> None:
    idx_q = torch.zeros(1, 2, 128)
    index_key_cache = torch.zeros(4, 128, 128)
    block_table = torch.arange(1040, dtype=torch.int32).view(1, 1040)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32)
    seq_lens = torch.tensor([129], dtype=torch.int32)
    context_lens = torch.tensor([128], dtype=torch.int32)
    start_loc = torch.tensor([1], dtype=torch.int32)
    causal_mask = torch.zeros(2048, 2048, dtype=torch.int8)
    score = torch.zeros(2, 1, 1040)
    expected = torch.zeros(2, 1, 2, dtype=torch.int32)

    with (
        patch(
            "vllm_ascend.models.minimax_m3.ops.msa_m3_npu._minimax_m3_index_score",
            return_value=score,
        ) as mock_score,
        patch(
            "vllm_ascend.models.minimax_m3.ops.msa_m3_npu._minimax_m3_index_prefill_topk",
            return_value=expected,
        ) as mock_topk,
    ):
        actual = minimax_m3_index_prefill(
            idx_q,
            index_key_cache,
            block_table,
            cu_seqlens_q,
            seq_lens,
            context_lens,
            start_loc,
            causal_mask,
            max_query_len=1,
            max_seq_len=max_seq_len,
            topk=2,
            init_blocks=1,
            local_blocks=1,
        )

    assert actual is expected
    mock_score.assert_called_once_with(
        idx_q,
        index_key_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        start_loc,
        causal_mask,
        init_blocks=1,
        local_blocks=1,
    )
    mock_topk.assert_called_once()
    topk_args = mock_topk.call_args.args
    limited_score = topk_args[0]
    assert limited_score.shape == (2, 1, expected_score_width)
    assert limited_score.data_ptr() == score.data_ptr()
    assert topk_args[1] is cu_seqlens_q
    assert topk_args[2] is context_lens
    assert topk_args[3:] == (1, 2, 1, 1)


def test_ascendc_index_decode_wraps_score_candidates() -> None:
    idx_q = torch.zeros(1, 2, 128)
    index_key_cache = torch.zeros(4, 128, 128)
    block_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32)
    seq_lens = torch.tensor([129], dtype=torch.int32)
    context_lens = torch.tensor([128], dtype=torch.int32)
    start_loc = torch.tensor([1], dtype=torch.int32)
    causal_mask = torch.zeros(2048, 2048, dtype=torch.int8)
    expected = torch.zeros(2, 1, 2, dtype=torch.int32)
    scores = torch.zeros(2, 1, 2)

    with patch(
        "vllm_ascend.models.minimax_m3.ops.msa_m3_npu._minimax_m3_index_decode",
        return_value=(expected, scores),
    ) as mock_decode:
        actual = minimax_m3_index_decode(
            idx_q,
            index_key_cache,
            block_table,
            cu_seqlens_q,
            seq_lens,
            context_lens,
            start_loc,
            causal_mask,
            topk=2,
            init_blocks=1,
            local_blocks=1,
            decode_query_len=1,
        )

    assert actual is expected
    mock_decode.assert_called_once_with(
        idx_q,
        index_key_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        context_lens,
        start_loc,
        causal_mask,
        topk=2,
        init_blocks=1,
        local_blocks=1,
        decode_query_len=1,
    )


@pytest.mark.parametrize("decode_query_len", [1, 2])
def test_decode_applies_forced_blocks_only_in_topk(decode_query_len: int) -> None:
    idx_q = torch.zeros(decode_query_len, 2, 128)
    index_key_cache = torch.zeros(4, 128, 128)
    block_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, decode_query_len], dtype=torch.int32)
    seq_lens = torch.tensor([128 + decode_query_len], dtype=torch.int32)
    context_lens = torch.tensor([128], dtype=torch.int32)
    start_loc = torch.tensor([1], dtype=torch.int32)
    causal_mask = torch.zeros(2048, 2048, dtype=torch.int8)
    score = torch.zeros(2, decode_query_len, 4)
    expected_indices = torch.zeros(2, decode_query_len, 2, dtype=torch.int32)
    expected_scores = torch.zeros(2, decode_query_len, 2)

    with (
        patch(
            "vllm_ascend.models.minimax_m3.ops.msa_m3_npu._minimax_m3_index_score",
            return_value=score,
        ) as mock_score,
        patch(
            "vllm_ascend.models.minimax_m3.ops.msa_m3_npu._index_score_topk_candidates",
            return_value=(expected_indices, expected_scores),
        ) as mock_topk,
    ):
        actual_indices, actual_scores = _minimax_m3_index_decode(
            idx_q,
            index_key_cache,
            block_table,
            cu_seqlens_q,
            seq_lens,
            context_lens,
            start_loc,
            causal_mask,
            topk=2,
            init_blocks=1,
            local_blocks=1,
            decode_query_len=decode_query_len,
        )

    assert actual_indices is expected_indices
    assert actual_scores is expected_scores
    assert mock_score.call_args.kwargs == {
        "init_blocks": 1,
        "local_blocks": 1,
    }
    assert mock_topk.call_args.kwargs["init_blocks"] == 1
    assert mock_topk.call_args.kwargs["local_blocks"] == 1


@pytest.mark.parametrize(
    ("tp_rank", "expected_blocks"),
    [(0, [0, 1]), (1, [2, 3])],
)
def test_tp_single_token_decode_uses_dense_mode(
    tp_rank: int,
    expected_blocks: list[int],
) -> None:
    class FakeTPGroup:
        world_size = 2
        rank_in_group = tp_rank

        @staticmethod
        def all_gather(tensor: torch.Tensor, dim: int) -> torch.Tensor:
            return torch.cat((tensor, tensor), dim=dim)

    idx_q = torch.zeros(1, 1, 128)
    index_key_cache = torch.zeros(4, 128, 128)
    block_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    context_lens = torch.tensor([511], dtype=torch.int32)
    causal_mask = torch.zeros(2048, 2048, dtype=torch.int8)
    local_topk = torch.zeros(2, 1, 1, dtype=torch.int32)
    local_scores = torch.ones(2, 1, 1)
    builder = _make_indexer_builder(torch.device("cpu"), tp_size=2)
    with patch(
        "vllm_ascend.models.minimax_m3.msa_m3.get_tp_group",
        return_value=FakeTPGroup(),
    ):
        tp_score = builder._build_tp_score_metadata(
            block_table,
            torch.tensor([0, 1], dtype=torch.int32),
            context_lens,
            max_seq_len=512,
            decode_query_len=1,
        )

    with patch(
        "vllm_ascend.models.minimax_m3.ops.msa_m3_npu._minimax_m3_index_decode",
        return_value=(local_topk, local_scores),
    ) as mock_decode:
        minimax_m3_index_tp_block_parallel_decode(
            idx_q,
            index_key_cache,
            tp_score,
            causal_mask,
            topk=1,
            init_blocks=0,
            local_blocks=0,
            tp_group=FakeTPGroup(),
        )

    decode_args = mock_decode.call_args.args
    assert decode_args[7] is None
    assert torch.equal(
        decode_args[2],
        torch.tensor([expected_blocks], dtype=torch.int32),
    )
    assert torch.equal(decode_args[3], torch.tensor([0, 1], dtype=torch.int32))
    assert torch.equal(decode_args[4], torch.tensor([256], dtype=torch.int32))
    assert decode_args[6].dtype == torch.int32


@pytest.mark.parametrize(
    ("tp_rank", "expected_block_table", "expected_k_lens"),
    [
        (0, [[0, 1], [2, 3], [4, 5]], [129, 13, 130]),
        (1, [[1], [3], [5]], [1, 0, 75]),
    ],
)
def test_tp_speculative_decode_uses_packed_causal_halo(
    tp_rank: int,
    expected_block_table: list[list[int]],
    expected_k_lens: list[int],
) -> None:
    class FakeTPGroup:
        world_size = 2
        rank_in_group = tp_rank

        @staticmethod
        def all_gather(tensor: torch.Tensor, dim: int) -> torch.Tensor:
            return torch.cat((tensor, tensor), dim=dim)

    # Request 0's query positions 126, 127 and 128 straddle the TP chunks
    # [0, 128) and [128, 256). Rank 0 includes rank 1's block as a score-only
    # halo, so packed right-down causal visibility is [127, 128, 129]. Its
    # halo score is discarded before TopK, leaving [127, 128, 128] for chunk0.
    # Request 2 starts at 200, so rank 0 uses klen=130: right-down visibility
    # begins at 128 and the entire owned block remains visible for every query.
    idx_q = torch.arange(9, dtype=torch.float32).view(9, 1, 1).expand(-1, -1, 128)
    causal_mask = torch.zeros(2048, 2048, dtype=torch.int8)
    score_width = len(expected_block_table[0])
    causal_score = torch.zeros(2, 9, score_width)
    local_topk = torch.zeros(2, 9, 1, dtype=torch.int32)
    local_scores = torch.ones(2, 9, 1)
    builder = _make_indexer_builder(torch.device("cpu"), tp_size=2)
    common = _create_common_attn_metadata(
        BatchSpec(seq_lens=[129, 13, 203], query_lens=[3, 3, 3]),
        block_size=128,
        device=torch.device("cpu"),
    )
    with (
        patch(
            "vllm_ascend.models.minimax_m3.msa_m3.get_tp_group",
            return_value=FakeTPGroup(),
        ),
        patch(
            "vllm_ascend.models.minimax_m3.msa_m3.split_decodes_and_prefills",
            return_value=(3, 0, 9, 0),
        ),
        patch.object(
            builder,
            "_build_tp_score_metadata",
            wraps=builder._build_tp_score_metadata,
        ) as mock_build_tp_score,
    ):
        metadata = builder.build(0, common)
    assert mock_build_tp_score.call_count == 1
    assert metadata.decode is not None
    assert metadata.decode.tp_score is not None
    tp_score = metadata.decode.tp_score

    with (
        patch(
            "vllm_ascend.models.minimax_m3.ops.msa_m3_npu._minimax_m3_index_score",
            return_value=causal_score,
        ) as mock_score,
        patch(
            "vllm_ascend.models.minimax_m3.ops.msa_m3_npu._index_score_topk_candidates",
            return_value=(local_topk, local_scores),
        ) as mock_topk,
    ):
        minimax_m3_index_tp_block_parallel_decode(
            idx_q,
            torch.zeros(2, 128, 128),
            tp_score,
            causal_mask,
            topk=1,
            init_blocks=0,
            local_blocks=0,
            tp_group=FakeTPGroup(),
        )

    mock_score.assert_called_once()
    score_args = mock_score.call_args.args
    assert score_args[6] is causal_mask
    assert torch.equal(
        score_args[2],
        torch.tensor(expected_block_table, dtype=torch.int32),
    )
    assert torch.equal(score_args[3], torch.tensor([0, 3, 6, 9], dtype=torch.int32))
    assert torch.equal(
        score_args[4],
        torch.tensor(expected_k_lens, dtype=torch.int32),
    )
    assert torch.equal(mock_topk.call_args.args[0], causal_score[..., :1])


def test_tp_score_metadata_reuses_graph_input_storage() -> None:
    """TP metadata consumed by ACLGraph must not point at per-build tensors."""

    builder = _make_indexer_builder(torch.device("cpu"), tp_size=2)
    block_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 4], dtype=torch.int32)
    context_lens = torch.tensor([125], dtype=torch.int32)

    tp_group = SimpleNamespace(rank_in_group=1)
    with patch(
        "vllm_ascend.models.minimax_m3.msa_m3.get_tp_group",
        return_value=tp_group,
    ):
        first = builder._build_tp_score_metadata(
            block_table,
            cu_seqlens_q,
            context_lens,
            max_seq_len=256,
            decode_query_len=4,
        )
        context_lens.copy_(torch.tensor([129], dtype=torch.int32))
        second = builder._build_tp_score_metadata(
            block_table,
            cu_seqlens_q,
            context_lens,
            max_seq_len=256,
            decode_query_len=4,
        )

    assert first.block_table is second.block_table is block_table
    assert first.context_lens is second.context_lens is context_lens
    assert first.cu_seqlens_q is second.cu_seqlens_q is cu_seqlens_q
    assert first.max_block_count == second.max_block_count == 2
    assert first.block_offset == second.block_offset == 1
    assert first.block_count == second.block_count == 1


def test_tp_block_parallel_forwards_global_forced_block_counts() -> None:
    class FakeTPGroup:
        world_size = 2
        rank_in_group = 1

        @staticmethod
        def all_gather(tensor: torch.Tensor, dim: int) -> torch.Tensor:
            return torch.cat((tensor, tensor), dim=dim)

    idx_q = torch.zeros(1, 1, 128)
    causal_mask = torch.zeros(2048, 2048, dtype=torch.int8)
    local_topk = torch.zeros(2, 1, 1, dtype=torch.int32)
    local_scores = torch.ones(2, 1, 1)
    tp_score = MiniMaxM3TPDecodeScoreMetadata(
        block_table=torch.tensor([[4, 5, 6, 7]], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        context_lens=torch.tensor([511], dtype=torch.int32),
        max_block_count=8,
        block_size=128,
        block_offset=4,
        block_count=4,
        decode_query_len=1,
    )

    with patch(
        "vllm_ascend.models.minimax_m3.ops.msa_m3_npu._minimax_m3_index_decode",
        return_value=(local_topk, local_scores),
    ) as mock_decode:
        minimax_m3_index_tp_block_parallel_decode(
            idx_q,
            torch.zeros(8, 128, 128),
            tp_score,
            causal_mask,
            topk=1,
            init_blocks=5,
            local_blocks=6,
            tp_group=FakeTPGroup(),
        )

    assert mock_decode.call_args.kwargs["init_blocks"] == 5
    assert mock_decode.call_args.kwargs["local_blocks"] == 6
    assert mock_decode.call_args.kwargs["block_offset"] == 4


def test_tp_topk_applies_init_blocks_by_global_block_id() -> None:
    score = torch.tensor([[[1.0, 2.0]]])

    indices, scores = _index_score_topk_candidates(
        score,
        context_lens=torch.tensor([256], dtype=torch.int32),
        decode_query_len=1,
        topk=2,
        block_offset=4,
        init_blocks=5,
    )

    assert torch.equal(indices, torch.tensor([[[4, 5]]], dtype=torch.int32))
    assert scores[0, 0, 0] == 1.0e30
    assert scores[0, 0, 1] == 2.0


def test_speculative_topk_masks_future_blocks_before_selection() -> None:
    score = torch.tensor(
        [
            [
                [9.0, 8.0, 7.0, 1.0e20],
                [9.0, 8.0, 7.0, 6.0],
            ]
        ]
    )

    indices, scores = _index_score_topk_candidates(
        score,
        context_lens=torch.tensor([383], dtype=torch.int32),
        decode_query_len=2,
        topk=3,
        local_blocks=1,
    )

    assert set(indices[0, 0].tolist()) == {0, 1, 2}
    assert torch.isfinite(scores[0, 0]).all()
    assert set(indices[0, 1].tolist()) == {0, 1, 3}


@patch("vllm_ascend.models.minimax_m3.msa_m3.get_tp_group")
@patch("vllm_ascend.models.minimax_m3.msa_m3.get_forward_context")
def test_indexer_speculative_decode_uses_tp_block_parallel_path(
    mock_get_forward_context: MagicMock,
    mock_get_tp_group: MagicMock,
) -> None:
    impl = object.__new__(AscendMiniMaxM3IndexerImpl)
    torch.nn.Module.__init__(impl)
    impl.num_index_heads = 1
    impl.index_head_dim = 4
    impl.topk_blocks = 2
    impl.init_blocks = 1
    impl.local_blocks = 1
    impl.index_cache = SimpleNamespace(
        prefix="layer.attn.index_cache",
        kv_cache=torch.zeros(4, 128, 4),
    )

    decode = AscendMiniMaxM3IndexerDecodeMetadata(
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([258], dtype=torch.int32),
        context_lens=torch.tensor([256], dtype=torch.int32),
        block_table=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        start_loc=torch.tensor([2], dtype=torch.int32),
        max_seq_len=258,
        decode_query_len=2,
        tp_score=MiniMaxM3TPDecodeScoreMetadata(
            block_table=torch.tensor([[0, 1, 2]], dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
            context_lens=torch.tensor([256], dtype=torch.int32),
            max_block_count=3,
            block_size=128,
            block_offset=0,
            block_count=2,
            decode_query_len=2,
        ),
    )
    metadata = AscendMiniMaxM3IndexerMetadata(
        seq_lens=decode.seq_lens,
        max_seq_len=decode.max_seq_len,
        slot_mapping=torch.arange(2, dtype=torch.int64),
        causal_mask=torch.zeros(2048, 2048, dtype=torch.int8),
        num_actual_tokens=2,
        num_decodes=1,
        num_decode_tokens=2,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=decode,
    )
    mock_get_forward_context.return_value = SimpleNamespace(
        attn_metadata={impl.index_cache.prefix: metadata},
    )
    mock_get_tp_group.return_value = SimpleNamespace(
        world_size=4,
        rank_in_group=0,
    )
    expected = torch.zeros((1, 2, 2), dtype=torch.int32)

    with patch(
        "vllm_ascend.models.minimax_m3.msa_m3.minimax_m3_index_tp_block_parallel_decode",
        return_value=expected,
    ) as mock_tp_block_parallel:
        actual, prefill = impl.forward(torch.zeros(2, 4))

    assert actual is expected
    assert prefill is None
    mock_tp_block_parallel.assert_called_once()
    call_kwargs = mock_tp_block_parallel.call_args.kwargs
    assert mock_tp_block_parallel.call_args.args[2] is decode.tp_score
    assert mock_tp_block_parallel.call_args.args[3] is metadata.causal_mask
    assert call_kwargs["tp_group"] is mock_get_tp_group.return_value


def test_speculative_decode_candidates_use_per_token_visibility() -> None:
    score = torch.tensor([[[5.0], [7.0]]])

    indices, scores = _index_score_topk_candidates(
        score,
        context_lens=torch.tensor([-1], dtype=torch.int32),
        decode_query_len=2,
        topk=2,
        block_offset=1,
    )

    assert torch.equal(indices, torch.tensor([[[-1, -1], [1, -1]]], dtype=torch.int32))
    assert torch.isneginf(scores[0, 0]).all()
    assert scores[0, 1, 0] == 7.0
    assert torch.isneginf(scores[0, 1, 1])


def test_speculative_decode_candidates_move_local_mask_per_token() -> None:
    score = torch.tensor([[[9.0, 1.0], [9.0, 1.0]]])

    indices, scores = _index_score_topk_candidates(
        score,
        context_lens=torch.tensor([127], dtype=torch.int32),
        decode_query_len=2,
        topk=1,
        local_blocks=1,
    )

    assert torch.equal(indices, torch.tensor([[[0], [1]]], dtype=torch.int32))
    assert torch.equal(scores, torch.full((1, 2, 1), 1.0e29))


def test_speculative_decode_candidates_do_not_overforce_earlier_tp_shard() -> None:
    score = torch.tensor([[[9.0, 8.0], [9.0, 8.0]]])

    indices, scores = _index_score_topk_candidates(
        score,
        context_lens=torch.tensor([510], dtype=torch.int32),
        decode_query_len=2,
        topk=1,
        block_offset=0,
        local_blocks=2,
    )

    assert torch.equal(indices, torch.tensor([[[0], [0]]], dtype=torch.int32))
    assert torch.equal(scores, torch.tensor([[[9.0], [9.0]]]))


@patch("vllm_ascend.models.minimax_m3.msa_m3.minimax_m3_sparse_attn")
@patch("vllm_ascend.models.minimax_m3.msa_m3.minimax_m3_sparse_attn_decode")
@patch("vllm_ascend.models.minimax_m3.msa_m3.get_forward_context")
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
    assert mock_sparse_attn_decode.call_args.kwargs["block_size"] == 128
    assert mock_sparse_attn_prefill.call_args.args[0].shape == (2, 2, 4)


@patch.object(torch.ops._C_ascend, "npu_sparse_attention_score", create=True)
def test_sparse_attn_decode_npu_forwards_runtime_metadata(
    mock_sparse_attention_score: MagicMock,
) -> None:
    q = torch.zeros(4, 2, 4)
    kv_cache = torch.zeros(2, 6, 128, 2, 4)
    topk_idx = torch.tensor(
        [
            [[0, 1], [0, -1], [2, 3], [-1, -1]],
            [[0, 1], [1, -1], [2, 3], [3, -1]],
        ],
        dtype=torch.int32,
    )
    block_table = torch.arange(8, dtype=torch.int32).view(2, 4)
    seq_lens = torch.tensor([129, 385], dtype=torch.int32)
    output = torch.empty_like(q)
    mock_sparse_attention_score.return_value = torch.ones_like(output)

    minimax_m3_sparse_attn_decode_npu(
        q,
        kv_cache,
        topk_idx,
        block_table,
        seq_lens,
        num_kv_heads=2,
        sm_scale=0.5,
        output=output,
        decode_query_len=2,
        block_size=128,
    )

    mock_sparse_attention_score.assert_called_once()
    kwargs = mock_sparse_attention_score.call_args.kwargs
    assert torch.equal(kwargs["actual_seq_lengths"], torch.tensor([2, 2], dtype=torch.int32))
    assert kwargs["actual_seq_lengths_kv"] is seq_lens
    assert torch.equal(
        kwargs["select_num_idx"],
        torch.tensor([[2, 1, 2, 0], [2, 1, 2, 1]], dtype=torch.int32),
    )
    assert kwargs["block_size"] == 128
    assert kwargs["top_k"] == 2
    assert torch.equal(output, torch.ones_like(output))


def test_m3_impls_provide_update_graph_params_protocol():
    """Both M3 impls must satisfy the update_graph_params protocol.

    The implementations are no-ops (replay parameters are refreshed by the
    metadata builders), but the method must exist so the full-graph
    parameter-update backend selection treats them uniformly.
    """
    from vllm_ascend.models.minimax_m3.msa_m3 import (
        AscendMiniMaxM3IndexerImpl,
        AscendMiniMaxM3SparseImpl,
    )

    for impl_cls in (AscendMiniMaxM3IndexerImpl, AscendMiniMaxM3SparseImpl):
        method = getattr(impl_cls, "update_graph_params", None)
        assert callable(method)
        # Signature parity with AscendAttentionBackendImpl.update_graph_params.
        params = inspect.signature(method).parameters
        assert list(params)[:5] == [
            "update_stream",
            "forward_context",
            "num_tokens",
            "vllm_config",
            "speculative_config",
        ]
        assert params["speculative_config"].default is None
