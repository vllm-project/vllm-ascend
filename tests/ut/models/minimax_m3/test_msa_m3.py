# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for MiniMax M3 sparse attention layer wiring in ``msa_m3``."""

from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_hardware_profile
from vllm_ascend.models.minimax_m3 import MiniMaxM3SparseAttention
from vllm_ascend.models.minimax_m3 import msa_m3 as msa_m3_module
from vllm_ascend.models.minimax_m3.minimax_m3 import (
    _cast_for_cache,
    _resolve_layer_kv_cache_dtype,
    _resolve_layer_kv_cache_dtypes,
    _scatter_index_cache,
)
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
    _sparse_proj_quant_type,
    _use_fused_qkv_indexer,
    minimax_m3_sparse_forward,
)
from vllm_ascend.models.minimax_m3.ops import msa_m3_npu as msa_m3_npu_module
from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import (
    _as_ascendc_index_kv_cache,
    _decode_gbsa_block_counts,
    _minimax_m3_index_score,
    _minimax_m3_sparse_attn_kv_gather_q,
    minimax_m3_index_decode,
    minimax_m3_index_prefill,
)
from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import (
    minimax_m3_sparse_attn_decode as minimax_m3_sparse_attn_decode_npu,
)
from vllm_ascend.ops.triton.msa import msa_m3_index_postprocess
from vllm_ascend.utils import AscendDeviceType


@pytest.fixture
def mock_gbsa_ops(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    gbsa_metadata_op = MagicMock()
    gbsa_metadata_op.return_value = torch.zeros(1024, dtype=torch.int32)
    gbsa_attention_op = MagicMock()
    gbsa_ops_module = ModuleType("cann_ops_transformer.ops")
    gbsa_ops_module.generic_block_sparse_attention_metadata = gbsa_metadata_op  # type: ignore[attr-defined]
    gbsa_ops_module.generic_block_sparse_attention = gbsa_attention_op  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cann_ops_transformer.ops", gbsa_ops_module)
    return gbsa_metadata_op, gbsa_attention_op


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_gbsa_prefix_normalization_preserves_valid_selection(dtype: torch.dtype, strided: bool) -> None:
    # count 外的正数属于残留数据，不能因排序进入有效选块集合。
    source = torch.tensor([[[8, 3, 99, 2], [4, -1, 9, -1], [6, 2, 1, 0], [5, 4, 3, 2]]], dtype=dtype)
    counts = torch.tensor([[2, 1, 4, 0]], dtype=torch.int32)
    if strided:
        storage = torch.zeros((*source.shape[:-1], source.shape[-1] * 2), dtype=dtype)
        storage[..., ::2].copy_(source)
        source = storage[..., ::2]
        count_storage = torch.zeros((1, counts.shape[-1] * 2), dtype=counts.dtype)
        count_storage[..., ::2].copy_(counts)
        counts = count_storage[..., ::2]
        assert not source.is_contiguous()
        assert not counts.is_contiguous()
    original, original_counts = source.clone(), counts.clone()

    normalized = msa_m3_npu_module._sort_valid_gbsa_prefix(source, counts)

    expected = torch.tensor([[[3, 8, -1, -1], [4, -1, -1, -1], [0, 1, 2, 6], [-1, -1, -1, -1]]], dtype=torch.int32)
    torch.testing.assert_close(normalized, expected, rtol=0, atol=0)
    assert normalized.dtype == torch.int32
    assert normalized.is_contiguous()
    assert torch.equal(source, original)
    assert torch.equal(counts, original_counts)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("tokens", [16, 64])
def test_gbsa_prefix_normalization_uses_float32_sort(dtype: torch.dtype, tokens: int) -> None:
    source = torch.tensor([1040, 0, 513, 99], dtype=dtype).repeat(1, tokens, 1)
    counts = torch.full((1, tokens), 3, dtype=torch.int32)
    with patch.object(torch, "sort", wraps=torch.sort) as sort:
        normalized = msa_m3_npu_module._sort_valid_gbsa_prefix(source, counts)
    assert sort.call_count == 1
    assert sort.call_args.args[0].dtype == torch.float32
    expected = torch.tensor([0, 513, 1040, -1], dtype=torch.int32).repeat(1, tokens, 1)
    torch.testing.assert_close(normalized, expected, rtol=0, atol=0)
    assert normalized.is_contiguous()


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
) -> AscendCommonAttentionMetadata:
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

    return AscendCommonAttentionMetadata(
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
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "layer0.attn": SimpleNamespace(num_heads=4, num_kv_heads=2, head_dim=128, topk_blocks=16),
            }
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            tensor_parallel_size=1,
        ),
    )


def _make_sparse_builder(
    device: torch.device, kv_cache_dtype: torch.dtype = torch.bfloat16
) -> AscendMiniMaxM3SparseMetadataBuilder:
    vllm_config = _make_vllm_config()
    vllm_config.speculative_config = SimpleNamespace(num_speculative_tokens=7, parallel_drafting=False)
    spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=128,
        head_size_v=128,
        dtype=kv_cache_dtype,
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


@patch(
    "vllm_ascend.models.minimax_m3.minimax_m3.kv_cache_dtype_str_to_dtype",
    return_value=torch.bfloat16,
)
def test_m3_kv_cache_dtype_skip_layers_keeps_gqa_bf16_and_msa_fp8(
    mock_kv_dtype: MagicMock,
) -> None:
    cache_config = SimpleNamespace(
        cache_dtype="fp8",
        kv_cache_dtype_skip_layers=["0", "1", "2"],
    )
    model_config = SimpleNamespace(dtype=torch.bfloat16)

    assert _resolve_layer_kv_cache_dtype(cache_config, "model.layers.0.self_attn") == "auto"
    assert _resolve_layer_kv_cache_dtype(cache_config, "model.layers.2.self_attn") == "auto"
    assert _resolve_layer_kv_cache_dtype(cache_config, "model.layers.3.self_attn") == "fp8"
    assert _resolve_layer_kv_cache_dtype(cache_config, "model.layers.59.self_attn") == "fp8"
    assert _resolve_layer_kv_cache_dtypes(
        cache_config,
        "model.layers.0.self_attn",
        model_config,
    ) == ("auto", torch.bfloat16)
    assert _resolve_layer_kv_cache_dtypes(
        cache_config,
        "model.layers.2.self_attn",
        model_config,
    ) == ("auto", torch.bfloat16)
    assert _resolve_layer_kv_cache_dtypes(
        cache_config,
        "model.layers.3.self_attn",
        model_config,
    ) == ("fp8", torch.float8_e4m3fn)
    assert _resolve_layer_kv_cache_dtypes(
        cache_config,
        "model.layers.59.self_attn",
        model_config,
    ) == ("fp8", torch.float8_e4m3fn)
    assert mock_kv_dtype.call_count == 2


def test_m3_fixed_scale_fp8_cache_cast_clamps_e4m3_range() -> None:
    source = torch.tensor([-500.0, -1.0, 1.0, 500.0], dtype=torch.float32)
    cache = torch.empty(1, dtype=torch.float8_e4m3fn)

    actual = _cast_for_cache(source, cache)

    assert actual.dtype == torch.float8_e4m3fn
    assert torch.equal(actual.float(), torch.tensor([-448.0, -1.0, 1.0, 448.0]))


@patch("vllm_ascend.models.minimax_m3.msa_m3.get_current_vllm_config")
def test_m3_indexer_cache_spec_follows_resolved_fp8_dtype(mock_get_vllm_config: MagicMock) -> None:
    mock_get_vllm_config.return_value = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=128))
    index_cache = AscendMiniMaxM3IndexerCache(
        head_dim=128,
        prefix="model.layers.3.self_attn.attn.index_cache",
        kv_cache_dtype="fp8",
        kv_cache_torch_dtype=torch.float8_e4m3fn,
    )

    spec = index_cache.get_kv_cache_spec(vllm_config)

    assert spec.dtype == torch.float8_e4m3fn
    assert spec.cache_dtype_str == "fp8"


def test_a5_decode_topk_keeps_fused_postprocess_fast_path() -> None:
    source_path = Path(msa_m3_module.__file__).with_name("ops") / "msa_m3_triton_a5.py"
    source = source_path.read_text(encoding="utf-8")
    decode_source = source[source.index("def minimax_m3_index_decode(") :]

    assert "_decode_index_score_pad_tail_kernel" not in source
    assert "score = torch.full(" in decode_source
    assert "_, topk_idx_raw = torch.topk(score, k=topk" in decode_source
    assert "_index_topk_postprocess_kernel" in decode_source
    assert "return topk_idx, select_num_idx" in decode_source


@pytest.mark.parametrize("sorted_output", [False, True])
@pytest.mark.parametrize("provided_output", [False, True])
def test_a5_decode_postprocess_dispatch_preserves_output_buffer(sorted_output, provided_output) -> None:
    from vllm_ascend.models.minimax_m3.ops import msa_m3_triton_a5 as a5_ops

    query_len, heads, topk, requests = 4, 2, 16, 3
    tokens = requests * query_len
    query = torch.zeros(tokens, heads, 128, dtype=torch.bfloat16)
    cache = torch.zeros(8, 128, 128, dtype=torch.bfloat16)
    table = torch.zeros(requests, 8, dtype=torch.int32)
    lengths = torch.tensor([129, 257, 0], dtype=torch.int32)
    output = torch.full((heads, tokens + 4, topk), -99, dtype=torch.int32) if provided_output else None
    score_kernel, legacy_post, sorted_post = MagicMock(), MagicMock(), MagicMock()

    def populate_scores(*args, **kwargs):
        scores = args[2]
        scores.copy_(torch.arange(scores.shape[-1], dtype=torch.float32))

    score_kernel.__getitem__.return_value.side_effect = populate_scores
    with (
        patch.object(a5_ops, "_decode_qk_score_kernel", score_kernel),
        patch.object(a5_ops, "_index_topk_postprocess_kernel", legacy_post),
        patch.object(a5_ops, "_index_topk_sorted_postprocess_kernel", sorted_post),
        patch.object(a5_ops, "_decode_qk_score_target_programs", return_value=32),
    ):
        indices, counts = a5_ops.minimax_m3_index_decode(
            query,
            cache,
            table,
            lengths,
            1024,
            topk,
            0,
            1,
            heads,
            query_len,
            out=output,
            sort_indices=sorted_output,
        )
    assert indices.shape == (heads, tokens, topk)
    assert counts.shape == (heads, tokens)
    assert indices.dtype == counts.dtype == torch.int32
    if output is not None:
        assert indices.data_ptr() == output.data_ptr()
        assert indices.stride() == output.stride()
        assert torch.all(output[:, tokens:] == -99)
    if sorted_output:
        legacy_post.__getitem__.assert_not_called()
        sorted_post.__getitem__.assert_called_once_with((3, heads))
        launch = sorted_post.__getitem__.return_value.call_args
        assert launch.args[1] is indices
        assert launch.args[10:13] == indices.stride()
        assert launch.kwargs["ROWS"] == 4
        assert launch.kwargs["TILE"] == 32
    else:
        sorted_post.__getitem__.assert_not_called()
        legacy_post.__getitem__.assert_called_once_with((tokens, heads))


def test_a5_sorted_decode_empty_batch_skips_kernels() -> None:
    from vllm_ascend.models.minimax_m3.ops import msa_m3_triton_a5 as a5_ops

    with patch.object(a5_ops, "_decode_qk_score_kernel") as score:
        indices, counts = a5_ops.minimax_m3_index_decode(
            torch.empty(0, 1, 128),
            torch.empty(1, 128, 128),
            torch.empty(0, 1, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            128,
            16,
            0,
            1,
            1,
            1,
            sort_indices=True,
        )
    score.__getitem__.assert_not_called()
    assert indices.shape == (1, 0, 16)
    assert counts.shape == (1, 0)


@pytest.mark.parametrize("invalid", ["dtype", "capacity", "fp32_range"])
def test_a5_sorted_decode_rejects_invalid_buffers_before_launch(invalid) -> None:
    from vllm_ascend.models.minimax_m3.ops import msa_m3_triton_a5 as a5_ops

    output = torch.empty(1, 1, 16, dtype=torch.int64 if invalid == "dtype" else torch.int32)
    if invalid == "capacity":
        output = output[:, :0]
    maximum = ((1 << 24) + 1) * 128 if invalid == "fp32_range" else 128
    with patch.object(a5_ops, "_decode_qk_score_kernel") as score, pytest.raises(ValueError):
        a5_ops.minimax_m3_index_decode(
            torch.empty(1, 1, 128),
            torch.empty(1, 128, 128),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            maximum,
            16,
            0,
            1,
            1,
            1,
            out=output,
            sort_indices=True,
        )
    score.__getitem__.assert_not_called()


def test_a3_decode_topk_uses_fused_prepare_and_postprocess() -> None:
    source_path = Path(msa_m3_index_postprocess.__file__)
    source = source_path.read_text(encoding="utf-8")

    assert "def _ascendc_decode_score_prepare_kernel(" in source
    assert "_, raw_indices = torch.topk(prepared, k=topk" in source
    assert "def _ascendc_decode_topk_postprocess_kernel(" in source
    assert "return indices, counts" in source


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
def test_sparse_metadata_builder(batch_spec: BatchSpec, mock_gbsa_ops) -> None:
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
        assert metadata.prefill.total_kv_blocks == 5
        assert metadata.prefill.max_kv_blocks == 3
        assert metadata.prefill.cu_seqlens_q.data_ptr() == builder.prefill_cu_seqlens_q_buffer.data_ptr()
        rebuilt = builder.build(0, common)
        assert rebuilt.prefill is not None
        assert rebuilt.prefill.cu_seqlens_q.data_ptr() == metadata.prefill.cu_seqlens_q.data_ptr()


@pytest.mark.parametrize("query_len", [1, 2, 4, 8])
@pytest.mark.parametrize("batch_size", [0, 1, 10, 128, 512])
def test_sparse_decode_prefix_matches_cumsum(batch_size: int, query_len: int) -> None:
    builder = _make_sparse_builder(torch.device("cpu"))
    prefix = builder._get_decode_cu_seqlens_q(batch_size, query_len)
    expected = torch.cat(
        [torch.zeros(1, dtype=torch.int64), torch.full((batch_size,), query_len, dtype=torch.int64).cumsum(0)]
    )
    assert prefix.dtype == torch.int64
    torch.testing.assert_close(prefix, expected, rtol=0, atol=0)


def test_sparse_decode_prefix_keeps_captured_buffers_alive() -> None:
    builder = _make_sparse_builder(torch.device("cpu"))
    captured = builder._get_decode_cu_seqlens_q(10, 4)
    expected = captured.clone()
    # Other graph sizes and query lengths must not overwrite captured storage.
    for batch_size in (1, 128, builder.context_len_buffer.numel()):
        assert builder._get_decode_cu_seqlens_q(batch_size, 4).data_ptr() == captured.data_ptr()
        assert builder._get_decode_cu_seqlens_q(batch_size, 1).data_ptr() != captured.data_ptr()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)


@pytest.mark.parametrize("batch_size,query_len", [(-1, 4), (8193, 4), (1, 0), (1, 9)])
def test_sparse_decode_prefix_rejects_invalid_dimensions(batch_size: int, query_len: int) -> None:
    builder = _make_sparse_builder(torch.device("cpu"))
    with pytest.raises(ValueError, match="Invalid uniform decode"):
        builder._get_decode_cu_seqlens_q(batch_size, query_len)


@pytest.mark.parametrize("query_len", [0, 1, 4])
def test_sparse_metadata_builder_forwards_uniform_decode_prefix(
    monkeypatch: pytest.MonkeyPatch, query_len: int, mock_gbsa_ops
) -> None:
    device = torch.device("cpu")
    builder = _make_sparse_builder(device)
    batch_size = 2
    common = _create_common_attn_metadata(
        BatchSpec(seq_lens=[65536] * batch_size, query_lens=[query_len] * batch_size), 128, device
    )
    monkeypatch.setattr(
        msa_m3_module, "split_decodes_and_prefills", lambda *args, **kwargs: (batch_size, 0, batch_size * query_len, 0)
    )
    metadata = builder.build(0, common)
    assert metadata.decode is not None
    if query_len == 0:
        assert metadata.num_decodes == 0
        assert metadata.decode.cu_seqlens_q is None
        assert builder.decode_cu_seqlens_q_buffers  # Buffers already exist before the first build.
    else:
        expected = common.query_start_loc.to(torch.int64)
        torch.testing.assert_close(metadata.decode.cu_seqlens_q, expected, rtol=0, atol=0)
        assert builder.build(0, common).decode.cu_seqlens_q.data_ptr() == metadata.decode.cu_seqlens_q.data_ptr()


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
    assert metadata.causal_mask is not None
    assert metadata.causal_mask.shape == (2048, 2048)
    if batch_spec.name == "decode_only":
        assert metadata.num_decodes == batch_spec.batch_size
        assert metadata.prefill is None
        assert metadata.decode is not None
        assert metadata.decode.start_loc is not None
        assert metadata.decode.context_lens is not None
        assert metadata.decode.start_loc.dtype == torch.int32
        assert torch.equal(metadata.decode.start_loc, metadata.decode.context_lens // 128)
        assert builder.decode_start_loc_buffer is not None
        assert metadata.decode.start_loc.data_ptr() == builder.decode_start_loc_buffer.data_ptr()
        rebuilt = builder.build(0, common)
        assert rebuilt.decode is not None
        assert rebuilt.decode.start_loc is not None
        assert rebuilt.decode.start_loc.data_ptr() == metadata.decode.start_loc.data_ptr()
    else:
        assert metadata.num_prefills == batch_spec.batch_size
        assert metadata.decode is None
        assert metadata.prefill is not None
        assert metadata.prefill.start_loc is not None
        assert torch.equal(
            metadata.prefill.start_loc,
            metadata.prefill.context_lens // 128,
        )
        assert builder.prefill_start_loc_buffer is not None
        assert metadata.prefill.start_loc.data_ptr() == builder.prefill_start_loc_buffer.data_ptr()
        assert metadata.prefill.cu_seqlens_q.data_ptr() == builder.prefill_cu_seqlens_q_buffer.data_ptr()
        rebuilt = builder.build(0, common)
        assert rebuilt.prefill is not None
        assert rebuilt.prefill.start_loc is not None
        assert rebuilt.prefill.start_loc.data_ptr() == metadata.prefill.start_loc.data_ptr()
        assert rebuilt.prefill.cu_seqlens_q.data_ptr() == metadata.prefill.cu_seqlens_q.data_ptr()


def test_sparse_metadata_builder_fia_padded_dummy_request(mock_gbsa_ops) -> None:
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

    padded_common = AscendCommonAttentionMetadata(
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


def test_indexer_metadata_builder_trims_graph_padded_spec_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(msa_m3_module, "_USE_ASCENDC_INDEX_SCORE_DECODE", True)
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
            "vllm_ascend.models.minimax_m3.msa_m3.split_decodes_and_prefills",
            return_value=(16, 0, 60, 0),
        ),
    ):
        first = builder.build(0, common)
        second = builder.build(0, common)

    assert first.num_decodes == 15
    assert first.num_decode_tokens == 60
    assert first.decode is not None
    assert first.decode.cu_seqlens_q is not None
    assert first.decode.context_lens is not None
    assert first.decode.cu_seqlens_q.shape == (16,)
    assert first.decode.cu_seqlens_q[-1] == 60
    assert first.decode.block_table.shape[0] == 15
    assert first.decode.context_lens.shape == (15,)
    assert second.decode is not None
    assert first.decode.block_table.data_ptr() == second.decode.block_table.data_ptr()
    assert second.decode.context_lens is not None
    assert second.decode.cu_seqlens_q is not None
    assert first.decode.context_lens.data_ptr() == second.decode.context_lens.data_ptr()
    assert first.decode.cu_seqlens_q.data_ptr() == second.decode.cu_seqlens_q.data_ptr()


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


def test_a5_index_score_uses_ascendc_prefill_and_triton_decode() -> None:
    module_source = inspect.getsource(msa_m3_module)
    a5_branch_start = module_source.index("if not _USE_ASCENDC_INDEX_SCORE_DECODE:")
    a5_branch_end = module_source.index("\n\ndef _active_decode_num_reqs", a5_branch_start)
    import_branches = module_source[a5_branch_start:a5_branch_end]

    assert msa_m3_module._USE_ASCENDC_INDEX_SCORE_PREFILL is True
    assert msa_m3_module._USE_ASCENDC_INDEX_SCORE_DECODE is (
        msa_m3_module.get_current_hardware_profile().device_adaptor_family != DeviceAdaptorFamily.FP8_OPTIMIZED
    )
    assert import_branches.count("minimax_m3_index_decode") == 1
    assert "msa_m3_triton_a5" in import_branches
    assert "msa_m3_triton" not in import_branches.replace("msa_m3_triton_a5", "")
    assert "_USE_ASCENDC_INDEX_SCORE_PREFILL = True" in module_source
    assert "get_ascend_device_type" not in module_source


def _make_mrv2_padding_context(
    *,
    is_padding: bool = True,
    num_tokens: int = 1,
    tokens_across_dp: tuple[int, ...] = (370, 1),
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
) -> SimpleNamespace:
    return SimpleNamespace(
        is_padding=torch.full((num_tokens,), is_padding, dtype=torch.bool),
        dp_metadata=SimpleNamespace(
            num_tokens_across_dp_cpu=torch.tensor(tokens_across_dp),
        ),
        additional_kwargs={},
        cudagraph_runtime_mode=cudagraph_mode,
    )


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
    impl.index_cache = cast(
        AscendMiniMaxM3IndexerCache,
        SimpleNamespace(
            prefix="layer.attn.index_cache",
            kv_cache=torch.zeros(4, 128, 4),
        ),
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
    expected_select_num_idx = torch.ones(1, 1, dtype=torch.int32)

    with (
        patch.object(msa_m3_module, "_USE_ASCENDC_INDEX_SCORE_DECODE", False),
        patch.object(
            msa_m3_module,
            "get_forward_context",
            return_value=SimpleNamespace(
                attn_metadata={impl.index_cache.prefix: metadata},
            ),
        ),
        patch.object(
            msa_m3_module,
            "minimax_m3_index_decode",
            return_value=(expected, expected_select_num_idx),
            create=True,
        ) as mock_decode,
    ):
        actual, prefill, select_num_idx = impl.forward(torch.zeros(1, 4))

    assert actual is expected
    assert prefill is None
    assert select_num_idx is expected_select_num_idx
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
    assert mock_decode.call_args.kwargs == {"sm_scale": impl.scale, "sort_indices": True}


def test_a3_indexer_forward_receives_fused_select_counts() -> None:
    impl = object.__new__(AscendMiniMaxM3IndexerImpl)
    torch.nn.Module.__init__(impl)
    impl.num_index_heads = 1
    impl.num_kv_heads = 1
    impl.index_head_dim = 4
    impl.block_size = 128
    impl.topk_blocks = 2
    impl.init_blocks = 1
    impl.local_blocks = 1
    impl.scale = 0.5
    impl.index_cache = cast(
        AscendMiniMaxM3IndexerCache,
        SimpleNamespace(
            prefix="layer.attn.index_cache",
            kv_cache=torch.zeros(4, 128, 4),
        ),
    )
    decode = AscendMiniMaxM3IndexerDecodeMetadata(
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([129], dtype=torch.int32),
        context_lens=torch.tensor([128], dtype=torch.int32),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        start_loc=torch.tensor([1], dtype=torch.int32),
        max_seq_len=129,
        decode_query_len=1,
    )
    metadata = AscendMiniMaxM3IndexerMetadata(
        seq_lens=decode.seq_lens,
        max_seq_len=decode.max_seq_len,
        slot_mapping=torch.zeros(1, dtype=torch.int64),
        causal_mask=torch.zeros(1, 1, dtype=torch.bool),
        num_actual_tokens=1,
        num_decodes=1,
        num_decode_tokens=1,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=decode,
    )
    expected = torch.zeros(1, 1, 2, dtype=torch.int32)
    expected_select_num_idx = torch.ones(1, 1, dtype=torch.int32)

    with (
        patch.object(msa_m3_module, "_USE_ASCENDC_INDEX_SCORE_DECODE", True),
        patch.object(
            msa_m3_module,
            "get_forward_context",
            return_value=SimpleNamespace(
                attn_metadata={impl.index_cache.prefix: metadata},
            ),
        ),
        patch.object(
            msa_m3_module,
            "minimax_m3_index_decode_ascendc",
            return_value=(expected, expected_select_num_idx),
        ),
    ):
        actual, prefill, select_num_idx = impl.forward(torch.zeros(1, 4))

    assert actual is expected
    assert prefill is None
    assert select_num_idx is expected_select_num_idx


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
    impl.index_cache = cast(
        AscendMiniMaxM3IndexerCache,
        SimpleNamespace(
            prefix="layer.attn.index_cache",
            kv_cache=torch.zeros(4, 128, 4),
        ),
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
        patch.object(msa_m3_module, "_USE_ASCENDC_INDEX_SCORE_PREFILL", False),
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
        decode, actual, select_num_idx = impl.forward(torch.zeros(1, 4))

    assert decode is None
    assert actual is expected
    assert select_num_idx is None
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
        "npu_scatter_nd_update_sk",
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


def test_cann_index_score_forwards_metadata_operands() -> None:
    idx_q = torch.zeros(1, 2, 128)
    index_key_cache = torch.zeros(4, 128, 128)
    block_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32)
    seq_lens = torch.tensor([129], dtype=torch.int32)
    start_loc = torch.tensor([1], dtype=torch.int32)
    causal_mask = torch.zeros(2048, 2048, dtype=torch.int8)
    expected = torch.zeros(2, 1, 4)

    mock_index_score = MagicMock(return_value=expected)
    with patch.dict("sys.modules", {"cann_ops_transformer.ops": SimpleNamespace(msa_index_score=mock_index_score)}):
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
    mock_index_score.assert_called_once()
    args, kwargs = mock_index_score.call_args
    assert args[0] is idx_q
    assert args[1].shape == (4, 128, 1, 128)
    assert args[1].data_ptr() == index_key_cache.data_ptr()
    assert args[2] is start_loc
    assert kwargs["block_table"] is block_table
    assert kwargs["atten_mask"] is causal_mask
    assert kwargs["actual_seq_qlen"] is cu_seqlens_q
    assert kwargs["actual_seq_klen"] is seq_lens
    assert kwargs["layout_key"] == "BBND"
    assert kwargs["sparse_mode"] == 3
    assert kwargs["init_blocks"] == 0
    assert kwargs["local_blocks"] == 0


def test_cann_index_score_casts_query_to_fp8_cache_dtype() -> None:
    idx_q = torch.ones(1, 2, 128, dtype=torch.bfloat16)
    index_key_cache = torch.zeros(4, 128, 128, dtype=torch.float8_e4m3fn)
    block_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32)
    seq_lens = torch.tensor([129], dtype=torch.int32)
    start_loc = torch.tensor([1], dtype=torch.int32)
    expected = torch.zeros(2, 1, 4)

    mock_index_score = MagicMock(return_value=expected)
    with patch.dict("sys.modules", {"cann_ops_transformer.ops": SimpleNamespace(msa_index_score=mock_index_score)}):
        actual = _minimax_m3_index_score(
            idx_q,
            index_key_cache,
            block_table,
            cu_seqlens_q,
            seq_lens,
            start_loc,
            None,
        )

    assert actual is expected
    args, kwargs = mock_index_score.call_args
    assert args[0].dtype == torch.float8_e4m3fn
    assert args[1].dtype == torch.float8_e4m3fn
    assert "scale" not in kwargs


def test_cann_index_score_uses_dense_mode_without_mask() -> None:
    idx_q = torch.zeros(1, 2, 128)
    index_key_cache = torch.zeros(4, 128, 128)
    block_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32)
    seq_lens = torch.tensor([128], dtype=torch.int32)
    start_loc = torch.tensor([1], dtype=torch.int32)

    mock_index_score = MagicMock(return_value=torch.zeros(2, 1, 4))
    with patch.dict("sys.modules", {"cann_ops_transformer.ops": SimpleNamespace(msa_index_score=mock_index_score)}):
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
    assert kwargs["init_blocks"] == 0
    assert kwargs["local_blocks"] == 0


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


def test_ascendc_index_decode_returns_fused_select_counts() -> None:
    idx_q = torch.zeros(1, 2, 128)
    index_key_cache = torch.zeros(4, 128, 128)
    block_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32)
    seq_lens = torch.tensor([129], dtype=torch.int32)
    context_lens = torch.tensor([128], dtype=torch.int32)
    start_loc = torch.tensor([1], dtype=torch.int32)
    causal_mask = torch.zeros(2048, 2048, dtype=torch.int8)
    score = torch.zeros(2, 1, 4)
    expected = torch.zeros(2, 1, 2, dtype=torch.int32)
    expected_counts = torch.ones(2, 1, dtype=torch.int32)

    with (
        patch(
            "vllm_ascend.models.minimax_m3.ops.msa_m3_npu._minimax_m3_index_score",
            return_value=score,
        ) as mock_score,
        patch(
            "vllm_ascend.models.minimax_m3.ops.msa_m3_npu.index_decode_topk_fused",
            return_value=(expected, expected_counts),
        ) as mock_fused_topk,
    ):
        actual, counts = minimax_m3_index_decode(
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
    assert counts is expected_counts
    mock_score.assert_called_once_with(
        idx_q,
        index_key_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        start_loc,
        causal_mask,
    )
    mock_fused_topk.assert_called_once_with(
        score,
        context_lens,
        1,
        2,
        block_table.shape[-1],
        init_blocks=1,
        local_blocks=1,
        sort_indices=False,
    )


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
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int64),
        ),
        prefill=AscendMiniMaxM3SparsePrefillMetadata(
            cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 7], dtype=torch.int32),
            seq_lens=torch.tensor([7], dtype=torch.int32),
            context_lens=torch.tensor([5], dtype=torch.int32),
            block_table=torch.tensor([[2, 3]], dtype=torch.int32),
            max_query_len=2,
            max_seq_len=7,
            total_kv_blocks=1,
            max_kv_blocks=1,
        ),
    )
    mock_get_forward_context.return_value = SimpleNamespace(attn_metadata={"layer.attn": metadata})
    layer = SimpleNamespace(layer_name="layer.attn")
    query = torch.arange(24, dtype=torch.float32).view(3, 8)
    kv_cache = torch.zeros(2, 4, 128, 2, 4)
    output = torch.zeros_like(query)
    decode_topk = torch.tensor([[0, 1]], dtype=torch.int32)
    prefill_topk = torch.tensor([[2, 3]], dtype=torch.int32)
    decode_select_num_idx = torch.tensor([[2]], dtype=torch.int32)

    result = impl.forward(
        layer,
        query,
        kv_cache,
        (decode_topk, prefill_topk, decode_select_num_idx),
        output,
    )

    assert result is output
    mock_sparse_attn_decode.assert_called_once()
    mock_sparse_attn_prefill.assert_called_once()
    assert mock_sparse_attn_decode.call_args.args[0].shape == (1, 2, 4)
    assert mock_sparse_attn_decode.call_args.kwargs["block_size"] == 128
    assert mock_sparse_attn_decode.call_args.kwargs["select_num_idx"] is decode_select_num_idx
    assert mock_sparse_attn_decode.call_args.kwargs["cu_seqlens_q"] is metadata.decode.cu_seqlens_q
    assert mock_sparse_attn_decode.call_args.kwargs["indices_are_sorted"] is True
    assert "query_lens" not in mock_sparse_attn_decode.call_args.kwargs
    assert mock_sparse_attn_prefill.call_args.args[0].shape == (2, 2, 4)
    assert mock_sparse_attn_prefill.call_args.kwargs["total_kv_blocks"] == 1
    assert mock_sparse_attn_prefill.call_args.kwargs["max_kv_blocks"] == 1


@pytest.mark.parametrize(("supports_fp8", "expected_inner_precise"), [(False, 0), (True, 1)])
@patch.object(
    torch.ops._C_ascend,
    "npu_sparse_attention_score_prefill",
    create=True,
)
@patch("vllm_ascend.models.minimax_m3.ops.msa_m3_npu._npu_k2q_csr")
def test_sparse_attn_prefill_kv_gather_q_forwards_csr_metadata(
    mock_k2q_csr: MagicMock,
    mock_sparse_attention_score_prefill: MagicMock,
    supports_fp8: bool,
    expected_inner_precise: int,
) -> None:
    q = torch.zeros(3, 4, 4, dtype=torch.bfloat16)
    kv_cache = torch.zeros(2, 8, 128, 2, 4, dtype=torch.bfloat16)
    topk_idx = torch.tensor(
        [
            [[0, 1], [0, -1], [2, 1]],
            [[0, 1], [1, -1], [2, 0]],
        ],
        dtype=torch.int32,
    )
    block_table = torch.arange(8, dtype=torch.int32).view(2, 4)
    original_topk_idx = topk_idx.clone()
    cu_seqlens_q = torch.tensor([0, 1, 3], dtype=torch.int32)
    seq_lens = torch.tensor([129, 257], dtype=torch.int32)
    output = torch.empty_like(q)
    k2q_row_ptr = torch.zeros(2, 6, dtype=torch.int32)
    k2q_q_indices = torch.zeros(2, 6, dtype=torch.int32)
    k2q_slot_indices = torch.zeros(2, 6, dtype=torch.int32)
    mock_k2q_csr.return_value = (k2q_row_ptr, k2q_q_indices, k2q_slot_indices)
    mock_sparse_attention_score_prefill.return_value = torch.ones_like(output)

    _minimax_m3_sparse_attn_kv_gather_q(
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens,
        num_kv_heads=2,
        sm_scale=0.5,
        output=output,
        block_size=128,
        total_kv_blocks=5,
        max_kv_blocks=3,
        supports_fp8=supports_fp8,
    )

    mock_k2q_csr.assert_called_once()
    k2q_kwargs = mock_k2q_csr.call_args.kwargs
    assert k2q_kwargs["order_method"] == 1
    assert k2q_kwargs["total_rows"] == 5
    assert k2q_kwargs["max_kv"] == 3
    assert k2q_kwargs["use_simt"] == 0
    assert k2q_kwargs["q_global_offset"] is True
    assert mock_k2q_csr.call_args.args[0] is topk_idx
    assert torch.equal(topk_idx, original_topk_idx)

    mock_sparse_attention_score_prefill.assert_called_once()
    args = mock_sparse_attention_score_prefill.call_args.args
    kwargs = mock_sparse_attention_score_prefill.call_args.kwargs
    assert args[0] is q
    assert args[3] is block_table
    assert args[4] is k2q_row_ptr
    assert args[7:12] == (2, 0.5, 128, 2, expected_inner_precise)
    assert torch.equal(kwargs["actual_seq_lengths"], torch.tensor([1, 2], dtype=torch.int32))
    assert torch.equal(kwargs["actual_seq_lengths_kv"], seq_lens)
    assert torch.equal(output, torch.ones_like(output))


@pytest.mark.parametrize("reuse_prefix", [False, True])
@pytest.mark.parametrize("device_type", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5])
def test_sparse_attn_decode_npu_forwards_runtime_metadata(
    mock_gbsa_ops: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
    reuse_prefix: bool,
    device_type: AscendDeviceType,
) -> None:
    monkeypatch.setattr(msa_m3_npu_module, "get_current_hardware_profile", lambda: get_hardware_profile(device_type))
    q = torch.zeros(4, 2, 4, dtype=torch.bfloat16)
    kv_cache = torch.zeros(2, 6, 128, 2, 4, dtype=torch.bfloat16)
    topk_idx = torch.tensor(
        [
            [[1, 0], [0, -1], [3, 2], [-1, -1]],
            [[1, 0], [1, -1], [3, 2], [3, -1]],
        ],
        dtype=torch.int32,
    )
    original_topk = topk_idx.clone()
    expected_topk = torch.tensor(
        [
            [[0, 1], [0, -1], [2, 3], [-1, -1]],
            [[0, 1], [1, -1], [2, 3], [3, -1]],
        ],
        dtype=torch.int32,
    )
    block_table = torch.arange(8, dtype=torch.int32).view(2, 4)
    seq_lens = torch.tensor([129, 385], dtype=torch.int32)
    output = torch.empty_like(q)
    metadata = torch.empty(1024, dtype=torch.int32)
    gbsa_metadata_op, gbsa_attention_op = mock_gbsa_ops
    gbsa_metadata_op.return_value = metadata
    gbsa_attention_op.return_value = (torch.ones_like(output), torch.empty(0))

    prefix = torch.tensor([0, 2, 4], dtype=torch.int64) if reuse_prefix else None
    if reuse_prefix:
        monkeypatch.setattr(
            msa_m3_npu_module,
            "_build_cu_seqlens_q",
            MagicMock(side_effect=AssertionError("The per-layer prefix must not be rebuilt")),
        )

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
        cu_seqlens_q=prefix,
    )

    gbsa_metadata_op.assert_called_once()
    if reuse_prefix:
        assert gbsa_metadata_op.call_args.kwargs["cu_seqlens_q"] is prefix
    metadata_args = gbsa_metadata_op.call_args.args
    metadata_kwargs = gbsa_metadata_op.call_args.kwargs
    assert torch.equal(metadata_args[0], expected_topk)
    assert torch.equal(topk_idx, original_topk)
    assert torch.equal(
        metadata_args[1],
        torch.tensor([[2, 1, 2, 0], [2, 1, 2, 1]], dtype=torch.int32),
    )
    assert metadata_args[2:6] == (2, 2, 4, [1, 128])
    assert torch.equal(
        metadata_kwargs["cu_seqlens_q"],
        torch.tensor([0, 2, 4], dtype=torch.int64),
    )
    assert torch.equal(metadata_kwargs["seqused_kv"], seq_lens)
    assert metadata_kwargs["max_seqlen_q"] == 2
    assert metadata_kwargs["layout_q"] == "TND"
    assert metadata_kwargs["layout_kv"] == "PA_BBND"
    assert metadata_kwargs["mask_mode"] == 1
    assert metadata_kwargs["quant_mode"] == 0
    assert metadata_kwargs["softmax_precision"] == 0
    assert "is_packed_gqa" not in metadata_kwargs

    gbsa_attention_op.assert_called_once()
    args = gbsa_attention_op.call_args.args
    kwargs = gbsa_attention_op.call_args.kwargs
    assert args[0] is q
    assert torch.equal(args[3], expected_topk)
    assert torch.equal(args[4], metadata_args[1])
    assert args[5] == [1, 128]
    assert kwargs["metadata"] is metadata
    assert torch.equal(kwargs["cu_seqlens_q"], metadata_kwargs["cu_seqlens_q"])
    assert torch.equal(kwargs["seqused_kv"], seq_lens)
    assert torch.equal(kwargs["block_table"], block_table)
    assert kwargs["softmax_scale"] == 0.5
    assert kwargs["mask_mode"] == 1
    assert kwargs["quant_mode"] == 0
    assert kwargs["softmax_precision"] == metadata_kwargs["softmax_precision"]
    assert kwargs["return_softmax_lse"] is False
    assert "is_packed_gqa" not in kwargs
    assert torch.equal(output, torch.ones_like(output))


@pytest.mark.parametrize("shape,dtype", [((2,), torch.int64), ((2, 1), torch.int64), ((3,), torch.int32)])
def test_sparse_attn_decode_rejects_invalid_reused_prefix(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    q = torch.zeros(4, 2, 4, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="GBSA cu_seqlens_q"):
        minimax_m3_sparse_attn_decode_npu(
            q,
            torch.zeros(2, 6, 128, 2, 4, dtype=torch.bfloat16),
            torch.zeros((2, 4, 2), dtype=torch.int32),
            torch.zeros((2, 4), dtype=torch.int32),
            torch.tensor([129, 385], dtype=torch.int32),
            num_kv_heads=2,
            sm_scale=0.5,
            output=torch.empty_like(q),
            decode_query_len=2,
            cu_seqlens_q=torch.zeros(shape, dtype=dtype),
        )


@pytest.mark.parametrize("query_len", [1, 4])
def test_sparse_decode_reuses_schedule_but_keeps_layer_inputs(
    mock_gbsa_ops: tuple[MagicMock, MagicMock], query_len: int
) -> None:
    metadata_op, attention_op = mock_gbsa_ops
    schedule = torch.zeros(1024, dtype=torch.int32)
    metadata_op.return_value = schedule
    cache: dict = {}
    counts = torch.ones((2, query_len), dtype=torch.int32)
    prefix = torch.tensor([0, query_len], dtype=torch.int64)
    seq_lens = torch.tensor([512], dtype=torch.int32)
    table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    for layer in range(3):
        q = torch.full((query_len, 2, 4), layer, dtype=torch.bfloat16)
        kv = torch.full((2, 4, 128, 2, 4), layer, dtype=torch.bfloat16)
        indices = torch.full((2, query_len, 1), layer, dtype=torch.int32)
        attention_op.return_value = (torch.full_like(q, layer + 1), torch.empty(0))
        output = torch.empty_like(q)
        minimax_m3_sparse_attn_decode_npu(
            q,
            kv,
            indices,
            table,
            seq_lens,
            2,
            0.5,
            output,
            query_len,
            select_num_idx=counts,
            cu_seqlens_q=prefix,
            gbsa_metadata_cache=cache,
        )
        args, kwargs = attention_op.call_args
        assert args[0] is q
        assert torch.equal(args[1], kv[0])
        assert torch.equal(args[2], kv[1])
        torch.testing.assert_close(args[3], indices, rtol=0, atol=0)
        assert args[4] is counts
        assert kwargs["metadata"] is schedule
        assert torch.equal(output, torch.full_like(q, layer + 1))
    assert metadata_op.call_count == 1
    assert attention_op.call_count == 3


def test_sparse_decode_new_forward_rebuilds_schedule_for_changed_counts(
    mock_gbsa_ops: tuple[MagicMock, MagicMock],
) -> None:
    metadata_op, attention_op = mock_gbsa_ops
    q = torch.zeros((4, 2, 4), dtype=torch.bfloat16)
    kv = torch.zeros((2, 4, 128, 2, 4), dtype=torch.bfloat16)
    attention_op.return_value = (q, torch.empty(0))
    schedules = [torch.full((1024,), i, dtype=torch.int32) for i in range(2)]
    metadata_op.side_effect = schedules
    for count in (1, 2):
        cache: dict = {}
        for _ in range(2):
            minimax_m3_sparse_attn_decode_npu(
                q,
                kv,
                torch.zeros((2, 4, 2), dtype=torch.int32),
                torch.zeros((1, 4), dtype=torch.int32),
                torch.tensor([128 * count], dtype=torch.int32),
                2,
                0.5,
                torch.empty_like(q),
                4,
                select_num_idx=torch.full((2, 4), count, dtype=torch.int32),
                cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int64),
                gbsa_metadata_cache=cache,
            )
            assert attention_op.call_args.kwargs["metadata"] is schedules[count - 1]
    assert metadata_op.call_count == 2


def test_sparse_decode_cache_separates_bf16_and_fp8_schedules(mock_gbsa_ops, monkeypatch):
    monkeypatch.setattr(
        msa_m3_npu_module, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A5)
    )
    metadata_op, attention_op = mock_gbsa_ops
    q = torch.zeros((1, 2, 128), dtype=torch.bfloat16)
    attention_op.return_value = (q, torch.empty(0))
    metadata_op.side_effect = [torch.full((1024,), i, dtype=torch.int32) for i in range(2)]
    cache: dict = {}
    schedules = []
    for dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.bfloat16):
        minimax_m3_sparse_attn_decode_npu(
            q,
            torch.zeros((2, 1, 128, 1, 128), dtype=dtype),
            torch.zeros((1, 1, 16), dtype=torch.int32),
            torch.zeros((1, 1), dtype=torch.int32),
            torch.tensor([128], dtype=torch.int32),
            1,
            0.5,
            torch.empty_like(q),
            1,
            select_num_idx=torch.ones((1, 1), dtype=torch.int32),
            gbsa_metadata_cache=cache,
        )
        schedules.append(attention_op.call_args.kwargs["metadata"])
        assert attention_op.call_args.kwargs["softmax_precision"] == (1 if dtype == torch.float8_e4m3fn else 0)
    assert metadata_op.call_count == 2 and len(cache) == 2
    assert schedules[0] is schedules[2] and schedules[0] is not schedules[1]


@pytest.mark.parametrize("changed", ["heads", "topk", "query_len", "head_dim"])
def test_sparse_decode_cache_separates_scheduling_configurations(
    mock_gbsa_ops: tuple[MagicMock, MagicMock],
    changed: str,
) -> None:
    metadata_op, attention_op = mock_gbsa_ops
    cache: dict = {}
    metadata_op.side_effect = [torch.zeros(1024, dtype=torch.int32) for _ in range(2)]
    for second in (False, True):
        qlen = 1 if second and changed == "query_len" else 4
        heads = 4 if second and changed == "heads" else 2
        width = 2 if second and changed == "topk" else 1
        dim = 8 if second and changed == "head_dim" else 4
        q = torch.zeros((qlen, heads, dim), dtype=torch.bfloat16)
        attention_op.return_value = (q, torch.empty(0))
        minimax_m3_sparse_attn_decode_npu(
            q,
            torch.zeros((2, 4, 128, 2, dim), dtype=torch.bfloat16),
            torch.zeros((2, qlen, width), dtype=torch.int32),
            torch.zeros((1, 4), dtype=torch.int32),
            torch.tensor([512], dtype=torch.int32),
            2,
            0.5,
            torch.empty_like(q),
            qlen,
            cu_seqlens_q=torch.tensor([0, qlen], dtype=torch.int64),
            gbsa_metadata_cache=cache,
        )
    assert metadata_op.call_count == 2


@pytest.mark.parametrize(
    "query_len,block_size,dtype",
    [(2, 128, torch.bfloat16), (4, 64, torch.bfloat16), (4, 128, torch.float16), (4, 128, torch.float8_e4m3fn)],
)
def test_sparse_decode_unverified_paths_do_not_reuse_metadata(
    mock_gbsa_ops: tuple[MagicMock, MagicMock],
    query_len: int,
    block_size: int,
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        msa_m3_npu_module, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A3)
    )
    metadata_op, attention_op = mock_gbsa_ops
    q = torch.zeros((query_len, 2, 4), dtype=torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype)
    attention_op.return_value = (q, torch.empty(0))
    cache: dict = {}
    for _ in range(2):
        minimax_m3_sparse_attn_decode_npu(
            q,
            torch.zeros((2, 4, block_size, 2, 4), dtype=dtype),
            torch.zeros((2, query_len, 1), dtype=torch.int32),
            torch.zeros((1, 4), dtype=torch.int32),
            torch.tensor([128], dtype=torch.int32),
            2,
            0.5,
            torch.empty_like(q),
            query_len,
            block_size=block_size,
            cu_seqlens_q=torch.tensor([0, query_len], dtype=torch.int64),
            gbsa_metadata_cache=cache,
        )
    assert not cache
    assert metadata_op.call_count == 2


@pytest.mark.parametrize(
    "device_type,enabled", [(AscendDeviceType.A3, True), (AscendDeviceType.A2, False), (AscendDeviceType.A5, True)]
)
def test_sparse_builder_metadata_cache_is_forward_local(
    device_type: AscendDeviceType, enabled: bool, mock_gbsa_ops
) -> None:
    with patch.object(msa_m3_module, "get_current_hardware_profile", return_value=get_hardware_profile(device_type)):
        builder = _make_sparse_builder(torch.device("cpu"))
        common = _create_common_attn_metadata(BatchSpec([512], [1]), 128, torch.device("cpu"))
        first = builder.build(0, common).decode
        second = builder.build(0, common).decode
    assert first is not None and second is not None
    if enabled:
        assert len(first.gbsa_metadata_cache) == len(second.gbsa_metadata_cache) == 1
        assert first.gbsa_metadata_cache is not second.gbsa_metadata_cache
        assert first.gbsa_metadata_prepared and second.gbsa_metadata_prepared
    else:
        assert first.gbsa_metadata_cache is None and second.gbsa_metadata_cache is None


@pytest.mark.parametrize("query_len", [1, 4])
@pytest.mark.parametrize("topk,logical_blocks", [(1, 8), (4, 8), (16, 8), (16, 0)])
@pytest.mark.parametrize("init_blocks,local_blocks", [(0, 0), (1, 2)])
def test_prepared_gbsa_counts_match_decode_topk_mask(query_len, topk, logical_blocks, init_blocks, local_blocks):
    seq_lens = torch.tensor([0, 1, 3, 4, 127, 128, 129, 131, 255, 256, 257, 2048], dtype=torch.int32)
    heads = 2
    width = max(logical_blocks, topk)
    scores = torch.randn(heads, seq_lens.numel() * query_len, width)
    # Reference the existing score-prepare / TopK / postprocess sequence.
    visible = torch.tensor(
        [max((int(length) - query_len + offset + 128) // 128, 0) for length in seq_lens for offset in range(query_len)]
    )
    columns = torch.arange(width)
    valid = (columns < logical_blocks) & (columns < visible[:, None])
    scores.masked_fill_(~valid, -torch.inf)
    scores = torch.where(valid & (columns < init_blocks), 1.0e30, scores)
    scores = torch.where(valid & (columns >= (visible - local_blocks).clamp_min(0)[:, None]), 1.0e29, scores)
    indices = scores.topk(topk, dim=-1).indices
    expected = ((indices < visible[:, None]) & (indices < logical_blocks)).sum(-1).to(torch.int32)
    actual = _decode_gbsa_block_counts(seq_lens, query_len, heads, topk, logical_blocks)
    assert actual.is_contiguous() and actual.dtype == torch.int32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("query_len", [1, 4])
@pytest.mark.parametrize(
    "device_type,kv_cache_dtype",
    [
        (AscendDeviceType.A3, torch.bfloat16),
        (AscendDeviceType.A5, torch.bfloat16),
        (AscendDeviceType.A5, torch.float8_e4m3fn),
    ],
)
def test_sparse_builder_prepares_gbsa_before_layers_and_refreshes_in_place(
    query_len, device_type, kv_cache_dtype, mock_gbsa_ops, monkeypatch
):
    metadata_op, attention_op = mock_gbsa_ops
    monkeypatch.setattr(msa_m3_module, "get_current_hardware_profile", lambda: get_hardware_profile(device_type))
    monkeypatch.setattr(msa_m3_npu_module, "get_current_hardware_profile", lambda: get_hardware_profile(device_type))
    builder = _make_sparse_builder(torch.device("cpu"), kv_cache_dtype)
    buffers = builder.decode_gbsa_metadata_buffers
    assert set(buffers) == {(4, 2, 128, 16, 1), (4, 2, 128, 16, 4)}
    assert all(buffer.shape == (1024,) and buffer.dtype == torch.int32 for buffer in buffers.values())
    preallocated = buffers[(4, 2, 128, 16, query_len)]
    mock_gbsa_ops[0].assert_not_called()
    monkeypatch.setattr(msa_m3_module, "split_decodes_and_prefills", lambda *args, **kwargs: (2, 0, 2 * query_len, 0))
    common = _create_common_attn_metadata(BatchSpec([128, 257], [query_len] * 2), 128, torch.device("cpu"))
    events = []

    def make_schedule(*args, **kwargs):
        events.append("metadata")
        return torch.full((1024,), metadata_op.call_count, dtype=torch.int32)

    metadata_op.side_effect = make_schedule
    md = builder.build(0, common)
    assert events == ["metadata"]  # No IndexScore / sparse layer has run yet.
    assert md.decode.gbsa_metadata_prepared
    buffer = next(iter(md.decode.gbsa_metadata_cache.values()))
    assert buffer is preallocated
    q = torch.zeros((2 * query_len, 4, 128), dtype=torch.bfloat16)

    def run_gbsa(*args, **kwargs):
        events.append("gbsa")
        return q, torch.empty(0)

    attention_op.side_effect = run_gbsa
    impl = AscendMiniMaxM3SparseImpl(4, 128, 0.5, 2, topk_blocks=16, sparse_block_size=128)
    kv = torch.zeros((2, 6, 128, 2, 128), dtype=kv_cache_dtype)
    with patch.object(
        msa_m3_module, "get_forward_context", return_value=SimpleNamespace(attn_metadata={"layer0.attn": md})
    ):
        # Re-entering the same layer models capture warmup; prepared entries
        # must not be cleared and regenerated inside the model graph.
        for _ in range(3):
            impl.forward(
                SimpleNamespace(layer_name="layer0.attn"),
                q,
                kv,
                (
                    torch.zeros((2, 2 * query_len, 16), dtype=torch.int32),
                    None,
                    torch.ones((2, 2 * query_len), dtype=torch.int32),
                ),
                torch.empty_like(q),
            )
            assert attention_op.call_args.kwargs["metadata"] is buffer
            assert attention_op.call_args.kwargs["quant_mode"] == (5 if kv_cache_dtype == torch.float8_e4m3fn else 0)
            assert attention_op.call_args.kwargs["softmax_precision"] == (
                1 if kv_cache_dtype == torch.float8_e4m3fn else 0
            )
    assert events == ["metadata", "gbsa", "gbsa", "gbsa"]
    assert metadata_op.call_args.kwargs["quant_mode"] == attention_op.call_args.kwargs["quant_mode"]
    assert metadata_op.call_args.kwargs["softmax_precision"] == attention_op.call_args.kwargs["softmax_precision"]
    common.seq_lens.add_(1)
    rebuilt = builder.build(0, common)
    refreshed = next(iter(rebuilt.decode.gbsa_metadata_cache.values()))
    assert refreshed.data_ptr() == buffer.data_ptr()
    assert torch.all(refreshed == 2) and metadata_op.call_count == 2
    expected_counts = _decode_gbsa_block_counts(common.seq_lens, query_len, 2, 16, common.block_table_tensor.shape[-1])
    torch.testing.assert_close(metadata_op.call_args.args[1], expected_counts, rtol=0, atol=0)
    monkeypatch.setattr(msa_m3_module, "split_decodes_and_prefills", lambda *args, **kwargs: (1, 0, query_len, 0))
    smaller = _create_common_attn_metadata(BatchSpec([512], [query_len]), 128, torch.device("cpu"))
    with patch.object(torch, "empty_like", side_effect=AssertionError("Persistent buffers must already exist")):
        changed_size = builder.build(0, smaller).decode
    assert next(iter(changed_size.gbsa_metadata_cache.values())) is preallocated
    assert len(buffers) == 2 and torch.all(preallocated == 3)


@pytest.mark.parametrize("set_graph_pad_size", [True, False])
@pytest.mark.parametrize("query_len", [1, 4])
@pytest.mark.parametrize("actual_reqs,padded_reqs", [(1, 2), (3, 4), (15, 16)])
@pytest.mark.parametrize(
    "device_type,kv_cache_dtype",
    [
        (AscendDeviceType.A3, torch.bfloat16),
        (AscendDeviceType.A5, torch.bfloat16),
        (AscendDeviceType.A5, torch.float8_e4m3fn),
    ],
)
def test_sparse_builder_refreshes_captured_gbsa_buffer_for_padded_decode(
    query_len, actual_reqs, padded_reqs, set_graph_pad_size, device_type, kv_cache_dtype, mock_gbsa_ops, monkeypatch
):
    metadata_op, _ = mock_gbsa_ops
    monkeypatch.setattr(msa_m3_module, "get_current_hardware_profile", lambda: get_hardware_profile(device_type))
    monkeypatch.setattr(msa_m3_npu_module, "get_current_hardware_profile", lambda: get_hardware_profile(device_type))
    monkeypatch.setattr(
        msa_m3_module,
        "split_decodes_and_prefills",
        lambda common, **kwargs: (common.num_reqs, 0, common.num_actual_tokens, 0),
    )
    builder = _make_sparse_builder(torch.device("cpu"), kv_cache_dtype)

    def make_schedule(*args, **kwargs):
        return torch.full((1024,), metadata_op.call_count, dtype=torch.int32)

    metadata_op.side_effect = make_schedule
    captured_common = _create_common_attn_metadata(
        BatchSpec([128] * padded_reqs, [query_len] * padded_reqs), 128, torch.device("cpu")
    )
    captured_common.graph_pad_size = padded_reqs * query_len if set_graph_pad_size else -1
    captured = builder.build(0, captured_common).decode
    buffer = next(iter(captured.gbsa_metadata_cache.values()))
    runtime_common = _create_common_attn_metadata(
        BatchSpec(
            [257] * actual_reqs + [0] * (padded_reqs - actual_reqs),
            [query_len] * actual_reqs + [0] * (padded_reqs - actual_reqs),
        ),
        128,
        torch.device("cpu"),
    )
    runtime_common.graph_pad_size = padded_reqs * query_len if set_graph_pad_size else -1
    runtime_common.num_input_tokens = padded_reqs * query_len
    runtime = builder.build(0, runtime_common).decode
    assert runtime.seq_lens.numel() == actual_reqs
    assert runtime.gbsa_metadata_prepared
    assert next(iter(runtime.gbsa_metadata_cache.values())).data_ptr() == buffer.data_ptr()
    assert torch.all(buffer == 2) and metadata_op.call_count == 2
    assert metadata_op.call_args.args[0].shape == (2, padded_reqs * query_len, 16)
    expected_counts = _decode_gbsa_block_counts(
        runtime_common.seq_lens, query_len, 2, 16, runtime_common.block_table_tensor.shape[-1]
    )
    torch.testing.assert_close(metadata_op.call_args.args[1], expected_counts, rtol=0, atol=0)
    assert metadata_op.call_args.kwargs["cu_seqlens_q"].shape == (padded_reqs + 1,)


@pytest.mark.parametrize(
    "dtype,block_size,query_len", [(torch.float16, 128, 1), (torch.bfloat16, 64, 1), (torch.bfloat16, 128, 2)]
)
def test_sparse_builder_keeps_unverified_gbsa_paths_in_layers(dtype, block_size, query_len, mock_gbsa_ops):
    config = _make_vllm_config()
    config.model_config.dtype = dtype
    spec = FullAttentionSpec(block_size=block_size, num_kv_heads=2, head_size=128, dtype=dtype)
    config.speculative_config = SimpleNamespace(num_speculative_tokens=3, parallel_drafting=False)
    with patch.object(
        msa_m3_module, "get_current_hardware_profile", return_value=get_hardware_profile(AscendDeviceType.A3)
    ):
        builder = AscendMiniMaxM3SparseMetadataBuilder(spec, ["layer0.attn"], config, torch.device("cpu"))
        common = _create_common_attn_metadata(BatchSpec([512], [query_len]), block_size, torch.device("cpu"))
        with patch.object(msa_m3_module, "split_decodes_and_prefills", return_value=(1, 0, query_len, 0)):
            decode = builder.build(0, common).decode
    assert not decode.gbsa_metadata_prepared and not decode.gbsa_metadata_cache
    mock_gbsa_ops[0].assert_not_called()


def test_sparse_impl_resets_reused_metadata_before_warmup_or_capture() -> None:
    decode = AscendMiniMaxM3SparseDecodeMetadata(
        seq_lens=torch.tensor([512], dtype=torch.int32),
        block_table=torch.zeros((1, 4), dtype=torch.int32),
        max_seq_len=512,
        decode_query_len=4,
        cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int64),
        gbsa_metadata_cache={},
    )
    md = AscendMiniMaxM3SparseMetadata(
        seq_lens=decode.seq_lens,
        max_seq_len=512,
        slot_mapping=torch.arange(4),
        num_actual_tokens=4,
        num_decodes=1,
        num_decode_tokens=4,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=decode,
    )
    impl = AscendMiniMaxM3SparseImpl(2, 4, 0.5, 2, topk_blocks=16, sparse_block_size=128)
    cache_sizes = []

    def fake_decode(*args, **kwargs):
        cache = kwargs["gbsa_metadata_cache"]
        cache_sizes.append(len(cache))
        cache[("schedule",)] = torch.zeros(1024, dtype=torch.int32)

    with (
        patch.object(
            msa_m3_module,
            "get_forward_context",
            return_value=SimpleNamespace(
                attn_metadata={"layer3": md, "layer4": md},
            ),
        ),
        patch.object(msa_m3_module, "minimax_m3_sparse_attn_decode", side_effect=fake_decode),
    ):
        for name in ["layer3", "layer4", "layer3", "layer4"]:
            impl.forward(
                SimpleNamespace(layer_name=name),
                torch.zeros((4, 8), dtype=torch.bfloat16),
                torch.empty(0),
                (torch.zeros((2, 4, 16), dtype=torch.int32), None, None),
                torch.empty((4, 8), dtype=torch.bfloat16),
            )
    assert cache_sizes == [0, 1, 0, 1]


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


def test_sparse_attn_decode_npu_uses_fp8_inputs_without_unsupported_scales(
    mock_gbsa_ops: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        msa_m3_npu_module, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A5)
    )
    q = torch.tensor([[[500.0, -500.0, 1.0, -1.0]]], dtype=torch.bfloat16)
    kv_cache = torch.zeros(2, 1, 128, 1, 4, dtype=torch.float8_e4m3fn)
    topk_idx = torch.tensor([[[0]]], dtype=torch.int32)
    select_num_idx = torch.ones(1, 1, dtype=torch.int32)
    block_table = torch.zeros(1, 1, dtype=torch.int32)
    seq_lens = torch.ones(1, dtype=torch.int32)
    dequant_scale = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    output = torch.empty_like(q)
    metadata = torch.empty(1024, dtype=torch.int32)
    gbsa_metadata_op, gbsa_attention_op = mock_gbsa_ops
    gbsa_metadata_op.return_value = metadata
    gbsa_attention_op.return_value = (torch.ones_like(output), torch.empty(0))

    minimax_m3_sparse_attn_decode_npu(
        q,
        kv_cache,
        topk_idx,
        block_table,
        seq_lens,
        num_kv_heads=1,
        sm_scale=0.5,
        output=output,
        decode_query_len=1,
        select_num_idx=select_num_idx,
        dequant_scale=dequant_scale,
    )

    metadata_kwargs = gbsa_metadata_op.call_args.kwargs
    assert metadata_kwargs["quant_mode"] == 5
    assert metadata_kwargs["softmax_precision"] == 1
    assert metadata_kwargs["max_seqlen_q"] == 1
    assert torch.equal(
        metadata_kwargs["cu_seqlens_q"],
        torch.tensor([0, 1], dtype=torch.int64),
    )

    args = gbsa_attention_op.call_args.args
    kwargs = gbsa_attention_op.call_args.kwargs
    assert args[0].dtype == torch.float8_e4m3fn
    assert args[1].dtype == torch.float8_e4m3fn
    assert args[2].dtype == torch.float8_e4m3fn
    assert torch.equal(args[4], select_num_idx)
    assert kwargs["quant_mode"] == 5
    assert kwargs["softmax_precision"] == 1
    assert "q_dequant_scale" not in kwargs
    assert "k_dequant_scale" not in kwargs
    assert "v_dequant_scale" not in kwargs
    assert kwargs["attention_out_dtype"] == torch.bfloat16
    assert torch.equal(output, torch.ones_like(output))


@patch.object(torch.ops._C_ascend, "npu_sparse_attention_score_prefill", create=True)
@patch("vllm_ascend.models.minimax_m3.ops.msa_m3_npu._npu_k2q_csr")
def test_sparse_attn_prefill_a5_uses_fp8_inputs(
    mock_k2q_csr: MagicMock,
    mock_sparse_attention_score_prefill: MagicMock,
) -> None:
    q = torch.tensor([[[500.0, -500.0, 1.0, -1.0]]], dtype=torch.bfloat16)
    kv_cache = torch.zeros(2, 1, 128, 1, 4, dtype=torch.float8_e4m3fn)
    topk_idx = torch.tensor([[[0]]], dtype=torch.int32)
    block_table = torch.zeros(1, 1, dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32)
    seq_lens = torch.ones(1, dtype=torch.int32)
    output = torch.empty_like(q)
    csr = torch.zeros(1, dtype=torch.int32)
    mock_k2q_csr.return_value = (csr, csr, csr)
    mock_sparse_attention_score_prefill.return_value = torch.ones_like(output)

    _minimax_m3_sparse_attn_kv_gather_q(
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens,
        num_kv_heads=1,
        sm_scale=0.5,
        output=output,
        block_size=128,
        total_kv_blocks=1,
        max_kv_blocks=1,
        supports_fp8=True,
    )
    args = mock_sparse_attention_score_prefill.call_args.args
    assert args[0].dtype == torch.float8_e4m3fn
    assert args[1].dtype == torch.float8_e4m3fn
    assert args[2].dtype == torch.float8_e4m3fn
    assert args[11] == 4


@pytest.mark.parametrize(
    ("device_type", "dtype"),
    [
        (AscendDeviceType.A2, torch.bfloat16),
        (AscendDeviceType.A3, torch.bfloat16),
        (AscendDeviceType.A5, torch.bfloat16),
        (AscendDeviceType.A5, torch.float8_e4m3fn),
    ],
)
@pytest.mark.parametrize("cache_layout", ["tensor", "tuple", "list"])
def test_sparse_attn_prefill_missing_split_kv_uses_cann_gbsa(
    device_type: AscendDeviceType,
    dtype,
    cache_layout,
    mock_gbsa_ops: tuple[MagicMock, MagicMock],
) -> None:
    q = torch.tensor([[[500.0, -500.0, 1.0, -1.0]]] * 5, dtype=torch.bfloat16)
    cache = torch.zeros(2, 3, 128, 1, 4, dtype=dtype)
    kv_cache = cache if cache_layout == "tensor" else (cache[0], cache[1])
    if cache_layout == "list":
        kv_cache = list(kv_cache)
    topk_idx = torch.tensor([[[0, -1]]] * 5, dtype=torch.int32)
    output = torch.empty_like(q)
    metadata = torch.empty(1024, dtype=torch.int32)
    gbsa_metadata_op, gbsa_attention_op = mock_gbsa_ops
    gbsa_metadata_op.return_value = metadata
    gbsa_attention_op.return_value = (torch.ones_like(q), torch.empty(0))
    with (
        patch.object(msa_m3_npu_module, "get_current_hardware_profile", return_value=get_hardware_profile(device_type)),
        patch.object(msa_m3_npu_module, "_is_minimax_sparse_attention_split_kv_available", return_value=False),
        patch.object(msa_m3_npu_module, "_npu_k2q_csr") as k2q,
        patch("torch.ops._C_ascend.npu_sparse_attention_score_prefill", create=True) as split_kv,
    ):
        msa_m3_npu_module.minimax_m3_sparse_attn(
            q,
            kv_cache,
            topk_idx,
            torch.tensor([[0, 1], [2, 0]], dtype=torch.int32),
            torch.tensor([0, 2, 5], dtype=torch.int32),
            torch.tensor([130, 3], dtype=torch.int32),
            torch.tensor([128, 0], dtype=torch.int32),
            3,
            1,
            0.5,
            output,
            total_kv_blocks=3,
            max_kv_blocks=2,
        )
    k2q.assert_not_called()
    split_kv.assert_not_called()
    gbsa_metadata_op.assert_called_once()
    metadata_kwargs = gbsa_metadata_op.call_args.kwargs
    assert metadata_kwargs["max_seqlen_q"] == 3
    assert metadata_kwargs["softmax_precision"] == (1 if dtype == torch.float8_e4m3fn else 0)
    assert torch.equal(metadata_kwargs["cu_seqlens_q"], torch.tensor([0, 2, 5], dtype=torch.int64))
    assert torch.equal(metadata_kwargs["seqused_kv"], torch.tensor([130, 3], dtype=torch.int32))

    gbsa_attention_op.assert_called_once()
    args, kwargs = gbsa_attention_op.call_args
    assert args[0].dtype == dtype
    assert args[1].dtype == dtype
    assert args[2].dtype == dtype
    assert torch.equal(args[3], topk_idx)
    torch.testing.assert_close(args[4], torch.ones(5, 1, dtype=torch.int32))
    assert kwargs["metadata"] is metadata
    assert kwargs["softmax_precision"] == metadata_kwargs["softmax_precision"]
    if dtype == torch.float8_e4m3fn:
        torch.testing.assert_close(args[0].float(), q.float().clamp(-448, 448))
        assert kwargs["attention_out_dtype"] == torch.bfloat16
        assert kwargs["quant_mode"] == 5
    else:
        assert args[0] is q
        assert kwargs["quant_mode"] == 0
        assert "attention_out_dtype" not in kwargs
    assert not {"q_dequant_scale", "k_dequant_scale", "v_dequant_scale"} & kwargs.keys()
    torch.testing.assert_close(output, torch.ones_like(q))


def test_sparse_attn_prefill_fallback_rejects_mismatched_fp8_cache() -> None:
    q = torch.zeros(1, 1, 4, dtype=torch.bfloat16)
    kv_cache = (
        torch.zeros(1, 128, 1, 4, dtype=torch.float8_e4m3fn),
        torch.zeros(1, 128, 1, 4, dtype=torch.bfloat16),
    )
    with pytest.raises(TypeError, match="FP8 sparse attention"):
        msa_m3_npu_module._minimax_m3_sparse_attn_a3(
            q,
            kv_cache,
            torch.zeros(1, 1, 1, dtype=torch.int32),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            1,
            0.5,
            torch.empty_like(q),
            128,
        )


@pytest.mark.parametrize(
    ("device_type", "split_kv_available", "expected_impl"),
    [
        (AscendDeviceType.A5, True, "kv_gather_q"),
        (AscendDeviceType.A5, False, "gbsa"),
        (AscendDeviceType.A3, True, "kv_gather_q"),
        (AscendDeviceType.A3, False, "gbsa"),
        (AscendDeviceType.A2, True, "kv_gather_q"),
        (AscendDeviceType.A2, False, "gbsa"),
    ],
)
def test_sparse_attn_prefill_dispatches_by_operator_availability(
    device_type: AscendDeviceType,
    split_kv_available: bool,
    expected_impl: str,
) -> None:
    with (
        patch.object(
            msa_m3_npu_module,
            "get_current_hardware_profile",
            return_value=get_hardware_profile(device_type),
        ),
        patch.object(
            msa_m3_npu_module,
            "_is_minimax_sparse_attention_split_kv_available",
            return_value=split_kv_available,
        ) as mock_is_available,
        patch.object(msa_m3_npu_module, "_minimax_m3_sparse_attn_a3") as mock_gbsa,
        patch.object(
            msa_m3_npu_module,
            "_minimax_m3_sparse_attn_kv_gather_q",
        ) as mock_kv_gather_q,
    ):
        msa_m3_npu_module.minimax_m3_sparse_attn(
            q=torch.empty(0),
            kv_cache=torch.empty(0),
            topk_idx=torch.empty(0),
            block_table=torch.empty(0),
            cu_seqlens_q=torch.empty(0),
            seq_lens=torch.empty(0),
            prefix_lens=torch.empty(0),
            max_query_len=0,
            num_kv_heads=2,
            sm_scale=0.5,
            output=torch.empty(0),
            total_kv_blocks=5,
            max_kv_blocks=3,
        )

    if expected_impl == "kv_gather_q":
        mock_kv_gather_q.assert_called_once()
        mock_gbsa.assert_not_called()
        assert mock_kv_gather_q.call_args.args[-2:] == (5, 3)
        assert mock_kv_gather_q.call_args.kwargs == {
            "supports_fp8": device_type == AscendDeviceType.A5,
        }
    else:
        mock_gbsa.assert_called_once()
        mock_kv_gather_q.assert_not_called()
        assert mock_gbsa.call_args.kwargs == {
            "max_query_len": 0,
        }

    mock_is_available.assert_called_once_with()


@pytest.mark.parametrize("tp_size", [1, 4, 8])
@pytest.mark.parametrize("query_len", [1, 4])
@pytest.mark.parametrize("num_prefills", [0, 1])
@pytest.mark.parametrize("idle_dp", [False, True])
@pytest.mark.parametrize("device_type", [AscendDeviceType.A2, AscendDeviceType.A3])
def test_ascendc_index_decode_uses_full_cache_without_tp_collectives(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    query_len: int,
    num_prefills: int,
    idle_dp: bool,
    device_type: AscendDeviceType,
) -> None:
    # Exercise metadata and forward together: an unguarded metadata, dummy,
    # or forward TP branch must fail even when the other branches are correct.
    monkeypatch.setattr(msa_m3_module, "get_current_hardware_profile", lambda: get_hardware_profile(device_type))
    monkeypatch.setattr(msa_m3_module, "_USE_ASCENDC_INDEX_SCORE_DECODE", True)
    monkeypatch.setattr(msa_m3_module, "_USE_ASCENDC_INDEX_SCORE_PREFILL", True)
    group = SimpleNamespace(world_size=tp_size, rank_in_group=0, all_gather=MagicMock())
    monkeypatch.setattr("vllm.distributed.get_tp_group", lambda: group)
    device = torch.device("cpu")
    common = _create_common_attn_metadata(
        BatchSpec(
            seq_lens=[16384 + query_len] * 2 + [128] * num_prefills,
            query_lens=[query_len] * 2 + [128] * num_prefills,
        ),
        block_size=128,
        device=device,
    )
    monkeypatch.setattr(
        msa_m3_module,
        "split_decodes_and_prefills",
        lambda *a, **kw: (2, num_prefills, 2 * query_len, 128 * num_prefills),
    )
    builder = _make_indexer_builder(device, tp_size=tp_size)
    forbidden = MagicMock(side_effect=AssertionError("AscendC decode must use the full cache"))
    metadata = builder.build(0, common)
    assert metadata.decode is not None
    assert metadata.decode.block_table.shape == (2, 129)

    impl = object.__new__(AscendMiniMaxM3IndexerImpl)
    torch.nn.Module.__init__(impl)
    impl.num_index_heads = 1
    impl.index_head_dim = 4
    impl.block_size = 128
    impl.topk_blocks = 2
    impl.init_blocks = 1
    impl.local_blocks = 1
    impl.index_cache = cast(
        AscendMiniMaxM3IndexerCache, SimpleNamespace(prefix="index", kv_cache=torch.zeros(258, 128, 4))
    )
    context = _make_mrv2_padding_context() if idle_dp else SimpleNamespace(additional_kwargs={})
    context.attn_metadata = {"index": metadata}
    monkeypatch.setattr(msa_m3_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(msa_m3_module, "minimax_m3_index_decode", forbidden, raising=False)
    expected = torch.zeros(1, 2 * query_len, 2, dtype=torch.int32)
    expected_select_num_idx = torch.full((1, 2 * query_len), 2, dtype=torch.int32)
    decode_op = MagicMock(return_value=(expected, expected_select_num_idx))
    prefill_op = MagicMock(return_value=torch.zeros(1, 128, 2, dtype=torch.int32))
    monkeypatch.setattr(msa_m3_module, "minimax_m3_index_decode_ascendc", decode_op)
    monkeypatch.setattr(msa_m3_module, "minimax_m3_index_prefill_ascendc", prefill_op)
    for _ in range(2):
        actual, prefill, select_num_idx = impl.forward(torch.zeros(common.num_actual_tokens, 4))
        assert actual is expected
        assert prefill is (prefill_op.return_value if num_prefills else None)
        assert select_num_idx is expected_select_num_idx
    assert decode_op.call_count == 2
    assert decode_op.call_args.args[0].shape == (2 * query_len, 1, 4)
    assert decode_op.call_args.args[1] is impl.index_cache.kv_cache
    assert decode_op.call_args.args[2] is metadata.decode.block_table
    assert torch.equal(decode_op.call_args.args[6], torch.tensor([128, 128], dtype=torch.int32))
    assert all(call.args[6] is metadata.decode.start_loc for call in decode_op.call_args_list)
    assert decode_op.call_args.kwargs["decode_query_len"] == query_len
    assert prefill_op.call_count == 2 * num_prefills
    forbidden.assert_not_called()
    group.all_gather.assert_not_called()


@pytest.mark.parametrize("query_len", [1, 4])
@pytest.mark.parametrize("mode", ["mixed", "piecewise", "eager"])
@pytest.mark.parametrize(
    "device_type,kv_cache_dtype",
    [
        (AscendDeviceType.A3, torch.bfloat16),
        (AscendDeviceType.A5, torch.bfloat16),
        (AscendDeviceType.A5, torch.float8_e4m3fn),
    ],
)
def test_sparse_schedule_ignores_prefill_and_token_only_padding(
    query_len, mode, device_type, kv_cache_dtype, mock_gbsa_ops, monkeypatch
):
    metadata_op, _ = mock_gbsa_ops
    monkeypatch.setattr(msa_m3_module, "get_current_hardware_profile", lambda: get_hardware_profile(device_type))
    monkeypatch.setattr(msa_m3_npu_module, "get_current_hardware_profile", lambda: get_hardware_profile(device_type))
    mixed = mode == "mixed"
    monkeypatch.setattr(
        msa_m3_module, "split_decodes_and_prefills", lambda *a, **k: (2, int(mixed), 2 * query_len, 8 if mixed else 0)
    )
    common = _create_common_attn_metadata(
        BatchSpec([128, 257] + ([512] if mixed else []), [query_len, query_len] + ([8] if mixed else [])),
        128,
        torch.device("cpu"),
    )
    common.graph_pad_size = -1
    common.num_input_tokens = 64
    builder = _make_sparse_builder(torch.device("cpu"), kv_cache_dtype)
    md = builder.build(0, common)
    assert md.num_prefills == int(mixed)
    assert md.num_decodes == 2
    assert md.decode.seq_lens.numel() == 2
    assert metadata_op.call_args.args[0].shape == (2, 2 * query_len, 16)
    assert metadata_op.call_args.kwargs["cu_seqlens_q"].shape == (3,)
    expected = _decode_gbsa_block_counts(common.seq_lens[:2], query_len, 2, 16, common.block_table_tensor.shape[-1])
    torch.testing.assert_close(metadata_op.call_args.args[1], expected, rtol=0, atol=0)


def test_sparse_decode_presorted_indices_bypass_normalization(mock_gbsa_ops, monkeypatch) -> None:
    _, attention_op = mock_gbsa_ops
    indices = torch.tensor([[[0, 1, -1]] * 4] * 2, dtype=torch.int32)
    counts = torch.full((2, 4), 2, dtype=torch.int32)
    q = torch.zeros((4, 2, 128), dtype=torch.bfloat16)
    attention_op.return_value = (torch.empty_like(q), torch.empty(0))
    kv = torch.zeros((2, 2, 128, 2, 128), dtype=torch.bfloat16)
    normalize = MagicMock(side_effect=AssertionError("Sorted producer output must not be sorted twice"))
    monkeypatch.setattr(msa_m3_npu_module, "_sort_valid_gbsa_prefix", normalize)
    minimax_m3_sparse_attn_decode_npu(
        q,
        kv,
        indices,
        torch.zeros((4, 2), dtype=torch.int32),
        torch.full((4,), 256, dtype=torch.int32),
        2,
        0.1,
        torch.empty_like(q),
        1,
        select_num_idx=counts,
        indices_are_sorted=True,
    )
    normalize.assert_not_called()
    assert attention_op.call_args.args[3] is indices
