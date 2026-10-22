# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for MiniMax M3 sparse attention layer wiring in ``msa_m3``."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.device.hardware_profile import get_hardware_profile
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
        assert metadata.prefill.total_kv_blocks == 5
        assert metadata.prefill.max_kv_blocks == 3
        assert metadata.prefill.cu_seqlens_q.data_ptr() == builder.prefill_cu_seqlens_q_buffer.data_ptr()
        rebuilt = builder.build(0, common)
        assert rebuilt.prefill.cu_seqlens_q.data_ptr() == metadata.prefill.cu_seqlens_q.data_ptr()


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
        assert metadata.decode.start_loc.dtype == torch.int32
        assert torch.equal(metadata.decode.start_loc, metadata.decode.context_lens // 128)
        assert builder.decode_start_loc_buffer is not None
        assert metadata.decode.start_loc.data_ptr() == builder.decode_start_loc_buffer.data_ptr()
        rebuilt = builder.build(0, common)
        assert rebuilt.decode.start_loc.data_ptr() == metadata.decode.start_loc.data_ptr()
    else:
        assert metadata.num_prefills == batch_spec.batch_size
        assert metadata.decode is None
        assert metadata.prefill is not None
        assert torch.equal(
            metadata.prefill.start_loc,
            metadata.prefill.context_lens // 128,
        )
        assert builder.prefill_start_loc_buffer is not None
        assert metadata.prefill.start_loc.data_ptr() == builder.prefill_start_loc_buffer.data_ptr()
        assert metadata.prefill.cu_seqlens_q.data_ptr() == builder.prefill_cu_seqlens_q_buffer.data_ptr()
        rebuilt = builder.build(0, common)
        assert rebuilt.prefill.start_loc.data_ptr() == metadata.prefill.start_loc.data_ptr()
        assert rebuilt.prefill.cu_seqlens_q.data_ptr() == metadata.prefill.cu_seqlens_q.data_ptr()


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
    assert first.decode.cu_seqlens_q.shape == (16,)
    assert first.decode.cu_seqlens_q[-1] == 60
    assert first.decode.block_table.shape[0] == 15
    assert first.decode.context_lens.shape == (15,)
    assert second.decode is not None
    assert first.decode.block_table.data_ptr() == second.decode.block_table.data_ptr()
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
    a5_branch_start = module_source.index("if get_ascend_device_type() == AscendDeviceType.A5:")
    a5_branch_end = module_source.index("\n\ndef _active_decode_num_reqs", a5_branch_start)
    import_branches = module_source[a5_branch_start:a5_branch_end]

    assert msa_m3_module._USE_ASCENDC_INDEX_SCORE_PREFILL is True
    assert msa_m3_module._USE_ASCENDC_INDEX_SCORE_DECODE is (
        msa_m3_module.get_ascend_device_type() != AscendDeviceType.A5
    )
    assert import_branches.count("minimax_m3_index_decode") == 1
    assert "msa_m3_triton_a5" in import_branches
    assert "msa_m3_triton" not in import_branches.replace("msa_m3_triton_a5", "")
    assert "_USE_ASCENDC_INDEX_SCORE_PREFILL = True" in module_source
    assert "_USE_ASCENDC_INDEX_SCORE_DECODE = get_ascend_device_type() != AscendDeviceType.A5" in module_source


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
    assert mock_decode.call_args.kwargs == {"sm_scale": impl.scale}


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
    impl.index_cache = SimpleNamespace(
        prefix="layer.attn.index_cache",
        kv_cache=torch.zeros(4, 128, 4),
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


@patch.object(torch.ops._C_ascend, "npu_sparse_attention_score", create=True)
def test_sparse_attn_decode_npu_uses_fp8_inputs_and_reused_scale(
    mock_sparse_attention_score: MagicMock,
) -> None:
    q = torch.tensor([[[500.0, -500.0, 1.0, -1.0]]], dtype=torch.bfloat16)
    kv_cache = torch.zeros(2, 1, 128, 1, 4, dtype=torch.float8_e4m3fn)
    topk_idx = torch.tensor([[[0]]], dtype=torch.int32)
    select_num_idx = torch.ones(1, 1, dtype=torch.int32)
    block_table = torch.zeros(1, 1, dtype=torch.int32)
    seq_lens = torch.ones(1, dtype=torch.int32)
    dequant_scale = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    output = torch.empty_like(q)
    mock_sparse_attention_score.return_value = torch.ones_like(output)

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

    args = mock_sparse_attention_score.call_args.args
    kwargs = mock_sparse_attention_score.call_args.kwargs
    assert args[0].dtype == torch.float8_e4m3fn
    assert args[1].dtype == torch.float8_e4m3fn
    assert args[2].dtype == torch.float8_e4m3fn
    assert kwargs["select_num_idx"] is select_num_idx
    assert kwargs["actual_seq_lengths"].dtype == torch.int32
    assert torch.equal(kwargs["actual_seq_lengths"], torch.ones_like(seq_lens))
    assert kwargs["q_dequant_scale"] is dequant_scale
    assert kwargs["k_dequant_scale"] is dequant_scale
    assert kwargs["v_dequant_scale"] is dequant_scale
    assert kwargs["attention_out_dtype"] == torch.bfloat16


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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("cache_layout", ["tensor", "tuple", "list"])
def test_sparse_attn_prefill_a5_missing_package_uses_q_gather_kv(dtype, cache_layout) -> None:
    q = torch.tensor([[[500.0, -500.0, 1.0, -1.0]]] * 5, dtype=torch.bfloat16)
    cache = torch.zeros(2, 3, 128, 1, 4, dtype=dtype)
    kv_cache = cache if cache_layout == "tensor" else (cache[0], cache[1])
    if cache_layout == "list":
        kv_cache = list(kv_cache)
    topk_idx = torch.tensor([[[0, -1]]] * 5, dtype=torch.int32)
    output = torch.empty_like(q)
    with (
        patch.object(
            msa_m3_npu_module, "get_current_hardware_profile", return_value=get_hardware_profile(AscendDeviceType.A5)
        ),
        patch.object(msa_m3_npu_module, "_is_minimax_sparse_attention_split_kv_available", return_value=False),
        patch.object(msa_m3_npu_module, "_npu_k2q_csr") as k2q,
        patch("torch.ops._C_ascend.npu_sparse_attention_score", create=True, return_value=torch.ones_like(q)) as op,
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
    op.assert_called_once()
    args, kwargs = op.call_args
    assert args[0].dtype == dtype
    assert args[1].dtype == dtype
    assert args[2].dtype == dtype
    assert args[3] is topk_idx
    torch.testing.assert_close(kwargs["actual_seq_lengths"], torch.tensor([2, 3], dtype=torch.int32))
    torch.testing.assert_close(kwargs["actual_seq_lengths_kv"], torch.tensor([130, 3], dtype=torch.int32))
    torch.testing.assert_close(kwargs["select_num_idx"], torch.ones(5, 1, dtype=torch.int32))
    assert kwargs["inner_precise"] == 4
    if dtype == torch.float8_e4m3fn:
        torch.testing.assert_close(args[0].float(), q.float().clamp(-448, 448))
        assert kwargs["attention_out_dtype"] == torch.bfloat16
        scale = kwargs["q_dequant_scale"]
        assert kwargs["k_dequant_scale"] is scale
        assert kwargs["v_dequant_scale"] is scale
        torch.testing.assert_close(scale, torch.ones(1, 1, 1, 1))
    else:
        assert args[0] is q
        assert "q_dequant_scale" not in kwargs
        assert "attention_out_dtype" not in kwargs
    torch.testing.assert_close(output, torch.ones_like(q))


@pytest.mark.parametrize("supports_fp8,value_dtype", [(False, torch.float8_e4m3fn), (True, torch.bfloat16)])
def test_sparse_attn_prefill_fallback_rejects_invalid_fp8(supports_fp8, value_dtype) -> None:
    q = torch.zeros(1, 1, 4, dtype=torch.bfloat16)
    kv_cache = (torch.zeros(1, 128, 1, 4, dtype=torch.float8_e4m3fn), torch.zeros(1, 128, 1, 4, dtype=value_dtype))
    with (
        patch("torch.ops._C_ascend.npu_sparse_attention_score", create=True) as op,
        pytest.raises(TypeError, match="FP8 sparse attention"),
    ):
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
            supports_fp8=supports_fp8,
        )
    op.assert_not_called()


@pytest.mark.parametrize(
    ("device_type", "split_kv_available", "expected_impl"),
    [
        (AscendDeviceType.A5, True, "kv_gather_q"),
        (AscendDeviceType.A5, False, "legacy"),
        (AscendDeviceType.A3, True, "kv_gather_q"),
        (AscendDeviceType.A3, False, "legacy"),
        (AscendDeviceType.A2, True, "kv_gather_q"),
        (AscendDeviceType.A2, False, "legacy"),
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
        patch.object(msa_m3_npu_module, "_minimax_m3_sparse_attn_a3") as mock_legacy,
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
        mock_legacy.assert_not_called()
        assert mock_kv_gather_q.call_args.args[-2:] == (5, 3)
        assert mock_kv_gather_q.call_args.kwargs == {
            "supports_fp8": device_type == AscendDeviceType.A5,
        }
    else:
        mock_legacy.assert_called_once()
        mock_kv_gather_q.assert_not_called()
        assert mock_legacy.call_args.kwargs == {
            "supports_fp8": device_type == AscendDeviceType.A5,
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
    monkeypatch.setattr(msa_m3_module, "get_ascend_device_type", lambda: device_type)
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
    impl.index_cache = SimpleNamespace(prefix="index", kv_cache=torch.zeros(258, 128, 4))
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
