# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

import vllm_ascend.attention.indexer_kpool as backend_module
from vllm_ascend.attention.indexer_kpool import (
    AscendIndexerKPoolMetadata,
    AscendIndexerKPoolTailMetadata,
    Glm5NextKPoolIndexerBackend,
)
from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import SparseAttnIndexerKpool, append_causal_tail


def test_cache_metadata_import_does_not_require_indexer_operators() -> None:
    module_name = f"{backend_module.__name__}_import_test"
    spec = importlib.util.spec_from_file_location(module_name, backend_module.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        "sys.modules",
        {
            module_name: module,
            "vllm_ascend.attention.indexer": None,
            "vllm_ascend.models.glm5next.sparse_attn_indexer_kpool": None,
        },
    ):
        spec.loader.exec_module(module)

    assert module.AscendIndexerKPoolBackend.get_builder_cls() is module.AscendIndexerKPoolMetadataBuilder
    assert module.AscendIndexerKPoolTailBackend.get_builder_cls() is module.AscendIndexerKPoolTailMetadataBuilder


@pytest.mark.parametrize("pool_size", [1, 4])
def test_causal_tail_follows_valid_history_without_holes(pool_size: int) -> None:
    topk = 8
    positions = torch.tensor([0, 1, 2, 3, 4, 6, 7, 8, 10, 15, 18])
    indices = torch.full((positions.numel(), topk + pool_size - 1), -1, dtype=torch.int32)
    expected_rows = []
    for row, position in enumerate(positions.tolist()):
        tail_start = (position + 1) // pool_size * pool_size
        # Reverse the selected pools to ensure their ranking is preserved.
        groups = list(reversed(range(min(tail_start // pool_size, topk // pool_size))))
        history = [group * pool_size + offset for group in groups for offset in range(pool_size)]
        indices[row, : len(history)] = torch.tensor(history, dtype=torch.int32)
        tail = list(range(tail_start, position + 1))
        if tail:
            indices[row, topk : topk + len(tail)] = torch.tensor(tail, dtype=torch.int32)
        expected_rows.append(history + tail + [-1] * (indices.shape[1] - len(history) - len(tail)))

    pointer = indices.data_ptr()
    append_causal_tail(indices, positions, topk, pool_size)

    assert indices.data_ptr() == pointer
    torch.testing.assert_close(indices, torch.tensor(expected_rows, dtype=torch.int32))


def _indexer_metadata(num_tokens: int = 8) -> AscendIndexerKPoolMetadata:
    return AscendIndexerKPoolMetadata(
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        slot_mapping=torch.tensor(
            [-1, -1, -1, 0, -1, -1, -1, 2, -1, -1],
            dtype=torch.int64,
        )[:num_tokens],
        seq_lens=torch.tensor([1, 1], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([1, 1], dtype=torch.int32),
        positions=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 0])[:num_tokens],
        block_size=2,
        compress_ratio=4,
        cum_query_lens=torch.tensor([4, 8], dtype=torch.int64),
        raw_seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        num_actual_tokens=8,
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        start_pos=torch.zeros(2, dtype=torch.int32),
        pool_tail=torch.zeros(2, dtype=torch.int64),
        pooled_key_indices=torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])[:num_tokens],
    )


def _tail_metadata() -> AscendIndexerKPoolTailMetadata:
    return AscendIndexerKPoolTailMetadata(
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        slot_mapping=torch.full((8,), -1, dtype=torch.int64),
        block_size=4,
    )


@pytest.mark.parametrize("compute_topk", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cann_indexer_updates_both_caches_and_masks_padding(monkeypatch, compute_topk, dtype):
    metadata = _indexer_metadata(num_tokens=10)
    tail_metadata = _tail_metadata()
    tail_metadata.slot_mapping = torch.arange(10)
    indexer_cache = torch.zeros(2, 2, 1, 2, dtype=torch.bfloat16)
    # Keep padded block strides, as in the hybrid cache allocator.
    backing = torch.zeros(2, 32)
    tail_cache = backing[:, :16].view(2, 4, 4)

    def compress(hidden, wk, gate, ape, state, table, starts, **kwargs):
        assert hidden.dtype == wk.dtype == gate.dtype == dtype
        assert state.dtype == ape.dtype == torch.float32
        assert kwargs["cmp_ratio"] == 4
        assert kwargs["cu_seqlens"] is metadata.query_start_loc
        assert starts is metadata.start_pos
        assert table is tail_metadata.block_table
        assert state.stride(0) == 32
        state[0, 0].fill_(7)
        return torch.tensor([[11, 11], [22, 22]], dtype=dtype)

    select = MagicMock(return_value=(torch.full((10, 7), -1, dtype=torch.int32), torch.empty(0)))
    monkeypatch.setattr(torch.ops._C_ascend, "npu_key_pool", compress, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_pool_key_indexer", select, raising=False)
    result = SparseAttnIndexerKpool(4, 2)(
        torch.zeros(10, 3, dtype=dtype),
        torch.zeros(10, 1, 2, dtype=dtype),
        torch.ones(10, 1, dtype=dtype),
        metadata.positions,
        indexer_cache,
        tail_cache,
        metadata,
        tail_metadata,
        key_weight=torch.zeros(2, 3, dtype=dtype),
        gate_weight=torch.zeros(2, 3, dtype=dtype),
        norm_weight=torch.ones(2),
        norm_bias=torch.zeros(2),
        norm_eps=1e-6,
        compress_ape=torch.zeros(4, 2),
        index_kpool=4,
        compute_topk=compute_topk,
    )
    torch.testing.assert_close(tail_cache[0, 0], torch.full_like(tail_cache[0, 0], 7))
    torch.testing.assert_close(indexer_cache[0, 0], torch.full_like(indexer_cache[0, 0], 11))
    torch.testing.assert_close(indexer_cache[1, 0], torch.full_like(indexer_cache[1, 0], 22))
    assert not backing[:, 16:].count_nonzero()
    if compute_topk:
        assert result.shape == (10, 1, 7)
        assert (result[8:] == -1).all()
        select.assert_called_once()
        kwargs = select.call_args.kwargs
        assert kwargs["actual_seq_q"] is metadata.cum_query_lens
        assert kwargs["actual_seq_k"] is metadata.seq_lens
        assert select.call_args.args[3] is metadata.pool_tail
        assert select.call_args.args[0].dtype == select.call_args.args[2].dtype == torch.bfloat16
        assert kwargs["layout_k"] == "PA_BBND"
    else:
        assert result is None
        select.assert_not_called()


@pytest.mark.parametrize(
    ("pcp_size", "dcp_size"),
    [(2, 1), (1, 2)],
)
def test_kpool_backend_rejects_context_parallelism(pcp_size: int, dcp_size: int) -> None:
    source = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                prefill_context_parallel_size=pcp_size,
                decode_context_parallel_size=dcp_size,
            )
        )
    )

    with pytest.raises(NotImplementedError, match="PCP or DCP"):
        Glm5NextKPoolIndexerBackend(source, qk_rope_head_dim=0)


@pytest.mark.parametrize("compute_topk", [False, True])
def test_backend_zero_token_batch_does_not_launch_operators(monkeypatch, compute_topk):
    compress = MagicMock()
    select = MagicMock()
    monkeypatch.setattr(torch.ops._C_ascend, "npu_key_pool", compress, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_pool_key_indexer", select, raising=False)
    result = SparseAttnIndexerKpool(4, 2)(
        torch.empty(0, 2),
        torch.empty(0, 1, 2),
        torch.empty(0, 1),
        torch.empty(0, dtype=torch.int64),
        torch.zeros(2, 2, 1, 2, dtype=torch.bfloat16),
        torch.zeros(2, 4, 4),
        _indexer_metadata(0),
        _tail_metadata(),
        key_weight=torch.empty(2, 3, dtype=torch.bfloat16),
        gate_weight=torch.empty(2, 3, dtype=torch.bfloat16),
        norm_weight=torch.ones(2),
        norm_bias=torch.zeros(2),
        norm_eps=1e-6,
        compress_ape=torch.zeros(4, 2),
        index_kpool=4,
        compute_topk=compute_topk,
    )
    if compute_topk:
        assert result.shape == (0, 1, 7)
    else:
        assert result is None
    compress.assert_not_called()
    select.assert_not_called()


@pytest.mark.parametrize("compute_topk", [False, True])
def test_cann_empty_pool_keeps_cache_and_packs_tail_before_padding(monkeypatch, compute_topk):
    metadata = _indexer_metadata(4)
    metadata.positions = torch.tensor([0, 1, 2, 3])
    metadata.query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    metadata.cum_query_lens = torch.tensor([3], dtype=torch.int64)
    metadata.start_pos = torch.tensor([0], dtype=torch.int32)
    metadata.seq_lens = torch.tensor([0], dtype=torch.int64)
    metadata.pool_tail = torch.tensor([3], dtype=torch.int64)
    metadata.pooled_key_indices = torch.zeros(4, dtype=torch.int64)
    metadata.slot_mapping = torch.full((4,), -1, dtype=torch.int64)
    cache = torch.full((2, 2, 1, 2), 9, dtype=torch.bfloat16)
    # No output row is defined when no pool completes. Reading a safe row
    # must not write that garbage into slot zero or propagate its NaNs.
    compress = MagicMock(return_value=torch.full((2, 2), float("nan"), dtype=torch.bfloat16))
    select = MagicMock(return_value=(torch.full((4, 7), -1, dtype=torch.int32), torch.empty(0)))
    monkeypatch.setattr(torch.ops._C_ascend, "npu_key_pool", compress, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_pool_key_indexer", select, raising=False)
    result = SparseAttnIndexerKpool(4, 2)(
        torch.zeros(4, 3, dtype=torch.bfloat16),
        torch.zeros(4, 1, 2, dtype=torch.bfloat16) if compute_topk else None,
        torch.ones(4, 1, dtype=torch.bfloat16) if compute_topk else None,
        metadata.positions,
        cache,
        torch.zeros(2, 4, 4),
        metadata,
        _tail_metadata(),
        key_weight=torch.zeros(2, 3, dtype=torch.bfloat16),
        gate_weight=torch.zeros(2, 3, dtype=torch.bfloat16),
        norm_weight=torch.ones(2),
        norm_bias=torch.zeros(2),
        norm_eps=1e-6,
        compress_ape=torch.zeros(4, 2),
        index_kpool=4,
        compute_topk=compute_topk,
    )
    torch.testing.assert_close(cache, torch.full_like(cache, 9))
    compress.assert_called_once()
    if compute_topk:
        assert result[:, 0].tolist() == [
            [0, -1, -1, -1, -1, -1, -1],
            [0, 1, -1, -1, -1, -1, -1],
            [0, 1, 2, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
        ]
    else:
        assert result is None
        select.assert_not_called()


@pytest.mark.parametrize("invalid", ["tail_dtype", "tail_shape", "metadata", "query", "ape"])
def test_invalid_cann_inputs_fail_before_mutating_cache(monkeypatch, invalid):
    metadata = _indexer_metadata(8)
    tail = torch.zeros(2, 4, 4)
    query = torch.zeros(8, 1, 2, dtype=torch.bfloat16)
    ape = torch.zeros(4, 2)
    if invalid == "tail_dtype":
        tail = tail.bfloat16()
    elif invalid == "tail_shape":
        tail = torch.zeros(2, 2, 4, 2)
    elif invalid == "metadata":
        metadata.query_start_loc = None
    elif invalid == "query":
        query = None
    else:
        ape = ape.bfloat16()
    compress = MagicMock()
    monkeypatch.setattr(torch.ops._C_ascend, "npu_key_pool", compress, raising=False)
    with pytest.raises((ValueError, TypeError)):
        SparseAttnIndexerKpool(4, 2)(
            torch.zeros(8, 3, dtype=torch.bfloat16),
            query,
            torch.ones(8, 1, dtype=torch.bfloat16),
            metadata.positions,
            torch.zeros(2, 2, 1, 2, dtype=torch.bfloat16),
            tail,
            metadata,
            _tail_metadata(),
            key_weight=torch.zeros(2, 3, dtype=torch.bfloat16),
            gate_weight=torch.zeros(2, 3, dtype=torch.bfloat16),
            norm_weight=torch.ones(2),
            norm_bias=torch.zeros(2),
            norm_eps=1e-6,
            compress_ape=ape,
            index_kpool=4,
            compute_topk=True,
        )
    compress.assert_not_called()


class _Projection(nn.Module):
    def forward(self, q_c):
        return q_c.repeat(1, 2), None


class _RecordingKPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.args = None
        self.kwargs = None

    def forward(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        if kwargs["compute_topk"]:
            return torch.zeros(args[0].shape[0], 1, 5, dtype=torch.int32)
        return None


def test_backend_uses_normalized_q_c_and_separate_tail_metadata(
    monkeypatch,
) -> None:
    backend = Glm5NextKPoolIndexerBackend.__new__(Glm5NextKPoolIndexerBackend)
    nn.Module.__init__(backend)
    backend.n_head = 2
    backend.head_dim = 2
    backend.topk_tokens = 2
    backend.index_kpool = 4
    backend.wq_b = _Projection()
    backend.wk_weights_proj = nn.Linear(3, 4, bias=False)
    backend.k_norm = nn.LayerNorm(2)
    backend.index_kpool_compress_ape = nn.Parameter(torch.zeros(4, 2))
    backend.index_kpool_compress_gate = nn.Parameter(torch.zeros(2, 3))
    backend.k_cache = SimpleNamespace(
        prefix="indexer.k_cache",
        kv_cache=torch.zeros(2, 2, 1, 2, dtype=torch.bfloat16),
    )
    backend.tail_cache = SimpleNamespace(
        prefix="indexer.tail",
        kv_cache=torch.zeros(2, 4, 4, dtype=torch.float32),
    )
    backend.topk_indices_buffer = None
    backend.softmax_scale = 0.5
    backend._key_weight = None
    backend.indexer_op = _RecordingKPool()
    tail_metadata = _tail_metadata()
    monkeypatch.setattr(
        backend_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={"indexer.tail": tail_metadata},
            cudagraph_runtime_mode=None,
            virtual_engine=0,
        ),
    )

    normalized_q_c = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    hidden = torch.ones(2, 3)
    metadata = _indexer_metadata(num_tokens=2)
    metadata.num_actual_tokens = 2
    metadata.cum_query_lens = torch.tensor([1, 2], dtype=torch.int32)
    metadata.raw_seq_lens = torch.tensor([1, 1], dtype=torch.int32)
    metadata.seq_lens = torch.tensor([0, 0], dtype=torch.int32)
    metadata.positions = torch.tensor([0, 0])
    metadata.slot_mapping = torch.tensor([-1, -1])

    result = backend.forward(
        hidden,
        normalized_q_c,
        None,
        None,
        hidden + 3,
        metadata,
        compute_topk=True,
    )

    assert result is not None
    assert backend.indexer_op.args is not None
    torch.testing.assert_close(backend.indexer_op.args[1], normalized_q_c.repeat(1, 2).view(2, 2, 2))
    torch.testing.assert_close(backend.indexer_op.args[0], hidden + 3)
    torch.testing.assert_close(backend.indexer_op.kwargs["key_weight"], backend.wk_weights_proj.weight[:2])
    torch.testing.assert_close(backend.indexer_op.kwargs["norm_weight"], backend.k_norm.weight.float())
    assert backend.indexer_op.kwargs["norm_eps"] == backend.k_norm.eps
    expected_weights = torch.nn.functional.linear(hidden, backend.wk_weights_proj.weight[2:]) * (0.5 * 2**-0.5)
    torch.testing.assert_close(backend.indexer_op.args[2], expected_weights)
    assert backend.indexer_op.args[7] is tail_metadata
    assert backend.indexer_op.kwargs is not None
    assert backend.indexer_op.kwargs["compute_topk"] is True
