# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch_npu
from torch import nn

import vllm_ascend.attention.indexer_kpool as backend_module
from vllm_ascend.attention.indexer_kpool import (
    AscendIndexerKPoolMetadata,
    AscendIndexerKPoolTailMetadata,
    Glm5NextKPoolIndexerBackend,
)
from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import (
    SparseAttnIndexerKpool,
    append_causal_tail,
)
from vllm_ascend.ops.paged_cache import write_pooled_cache


@pytest.fixture(autouse=True)
def mock_cann_package(monkeypatch):
    # CPU unit tests mock the external package; production owns no kernels.
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", ModuleType("cann_ops_transformer"))
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "key_pool", MagicMock(), raising=False)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "pool_key_indexer", MagicMock(), raising=False)

    def scatter(cache, indices, values):
        for index, value in zip(indices[:, 0].tolist(), values):
            if index >= 0:
                cache[index].copy_(value)

    monkeypatch.setattr(torch_npu, "npu_scatter_nd_update_", scatter)


@pytest.mark.parametrize("page_padding", [0, 16])
def test_pooled_cache_scatter_preserves_padded_pages_and_invalid_slots(page_padding):
    backing = torch.full((3 * (8 + page_padding) + 8,), 17, dtype=torch.bfloat16)
    cache = backing.as_strided((3, 4, 1, 2), (8 + page_padding, 2, 2, 1), storage_offset=4)
    expected = backing.clone()
    expected_cache = expected.as_strided(cache.shape, cache.stride(), storage_offset=4)
    expected_cache[0, 0] = 11
    expected_cache[1, 3] = 22
    expected_cache[2, 3] = 33
    values = torch.tensor([[float("nan")] * 2, [11] * 2, [22] * 2, [33] * 2, [float("nan")] * 2]).bfloat16()
    write_pooled_cache(cache, torch.tensor([-1, 0, 7, 11, 12]), values)
    torch.testing.assert_close(backing, expected, rtol=0, atol=0)


@pytest.mark.parametrize("missing", ["package", "key_pool", "pool_key_indexer"])
def test_missing_cann_dependency_fails_at_backend_construction(monkeypatch, missing):
    if missing == "package":
        monkeypatch.setitem(sys.modules, "cann_ops_transformer", None)
    else:
        monkeypatch.delattr(torch.ops.cann_ops_transformer, missing)
    with pytest.raises(RuntimeError, match="matching CANN ops-transformer operator package"):
        SparseAttnIndexerKpool(2048, 128)


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
            "cann_ops_transformer": None,
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
        torch.testing.assert_close(starts, metadata.start_pos % 4)
        torch.testing.assert_close(table[:, 0], torch.tensor([1, 2], dtype=torch.int32))
        assert (table[:, 1:] == torch.tensor([[3], [4]], dtype=torch.int32)).all()
        assert state.data_ptr() != tail_cache.data_ptr()
        state[3:, 0].fill_(7)
        return torch.tensor([[11, 11], [22, 22]], dtype=dtype)

    select = MagicMock(return_value=(torch.full((10, 7), -1, dtype=torch.int32), torch.empty(0)))
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "key_pool", compress, raising=False)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "pool_key_indexer", select, raising=False)
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
    # Both requests completed their pools, so scratch tail writes are not live.
    assert not tail_cache.count_nonzero()
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


@pytest.mark.parametrize("lookahead", [0, 1, 3, 7])
def test_kpool_backend_constructs_with_existing_cache_modules(monkeypatch, lookahead):
    monkeypatch.setattr(
        backend_module,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(attention_backend_family=backend_module.AttentionBackendFamily.STANDARD),
    )
    source = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
            speculative_config=SimpleNamespace(num_speculative_tokens=lookahead) if lookahead else None,
        ),
        n_head=1,
        head_dim=2,
        topk_tokens=4,
        q_lora_rank=2,
        index_kpool=4,
        wq_b=nn.Linear(2, 2, bias=False),
        wk_weights_proj=nn.Linear(3, 3, bias=False),
        k_norm=nn.LayerNorm(2),
        softmax_scale=0.5,
        index_kpool_compress_ape=nn.Parameter(torch.zeros(4, 2)),
        index_kpool_compress_gate=nn.Parameter(torch.zeros(2, 3)),
        k_cache=SimpleNamespace(),
        tail_cache=SimpleNamespace(),
        topk_indices_buffer=None,
    )

    backend = Glm5NextKPoolIndexerBackend(source, qk_rope_head_dim=0)

    assert isinstance(backend.indexer_op, SparseAttnIndexerKpool)
    assert backend.wq_b is source.wq_b
    assert backend.k_cache is source.k_cache
    assert backend.tail_cache is source.tail_cache


@pytest.mark.parametrize("compute_topk", [False, True])
def test_backend_zero_token_batch_does_not_launch_operators(monkeypatch, compute_topk):
    compress = MagicMock()
    select = MagicMock()
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "key_pool", compress, raising=False)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "pool_key_indexer", select, raising=False)
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
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "key_pool", compress, raising=False)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "pool_key_indexer", select, raising=False)
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
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "key_pool", compress, raising=False)
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


@pytest.mark.parametrize("prefix_length", [3, 4, 27])
@pytest.mark.parametrize(("lookahead", "accepted"), [(0, 4), (3, 1), (3, 2), (3, 3), (3, 4)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cann_tail_survives_speculative_rejection(monkeypatch, prefix_length, lookahead, accepted, dtype):
    """Exercise the real framework around CANN's incomplete-pool contract."""
    torch.manual_seed(13)
    ratio, capacity, dim = 4, 4 + lookahead, 2
    key_weight = torch.randn(dim, 3).to(dtype)
    gate_weight = torch.randn(dim, 3).to(dtype)
    norm_weight, norm_bias = torch.randn(dim), torch.randn(dim)
    ape = torch.randn(ratio, dim)
    backing = torch.full((3, capacity * 2 * dim + 8), 17.0)
    tail = backing[:, : capacity * 2 * dim].view(3, capacity, 2 * dim)
    pooled = torch.zeros(1, 64, 1, dim, dtype=torch.bfloat16)

    def project(hidden):
        key = (hidden.float() @ key_weight.float().T).to(dtype).float()
        mean = key.mean(-1, keepdim=True)
        variance = ((key - mean) ** 2).mean(-1, keepdim=True)
        key = (key - mean) * torch.rsqrt(variance + 1e-6) * norm_weight + norm_bias
        gate = (hidden.float() @ gate_weight.float().T).to(dtype).float()
        return key, gate

    def pool(keys, gates):
        probability = (gates + ape).softmax(0).to(dtype).float()
        return (keys * probability).sum(0).bfloat16()

    def cann_compress(hidden, wk, gate_w, position_bias, state, table, starts, **kwargs):
        start = int(starts[0])
        end = start + hidden.shape[0]
        key, gate = project(hidden)
        # The external implementation can save the tail before consuming the
        # historical prefix. Stage separate pages to avoid ring aliasing.
        for position in range(max(start, end - end % ratio), end):
            page = table[0, position // state.shape[1]]
            row = position % state.shape[1]
            state[page, row, :dim] = key[position - start]
            state[page, row, dim:] = gate[position - start]
        values = []
        for group in range(start // ratio, end // ratio):
            keys, gates = [], []
            for position in range(group * ratio, (group + 1) * ratio):
                if position >= start:
                    keys.append(key[position - start])
                    gates.append(gate[position - start])
                else:
                    page = table[0, position // state.shape[1]]
                    row = position % state.shape[1]
                    keys.append(state[page, row, :dim])
                    gates.append(state[page, row, dim:])
            values.append(pool(torch.stack(keys), torch.stack(gates)))
        # This deliberately matches the external dependency: completed pools
        # do not persist their raw rows. The framework must retain them.
        return torch.stack(values) if values else hidden.new_zeros(1, dim)

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "key_pool", cann_compress, raising=False)
    op = SparseAttnIndexerKpool(4, dim)

    def run(hidden, start, committed):
        count = hidden.shape[0]
        positions = torch.arange(start, start + count)
        boundaries = (positions + 1) % ratio == 0
        retained = torch.arange(count - capacity, count).clamp_min(-1).view(1, -1)
        metadata = AscendIndexerKPoolMetadata(
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            slot_mapping=torch.where(boundaries, positions // ratio, -1),
            seq_lens=torch.tensor([(start + count) // ratio]),
            positions=positions,
            block_size=64,
            compress_ratio=ratio,
            cum_query_lens=torch.tensor([count]),
            query_start_loc=torch.tensor([0, count], dtype=torch.int32),
            start_pos=torch.tensor([start], dtype=torch.int32),
            pool_tail=torch.tensor([(start + count) % ratio]),
            pooled_key_indices=boundaries.long().cumsum(0).sub(1).clamp_min(0),
            retained_tail_indices=retained if lookahead else None,
        )
        tail_metadata = AscendIndexerKPoolTailMetadata(
            block_table=torch.ones(1, 64, dtype=torch.int32),
            slot_mapping=capacity + positions % capacity,
            block_size=capacity,
        )
        assert (
            op(
                hidden,
                None,
                None,
                positions,
                pooled,
                tail,
                metadata,
                tail_metadata,
                key_weight=key_weight,
                gate_weight=gate_weight,
                norm_weight=norm_weight,
                norm_bias=norm_bias,
                norm_eps=1e-6,
                compress_ape=ape,
                index_kpool=ratio,
                compute_topk=False,
            )
            is None
        )
        keys, gates = project(torch.cat((committed, hidden)))
        for group in range(keys.shape[0] // ratio):
            expected = pool(keys[group * ratio : (group + 1) * ratio], gates[group * ratio : (group + 1) * ratio])
            torch.testing.assert_close(pooled[0, group, 0], expected, atol=0.02, rtol=0.02)
        first_live = max(0, keys.shape[0] - capacity) if lookahead else keys.shape[0] // ratio * ratio
        for position in range(first_live, keys.shape[0]):
            torch.testing.assert_close(
                tail[1, position % capacity], torch.cat((keys[position], gates[position])), atol=1e-5, rtol=1e-5
            )
        torch.testing.assert_close(backing[[0, 2]], torch.full_like(backing[[0, 2]], 17))
        torch.testing.assert_close(backing[:, -8:], torch.full_like(backing[:, -8:], 17))

    prefix = torch.randn(prefix_length, 3).to(dtype)
    run(prefix, 0, prefix[:0])
    draft = torch.randn(4, 3).to(dtype)
    run(draft, prefix_length, prefix)
    committed = torch.cat((prefix, draft[:accepted]))
    continuation = torch.randn(8, 3).to(dtype)
    run(continuation, committed.shape[0], committed)


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
        hidden + 3,
        metadata,
        compute_topk=True,
    )

    assert result is not None
    assert backend.indexer_op.args is not None
    assert backend.indexer_op.kwargs is not None
    torch.testing.assert_close(backend.indexer_op.args[1], normalized_q_c.repeat(1, 2).view(2, 2, 2))
    torch.testing.assert_close(backend.indexer_op.args[0], hidden + 3)
    torch.testing.assert_close(backend.indexer_op.kwargs["key_weight"], backend.wk_weights_proj.weight[:2])
    torch.testing.assert_close(backend.indexer_op.kwargs["norm_weight"], backend.k_norm.weight.float())
    assert backend.indexer_op.kwargs["norm_eps"] == backend.k_norm.eps
    expected_weights = torch.nn.functional.linear(hidden, backend.wk_weights_proj.weight[2:]) * (0.5 * 2**-0.5)
    torch.testing.assert_close(backend.indexer_op.args[2], expected_weights)
    assert backend.indexer_op.args[7] is tail_metadata
    assert backend.indexer_op.kwargs["compute_topk"] is True
