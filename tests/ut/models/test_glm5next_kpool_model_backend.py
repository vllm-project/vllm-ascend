# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

import vllm_ascend.attention.indexer_kpool as backend_module
from vllm_ascend.attention.indexer_kpool import (
    AscendIndexerKPoolMetadata,
    AscendIndexerKPoolStateMetadata,
    Glm5NextKPoolIndexerBackend,
)

# The operator PR owns these kernels. Each orchestration test below replaces
# the kernel call with an asserting CPU implementation.
with patch.dict(
    "sys.modules",
    {
        "vllm_ascend.ops.triton.glm5_next_kpool_state_compress": MagicMock(),
        "vllm_ascend.ops.triton.glm5_next_lightning_indexer": MagicMock(),
    },
):
    import vllm_ascend.models.glm5next.sparse_attn_indexer_kpool as kpool_module
    from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import SparseAttnIndexerKpool, append_causal_tail


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
        seq_lens=torch.tensor([1, 1], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([1, 1], dtype=torch.int32),
        positions=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 0])[:num_tokens],
        block_size=2,
        compress_ratio=4,
        cum_query_lens=torch.tensor([4, 8], dtype=torch.int32),
        raw_seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        num_actual_tokens=8,
    )


def _state_metadata() -> AscendIndexerKPoolStateMetadata:
    return AscendIndexerKPoolStateMetadata(
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        slot_mapping=torch.full((8,), -1, dtype=torch.int64),
        block_size=4,
        cache_role="indexer_state",
    )


@pytest.mark.parametrize("compute_topk", [False, True])
def test_triton_indexer_updates_both_caches_and_masks_padding(monkeypatch, compute_topk):
    metadata = _indexer_metadata(num_tokens=10)
    state_metadata = _state_metadata()
    state_metadata.slot_mapping = torch.arange(10)
    indexer_cache = torch.zeros(2, 2, 1, 2, dtype=torch.bfloat16)
    state_cache = torch.zeros(2, 4, 1, 4)

    def compress(state, cache, k, gate, ape, positions, query_ends, seq_lens, state_slots, table, indexer_slots, pool):
        assert k.dtype == gate.dtype == state.dtype == torch.float32
        assert pool == 4
        torch.testing.assert_close(query_ends, metadata.cum_query_lens)
        torch.testing.assert_close(seq_lens, metadata.raw_seq_lens)
        assert table is state_metadata.block_table
        torch.testing.assert_close(indexer_slots, metadata.slot_mapping)
        state[0, 0].fill_(7)
        cache[0, 0].fill_(11)

    select = MagicMock(return_value=torch.full((10, 1, 7), -1, dtype=torch.int32))
    monkeypatch.setattr(kpool_module, "glm5_next_kpool_state_compress_and_write_cache_triton", compress)
    monkeypatch.setattr(kpool_module, "glm5_next_lightning_indexer_triton", select)
    result = SparseAttnIndexerKpool(4, 2)(
        torch.zeros(10, 2),
        torch.zeros(10, 1, 2, dtype=torch.bfloat16),
        torch.ones(10, 1, dtype=torch.bfloat16),
        metadata.positions,
        indexer_cache,
        state_cache,
        metadata,
        state_metadata,
        gate_score=torch.zeros(10, 2),
        compress_ape=torch.zeros(4, 2),
        index_kpool=4,
        max_pool_seq_len=1,
        compute_topk=compute_topk,
    )
    torch.testing.assert_close(state_cache[0, 0], torch.full_like(state_cache[0, 0], 7))
    torch.testing.assert_close(indexer_cache[0, 0], torch.full_like(indexer_cache[0, 0], 11))
    if compute_topk:
        assert result.shape == (10, 1, 7)
        assert (result[8:] == -1).all()
        select.assert_called_once()
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


def test_backend_uses_normalized_q_c_and_separate_state_metadata(
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
    backend.state_cache = SimpleNamespace(
        prefix="indexer.state",
        kv_cache=torch.zeros(2, 4, 1, 4, dtype=torch.float32),
    )
    backend.topk_indices_buffer = None
    backend.softmax_scale = 0.5
    backend._wk_weight_f32 = None
    backend.indexer_op = _RecordingKPool()
    state_metadata = _state_metadata()
    monkeypatch.setattr(
        backend_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={"indexer.state": state_metadata},
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
    expected_k = torch.nn.functional.layer_norm(
        torch.nn.functional.linear(hidden + 3, backend.wk_weights_proj.weight)[:, :2],
        (2,),
        backend.k_norm.weight,
        backend.k_norm.bias,
        backend.k_norm.eps,
    )
    torch.testing.assert_close(backend.indexer_op.args[0], expected_k)
    assert backend.indexer_op.args[0].dtype == torch.float32
    expected_weights = torch.nn.functional.linear(hidden, backend.wk_weights_proj.weight[2:]) * (0.5 * 2**-0.5)
    torch.testing.assert_close(backend.indexer_op.args[2], expected_weights)
    assert backend.indexer_op.args[7] is state_metadata
    assert backend.indexer_op.kwargs is not None
    assert backend.indexer_op.kwargs["compute_topk"] is True
