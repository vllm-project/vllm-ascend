# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from vllm.config.compilation import CUDAGraphMode

import vllm_ascend.attention.indexer_kpool as backend_module
import vllm_ascend.ops.linear as linear_module
import vllm_ascend.utils as utils_module
from vllm_ascend.attention.indexer_kpool import (
    AscendIndexerKPoolMetadata,
    AscendIndexerKPoolTailMetadata,
    Glm5NextKPoolIndexerBackend,
)
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile

with patch.dict(
    "sys.modules",
    {
        "vllm_ascend.ops.triton.glm5_next_kpool_tail_compress": MagicMock(),
        "vllm_ascend.ops.triton.glm5_next_lightning_indexer": MagicMock(),
    },
):
    import vllm_ascend.models.glm5next.sparse_attn_indexer_kpool as kpool_module
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
        seq_lens=torch.tensor([1, 1], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([1, 1], dtype=torch.int32),
        positions=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 0])[:num_tokens],
        block_size=2,
        compress_ratio=4,
        cum_query_lens=torch.tensor([4, 8], dtype=torch.int32),
        raw_seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        num_actual_tokens=8,
    )


def _tail_metadata() -> AscendIndexerKPoolTailMetadata:
    return AscendIndexerKPoolTailMetadata(
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        slot_mapping=torch.full((8,), -1, dtype=torch.int64),
        block_size=4,
    )


@pytest.mark.parametrize("compute_topk,allow_cache_packing", [(False, True), (True, True), (True, False)])
def test_triton_indexer_updates_both_caches_and_masks_padding(monkeypatch, compute_topk, allow_cache_packing):
    metadata = _indexer_metadata(num_tokens=10)
    tail_metadata = _tail_metadata()
    tail_metadata.slot_mapping = torch.arange(10)
    indexer_cache = torch.zeros(2, 2, 1, 2, dtype=torch.bfloat16)
    tail_cache = torch.zeros(2, 2, 4, 2)

    def compress(state, cache, k, gate, ape, positions, query_ends, seq_lens, state_slots, table, indexer_slots, pool):
        assert k.dtype == gate.dtype == state.dtype == torch.float32
        assert pool == 4
        torch.testing.assert_close(query_ends, metadata.cum_query_lens)
        torch.testing.assert_close(seq_lens, metadata.raw_seq_lens)
        assert table is tail_metadata.block_table
        torch.testing.assert_close(indexer_slots, metadata.slot_mapping)
        state[0, 0].fill_(7)
        cache[0, 0].fill_(11)

    select = MagicMock(return_value=torch.full((10, 1, 7), -1, dtype=torch.int32))
    output_buffer = torch.empty(10, 16, dtype=torch.int32)
    monkeypatch.setattr(kpool_module, "glm5_next_kpool_tail_compress_and_write_cache_triton", compress)
    monkeypatch.setattr(kpool_module, "glm5_next_lightning_indexer_triton", select)
    result = SparseAttnIndexerKpool(4, 2)(
        torch.zeros(10, 2),
        torch.zeros(10, 1, 2, dtype=torch.bfloat16),
        torch.ones(10, 1, dtype=torch.bfloat16),
        metadata.positions,
        indexer_cache,
        tail_cache,
        metadata,
        tail_metadata,
        gate_score=torch.zeros(10, 2),
        compress_ape=torch.zeros(4, 2),
        index_kpool=4,
        max_pool_seq_len=1,
        compute_topk=compute_topk,
        output_buffer=output_buffer,
        allow_cache_packing=allow_cache_packing,
    )
    torch.testing.assert_close(tail_cache[0, 0], torch.full_like(tail_cache[0, 0], 7))
    torch.testing.assert_close(indexer_cache[0, 0], torch.full_like(indexer_cache[0, 0], 11))
    if compute_topk:
        assert result.shape == (10, 1, 7)
        assert (result[8:] == -1).all()
        select.assert_called_once()
        assert select.call_args.kwargs["output_buffer"] is output_buffer
        assert select.call_args.kwargs["pack_tail"] is True
        assert select.call_args.kwargs["allow_cache_packing"] is allow_cache_packing
    else:
        assert result is None
        select.assert_not_called()


def test_kpool_cache_backends_support_pcp() -> None:
    assert backend_module.AscendIndexerKPoolBackend.supports_pcp()
    assert backend_module.AscendIndexerKPoolTailBackend.supports_pcp()


@pytest.mark.parametrize("weight_nz_mode", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_kpool_weight_postprocessing_keeps_nd_before_fp32_cast(monkeypatch, weight_nz_mode, dtype):
    profile = get_hardware_profile(AscendDeviceType.A5)
    for module in (backend_module, linear_module, utils_module):
        monkeypatch.setattr(module, "get_current_hardware_profile", lambda: profile)
    monkeypatch.setattr(utils_module, "get_ascend_config", lambda: SimpleNamespace(weight_nz_mode=weight_nz_mode))
    monkeypatch.setattr(Glm5NextKPoolIndexerBackend, "_init_cann_ops", lambda self: None)

    source = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1)
        ),
        n_head=2,
        head_dim=4,
        topk_tokens=8,
        q_lora_rank=4,
        index_kpool=4,
        wq_b=nn.Linear(4, 8, bias=False, dtype=dtype),
        wk_weights_proj=nn.Linear(8, 6, bias=False, dtype=dtype),
        k_norm=nn.LayerNorm(4, dtype=dtype),
        softmax_scale=0.5,
        index_kpool_compress_ape=nn.Parameter(torch.zeros(4, 4, dtype=dtype)),
        index_kpool_compress_gate=nn.Parameter(torch.randn(4, 8, dtype=dtype)),
        k_cache=SimpleNamespace(prefix="indexer.k_cache"),
        tail_cache=SimpleNamespace(prefix="indexer.tail"),
        topk_indices_buffer=None,
    )
    source.wk_weights_proj.prefix = "model.layers.0.indexer.wk_weights_proj"
    source.wq_b.prefix = "model.layers.0.indexer.wq_b"
    backend = Glm5NextKPoolIndexerBackend(source, qk_rope_head_dim=0)
    original_weight = source.wk_weights_proj.weight.detach().clone()
    nz_weights = []

    def format_cast(weight, fmt, **kwargs):
        # Fail at the device boundary if Linear tries to convert the source
        # before KPool can derive its FP32 head weights (the A5 failure path).
        if weight.data_ptr() == source.wk_weights_proj.weight.data_ptr():
            raise AssertionError("KPool source converted to NZ before FP32 head-weight extraction")
        nz_weights.append(weight.data_ptr())
        return weight

    monkeypatch.setattr(utils_module.torch_npu, "npu_format_cast", format_cast)
    method = linear_module.AscendUnquantizedLinearMethod()
    # Match the loader's ordering: Linear hooks first, then attention hooks.
    for layer in (source.wk_weights_proj, source.wq_b):
        method.process_weights_after_loading(layer)
    expected_nz_weights = [source.wq_b.weight.data_ptr()] if weight_nz_mode == 2 else []
    assert nz_weights == expected_nz_weights

    # Repeated post-processing must not encounter an NZ source on the next cast.
    for _ in range(2):
        backend.process_weights_after_loading()
        # CANN key weights and FP32 head weights must stay in ND, including
        # when post-processing is repeated.
        assert nz_weights == expected_nz_weights
        assert backend._key_weight.dtype == dtype
        assert backend._head_weight_f32.dtype == torch.float32
        assert backend._key_weight.is_contiguous()
        assert backend._head_weight_f32.is_contiguous()
        torch.testing.assert_close(backend._key_weight, original_weight[: source.head_dim])
        torch.testing.assert_close(backend._head_weight_f32, original_weight[source.head_dim :].float())
        torch.testing.assert_close(backend._gate_weight, source.index_kpool_compress_gate.detach())
        torch.testing.assert_close(backend._norm_weight_f32, source.k_norm.weight.detach().float())
        torch.testing.assert_close(backend._norm_bias_f32, source.k_norm.bias.detach().float())
    torch.testing.assert_close(source.wk_weights_proj.weight, original_weight)


@pytest.mark.parametrize("compute_topk", [False, True])
def test_backend_zero_token_batch_does_not_launch_operators(monkeypatch, compute_topk):
    compress = MagicMock()
    select = MagicMock()
    monkeypatch.setattr(kpool_module, "glm5_next_kpool_tail_compress_and_write_cache_triton", compress)
    monkeypatch.setattr(kpool_module, "glm5_next_lightning_indexer_triton", select)
    result = SparseAttnIndexerKpool(4, 2)(
        torch.empty(0, 2),
        torch.empty(0, 1, 2),
        torch.empty(0, 1),
        torch.empty(0, dtype=torch.int64),
        torch.zeros(2, 2, 1, 2, dtype=torch.bfloat16),
        torch.zeros(2, 2, 4, 2),
        _indexer_metadata(0),
        _tail_metadata(),
        gate_score=torch.empty(0, 2),
        compress_ape=torch.zeros(4, 2),
        index_kpool=4,
        max_pool_seq_len=0,
        compute_topk=compute_topk,
    )
    if compute_topk:
        assert result.shape == (0, 1, 7)
    else:
        assert result is None
    compress.assert_not_called()
    select.assert_not_called()


def test_kpool_operator_uses_global_pcp_write_view(monkeypatch) -> None:
    metadata = _indexer_metadata(num_tokens=2)
    metadata.cum_query_lens = torch.tensor([2], dtype=torch.int32)
    metadata.raw_seq_lens = torch.tensor([2], dtype=torch.int32)
    tail_metadata = _tail_metadata()
    write_k = torch.arange(8, dtype=torch.float32).view(4, 2)
    write_gate = write_k + 20
    write_positions = torch.arange(4)
    write_query_ends = torch.tensor([4], dtype=torch.int32)
    write_seq_lens = torch.tensor([4], dtype=torch.int32)
    write_indexer_slots = torch.tensor([-1, -1, -1, 0])
    write_tail_slots = torch.arange(4)
    write_tail_table = torch.tensor([[1]], dtype=torch.int32)

    def compress(state, cache, k, gate, ape, positions, query_ends, seq_lens, tail_slots, table, slots, pool):
        assert k is write_k
        assert gate is write_gate
        assert positions is write_positions
        assert query_ends is write_query_ends
        assert seq_lens is write_seq_lens
        torch.testing.assert_close(tail_slots, write_tail_slots)
        assert table is write_tail_table
        torch.testing.assert_close(slots, write_indexer_slots)
        assert pool == 4

    monkeypatch.setattr(kpool_module, "glm5_next_kpool_tail_compress_and_write_cache_triton", compress)
    result = SparseAttnIndexerKpool(4, 2)(
        torch.zeros(2, 2),
        None,
        None,
        metadata.positions,
        torch.zeros(2, 2, 1, 2, dtype=torch.bfloat16),
        torch.zeros(2, 2, 4, 2),
        metadata,
        tail_metadata,
        gate_score=torch.zeros(2, 2),
        compress_ape=torch.zeros(4, 2),
        index_kpool=4,
        max_pool_seq_len=1,
        compute_topk=False,
        write_k=write_k,
        write_gate_score=write_gate,
        write_positions=write_positions,
        write_cum_query_lens=write_query_ends,
        write_raw_seq_lens=write_seq_lens,
        write_indexer_slot_mapping=write_indexer_slots,
        write_tail_slot_mapping=write_tail_slots,
        write_tail_block_table=write_tail_table,
    )

    assert result is None


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
            return torch.zeros(args[1].shape[0], 1, 5, dtype=torch.int32)
        return None


@pytest.mark.parametrize("graph_mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
def test_backend_uses_normalized_q_c_and_separate_tail_metadata(
    monkeypatch,
    graph_mode,
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
    backend.topk_indices_buffer = torch.empty(64, 16, dtype=torch.int32)
    backend.softmax_scale = 0.5
    backend._key_weight = None
    recorder = _RecordingKPool()
    backend._forward_cann = recorder.forward
    tail_metadata = _tail_metadata()
    monkeypatch.setattr(
        backend_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={"indexer.tail": tail_metadata},
            cudagraph_runtime_mode=graph_mode,
            virtual_engine=0,
        ),
    )

    # A 64-row graph bucket does not imply reuse for eight one-token requests.
    normalized_q_c = torch.arange(128, dtype=torch.float32).reshape(64, 2)
    hidden = torch.ones(64, 3)
    metadata = _indexer_metadata()
    metadata.cum_query_lens = torch.arange(1, 9, dtype=torch.int32)
    metadata.raw_seq_lens = torch.ones(8, dtype=torch.int32)
    metadata.seq_lens = torch.zeros(8, dtype=torch.int32)
    metadata.block_table = torch.zeros(8, 1, dtype=torch.int32)
    metadata.positions = torch.zeros(64, dtype=torch.int64)
    metadata.slot_mapping = torch.full((64,), -1, dtype=torch.int64)

    result = backend.forward(
        hidden,
        normalized_q_c,
        hidden + 3,
        metadata,
        compute_topk=True,
    )

    assert result is not None
    assert recorder.args is not None
    assert recorder.kwargs is not None
    expected_rows = 64 if graph_mode == CUDAGraphMode.FULL else 8
    assert result.shape[0] == expected_rows
    torch.testing.assert_close(recorder.args[1], normalized_q_c[:expected_rows].repeat(1, 2).view(expected_rows, 2, 2))
    torch.testing.assert_close(recorder.args[0], hidden[:expected_rows] + 3)
    torch.testing.assert_close(backend._key_weight, backend.wk_weights_proj.weight[:2])
    torch.testing.assert_close(backend._norm_weight_f32, backend.k_norm.weight.float())
    expected_weights = torch.nn.functional.linear(hidden[:expected_rows], backend.wk_weights_proj.weight[2:]) * (
        0.5 * 2**-0.5
    )
    torch.testing.assert_close(recorder.args[2], expected_weights)
    assert recorder.args[-1] is tail_metadata
    assert recorder.kwargs["compute_topk"] is True


@pytest.mark.parametrize("missing", ["cann_ops_transformer", "vllm_ascend.ops.triton.glm5_next_cann_keypool"])
def test_cann_backend_reports_missing_operator_dependencies(monkeypatch, missing):
    def load(name):
        if name == missing:
            raise ImportError(name)
        return MagicMock()

    loader = MagicMock(side_effect=load)
    monkeypatch.setattr(backend_module, "import_module", loader)
    with pytest.raises(RuntimeError, match="CANN ops-transformer operator package and Triton cache/index adapters"):
        Glm5NextKPoolIndexerBackend.__new__(Glm5NextKPoolIndexerBackend)._init_cann_ops()
    assert missing in [call.args[0] for call in loader.call_args_list]


@pytest.mark.parametrize("compute_topk", [False, True])
@pytest.mark.parametrize(
    "actual_query_tokens,runtime_mode,query_rows",
    [
        (8, CUDAGraphMode.NONE, 8),
        (4, CUDAGraphMode.NONE, 4),
        (3, CUDAGraphMode.NONE, 4),
        (0, CUDAGraphMode.NONE, 4),
        (0, CUDAGraphMode.PIECEWISE, 4),
        (0, CUDAGraphMode.FULL, 4),
        (0, CUDAGraphMode.NONE, 0),
        (0, CUDAGraphMode.FULL, 0),
    ],
)
@pytest.mark.parametrize("reuse_output", [False, True])
@pytest.mark.parametrize("pcp", [False, True])
def test_cann_indexer_writes_caches_before_optional_selection(
    monkeypatch, compute_topk, actual_query_tokens, runtime_mode, query_rows, reuse_output, pcp
):
    monkeypatch.setattr(
        backend_module, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=runtime_mode)
    )
    op = Glm5NextKPoolIndexerBackend.__new__(Glm5NextKPoolIndexerBackend)
    nn.Module.__init__(op)
    op.topk_tokens, op.head_dim, op.index_kpool = 4, 2, 4
    op._key_weight = torch.empty(2, 3, dtype=torch.bfloat16)
    op._gate_weight = torch.empty(2, 3, dtype=torch.bfloat16)
    op._norm_weight_f32 = op._norm_bias_f32 = None
    op.k_norm = SimpleNamespace(eps=1e-6)
    op.index_kpool_compress_ape = torch.empty(4, 2)
    storage = torch.full((20, 22), 77, dtype=torch.int32)
    op.topk_indices_buffer = storage[::2, :11] if reuse_output else None
    pooled = torch.empty(2, 2, dtype=torch.bfloat16)
    state = torch.empty(5, 4, 4)
    packed = torch.empty(query_rows, 1, 7, dtype=torch.int32)
    calls = MagicMock()
    for name, result in (
        ("prepare_keypool_state", state),
        ("key_pool", pooled),
        ("write_keypool_caches", None),
        ("pool_key_indexer", (torch.empty(query_rows, 7, dtype=torch.int32), None)),
        ("pack_keypool_indices", packed),
    ):
        mock = MagicMock(return_value=result)
        setattr(op, "_" + name, mock)
        calls.attach_mock(mock, name)
    metadata = _indexer_metadata()
    metadata.query_start_loc = torch.tensor([0, 4, 8], dtype=torch.int32)
    metadata.start_pos = torch.zeros(2, dtype=torch.int32)
    metadata.pool_tail = torch.zeros(2, dtype=torch.int64)
    metadata.pooled_slot_mapping = torch.tensor([0, 2], dtype=torch.int64)
    metadata.keypool_state_table = torch.empty(2, 2, dtype=torch.int32)
    metadata.keypool_start_offsets = torch.empty(2, dtype=torch.int32)
    if pcp:
        metadata.write_metadata = replace(
            metadata,
            query_start_loc=torch.tensor([0, 8], dtype=torch.int32),
            start_pos=torch.zeros(1, dtype=torch.int32),
            keypool_state_table=torch.empty(1, 3, dtype=torch.int32),
        )
    cache_metadata = metadata.write_metadata or metadata
    metadata.positions = metadata.positions[:query_rows]
    metadata.num_actual_tokens = actual_query_tokens
    tail_metadata = _tail_metadata()
    tail_metadata.cann_inputs = SimpleNamespace(retained_indices=None)
    cache = torch.empty(2, 2, 1, 2, dtype=torch.bfloat16)
    tail = torch.empty(2, 4, 4)
    with patch.object(torch.Tensor, "item", side_effect=AssertionError("unexpected device synchronization")):
        result = op._forward_cann(
            torch.empty(8, 3, dtype=torch.bfloat16),
            torch.empty(query_rows, 1, 2, dtype=torch.bfloat16),
            torch.empty(query_rows, 1, dtype=torch.bfloat16),
            cache,
            tail,
            metadata,
            tail_metadata,
            compute_topk=compute_topk,
        )
    names = [call[0] for call in calls.mock_calls]
    assert names[:3] == ["prepare_keypool_state", "key_pool", "write_keypool_caches"]
    assert op._key_pool.call_args.args[1:3] == (op._key_weight, op._gate_weight)
    assert op._key_pool.call_args.kwargs["cu_seqlens"] is cache_metadata.query_start_loc
    assert op._key_pool.call_args.args[5] is cache_metadata.keypool_state_table
    assert op._prepare_keypool_state.call_args.args[1] is cache_metadata.start_pos
    assert op._write_keypool_caches.call_args.args[:2] == (tail, cache)
    assert op._write_keypool_caches.call_args.args[4] is pooled
    assert op._write_keypool_caches.call_args.args[5] is cache_metadata
    skip_empty_query = query_rows == 0 or (actual_query_tokens == 0 and runtime_mode != CUDAGraphMode.FULL)
    if compute_topk and skip_empty_query:
        assert result.shape == (query_rows, 1, 7)
        assert result.dtype == torch.int32
        assert torch.all(result == -1)
        assert names[3:] == []
        op._pool_key_indexer.assert_not_called()
        op._pack_keypool_indices.assert_not_called()
        if reuse_output:
            assert result.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
            assert torch.all(op.topk_indices_buffer[:query_rows] == -1)
            assert torch.all(op.topk_indices_buffer[query_rows:] == 77)
            assert torch.all(storage[1::2] == 77)
            assert torch.all(storage[:, 11:] == 77)
    elif compute_topk:
        assert result is packed
        assert names[3:] == ["pool_key_indexer", "pack_keypool_indices"]
        assert op._pool_key_indexer.call_args.args[1] is cache
        assert op._pool_key_indexer.call_args.kwargs["block_table"] is metadata.block_table
        assert op._pool_key_indexer.call_args.kwargs["actual_seq_q"] is metadata.cum_query_lens
        assert op._pool_key_indexer.call_args.kwargs["actual_seq_k"] is metadata.seq_lens
        assert op._pool_key_indexer.call_args.args[3] is metadata.pool_tail
        assert op._pack_keypool_indices.call_args.args[1] is metadata
    else:
        assert result is None
        op._pool_key_indexer.assert_not_called()
        op._pack_keypool_indices.assert_not_called()


@pytest.mark.parametrize("pcp_local_token_count", [2, 4])
@pytest.mark.parametrize("actual_tokens", [0, 2])
def test_backend_restores_pcp_keys_before_kpool_cache_write(monkeypatch, pcp_local_token_count, actual_tokens) -> None:
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
    recorder = _RecordingKPool()
    backend._forward_cann = recorder.forward
    tail_metadata = _tail_metadata()
    metadata = _indexer_metadata(num_tokens=2)
    metadata.num_actual_tokens = actual_tokens
    metadata.cum_query_lens = torch.tensor([actual_tokens], dtype=torch.int32)
    metadata.raw_seq_lens = torch.tensor([4], dtype=torch.int32)
    metadata.seq_lens = torch.tensor([1], dtype=torch.int32)
    metadata.positions = torch.tensor([0, 2])
    metadata.slot_mapping = torch.tensor([-1, -1])
    metadata.pcp_local_token_count = pcp_local_token_count
    metadata.pcp_hidden_restore_indices = torch.tensor([0, pcp_local_token_count, 1, pcp_local_token_count + 1])
    metadata.write_metadata = _indexer_metadata(num_tokens=4)
    tail_metadata.write_slot_mapping = torch.arange(4)

    pcp_group = MagicMock()
    pcp_group.all_gather.side_effect = lambda tensor, dim: torch.cat((tensor, tensor + 10), dim=dim)
    monkeypatch.setattr(backend_module, "get_pcp_group", lambda: pcp_group)
    monkeypatch.setattr(
        backend_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={"indexer.tail": tail_metadata},
            cudagraph_runtime_mode=None,
            virtual_engine=0,
        ),
    )

    result = backend.forward(
        torch.ones(2, 3),
        torch.ones(2, 2),
        torch.full((2, 3), 2.0),
        metadata,
        compute_topk=True,
    )

    pcp_group.all_gather.assert_called_once()
    recorded_args = recorder.args
    recorded_kwargs = recorder.kwargs
    assert recorded_args is not None
    assert recorded_kwargs is not None
    local_hidden = torch.full((actual_tokens, 3), 2.0)
    padding = pcp_local_token_count - actual_tokens
    padded_hidden = torch.nn.functional.pad(local_hidden, (0, 0, 0, padding))
    restore = metadata.pcp_hidden_restore_indices
    assert restore is not None
    assert pcp_group.all_gather.call_args_list[0].args[0].shape[0] == pcp_local_token_count
    torch.testing.assert_close(
        recorded_args[0],
        torch.cat((padded_hidden, padded_hidden + 10), dim=0)[restore],
    )
    assert recorded_args[-2] is metadata
    assert result.shape[0] == recorded_args[1].shape[0] == actual_tokens
