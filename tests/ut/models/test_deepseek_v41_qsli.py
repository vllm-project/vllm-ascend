# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the model's selector with CPU tensors and mocked native operators."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def selector():
    # Load the production method without initializing model weights, Triton or
    # NPU modules. Native operator numerics are covered by the NPU tests.
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/indexer.py"
    tree = ast.parse(path.read_text())
    indexer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41Indexer")
    method = next(
        node for node in indexer.body if isinstance(node, ast.FunctionDef) and node.name == "select_projected"
    )
    native = SimpleNamespace(
        npu_quant_lightning_indexer_v3=Mock(),
        npu_quant_sparse_lightning_indexer_metadata=Mock(return_value=torch.empty(1024, dtype=torch.int32)),
    )
    events = Mock()
    events.attach_mock(native.npu_quant_lightning_indexer_v3, "indexer")
    events.attach_mock(native.npu_quant_sparse_lightning_indexer_metadata, "metadata")
    events.attach_mock(Mock(), "wait")
    namespace = dict(
        torch=SimpleNamespace(
            ops=SimpleNamespace(_C_ascend=native), int32=torch.int32, float16=torch.float16, full=torch.full
        ),
        DeviceMetadataStage=SimpleNamespace(INDEXER="indexer"),
        wait_for_device_metadata=events.wait,
        prepare_indexer_indices=Mock(side_effect=lambda indices, *args, **kwargs: indices),
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    model = SimpleNamespace(packed_cache_ops=None, compress_ratio=1, index_topk=4)
    rows = 3
    inputs = dict(
        query=torch.empty(rows, 32, 128),
        weights=torch.empty(rows, 32),
        positions=torch.arange(rows),
        source_cache=(torch.empty(2, 128, 1, 128, dtype=torch.int8), torch.ones(2, 128, 1, 1)),
        source_metadata=SimpleNamespace(
            query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
            cache_seq_lens=torch.tensor([128, 256], dtype=torch.int32),
            cmp_residual=torch.zeros(2, dtype=torch.int32),
            max_query_len=2,
            max_cache_seq_len=256,
            qli_metadata=torch.empty(1024, dtype=torch.int32),
            block_table=torch.tensor([[0, 0], [0, 1]], dtype=torch.int32),
        ),
        is_candidate_source=False,
        uses_candidate_filter=False,
        candidates=None,
        candidate_topk_blocks=2,
        candidate_block_size=8,
        quantized_query=torch.empty(rows, 32, 128, dtype=torch.int8),
        query_scale=torch.ones(rows, 32),
    )
    native.npu_quant_lightning_indexer_v3.return_value = (
        torch.arange(rows * 4, dtype=torch.int32).reshape(rows, 1, 4),
        torch.empty(0),
        torch.tensor([[[1, -1]], [[0, 2]], [[3, -1]]], dtype=torch.int32),
        torch.tensor([[1], [2], [1]], dtype=torch.int32),
    )
    return SimpleNamespace(
        run=lambda **kwargs: namespace["select_projected"](model, **(inputs | kwargs)),
        model=model,
        inputs=inputs,
        native=native,
        events=events,
    )


def test_source_lengths_reach_consumer_metadata_and_indexer(selector):
    lengths = torch.zeros(3, 1, dtype=torch.int32)
    _, candidates = selector.run(is_candidate_source=True, candidate_lengths=lengths)
    torch.testing.assert_close(lengths, torch.tensor([[1], [2], [1]], dtype=torch.int32))
    source_call = selector.native.npu_quant_lightning_indexer_v3.call_args
    assert source_call.kwargs["candidate_mode"] == 1
    assert source_call.kwargs["candidate_block_length"] is None
    selector.native.npu_quant_sparse_lightning_indexer_metadata.assert_not_called()

    # Reuse the same graph-stable buffer with different row lengths. Each
    # consumer must see this forward's values, not cached source metadata.
    for values in ([[1], [2], [1]], [[0], [1], [2]]):
        lengths.copy_(torch.tensor(values, dtype=torch.int32))
        selector.events.reset_mock()
        _, returned = selector.run(uses_candidate_filter=True, candidates=candidates, candidate_lengths=lengths)
        metadata_call = selector.native.npu_quant_sparse_lightning_indexer_metadata.call_args
        assert metadata_call.args == (32, 1, 128, 4, 2)
        assert metadata_call.kwargs["candidate_block_length"] is lengths
        assert metadata_call.kwargs["batch_size"] == 2
        assert metadata_call.kwargs["max_seqlen_k"] == 256
        assert metadata_call.kwargs["cu_seqlens_q"] is selector.inputs["source_metadata"].query_start_loc
        call = selector.native.npu_quant_lightning_indexer_v3.call_args
        assert call.kwargs["metadata"] is selector.native.npu_quant_sparse_lightning_indexer_metadata.return_value
        assert call.kwargs["candidate_block_length"] is lengths
        assert call.kwargs["candidate_topk_index"] is candidates
        assert call.kwargs["candidate_mode"] == 2
        assert returned is candidates
        assert [event[0] for event in selector.events.mock_calls] == ["wait", "metadata", "indexer"]


def test_consumer_rejects_missing_candidate_lengths(selector):
    with pytest.raises(ValueError, match="requires source candidate lengths"):
        selector.run(uses_candidate_filter=True)
    selector.native.npu_quant_sparse_lightning_indexer_metadata.assert_not_called()
    selector.native.npu_quant_lightning_indexer_v3.assert_not_called()


def test_unfiltered_selection_uses_ordinary_metadata(selector):
    selector.run()
    selector.native.npu_quant_sparse_lightning_indexer_metadata.assert_not_called()
    call = selector.native.npu_quant_lightning_indexer_v3.call_args
    assert call.kwargs["candidate_mode"] == 3
    assert call.kwargs["metadata"] is selector.inputs["source_metadata"].qli_metadata
    assert call.kwargs["candidate_block_length"] is None


def test_empty_cache_clears_source_lengths_without_native_call(selector):
    selector.inputs["source_metadata"].max_cache_seq_len = 0
    lengths = torch.full((3, 1), 2, dtype=torch.int32)
    selected, candidates = selector.run(is_candidate_source=True, candidate_lengths=lengths)
    assert selected.shape == (3, 0)
    assert torch.count_nonzero(lengths) == 0
    assert torch.all(candidates == -1)
    assert not selector.events.mock_calls


def test_empty_query_skips_native_calls(selector):
    selected, candidates = selector.run(query=torch.empty(0, 32, 128), is_candidate_source=True)
    assert selected.shape == (0, 4)
    assert candidates.shape == (0, 1, 2)
    assert not selector.events.mock_calls


def test_a5_keeps_packed_cache_dispatch(selector):
    packed = SimpleNamespace(run_a5_indexer=Mock(return_value=(object(), object())))
    selector.model.packed_cache_ops = packed
    lengths = torch.ones(3, 1, dtype=torch.int32)
    assert selector.run(uses_candidate_filter=True, candidate_lengths=lengths) == packed.run_a5_indexer.return_value
    assert packed.run_a5_indexer.call_args.kwargs["candidate_lengths"] is lengths
    assert not selector.events.mock_calls
