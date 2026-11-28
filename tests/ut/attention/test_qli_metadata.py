# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.attention.indexer import QLIMetadataBuffers, QLIMetadataConfig, get_qli_metadata


def _metadata_op(**kwargs):
    # A fixed-size result whose contents depend on the actual device lengths.
    return torch.stack((kwargs["cu_seqlens_q"][-1], kwargs["seqused_k"].sum().to(torch.int32)))


@pytest.mark.parametrize("quant_mode", [1, 5])
def test_layers_share_one_producer_per_build(quant_mode):
    config = QLIMetadataConfig(32, 128, quant_mode)
    buffers = QLIMetadataBuffers(config, 4)
    queries, lengths = torch.tensor([0, 3, 5]), torch.tensor([32, 128])
    op = Mock(side_effect=_metadata_op)
    prepared = buffers.build("target", queries, lengths, op)
    for _ in range(8):
        actual = get_qli_metadata(prepared, config, queries[1:], lengths, op)
        assert actual is prepared
    op.assert_called_once()
    torch.testing.assert_close(prepared.query_start_loc, queries.to(torch.int32))
    torch.testing.assert_close(prepared.seq_lens, lengths.to(torch.int32))
    torch.testing.assert_close(prepared.metadata, torch.tensor([5, 160], dtype=torch.int32))
    assert op.call_args.kwargs["quant_mode"] == quant_mode


def test_rebuild_refreshes_existing_graph_references_with_new_lengths():
    buffers = QLIMetadataBuffers(QLIMetadataConfig(32, 128, 1), 4)
    queries = torch.tensor([0, 1, 2], dtype=torch.int32)
    lengths = torch.tensor([32, 128], dtype=torch.int32)
    op = Mock(side_effect=_metadata_op)
    captured = buffers.build("target", queries, lengths, op)
    pointers = [t.data_ptr() for t in (captured.query_start_loc, captured.seq_lens, captured.metadata)]
    # Same input addresses and shapes, different values on every decode step.
    for step in range(1, 4):
        lengths.add_(1)
        current = buffers.build("target", queries, lengths, op)
        assert [t.data_ptr() for t in (current.query_start_loc, current.seq_lens, current.metadata)] == pointers
        torch.testing.assert_close(captured.metadata, torch.tensor([2, 160 + step * 2], dtype=torch.int32))
        torch.testing.assert_close(captured.seq_lens, lengths)
    assert op.call_count == 4


def test_batch_shape_changes_keep_graph_storage():
    buffers = QLIMetadataBuffers(QLIMetadataConfig(32, 128, 1), 4)
    first = buffers.build("target", torch.tensor([0, 1, 2]), torch.tensor([11, 12]), _metadata_op)
    small = buffers.build("target", torch.tensor([0, 3]), torch.tensor([27]), _metadata_op)
    assert small.metadata.data_ptr() == first.metadata.data_ptr()
    assert small.query_start_loc.data_ptr() == first.query_start_loc.data_ptr()
    assert small.seq_lens.shape == (1,)
    torch.testing.assert_close(first.metadata, torch.tensor([3, 27], dtype=torch.int32))
    large = buffers.build("target", torch.tensor([0, 1, 2, 3]), torch.tensor([31, 41, 51]), _metadata_op)
    assert large.metadata.data_ptr() == first.metadata.data_ptr()
    torch.testing.assert_close(large.seq_lens, torch.tensor([31, 41, 51], dtype=torch.int32))


def test_draft_steps_do_not_overwrite_each_other():
    buffers = QLIMetadataBuffers(QLIMetadataConfig(32, 128, 1), 4)
    queries, lengths = torch.tensor([0, 1, 2]), torch.tensor([32, 128])
    first = buffers.build(("slot_mapping", 10), queries, lengths, _metadata_op)
    lengths.add_(1)
    second = buffers.build(("slot_mapping", 20), queries, lengths, _metadata_op)
    assert first.metadata.data_ptr() != second.metadata.data_ptr()
    assert first.seq_lens.data_ptr() != second.seq_lens.data_ptr()
    torch.testing.assert_close(first.metadata, torch.tensor([2, 160], dtype=torch.int32))
    torch.testing.assert_close(second.metadata, torch.tensor([2, 162], dtype=torch.int32))


@pytest.mark.parametrize("change", [{"head_dim": 64}, {"num_heads_q": 64}, {"topk": 1024}, {"quant_mode": 5}])
def test_incompatible_layer_cannot_consume_group_metadata(change):
    config = QLIMetadataConfig(32, 128, 1)
    prepared = QLIMetadataBuffers(config, 2).build("target", torch.tensor([0, 1]), torch.tensor([32]), _metadata_op)
    op = Mock()
    with pytest.raises(ValueError, match="configuration"):
        get_qli_metadata(prepared, replace(config, **change), torch.tensor([1]), torch.tensor([32]), op)
    op.assert_not_called()


@pytest.mark.parametrize("queries,lengths", [([0, 1], [1, 2]), ([0, 1, 2, 3], [1, 2, 3])])
def test_invalid_layout_or_capacity_fails_before_launch(queries, lengths):
    op = Mock()
    with pytest.raises(ValueError):
        QLIMetadataBuffers(QLIMetadataConfig(32, 128, 1), 2).build(
            "target", torch.tensor(queries), torch.tensor(lengths), op
        )
    op.assert_not_called()


def test_output_layout_change_does_not_replace_captured_storage():
    buffers = QLIMetadataBuffers(QLIMetadataConfig(32, 128, 1), 2)
    captured = buffers.build("target", torch.tensor([0, 1]), torch.tensor([32]), _metadata_op)
    with pytest.raises(RuntimeError, match="layout changed"):
        buffers.build(
            "target", torch.tensor([0, 1]), torch.tensor([32]), lambda **kw: torch.zeros(3, dtype=torch.int32)
        )
    torch.testing.assert_close(captured.metadata, torch.tensor([1, 32], dtype=torch.int32))
