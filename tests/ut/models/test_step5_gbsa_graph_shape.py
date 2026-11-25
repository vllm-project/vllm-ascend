# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The Step5 GBSA builder has to recognise the captured graph shapes.

The graph manager rounds every configured capture size up to a multiple of the
decode query width, so the token counts the captured graphs run with differ
from ``compilation_config.cudagraph_capture_sizes`` (10 configured, query width
8 -> 16 captured). A step that runs under a captured graph has to refresh the
persistent GBSA metadata buffer, or the graph replays the metadata of the step
that captured it.
"""

from types import SimpleNamespace as NS

import pytest
import torch

import vllm_ascend.models.step5.sparse_attention as step5
from vllm_ascend.models.step5.sparse_attention import AscendStep5DSAMetadataBuilder


def _builder(capture_sizes: set[int]) -> AscendStep5DSAMetadataBuilder:
    """A builder with the state the metadata path reads."""
    builder = object.__new__(AscendStep5DSAMetadataBuilder)
    builder.capture_sizes = set(capture_sizes)
    # The MTP query width: a uniform verify batch is a decode batch.
    builder.reorder_batch_threshold = 8
    builder.gbsa_num_heads = 4
    builder.gbsa_num_kv_groups = 2
    builder.gbsa_head_size = 128
    builder.gbsa_region_size = 64
    builder.gbsa_topk = 8
    builder.gbsa_block_size = 128
    builder.gbsa_meta_buffers = {}
    builder.query_start_loc_int64_buffer = torch.zeros(17, dtype=torch.int64)
    builder.context_len_buffer = torch.zeros(32, dtype=torch.int32)
    return builder


def _common(num_reqs: int = 2) -> NS:
    """A uniform MTP verify batch: ``num_reqs`` requests of 8 query tokens."""
    tokens = num_reqs * 8
    return NS(
        num_reqs=num_reqs,
        num_actual_tokens=tokens,
        num_input_tokens=tokens,
        max_query_len=8,
        query_start_loc=torch.arange(0, tokens + 1, 8, dtype=torch.int32),
        query_start_loc_cpu=torch.arange(0, tokens + 1, 8, dtype=torch.int32),
        seq_lens=torch.tensor([137, 73][:num_reqs], dtype=torch.int32),
        block_table_tensor=torch.zeros((num_reqs, 1), dtype=torch.int32),
        slot_mapping=torch.arange(tokens),
    )


@pytest.fixture
def _metadata_op(monkeypatch):
    """The AICPU scheduling op stands in for the captured metadata."""
    monkeypatch.setattr(step5, "gbsa_sparse_attention_metadata", lambda *args, **kwargs: kwargs["seqused_kv"].clone())


def test_step5_gbsa_refreshes_metadata_at_the_rounded_graph_size(monkeypatch, _metadata_op):
    """A captured shape the configuration never mentions is still refreshed.

    The graph manager captured 16 tokens (10 configured, query width 8), so
    the builder has to recognise the 16-token step as a graph shape: the
    persistent buffer is refreshed in place and the captured graph reads the
    current KV lengths instead of the ones from the capture step.
    """
    monkeypatch.setattr(step5, "get_graph_params", lambda: NS(events={16: []}))
    builder = _builder({10, 24})
    common = _common()

    first = builder.build(0, common)
    common.seq_lens.copy_(torch.tensor([145, 81], dtype=torch.int32))
    second = builder.build(0, common)

    assert first.decode.gbsa_metadata is second.decode.gbsa_metadata
    torch.testing.assert_close(second.decode.gbsa_metadata, torch.tensor([145, 81], dtype=torch.int32))
    assert len(builder.gbsa_meta_buffers) == 1


def test_step5_gbsa_uses_the_configured_sizes_without_captured_graphs(monkeypatch, _metadata_op):
    """Without captured graphs the configured capture sizes decide."""
    monkeypatch.setattr(step5, "get_graph_params", lambda: None)
    builder = _builder({16})
    common = _common()

    first = builder.build(0, common)
    common.seq_lens.copy_(torch.tensor([145, 81], dtype=torch.int32))
    second = builder.build(0, common)

    assert first.decode.gbsa_metadata is second.decode.gbsa_metadata
    torch.testing.assert_close(second.decode.gbsa_metadata, torch.tensor([145, 81], dtype=torch.int32))


def test_step5_gbsa_ignores_a_configured_size_that_was_not_captured(monkeypatch, _metadata_op):
    """A configured size the graph manager did not capture is not a graph shape.

    24 tokens is in the configured set, but the captured graph runs at 16
    tokens, so a 24-token step is not graph shaped and takes the fresh-tensor
    path.
    """
    monkeypatch.setattr(step5, "get_graph_params", lambda: NS(events={16: []}))
    builder = _builder({10, 24})
    common = _common(num_reqs=3)

    first = builder.build(0, common)
    common.seq_lens.copy_(torch.tensor([145, 81, 97], dtype=torch.int32))
    second = builder.build(0, common)

    assert first.decode.gbsa_metadata is not second.decode.gbsa_metadata
    assert builder.gbsa_meta_buffers == {}
