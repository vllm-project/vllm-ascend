# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.dspark.buffers import IndexedConfidenceBuffer, IndexedDraftTokenBuffer


@pytest.mark.parametrize("batch", [0, 1, 4, 16])
@pytest.mark.parametrize("active_k", [1, 2, 3, 4])
def test_indexed_writes_preserve_backing_and_inactive_elements(batch, active_k):
    tokens = torch.full((16, 5), -1, dtype=torch.int64)
    confidence = torch.full((16, 5), -1.0)
    token_writer = IndexedDraftTokenBuffer(tokens)
    confidence_writer = IndexedConfidenceBuffer(confidence, active_k)
    token_pointer, confidence_pointer = tokens.data_ptr(), confidence.data_ptr()
    for offset in [0, 100]:
        # Strided inputs exercise the same per-column layout as upstream.
        values = torch.arange(batch * active_k).reshape(batch, active_k) + offset
        for col in range(active_k):
            token_writer[:batch, col] = values[:, col]
        confidence_writer[:batch] = values.float() / 100
        torch.testing.assert_close(tokens[:batch, :active_k], values)
        torch.testing.assert_close(confidence[:batch, :active_k], values.float() / 100)
        assert torch.all(tokens[:, active_k:] == -1)
        assert torch.all(confidence[:, active_k:] == -1)
        assert torch.all(tokens[batch:] == -1)
        assert torch.all(confidence[batch:] == -1)
    assert (tokens.data_ptr(), confidence.data_ptr()) == (token_pointer, confidence_pointer)


def test_reject_narrow_backing_copy_and_invalid_indices():
    backing = torch.zeros(16, 5)
    with pytest.raises(ValueError, match="contiguous"):
        IndexedConfidenceBuffer(backing[:, :4], 4)
    with pytest.raises(ValueError, match="contiguous"):
        IndexedDraftTokenBuffer(backing[:, :4])
    writer = IndexedDraftTokenBuffer(backing)
    with pytest.raises(TypeError):
        writer[1:2, 0] = torch.zeros(1)
    with pytest.raises(IndexError):
        writer[:17, 0] = torch.zeros(17)
    with pytest.raises(IndexError):
        writer[:1, 5] = torch.zeros(1)


def test_preallocated_indices_reused_across_width_and_batch_changes():
    backing = torch.zeros(16, 5)
    writers = {k: IndexedConfidenceBuffer(backing, k) for k in range(1, 5)}
    pointers = {k: writer._indices.data_ptr() for k, writer in writers.items()}
    expected = backing.clone()
    for batch, width in [(16, 4), (1, 2), (4, 3), (16, 1), (16, 4)]:
        value = torch.full((batch, width), float(batch + width))
        writers[width][:batch] = value
        expected[:batch, :width] = value
        torch.testing.assert_close(backing, expected)
        assert writers[width]._indices.data_ptr() == pointers[width]
