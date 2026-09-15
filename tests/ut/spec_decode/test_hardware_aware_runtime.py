# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tensor storage and runtime-width tests for hardware-aware decoding."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.hardware_aware import (
    IndexedConfidenceBuffer,
    IndexedDraftTokenBuffer,
    adaptive_verification_gate_wrapper,
    configured_capture_k,
    physical_k_scope,
)


def make_speculator():
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={
                "dynamic_spec_config": {
                    "method": "dspark",
                    "policy": "hardware_aware",
                    "physical_k": {"capture_k": [2, 5]},
                }
            }
        ),
        num_speculative_steps=5,
        num_query_per_req=5,
        max_num_reqs=2,
        sample_from_anchor=True,
        sample_col=torch.arange(5).repeat(2),
        draft_token_confidence_probs=torch.zeros((2, 5)),
        _anchor_idx=torch.arange(2) * 5,
    )


def test_capture_widths_use_compact_config():
    assert configured_capture_k(make_speculator().vllm_config, 5) == (2, 5)


def test_full_mode_preserves_upstream_adaptive_verification_factory():
    def original_factory(**kwargs):
        return kwargs

    runner_module = SimpleNamespace(
        maybe_create_adaptive_verification_manager=original_factory
    )
    with adaptive_verification_gate_wrapper(runner_module, CUDAGraphMode.FULL):
        assert runner_module.maybe_create_adaptive_verification_manager is original_factory
    assert runner_module.maybe_create_adaptive_verification_manager is original_factory


def test_piecewise_mode_temporarily_installs_ascend_adapter():
    def original_factory(**kwargs):
        return kwargs

    runner_module = SimpleNamespace(
        maybe_create_adaptive_verification_manager=original_factory
    )
    with adaptive_verification_gate_wrapper(runner_module, CUDAGraphMode.PIECEWISE):
        assert runner_module.maybe_create_adaptive_verification_manager is not original_factory
    assert runner_module.maybe_create_adaptive_verification_manager is original_factory


def test_physical_k_scope_updates_and_restores_query_layout():
    speculator = make_speculator()
    batch = SimpleNamespace(_vllm_ascend_physical_draft_k=2)
    with physical_k_scope(speculator, batch) as active_k:
        assert active_k == 2
        assert speculator.num_speculative_steps == 2
        assert speculator.num_query_per_req == 2
        assert speculator.sample_col.tolist() == [0, 1, 0, 1]
        assert tuple(speculator.draft_token_confidence_probs.shape) == (2, 2)
        assert speculator._anchor_idx.tolist() == [0, 2]
    assert speculator.num_speculative_steps == 5
    assert speculator.num_query_per_req == 5
    assert tuple(speculator.draft_token_confidence_probs.shape) == (2, 5)


def test_mixed_batch_falls_back_to_fixed_width():
    speculator = make_speculator()
    batch = SimpleNamespace(num_draft_tokens_per_req=torch.tensor([2, 3]))
    with physical_k_scope(speculator, batch) as active_k:
        assert active_k == 5


def test_preallocated_indices_are_reused():
    speculator = make_speculator()
    batch = SimpleNamespace(_vllm_ascend_physical_draft_k=2)
    with physical_k_scope(speculator, batch):
        pointers = (speculator.sample_col.data_ptr(), speculator._anchor_idx.data_ptr())
    with physical_k_scope(speculator, batch):
        assert (speculator.sample_col.data_ptr(), speculator._anchor_idx.data_ptr()) == pointers


def test_dflash_writes_only_active_prefix():
    speculator = SimpleNamespace(
        num_speculative_steps=4,
        sample_indices=torch.arange(8, dtype=torch.int64),
        sample_pos=torch.zeros(8, dtype=torch.int64),
        sample_idx_mapping=torch.zeros(8, dtype=torch.int32),
        temperature=torch.zeros(2),
        seeds=torch.zeros(2, dtype=torch.int64),
        sample_col=torch.arange(4, dtype=torch.int32).repeat(2),
        draft_logits=None,
        draft_tokens=torch.full((2, 5), -1, dtype=torch.int64),
        _run_model=lambda *args: torch.zeros((8, 1)),
        sample_draft=lambda *args: torch.arange(8, dtype=torch.int64),
    )
    AscendDFlashSpeculator._generate_draft(
        speculator,
        num_reqs=2,
        num_tokens_padded=8,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
    )
    assert speculator.draft_tokens.tolist() == [[0, 1, 2, 3, -1], [4, 5, 6, 7, -1]]


@pytest.mark.parametrize("batch", [0, 1, 4, 16])
@pytest.mark.parametrize("active_k", [1, 2, 3, 4])
def test_indexed_writes_preserve_backing(batch, active_k):
    tokens = torch.full((16, 5), -1, dtype=torch.int64)
    confidence = torch.full((16, 5), -1.0)
    token_writer = IndexedDraftTokenBuffer(tokens)
    confidence_writer = IndexedConfidenceBuffer(confidence, active_k)
    values = torch.arange(batch * active_k).reshape(batch, active_k)
    for col in range(active_k):
        token_writer[:batch, col] = values[:, col]
    confidence_writer[:batch] = values.float()
    torch.testing.assert_close(tokens[:batch, :active_k], values)
    torch.testing.assert_close(confidence[:batch, :active_k], values.float())
    assert torch.all(tokens[:, active_k:] == -1)
    assert torch.all(confidence[:, active_k:] == -1)


def test_indexed_writers_reject_invalid_storage_and_indices():
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
