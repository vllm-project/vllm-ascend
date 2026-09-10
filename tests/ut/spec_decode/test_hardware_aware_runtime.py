# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tensor storage, width dispatch and diagnostics for hardware-aware decoding."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.hardware_aware import (
    IndexedConfidenceBuffer,
    IndexedDraftTokenBuffer,
    enable_budget_debug,
    enable_draft_graph_debug,
    physical_k_scope,
)


@pytest.mark.parametrize(
    "dynamic_config",
    [
        {"method_params": {"v2_varlen_physical_k": True}},
        {"method": "dspark", "policy": "hardware_aware", "physical_k": {"min_k": 3, "capture_k": [3, 5]}},
    ],
)
def test_v2_physical_k_scope_updates_query_layout_atomically(dynamic_config) -> None:
    speculator = SimpleNamespace(
        vllm_config=SimpleNamespace(additional_config={"dynamic_spec_config": dynamic_config}),
        num_speculative_steps=5,
        num_query_per_req=5,
        max_num_reqs=2,
        sample_from_anchor=True,
        sample_col=torch.arange(5).repeat(2),
        draft_token_confidence_probs=torch.zeros((2, 5)),
        _anchor_idx=torch.arange(2) * 5,
    )
    batch = SimpleNamespace(num_draft_tokens_per_req=torch.tensor([2, 2]))

    with physical_k_scope(speculator, batch) as active_k:
        assert active_k == 2
        assert speculator.num_speculative_steps == 2
        assert speculator.num_query_per_req == 2
        assert speculator.sample_col.tolist() == [0, 1, 0, 1]
        assert tuple(speculator.draft_token_confidence_probs.shape) == (2, 2)
        assert speculator._anchor_idx.tolist() == [0, 2]

    assert speculator.num_speculative_steps == 5
    assert speculator.num_query_per_req == 5
    assert speculator.sample_col.tolist() == [0, 1, 2, 3, 4] * 2
    assert tuple(speculator.draft_token_confidence_probs.shape) == (2, 5)
    assert speculator._anchor_idx.tolist() == [0, 5]


def test_v2_physical_k_scope_keeps_mixed_batch_on_safe_width() -> None:
    speculator = SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={"dynamic_spec_config": {"method_params": {"v2_varlen_physical_k": True}}}
        ),
        num_speculative_steps=5,
        num_query_per_req=5,
        max_num_reqs=2,
        sample_from_anchor=True,
        sample_col=torch.arange(5).repeat(2),
        _anchor_idx=torch.arange(2) * 5,
    )
    batch = SimpleNamespace(num_draft_tokens_per_req=torch.tensor([2, 3]))

    with physical_k_scope(speculator, batch) as active_k:
        assert active_k == 5
        assert speculator.num_speculative_steps == 5
        assert speculator.num_query_per_req == 5


def test_v2_physical_k_scope_uses_next_draft_width_from_scheduler() -> None:
    speculator = SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={"dynamic_spec_config": {"method_params": {"v2_varlen_physical_k": True}}}
        ),
        num_speculative_steps=5,
        num_query_per_req=5,
        max_num_reqs=2,
        sample_from_anchor=True,
        sample_col=torch.arange(5).repeat(2),
        _anchor_idx=torch.arange(2) * 5,
    )
    batch = SimpleNamespace(
        num_draft_tokens_per_req=torch.tensor([5, 5]),
        _vllm_ascend_physical_draft_k=2,
    )

    with physical_k_scope(speculator, batch) as active_k:
        assert active_k == 2
        assert speculator.num_speculative_steps == 2
        assert speculator.num_query_per_req == 2


def test_v2_physical_k_scope_reuses_preallocated_device_indices() -> None:
    speculator = SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={"dynamic_spec_config": {"method_params": {"v2_varlen_physical_k": True}}}
        ),
        num_speculative_steps=5,
        num_query_per_req=5,
        max_num_reqs=2,
        sample_from_anchor=True,
        sample_col=torch.arange(5).repeat(2),
        _anchor_idx=torch.arange(2) * 5,
    )
    batch = SimpleNamespace(num_draft_tokens_per_req=torch.tensor([2, 2]))

    with physical_k_scope(speculator, batch):
        sample_col = speculator.sample_col
        anchor_idx = speculator._anchor_idx

    with physical_k_scope(speculator, batch):
        assert speculator.sample_col.data_ptr() == sample_col.data_ptr()
        assert speculator._anchor_idx.data_ptr() == anchor_idx.data_ptr()


def test_v2_dflash_physical_k_writes_only_active_prefix() -> None:
    """A smaller captured K must not resize the fixed draft-token buffer."""
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
    )
    speculator._run_model = lambda *args: torch.zeros((8, 1))
    speculator.sample_draft = lambda *args: torch.arange(8, dtype=torch.int64)

    AscendDFlashSpeculator._generate_draft(
        speculator,
        num_reqs=2,
        num_tokens_padded=8,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
    )

    assert speculator.draft_tokens.tolist() == [
        [0, 1, 2, 3, -1],
        [4, 5, 6, 7, -1],
    ]


def test_v1_scheduler_rejects_removed_hardware_policy() -> None:
    from vllm_ascend.spec_decode.utils import DynamicSpecScheduler

    with pytest.raises(ValueError, match="legacy V1 scheduler"):
        DynamicSpecScheduler(
            method="dspark",
            policy="hardware_aware",
            method_params={},
            max_batch_size=2,
            num_speculative_tokens=5,
            device=torch.device("cpu"),
        )


def test_v1_confidence_budget_handles_smaller_draft_width() -> None:
    from vllm_ascend.spec_decode.utils import DynamicSpecScheduler

    scheduler = DynamicSpecScheduler(
        method="dflash",
        method_params={},
        max_batch_size=2,
        num_speculative_tokens=5,
        device=torch.device("cpu"),
    )
    result = scheduler.update(logits=torch.zeros((4, 8)), num_reqs=2)
    assert result.tolist() == [2, 2]


@pytest.mark.parametrize("enable", [enable_budget_debug, enable_draft_graph_debug])
def test_info_mode_does_not_wrap_or_touch_manager(enable):
    logger = Mock()
    logger.isEnabledFor.return_value = False
    manager = SimpleNamespace()
    enable(manager, logger)
    assert vars(manager) == {}
    logger.debug.assert_not_called()


def test_budget_debug_preserves_arguments_result_and_state():
    logger = Mock()
    logger.isEnabledFor.return_value = True
    state = ({"a": 3, "b": 3}, {"a": 1, "b": 1}, 4)
    original = Mock(return_value=6)
    manager = SimpleNamespace(get_num_tokens=original, _batch_budget=state)
    enable_budget_debug(manager, logger)
    assert manager.get_num_tokens("tokens", drafts="drafts") == 6
    original.assert_called_once_with("tokens", drafts="drafts")
    assert manager._batch_budget is state
    assert state == ({"a": 3, "b": 3}, {"a": 1, "b": 1}, 4)
    assert logger.debug.call_args.args[1:7] == (2, 6, 4, 3, 3, 2)


def test_graph_debug_preserves_descriptor_identity():
    logger = Mock()
    logger.isEnabledFor.return_value = True
    desc = SimpleNamespace(cg_mode="FULL", num_tokens=48, num_reqs=16, uniform_token_count=3)
    original = Mock(return_value=desc)
    manager = SimpleNamespace(dispatch=original, _capture_descs={"FULL": [desc]})
    enable_draft_graph_debug(manager, logger)
    assert manager.dispatch(num_tokens=48, num_reqs=16, uniform_token_count=3) is desc
    original.assert_called_once_with(num_tokens=48, num_reqs=16, uniform_token_count=3)
    assert logger.debug.call_args.args[1:] == ("FULL", 48, 16, 3)


def test_debug_does_not_swallow_upstream_errors():
    logger = Mock()
    logger.isEnabledFor.return_value = True
    manager = SimpleNamespace(get_num_tokens=Mock(side_effect=ValueError("upstream")))
    enable_budget_debug(manager, logger)
    with pytest.raises(ValueError, match="upstream"):
        manager.get_num_tokens({}, {})


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
