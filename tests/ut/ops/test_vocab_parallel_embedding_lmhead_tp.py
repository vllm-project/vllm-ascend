"""Unit tests for lmhead-TP row alignment at the LM-head layer.

Pure-mock tests (CPU tensors, no NPU): they lock the pad/trim contract of
``AscendLogitsProcessor._get_logits_lmheadtp`` when the LM head carries a
``lmhead_tp_capacity`` (the DSpark draft LMHead), and verify that heads
without the attribute keep the original passthrough behavior. Collective
behavior itself is validated on real hardware.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.ops.vocab_parallel_embedding import (
    AscendLogitsProcessor,
    lmhead_all_to_all,
)


def _make_head(capacity=None):
    head = MagicMock()
    head.tp_size = 2
    head.lmhead_tp_capacity = capacity
    head.num_org_embeddings_per_partition = 4
    return head


def _make_processor(vocab=8):
    proc = object.__new__(AscendLogitsProcessor)
    proc.org_vocab_size = vocab
    return proc


def _gather_comm(hidden_states, dim=0):
    """Fake lmhead-TP group: all ranks feed the same rows, so the gather is a
    no-op that keeps the padded shape. The real all_gather would concat across
    ranks; here we emulate the contract that every rank pads to capacity first."""
    return hidden_states


def _all_to_all(logits, group):
    """Fake all_to_all: keeps the row count (single-rank simulation)."""
    return logits


def test_lmhead_tp_capacity_pads_then_trims():
    """With lmhead_tp_capacity set, _get_logits_lmheadtp must feed the head
    the group-agreed capacity rows and hand back only the real rows, mirroring
    V1's dspark branch (pad token_indices_to_sample -> trim raw_logits)."""
    proc = _make_processor(vocab=8)
    head = _make_head(capacity=8)
    hidden_states = torch.randn(3, 5)
    proc._apply_head = MagicMock(side_effect=lambda lm_head, hidden, bias: torch.zeros(hidden.shape[0], 8))

    with (
        patch(
            "vllm_ascend.ops.vocab_parallel_embedding.get_lmhead_tp_group",
            return_value=MagicMock(all_gather=_gather_comm),
        ),
        patch(
            "vllm_ascend.ops.vocab_parallel_embedding.get_ascend_config",
            return_value=MagicMock(enable_reduce_sample=False),
        ),
        patch(
            "vllm_ascend.ops.vocab_parallel_embedding.lmhead_all_to_all",
            side_effect=_all_to_all,
        ),
    ):
        logits = proc._get_logits_lmheadtp(hidden_states, head, None)

    # the head was fed the padded capacity rows
    padded = proc._apply_head.call_args.args[1]
    assert padded.shape == (8, 5)
    torch.testing.assert_close(padded[:3], hidden_states)
    assert torch.all(padded[3:] == 0)
    # only the real rows come back
    assert logits.shape[0] == 3


def test_lmhead_tp_capacity_exact_passthrough():
    """At capacity the hidden states pass through untouched (no copy)."""
    proc = _make_processor(vocab=8)
    head = _make_head(capacity=8)
    hidden_states = torch.randn(8, 5)
    proc._apply_head = MagicMock(side_effect=lambda lm_head, hidden, bias: torch.zeros(hidden.shape[0], 8))

    with (
        patch(
            "vllm_ascend.ops.vocab_parallel_embedding.get_lmhead_tp_group",
            return_value=MagicMock(all_gather=_gather_comm),
        ),
        patch(
            "vllm_ascend.ops.vocab_parallel_embedding.get_ascend_config",
            return_value=MagicMock(enable_reduce_sample=False),
        ),
        patch(
            "vllm_ascend.ops.vocab_parallel_embedding.lmhead_all_to_all",
            side_effect=_all_to_all,
        ),
    ):
        logits = proc._get_logits_lmheadtp(hidden_states, head, None)

    assert proc._apply_head.call_args.args[1] is hidden_states
    assert logits.shape[0] == 8


def test_lmhead_tp_no_capacity_is_passthrough():
    """Heads without lmhead_tp_capacity (target / MTP / DFlash LMHeads) keep
    the original behavior: no padding, no trimming."""
    proc = _make_processor(vocab=8)
    head = _make_head(capacity=None)
    hidden_states = torch.randn(3, 5)
    proc._apply_head = MagicMock(side_effect=lambda lm_head, hidden, bias: torch.zeros(hidden.shape[0], 8))

    with (
        patch(
            "vllm_ascend.ops.vocab_parallel_embedding.get_lmhead_tp_group",
            return_value=MagicMock(all_gather=_gather_comm),
        ),
        patch(
            "vllm_ascend.ops.vocab_parallel_embedding.get_ascend_config",
            return_value=MagicMock(enable_reduce_sample=False),
        ),
        patch(
            "vllm_ascend.ops.vocab_parallel_embedding.lmhead_all_to_all",
            side_effect=_all_to_all,
        ),
    ):
        logits = proc._get_logits_lmheadtp(hidden_states, head, None)

    assert proc._apply_head.call_args.args[1] is hidden_states
    assert logits.shape[0] == 3


def test_lmhead_tp_capacity_raises_when_rows_exceed():
    proc = _make_processor(vocab=8)
    head = _make_head(capacity=8)

    with pytest.raises(ValueError, match="exceed the group-agreed capacity"):
        proc._get_logits_lmheadtp(torch.randn(9, 5), head, None)


def test_lmhead_tp_capacity_attribute_default_none():
    """The new constructor kwarg must default to None so existing callers
    (target / MTP / DFlash) construct heads without any behavior change."""
    from vllm_ascend.ops.vocab_parallel_embedding import AscendParallelLMHead

    head = object.__new__(AscendParallelLMHead)
    assert head.lmhead_tp_capacity is None
    head2 = object.__new__(AscendParallelLMHead)
    head2.lmhead_tp_capacity = 16
    assert head2.lmhead_tp_capacity == 16


def test_lmhead_all_to_all_requires_divisible_rows():
    """The all_to_all collective requires N % world_size == 0; the padding
    capacity must satisfy this (checked on the real path by construction)."""
    group = MagicMock(world_size=2)
    logits = torch.zeros(5, 4)
    with pytest.raises(ValueError, match="must be divisible"):
        lmhead_all_to_all(logits, group)
