"""Tests for the physical-prefill adapter used by recompute scheduling."""

from dataclasses import dataclass

import numpy as np

from vllm_ascend.worker.v2.spec_decode.recompute_rejection_sampler import (
    RecomputeAwareRejectionSampler,
)


@dataclass
class Batch:
    num_draft_tokens: int
    is_prefilling_np: np.ndarray
    num_computed_prefill_tokens_np: np.ndarray
    prefill_len_np: np.ndarray
    has_prefill: bool


def _batch(computed=(99, 100), prefill=(False, False), drafts=14):
    return Batch(
        num_draft_tokens=drafts,
        is_prefilling_np=np.array(prefill, dtype=bool),
        num_computed_prefill_tokens_np=np.array(computed),
        prefill_len_np=np.array((100, 100)),
        has_prefill=any(prefill),
    )


def test_masks_recompute_tail_only_for_sampler():
    original = _batch()
    seen = {}

    def sampler(logits, batch, draft_logits):
        seen["batch"] = batch
        return logits, draft_logits

    result = RecomputeAwareRejectionSampler(sampler)("logits", original, "draft")

    assert result == ("logits", "draft")
    assert seen["batch"] is not original
    np.testing.assert_array_equal(seen["batch"].is_prefilling_np, [True, False])
    assert seen["batch"].has_prefill is True
    np.testing.assert_array_equal(original.is_prefilling_np, [False, False])
    assert original.has_prefill is False


def test_decode_and_ordinary_prefill_preserve_batch_identity():
    batches = (
        _batch(computed=(100, 100)),
        _batch(computed=(99, 100), prefill=(True, False)),
    )
    for batch in batches:
        sampler = lambda logits, actual, draft: actual
        assert RecomputeAwareRejectionSampler(sampler)(None, batch, None) is batch


def test_no_draft_preserves_batch_identity():
    batch = _batch(drafts=0)
    seen = {}
    sampler = lambda logits, actual, draft: seen.setdefault("batch", actual)
    RecomputeAwareRejectionSampler(sampler)(None, batch, None)
    assert seen["batch"] is batch
