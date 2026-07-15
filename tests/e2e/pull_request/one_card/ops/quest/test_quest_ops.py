"""PR smoke tests for QUEST metadata and block-selection kernels.

The nightly suite covers the broad dtype/shape/fuzz matrix. This file selects
the core PR-gating cases: metadata refresh, structured block selection,
prefill-to-select chaining, and an input-validation guard.
"""

import torch

from tests.e2e.nightly.single_node.ops.singlecard_ops import (
    test_quest_ops as nightly_quest,
)


def test_quest_prefill_metadata_matches_reference_fp16():
    nightly_quest.test_prefill_metadata_matches_reference(torch.float16)


def test_quest_block_select_structured_exact_bf16_with_anchors():
    nightly_quest.test_block_select_structured_exact(
        torch.bfloat16,
        k=8,
        tokens_since_metadata_update=0,
        num_heads=4,
        num_kv_heads=2,
    )


def test_quest_prefill_then_select_end_to_end_fp16():
    nightly_quest.test_prefill_then_select_end_to_end(
        torch.float16,
        tokens_since_metadata_update=0,
    )


def test_quest_block_select_rejects_unaligned_k():
    nightly_quest.test_block_select_rejects_unaligned_k(torch.float16, k=9)
