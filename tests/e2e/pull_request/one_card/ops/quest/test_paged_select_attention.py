"""PR smoke tests for QUEST paged-select attention kernels.

The exhaustive matrix lives in the nightly single-card op suite. These tests
reuse a few representative nightly cases so pull-request CI catches broken op
registration, core correctness, and end-to-end QUEST kernel wiring without
running the full matrix.
"""

import math

import torch

from tests.e2e.nightly.single_node.ops.singlecard_ops import (
    test_paged_select_attention as nightly_psa,
)


def test_paged_select_attention_matches_reference_fp16_gqa():
    nightly_psa.test_matches_reference(
        torch.float16,
        num_heads=8,
        num_kv_heads=2,
        k=8,
        scale=1.0 / math.sqrt(nightly_psa.HEAD_DIM),
    )


def test_paged_select_attention_end_to_end_select_then_attention_bf16():
    nightly_psa.test_end_to_end_select_then_attention(torch.bfloat16)


def test_paged_select_attention_rejects_non_int32_selected_indices():
    nightly_psa.test_rejects_non_int32_selected_indices()
