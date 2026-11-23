# SPDX-License-Identifier: Apache-2.0
"""Packed segments preserve request, token and recurrence-chunk identity."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

# This module only depends on torch. Avoid importing the unrelated fused-MoE
# registry when running the CPU metadata tests without an NPU installation.
_MODULE_NAME = "gdn_ascendc_checkpoint_under_test"
_SPEC = importlib.util.spec_from_file_location(
    _MODULE_NAME, Path(__file__).parents[3] / "vllm_ascend/ops/gdn_ascendc_checkpoint.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = _MODULE
_SPEC.loader.exec_module(_MODULE)
build_ascendc_checkpoint_plan = _MODULE.build_ascendc_checkpoint_plan


@pytest.mark.parametrize(
    "lengths,offsets",
    [([1464], [1408]), ([1336], [1280]), ([315], [256]), ([145, 0, 65, 209], [128, 0, 0, 192])],
)
def test_checkpoint_segments_partition_tokens_and_chunks(lengths, offsets):
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    plan = build_ascendc_checkpoint_plan(tuple(cu), offsets, torch.device("cpu"))
    tokens = torch.cat((plan.prefix.tokens, plan.tail.tokens)).sort().values
    chunks = torch.cat((plan.prefix.chunks, plan.tail.chunks)).sort().values
    torch.testing.assert_close(tokens, torch.arange(sum(lengths)), rtol=0, atol=0)
    torch.testing.assert_close(chunks, torch.arange(plan.num_chunks), rtol=0, atol=0)
    assert plan.tail.rows.tolist() == [row for row, offset in enumerate(offsets) if offset]
    for row, start, end in zip(plan.tail.rows.tolist(), plan.tail.cu_seqlens, plan.tail.cu_seqlens[1:]):
        assert end - start == lengths[row] - offsets[row]


@pytest.mark.parametrize("offset", [-1, 1, 63, 129, 192, 256])
def test_invalid_checkpoint_boundary_is_rejected(offset):
    with pytest.raises(ValueError, match="64-token boundary"):
        build_ascendc_checkpoint_plan((0, 192), [offset], torch.device("cpu"))


def test_segment_chunk_indices_follow_original_sequence_rows():
    plan = build_ascendc_checkpoint_plan((0, 145, 145, 210, 419), [128, 0, 0, 192], torch.device("cpu"))
    assert plan.prefix.rows.tolist() == [0, 2, 3]
    assert plan.tail.rows.tolist() == [0, 3]
    assert plan.prefix.chunk_indices == (0, 0, 0, 1, 1, 0, 1, 1, 2, 0, 2, 1, 2, 2)
    assert plan.tail.chunk_indices == (0, 0, 1, 0)
    assert plan.tail.chunks.tolist() == [2, 8]
