# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm_ascend.spec_decode.suffix_proposer.AscendSuffixDecodingProposer.

The Ascend subclass only overrides ``dummy_run`` (a no-op) and ``propose``
(which forwards ``self.runner.input_batch`` to the upstream proposer). These
tests pin both behaviours without running on NPU.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vllm_ascend.spec_decode.suffix_proposer import AscendSuffixDecodingProposer


def _make_proposer(**attrs):
    """Build an AscendSuffixDecodingProposer bypassing the upstream __init__."""
    proposer = AscendSuffixDecodingProposer.__new__(AscendSuffixDecodingProposer)
    for name, value in attrs.items():
        setattr(proposer, name, value)
    return proposer


def test_dummy_run_is_noop_and_returns_none():
    # dummy_run body is a literal ``pass``; it must not touch any attribute.
    proposer = _make_proposer()
    assert proposer.dummy_run(num_tokens=4) is None


def test_propose_delegates_to_super_with_runner_input_batch():
    input_batch = object()
    proposer = _make_proposer(runner=SimpleNamespace(input_batch=input_batch))
    sampled_token_ids = [[1, 2]]
    slot_mappings = {"a": torch.tensor([1, 2])}

    sentinel = object()
    super_propose = Mock(return_value=sentinel)
    with patch(
        "vllm.v1.spec_decode.suffix_decoding.SuffixDecodingProposer.propose",
        new=super_propose,
    ):
        out = proposer.propose(
            num_speculative_tokens=3,
            sampled_token_ids=sampled_token_ids,
            slot_mappings=slot_mappings,
        )

    assert out is sentinel
    super_propose.assert_called_once_with(3, input_batch, sampled_token_ids, slot_mappings)

