# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm_ascend.spec_decode.medusa_proposer.AscendMedusaProposer.

Covers the two ``propose`` branches (passthrough when the target input
excludes draft tokens, and the cumsum-based index gather otherwise) and the
``dummy_run`` orchestration, without instantiating the upstream MedusaProposer
``__init__`` or running on NPU.
"""

from contextlib import nullcontext
from unittest.mock import Mock, patch

import torch

from vllm_ascend.spec_decode.medusa_proposer import AscendMedusaProposer


def _make_proposer(**attrs):
    """Build an AscendMedusaProposer bypassing the upstream __init__."""
    proposer = AscendMedusaProposer.__new__(AscendMedusaProposer)
    for name, value in attrs.items():
        setattr(proposer, name, value)
    return proposer


def test_propose_passes_through_hidden_states_when_shape_matches():
    # sample_hidden_states.shape[0] == len(valid_sampled_token_ids) -> branch A.
    sample_hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    valid = [[1], [2], [3]]
    spec_decode_metadata = Mock(num_draft_tokens=[1, 1, 1])
    sampling_metadata = Mock()
    proposer = _make_proposer(device="cpu")

    sentinel = object()
    super_propose = Mock(return_value=sentinel)
    with patch("vllm.v1.spec_decode.medusa.MedusaProposer.propose", new=super_propose):
        out = proposer.propose(
            num_speculative_tokens=2,
            valid_sampled_token_ids=valid,
            sampling_metadata=sampling_metadata,
            spec_decode_metadata=spec_decode_metadata,
            sample_hidden_states=sample_hidden,
        )

    assert out is sentinel
    super_propose.assert_called_once()
    assert super_propose.call_args.args[0] == 2
    assert torch.equal(
        super_propose.call_args.kwargs["target_hidden_states"], sample_hidden
    )


def test_propose_gathers_target_hidden_states_with_computed_indices():
    # shape[0] (4) != len(valid) (2) -> branch B: cumsum-based index gather.
    sample_hidden = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    valid = [[1], [2, 3]]  # num_accepted_tokens = [1, 2]
    spec_decode_metadata = Mock(num_draft_tokens=[1, 1])  # num_draft + 1 = [2, 2]
    sampling_metadata = Mock()
    proposer = _make_proposer(device="cpu")

    # offsets = cumsum([2, 2]) - [2, 2] = [0, 2]
    # indices = offsets + num_accepted_tokens - 1 = [0, 2] + [1, 2] - 1 = [0, 3]
    expected_indices = torch.tensor([0, 3], dtype=torch.long)
    sentinel = object()
    super_propose = Mock(return_value=sentinel)
    with patch("vllm.v1.spec_decode.medusa.MedusaProposer.propose", new=super_propose):
        out = proposer.propose(
            num_speculative_tokens=2,
            valid_sampled_token_ids=valid,
            sampling_metadata=sampling_metadata,
            spec_decode_metadata=spec_decode_metadata,
            sample_hidden_states=sample_hidden,
        )

    assert out is sentinel
    gathered = super_propose.call_args.kwargs["target_hidden_states"]
    assert torch.equal(gathered, sample_hidden[expected_indices])


def test_dummy_run_runs_model_and_logits_on_zero_hidden_states():
    proposer = _make_proposer(
        max_num_tokens=8,
        hidden_size=4,
        dtype=torch.float32,
        device="cpu",
        vllm_config=Mock(),
        model=Mock(),
    )
    captured = {}

    def dummy_compute_logits(hidden_states):
        captured["logits"] = hidden_states
        return None

    def fake_forward_context(*args, **kwargs):
        return nullcontext()

    with patch(
        "vllm_ascend.spec_decode.medusa_proposer.set_ascend_forward_context",
        side_effect=fake_forward_context,
    ):
        proposer.dummy_run(num_tokens=8, dummy_compute_logits=dummy_compute_logits)

    proposer.model.assert_called_once()
    model_input = proposer.model.call_args.args[0]
    assert model_input.shape == (8, 4)
    assert model_input.dtype == torch.float32
    assert captured["logits"].shape == (8, 4)
    assert torch.equal(captured["logits"], model_input)

