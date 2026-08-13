# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU correctness tests for Model Runner V2 rejection sampling."""

import math

import pytest
import torch
import torch_npu  # noqa: F401  # Registers the NPU and ACLGraph APIs.

from tests.ut.sample.custom_op_utils import require_categorical_sampling_operator
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample

DEVICE = torch.device("npu")


@pytest.fixture(scope="module", autouse=True)
def _require_categorical_sampling_operator() -> None:
    require_categorical_sampling_operator()


def _one_step_metadata(
    num_reqs: int,
    *,
    seed: int = 1234,
    first_position: int = 0,
) -> tuple[torch.Tensor, ...]:
    num_logits = 2 * num_reqs
    return (
        torch.arange(0, num_logits + 1, 2, dtype=torch.int32, device=DEVICE),
        torch.arange(first_position, first_position + num_logits, dtype=torch.int64, device=DEVICE),
        torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        torch.zeros(num_logits, dtype=torch.int32, device=DEVICE),
        torch.tensor([0, 1], dtype=torch.int32, device=DEVICE).repeat(num_reqs),
        torch.ones(1, dtype=torch.float32, device=DEVICE),
        torch.tensor([seed], dtype=torch.int64, device=DEVICE),
    )


def test_probabilistic_rejection_uses_nonzero_uniform_draws() -> None:
    """A p/q=0.25 draft must not degenerate into the previous all-accept path."""
    num_reqs = 4_096
    target_row = torch.tensor([0.0, math.log(3.0)], dtype=torch.float32, device=DEVICE)
    target_logits = target_row.repeat(2 * num_reqs, 1)
    draft_sampled = torch.zeros(2 * num_reqs, dtype=torch.int64, device=DEVICE)
    cu_num_logits, pos, idx_mapping, expanded_idx_mapping, expanded_local_pos, temperature, seed = _one_step_metadata(
        num_reqs
    )

    _, num_sampled = rejection_sample(
        target_logits,
        None,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seed,
        num_speculative_steps=1,
    )
    torch.npu.synchronize()

    accepted = torch.count_nonzero(num_sampled == 2).item()
    expected = num_reqs * 0.25
    standard_deviation = math.sqrt(num_reqs * 0.25 * 0.75)
    assert abs(accepted - expected) < 8.0 * standard_deviation


def test_fp64_rejection_resolves_sub_fp32_residual_interval() -> None:
    """A forced rejection must use the AscendC FP64 residual sampler."""
    seed_value = 0x0123456789ABCDEF
    position = 482_600
    num_rare_tokens = 32_768
    rejected_token = num_rare_tokens + 1
    vocab_size = rejected_token + 1
    rare_logit = -26.0 * math.log(2.0)

    target_logits = torch.full((2, vocab_size), -float("inf"), dtype=torch.float32, device=DEVICE)
    target_logits[0, 0] = 0.0
    target_logits[0, 1 : num_rare_tokens + 1] = rare_logit
    target_logits[1, 0] = 0.0
    draft_sampled = torch.tensor([0, rejected_token], dtype=torch.int64, device=DEVICE)
    cu_num_logits, pos, idx_mapping, expanded_idx_mapping, expanded_local_pos, temperature, seed = _one_step_metadata(
        1, seed=seed_value, first_position=position
    )

    sampled, num_sampled = rejection_sample(
        target_logits,
        None,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seed,
        num_speculative_steps=1,
        use_fp64=True,
    )
    torch.npu.synchronize()

    # This seed and position land inside an interval narrower than the FP32
    # random grid. The categorical operator's 64-bit fixed-mass reference
    # selects token 2310; its 24-bit midpoint approximation selects 2312.
    assert num_sampled.item() == 1
    assert sampled[0, 0].item() == 2310


def test_fp64_rejection_aclgraph_replay_matches_eager() -> None:
    vocab_size = 5
    rejected_token = vocab_size - 1
    target_logits = torch.full((2, vocab_size), -float("inf"), dtype=torch.float32, device=DEVICE)
    target_logits[:, :2] = 0.0
    draft_sampled = torch.tensor([0, rejected_token], dtype=torch.int64, device=DEVICE)
    cu_num_logits, pos, idx_mapping, expanded_idx_mapping, expanded_local_pos, temperature, seed = _one_step_metadata(
        1, seed=9876
    )

    def run_rejection() -> tuple[torch.Tensor, torch.Tensor]:
        return rejection_sample(
            target_logits,
            None,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            seed,
            num_speculative_steps=1,
            use_fp64=True,
        )

    for _ in range(2):
        run_rejection()
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_sampled, graph_num_sampled = run_rejection()
    torch.npu.synchronize()

    for first_position in (7, 31, 99):
        pos.copy_(torch.arange(first_position, first_position + 2, dtype=torch.int64, device=DEVICE))
        expected_sampled, expected_num_sampled = run_rejection()
        torch.npu.synchronize()
        expected_sampled = expected_sampled.cpu()
        expected_num_sampled = expected_num_sampled.cpu()

        graph.replay()
        torch.npu.synchronize()
        actual_sampled = graph_sampled.cpu()
        actual_num_sampled = graph_num_sampled.cpu()
        torch.testing.assert_close(actual_num_sampled, expected_num_sampled, rtol=0, atol=0)
        for req_idx, count in enumerate(actual_num_sampled.tolist()):
            # As upstream, only sampled[:, :num_sampled] is initialized.
            torch.testing.assert_close(
                actual_sampled[req_idx, :count],
                expected_sampled[req_idx, :count],
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("use_fp64", [False, True])
@pytest.mark.parametrize("has_draft_logits", [False, True])
def test_large_batch_rejection_splits_oversized_grids(use_fp64: bool, has_draft_logits: bool) -> None:
    num_reqs = 128
    vocab_size = 1_048_576
    num_speculative_steps = 5
    logits_per_req = num_speculative_steps + 1
    num_logits = num_reqs * logits_per_req

    target_row = torch.full((1, vocab_size), -float("inf"), dtype=torch.float32, device=DEVICE)
    target_row[:, 1] = 0.0
    target_logits = target_row.expand(num_logits, -1)
    draft_logits = None
    if has_draft_logits:
        draft_row = torch.full((1, 1, vocab_size), -float("inf"), dtype=torch.float32, device=DEVICE)
        draft_row[:, :, 0] = 0.0
        draft_logits = draft_row.expand(num_reqs, num_speculative_steps, -1)
    draft_sampled = torch.zeros(num_logits, dtype=torch.int64, device=DEVICE)
    cu_num_logits = torch.arange(
        0,
        num_logits + 1,
        logits_per_req,
        dtype=torch.int32,
        device=DEVICE,
    )
    pos = torch.arange(num_logits, dtype=torch.int64, device=DEVICE)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    expanded_idx_mapping = idx_mapping.repeat_interleave(logits_per_req)
    expanded_local_pos = torch.arange(logits_per_req, dtype=torch.int32, device=DEVICE).repeat(num_reqs)
    temperature = torch.ones(num_reqs, dtype=torch.float32, device=DEVICE)
    seed = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE) + 12_345

    sampled, num_sampled = rejection_sample(
        target_logits,
        draft_logits,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seed,
        num_speculative_steps=num_speculative_steps,
        use_fp64=use_fp64,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(num_sampled, torch.ones_like(num_sampled), rtol=0, atol=0)
    torch.testing.assert_close(sampled[:, 0], torch.ones_like(sampled[:, 0]), rtol=0, atol=0)
