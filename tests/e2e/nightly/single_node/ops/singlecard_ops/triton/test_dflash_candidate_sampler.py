# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import CandidateSampler

import vllm_ascend.patch.worker.patch_v2.patch_dflash_speculator  # noqa: F401
from vllm_ascend.worker.v2.sample.gumbel import _gumbel_sample


@pytest.mark.parametrize("probabilistic", [False, True])
def test_candidate_walk_preserves_conditional_scores_and_clears_old_cache(probabilistic):
    """Zero-temperature walks must preserve q along the chosen path for rejection."""
    max_reqs, num_reqs, steps, topk, vocab = 4, 3, 2, 4, 64
    device = "npu"
    sampler = CandidateSampler(max_reqs, steps, topk, torch.device(device))
    logits = torch.full((max_reqs, steps, vocab), -float("inf"), device=device) if probabilistic else None
    mapping = torch.tensor([[2, 2], [0, 0], [-1, -1]], dtype=torch.int32, device=device).flatten()
    positions = torch.full((num_reqs * steps,), 10, dtype=torch.int32, device=device)
    temperatures = torch.zeros(max_reqs, device=device)
    seeds = torch.arange(max_reqs, dtype=torch.int64, device=device)
    tokens = torch.full((num_reqs * steps,), -1, dtype=torch.int64, device=device)
    scores = torch.zeros(num_reqs, steps, topk, topk)
    scores[:, 0, 0] = torch.tensor([0.0, 2.0, 2.0, 1.0])  # Lowest index wins the tie.
    scores[:, 1, 1] = torch.tensor([0.0, 1.0, 2.0, 3.0])

    for offset in (1, 25):
        candidates = torch.arange(offset, offset + num_reqs * steps * topk).view(num_reqs, steps, topk)
        sampler.sample(
            candidates.to(device),
            scores.to(device),
            num_reqs,
            positions,
            mapping,
            temperatures,
            seeds,
            tokens,
            logits,
            use_fp64=False,
        )
        expected_tokens = torch.tensor(
            [candidates[0, 0, 1], candidates[0, 1, 3], candidates[1, 0, 1], candidates[1, 1, 3], -1, -1]
        )
        torch.testing.assert_close(tokens.cpu(), expected_tokens)
        if logits is not None:
            expected = torch.full((max_reqs, steps, vocab), -float("inf"))
            for row, state in ((0, 2), (1, 0)):
                for step, previous in ((0, 0), (1, 1)):
                    expected[state, step, candidates[row, step]] = scores[row, step, previous]
            torch.testing.assert_close(logits.cpu(), expected)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_probabilistic_candidate_walk_matches_proposal_distribution(temperature):
    """The NPU walk must draw from the same softmax q supplied to rejection."""
    num_reqs, steps, topk = 8192, 1, 4
    device = "npu"
    sampler = CandidateSampler(num_reqs, steps, topk, torch.device(device))
    unary = torch.tensor([-1.0, 0.0, 0.5, 1.0])
    scores = unary.expand(num_reqs, steps, topk, topk).contiguous().to(device)
    candidates = torch.arange(topk).expand(num_reqs, steps, topk).contiguous().to(device)
    tokens = torch.empty(num_reqs, dtype=torch.int64, device=device)
    logits = torch.full((num_reqs, steps, topk), -float("inf"), device=device)
    seeds = torch.arange(num_reqs, dtype=torch.int64, device=device)
    sampler.sample(
        candidates,
        scores,
        num_reqs,
        torch.full((num_reqs,), 10, dtype=torch.int32, device=device),
        torch.arange(num_reqs, dtype=torch.int32, device=device),
        torch.full((num_reqs,), temperature, device=device),
        seeds,
        tokens,
        logits,
        use_fp64=False,
    )
    frequencies = torch.bincount(tokens.cpu(), minlength=topk).float() / num_reqs
    expected = (unary / temperature).softmax(dim=-1)
    torch.testing.assert_close(frequencies, expected, atol=0.025, rtol=0.0)
    torch.testing.assert_close(logits.cpu(), unary.expand(num_reqs, steps, topk))

    target_tokens = _gumbel_sample(
        unary.expand(num_reqs, topk).contiguous().to(device),
        torch.arange(num_reqs, dtype=torch.int32, device=device),
        torch.full((num_reqs,), temperature, device=device),
        seeds,
        torch.full((num_reqs,), 9, dtype=torch.int32, device=device),
        apply_temperature=True,
        is_drafting=False,
    )
    # Independent draws from the same q agree with probability sum(q**2).
    agreement = (tokens == target_tokens).float().mean().cpu()
    torch.testing.assert_close(agreement, expected.square().sum(), atol=0.025, rtol=0.0)
