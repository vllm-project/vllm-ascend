# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID

from vllm_ascend.sample.rejection_sampler import rejection_random_sample_logits_pytorch
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample


def _run_extreme_rejection_case() -> tuple[torch.Tensor, torch.Tensor]:
    num_requests = 16
    vocab_size = 4
    num_speculative_steps = 1
    device = torch.device("npu")

    target_logits = torch.full((num_requests * 2, vocab_size), -8.0, device=device)
    target_logits[0::2, 1] = 8.0
    target_logits[1::2, 2] = 8.0
    draft_logits = torch.full((num_requests, num_speculative_steps, vocab_size), -8.0, device=device)
    draft_logits[:, 0, 0] = 8.0

    draft_sampled = torch.zeros(num_requests * 2, dtype=torch.int64, device=device)
    cu_num_logits = torch.arange(0, num_requests * 2 + 1, 2, dtype=torch.int32, device=device)
    idx_mapping = torch.arange(num_requests, dtype=torch.int32, device=device)
    expanded_idx_mapping = idx_mapping.repeat_interleave(2)
    expanded_local_pos = torch.tensor([0, 1], dtype=torch.int32, device=device).repeat(num_requests)
    positions = torch.arange(num_requests * 2, dtype=torch.int32, device=device)
    temperature = torch.ones(num_requests, device=device)
    seeds = torch.arange(100, 100 + num_requests, dtype=torch.int64, device=device)

    return rejection_sample(
        target_logits,
        draft_logits,
        draft_sampled,
        cu_num_logits,
        positions,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seeds,
        num_speculative_steps,
    )


def test_probabilistic_rejection_uses_nonzero_uniform_and_is_repeatable():
    sampled_first, num_sampled_first = _run_extreme_rejection_case()
    sampled_second, num_sampled_second = _run_extreme_rejection_case()
    torch.npu.synchronize()

    assert torch.equal(num_sampled_first, num_sampled_second)
    assert torch.equal(sampled_first[:, 0], sampled_second[:, 0])
    assert (sampled_first[:, 0] == 1).all()
    assert (num_sampled_first == 1).all()


def test_logits_space_recovered_token_on_npu():
    device = torch.device("npu")
    output_token_ids = torch.full((1, 3), PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=device)
    draft_probs = torch.tensor([[0.5, 0.4, 0.1], [0.8, 0.1, 0.1]], device=device)
    target_probs = torch.tensor([[0.7, 0.2, 0.1], [0.2, 0.1, 0.7]], device=device)

    rejection_random_sample_logits_pytorch(
        output_token_ids,
        torch.tensor([2], dtype=torch.int32, device=device),
        torch.tensor([0, 0], dtype=torch.int32, device=device),
        draft_probs.log(),
        target_probs.log(),
        torch.tensor([[9]], dtype=torch.int32, device=device),
        torch.tensor([0.8, 0.5], dtype=torch.float64, device=device),
        torch.tensor([False], device=device),
        max_spec_len=2,
        num_draft_tokens=[2],
        generators={},
    )
    torch.npu.synchronize()

    assert output_token_ids.cpu().tolist() == [[0, 2, PLACEHOLDER_TOKEN_ID]]
