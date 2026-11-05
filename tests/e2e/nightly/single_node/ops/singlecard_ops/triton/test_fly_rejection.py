# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample


@pytest.mark.parametrize(
    "target_ids,window,threshold,expected",
    [
        ([0, 1, 1, 2], 2, 0.0, [1, 1, 1, 2]),
        ([0, 0, 1, 2], 2, 0.0, [0]),
        ([1, 0, 1, 2], 2, 0.0, [1, 0]),
        ([0, 1, 1, 2], 2, 100.0, [0]),
        ([0, 1, 1, 2], 0, 0.0, [0]),
    ],
)
def test_fly_greedy_windows_restore_original_rejection(target_ids, window, threshold, expected):
    device = "npu"
    logits = torch.zeros(4, 17, device=device)
    logits[torch.arange(4, device=device), torch.tensor(target_ids, device=device)] = 2
    sampled, counts = rejection_sample(
        target_logits=logits,
        draft_logits=None,
        draft_sampled=torch.tensor([0, 1, 1, 1], device=device),
        cu_num_logits=torch.tensor([0, 4], dtype=torch.int32, device=device),
        pos=torch.arange(4, dtype=torch.int32, device=device),
        idx_mapping=torch.zeros(1, dtype=torch.int32, device=device),
        expanded_idx_mapping=torch.zeros(4, dtype=torch.int32, device=device),
        expanded_local_pos=torch.arange(4, dtype=torch.int32, device=device),
        temperature=torch.zeros(1, device=device),
        seed=torch.zeros(1, dtype=torch.int64, device=device),
        num_speculative_steps=3,
        use_entropy_verify=False,
        fly_window_size=window,
        fly_entropy_threshold=threshold,
    )
    assert counts.item() == len(expected)
    assert sampled[0, : len(expected)].tolist() == expected


@pytest.mark.parametrize("probabilistic_draft", [False, True])
def test_random_sampling_unchanged_when_entropy_gate_never_opens(probabilistic_draft):
    torch.manual_seed(17)
    device = "npu"
    inputs = dict(
        target_logits=torch.randn(16, 17, device=device),
        draft_logits=torch.randn(4, 3, 17, device=device) if probabilistic_draft else None,
        draft_sampled=torch.randint(0, 17, (16,), device=device),
        cu_num_logits=torch.arange(5, dtype=torch.int32, device=device) * 4,
        pos=torch.arange(16, dtype=torch.int32, device=device),
        idx_mapping=torch.arange(4, dtype=torch.int32, device=device),
        expanded_idx_mapping=torch.arange(4, dtype=torch.int32, device=device).repeat_interleave(4),
        expanded_local_pos=torch.arange(4, dtype=torch.int32, device=device).repeat(4),
        temperature=torch.ones(4, device=device),
        seed=torch.arange(4, dtype=torch.int64, device=device),
        num_speculative_steps=3,
        use_entropy_verify=False,
    )
    baseline, baseline_counts = rejection_sample(**inputs)
    actual, counts = rejection_sample(**inputs, fly_window_size=2, fly_entropy_threshold=100)
    torch.testing.assert_close(counts, baseline_counts, rtol=0, atol=0)
    for row, count in enumerate(counts.tolist()):
        torch.testing.assert_close(actual[row, :count], baseline[row, :count], rtol=0, atol=0)
