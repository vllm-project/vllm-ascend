# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import _TL_RAND_MIN

from vllm_ascend.worker.v2.spec_decode import rejection_sampler_utils as utils
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample


@triton.jit
def _acceptance_uniforms(output, seeds, positions, mapping):
    row = tl.program_id(0)
    slot = tl.load(mapping + row).to(tl.int64)
    seed = tl.load(seeds + slot)
    position = tl.load(positions + row).to(tl.int32)
    uniform_seed = tl.randint(seed, position)
    uniform = tl.max(tl.rand(uniform_seed, tl.arange(0, 1)).to(tl.float32), axis=0)
    tl.store(output + row, tl.maximum(uniform, _TL_RAND_MIN))


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


@pytest.mark.parametrize("probabilistic_draft", [False, True])
def test_random_active_windows_restore_original_resampling_statistics(monkeypatch, probabilistic_draft):
    # Select probabilities around the actual NPU acceptance draws, with a
    # generous margin. This makes success/failure deterministic while testing
    # the real random kernel, permuted request slots and ragged sequences.
    device = "npu"
    vocab = 17
    patterns = [[False, True, True, True], [True, False, True, False], [False, True, False], [True, False]]
    slots = [3, 0, 4, 1]
    lengths = [len(pattern) + 1 for pattern in patterns]
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    total = starts[-1]
    mapping_cpu = [slot for slot, length in zip(slots, lengths) for _ in range(length)]
    mapping = torch.tensor(mapping_cpu, dtype=torch.int32, device=device)
    positions = torch.arange(total, dtype=torch.int32, device=device)
    seeds = torch.arange(6, dtype=torch.int64, device=device) + 17
    temperatures = torch.tensor([0.7, 1.3, 0.8, 1.1, 0.9, 1.2], device=device)
    uniforms = torch.empty(total, device=device)
    _acceptance_uniforms[(total,)](uniforms, seeds, positions, mapping, num_warps=1)
    draws = uniforms.cpu().tolist()
    logits = torch.empty(total, vocab, device=device)
    draft_logits = torch.empty(6, 4, vocab, device=device) if probabilistic_draft else None
    draft_ids = []
    local_positions = []
    for request, (pattern, slot) in enumerate(zip(patterns, slots)):
        draft_ids.extend([0] + [1] * len(pattern))
        local_positions.extend(range(len(pattern) + 1))
        for step in range(len(pattern) + 1):
            row = starts[request] + step
            accepted = step == len(pattern) or pattern[step]
            q = 0.5 if probabilistic_draft else 1.0
            p = (0.75 if probabilistic_draft else 1.0) if accepted else q * draws[row] * 0.1
            probabilities = torch.full((vocab,), (1 - p) / (vocab - 1), device=device)
            probabilities[1] = p
            logits[row] = probabilities.log() + 5 + row * 0.25
            if draft_logits is not None and step < len(pattern):
                draft_probabilities = torch.full((vocab,), 0.5 / (vocab - 1), device=device)
                draft_probabilities[1] = 0.5
                draft_logits[slot, step] = temperatures[slot] * (draft_probabilities.log() + 31 + slot + step)
    inputs = dict(
        target_logits=logits,
        draft_logits=draft_logits,
        draft_sampled=torch.tensor(draft_ids, device=device),
        cu_num_logits=torch.tensor(starts, dtype=torch.int32, device=device),
        pos=positions,
        idx_mapping=torch.tensor(slots, dtype=torch.int32, device=device),
        expanded_idx_mapping=mapping,
        expanded_local_pos=torch.tensor(local_positions, dtype=torch.int32, device=device),
        temperature=temperatures,
        seed=seeds,
        num_speculative_steps=4,
        use_entropy_verify=False,
    )
    baseline, baseline_counts = rejection_sample(**inputs)
    captured = []
    original_resample = utils.resample

    def capture_resample(*args, **kwargs):
        captured.append((args[1].clone(), args[3].clone(), args[5].clone()))
        return original_resample(*args, **kwargs)

    monkeypatch.setattr(utils, "resample", capture_resample)
    sampled, counts = rejection_sample(**inputs, fly_window_size=2, fly_entropy_threshold=0)
    accepted, target_lse, draft_lse = captured[0]
    assert accepted.tolist() == [4, 1, 0, 1]
    assert counts.tolist() == [5, 2, 1, 2]
    for request, rejection in enumerate([4, 1, 0, 1]):
        if rejection == len(patterns[request]):
            continue
        count = int(counts[request])
        assert int(baseline_counts[request]) == count
        torch.testing.assert_close(sampled[request, :count], baseline[request, :count], rtol=0, atol=0)
        expected_target = logits[starts[request] + rejection].cpu().logsumexp(0)
        torch.testing.assert_close(target_lse[request].cpu(), expected_target, rtol=1e-5, atol=1e-5)
        if draft_logits is not None:
            slot = slots[request]
            expected_draft = (draft_logits[slot, rejection] / temperatures[slot]).cpu().logsumexp(0)
            torch.testing.assert_close(draft_lse[request].cpu(), expected_draft, rtol=1e-5, atol=1e-5)
