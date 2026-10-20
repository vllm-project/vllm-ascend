# Adapt from https://github.com/vllm-project/vllm/blob/main/tests/v1/spec_decode/test_rejection_sampler_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc

import pytest
import torch
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates

from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample

VOCAB_SIZE = 4096

pytest.importorskip("triton")
if not (hasattr(torch, "npu") and torch.npu.is_available()):
    pytest.skip("NPU required for MRV2 rejection sampler tests", allow_module_level=True)


def _build_rejection_sample_inputs(
    target_logits_1d: torch.Tensor,
    draft_logits_1d: torch.Tensor,
    num_speculative_steps: int,
    temperature: float,
    num_trials: int,
) -> dict:
    """Build rejection_sample kwargs from a fixed target and draft distribution.

    target_logits_1d must already have temperature applied (the sampler applies
    sampling params before verification), whereas draft_logits_1d must not:
    rejection_sample divides the draft logits by the temperature on load.
    """
    device = target_logits_1d.device
    vocab_size = target_logits_1d.shape[0]
    K = num_speculative_steps
    num_logits = num_trials * (K + 1)

    target_logits = target_logits_1d.unsqueeze(0).expand(num_logits, -1).contiguous()
    draft_logits = draft_logits_1d.view(1, 1, vocab_size).expand(num_trials, K, -1).contiguous()

    scaled_draft_logits_1d = draft_logits_1d.float()
    if temperature > 0:
        scaled_draft_logits_1d = scaled_draft_logits_1d / temperature
    draft_probs = torch.softmax(scaled_draft_logits_1d, dim=0)
    # Sample on CPU: torch.multinomial is not reliably available on NPU.
    draft_tokens = torch.multinomial(draft_probs.cpu().expand(num_trials, -1), K, replacement=True).to(device)
    draft_sampled_2d = torch.zeros(num_trials, K + 1, dtype=torch.int64, device=device)
    draft_sampled_2d[:, 1:] = draft_tokens
    draft_sampled = draft_sampled_2d.reshape(-1)

    cu_num_logits = torch.arange(num_trials + 1, dtype=torch.int32, device=device) * (K + 1)
    pos = torch.arange(num_logits, dtype=torch.int32, device=device)
    idx_mapping = torch.arange(num_trials, dtype=torch.int32, device=device)
    expanded_idx_mapping = torch.arange(num_trials, dtype=torch.int32, device=device).repeat_interleave(K + 1)
    expanded_local_pos = torch.arange(K + 1, dtype=torch.int32, device=device).repeat(num_trials)
    temp_tensor = torch.full((num_trials,), temperature, dtype=torch.float32, device=device)
    seed = torch.arange(num_trials, dtype=torch.int64, device=device)

    return dict(
        target_logits=target_logits,
        draft_logits=draft_logits,
        draft_sampled=draft_sampled,
        cu_num_logits=cu_num_logits,
        pos=pos,
        idx_mapping=idx_mapping,
        expanded_idx_mapping=expanded_idx_mapping,
        expanded_local_pos=expanded_local_pos,
        temperature=temp_tensor,
        seed=seed,
    )


@pytest.mark.parametrize(
    "num_speculative_steps,temperature,unconditional_rates",
    [
        (3, 1.0, [0.9, 0.5, 0.2]),
        (3, 0.0, [0.9, 0.5, 0.2]),
        (3, 1.0, [1.0, 1.0, 1.0]),
        (3, 0.0, [1.0, 1.0, 1.0]),
        (3, 1.0, [0.0, 0.0, 0.0]),
        (3, 0.0, [0.0, 0.0, 0.0]),
        (1, 1.0, [0.7]),
        (1, 0.0, [0.7]),
    ],
)
@torch.inference_mode()
def test_synthetic_rejection_sample(
    num_speculative_steps: int,
    temperature: float,
    unconditional_rates: list[float],
):
    """
    Verify that synthetic rejection sampling produces the expected
    per-position acceptance rates. The unconditional rate at position i
    is P(all draft steps 0..i accepted) = product(conditional_rates[0:i+1]).
    This is approximately mean(num accepted >= i + 1) over many trials.
    """
    torch.manual_seed(42)
    device = "npu"
    # NPU: triton-ascend caps the flattened grid at 65535. The block-stats
    # kernel launches one program per logit row, and the resample kernel
    # launches num_reqs * cdiv(vocab, 1024) programs, so a single call cannot
    # hold upstream's 10 * VOCAB_SIZE trials. Split the trials into chunks
    # below every grid bound and aggregate the acceptance statistics —
    # rejection_sample is a pure function, so this is equivalent to one
    # large call (upstream uses 10 * VOCAB_SIZE trials in one shot).
    num_trials = 10 * VOCAB_SIZE
    TRIALS_PER_CALL = 16000
    deviation_tol = 1e-2

    target_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)
    draft_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)

    if temperature > 0:
        target_logits_1d /= temperature

    conditional_rates = unconditional_to_conditional_rates(unconditional_rates)
    synthetic_conditional_rates = torch.tensor(conditional_rates, dtype=torch.float32, device=device)

    num_accepted_chunks = []
    for start in range(0, num_trials, TRIALS_PER_CALL):
        chunk_trials = min(TRIALS_PER_CALL, num_trials - start)
        inputs = _build_rejection_sample_inputs(
            target_logits_1d,
            draft_logits_1d,
            num_speculative_steps,
            temperature=temperature,
            num_trials=chunk_trials,
        )
        # Synthetic acceptance is driven by u = f(seed, pos) only, so chunks
        # that restart the seed/pos sequences would replay identical draws.
        # Offset both so every chunk consumes a fresh noise stream.
        inputs["seed"] = inputs["seed"] + start
        inputs["pos"] = inputs["pos"] + start * (num_speculative_steps + 1)

        _, num_sampled = rejection_sample(
            **inputs,
            num_speculative_steps=num_speculative_steps,
            synthetic_conditional_rates=synthetic_conditional_rates,
        )
        # num_sampled includes the resampled/bonus token.
        num_accepted_chunks.append(num_sampled - 1)
        gc.collect()
        torch.npu.empty_cache()

    num_accepted = torch.cat(num_accepted_chunks)
    for i, expected_rate in enumerate(unconditional_rates):
        observed_rate = (num_accepted >= i + 1).float().mean().item()
        assert abs(observed_rate - expected_rate) < deviation_tol, (
            f"Step {i}: observed rate {observed_rate:.4f} deviates from "
            f"expected rate {expected_rate:.4f} by more than {deviation_tol}."
        )

    gc.collect()
    torch.npu.empty_cache()


@pytest.mark.parametrize("has_draft_logits", [True, False])
@pytest.mark.parametrize("use_block_verification", [False, True])
@torch.inference_mode()
def test_draft_vocab_narrower_than_target_vocab(has_draft_logits: bool, use_block_verification: bool):
    """A draft vocab narrower than the target's must not read out of bounds.

    Some draft/target pairs (e.g. MiMo v2.5 Pro + DFlash) pad the target's
    vocab beyond the draft's. The kernels derive vocab_size from
    target_logits, so without clamping to the draft's width every draft
    read runs past the draft logits' last dimension. Padding the target
    with -inf keeps the padding columns unsampleable, which makes the
    padded run bitwise-comparable to a baseline over the truncated target.
    """
    torch.manual_seed(11)
    device = "npu"
    num_speculative_steps = 3
    num_trials = 512
    draft_vocab_size = 1024
    target_vocab_size = draft_vocab_size + 4

    target_logits_1d = torch.randn(draft_vocab_size, device=device, dtype=torch.float32)
    draft_logits_1d = torch.randn(draft_vocab_size, device=device, dtype=torch.float32)

    inputs = _build_rejection_sample_inputs(
        target_logits_1d,
        draft_logits_1d,
        num_speculative_steps,
        temperature=1.0,
        num_trials=num_trials,
    )
    # The padded run keeps every seed/pos/noise input identical to the
    # baseline, and the kernels only read their inputs, so the outputs must
    # match bitwise on the slots the kernels write.
    padded_inputs = dict(inputs)
    padded_target = torch.full(
        (inputs["target_logits"].shape[0], target_vocab_size),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )
    padded_target[:, :draft_vocab_size] = inputs["target_logits"]
    padded_inputs["target_logits"] = padded_target
    if not has_draft_logits:
        inputs["draft_logits"] = None
        padded_inputs["draft_logits"] = None

    sampled, num_sampled = rejection_sample(
        **padded_inputs, num_speculative_steps=num_speculative_steps, use_block_verification=use_block_verification
    )
    baseline_sampled, baseline_num_sampled = rejection_sample(
        **inputs, num_speculative_steps=num_speculative_steps, use_block_verification=use_block_verification
    )

    assert torch.equal(num_sampled, baseline_num_sampled)
    # Slots after the first rejection are never written; compare the rest.
    steps = torch.arange(num_speculative_steps + 1, device=device)
    valid = steps.unsqueeze(0) < num_sampled.unsqueeze(1)
    assert torch.equal(sampled[valid], baseline_sampled[valid]), "Padded target vocab changed the sampling result."
    assert (sampled[valid] < draft_vocab_size).all(), "Sampled a target padding column."

    gc.collect()
    torch.npu.empty_cache()


@pytest.mark.parametrize("has_draft_logits", [True, False])
@torch.inference_mode()
def test_placeholder_draft_tokens_are_rejected(has_draft_logits: bool):
    """A -1 placeholder draft token must be rejected, never accepted and
    never used as a pointer index, even though a valid draft follows it.

    The draft matches the target exactly, so every real draft token passes
    the probability ratio test with probability ~1. Any acceptance at or
    after the placeholder therefore means the guard is missing; before the
    fix the placeholder also indexed target_logits out of bounds.
    """
    torch.manual_seed(0)
    device = "npu"
    num_speculative_steps = 3
    num_trials = 512

    if has_draft_logits:
        target_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)
    else:
        # One-hot drafts have q(x) = 1 at the drafted token, so acceptance
        # needs p(x) ~ 1: make the target nearly deterministic at one token.
        target_logits_1d = torch.full((VOCAB_SIZE,), -20.0, device=device, dtype=torch.float32)
        target_logits_1d[7] = 20.0
    draft_logits_1d = target_logits_1d

    inputs = _build_rejection_sample_inputs(
        target_logits_1d,
        draft_logits_1d,
        num_speculative_steps,
        temperature=1.0,
        num_trials=num_trials,
    )
    # Column 0 anchors the last verified token; columns 1..K hold the draft
    # proposals. Put the placeholder at draft step 1 (column 2).
    draft_rows = inputs["draft_sampled"].view(num_trials, num_speculative_steps + 1)
    draft_rows[:, 2] = -1
    if not has_draft_logits:
        inputs["draft_logits"] = None
        # p(7) ~ 1, so draft step 0 (column 1) is accepted ~surely and the
        # rejection lands exactly on the placeholder.
        draft_rows[:, 1] = 7

    sampled, num_sampled = rejection_sample(**inputs, num_speculative_steps=num_speculative_steps)

    # Only draft step 0 can be accepted, plus one resampled token.
    assert (num_sampled <= 2).all(), "Accepted a draft token past a placeholder."
    assert (num_sampled == 2).float().mean().item() > 0.9, (
        "The first draft was rarely accepted; the test is not exercising acceptance past the placeholder."
    )
    # The rejected slot is resampled from the target, never the placeholder.
    resampled = num_sampled >= 2
    assert (sampled[resampled, 1] >= 0).all()
    assert (sampled[resampled, 1] < VOCAB_SIZE).all()

    gc.collect()
    torch.npu.empty_cache()


@pytest.mark.parametrize("num_speculative_steps", [1, 3])
@pytest.mark.parametrize("temperature", [0.6, 1.0])
@pytest.mark.parametrize("has_draft_logits", [True, False])
@torch.inference_mode()
def test_block_verification_preserves_target_distribution(num_speculative_steps, temperature, has_draft_logits):
    """Check the complete verify/recover path, including independence of its draws."""
    torch.manual_seed(42)
    target = torch.tensor([0.2, 0.3, 0.5], device="npu")
    draft = torch.tensor([0.5, 0.3, 0.2], device="npu")
    inputs = _build_rejection_sample_inputs(
        target.log(), draft.log() * temperature, num_speculative_steps, temperature, num_trials=8192
    )
    if not has_draft_logits:
        inputs["draft_logits"] = None
        # Deterministic one-hot proposals expose acceptance/recovery RNG
        # coupling: reusing u would produce [0.2, 0.175, 0.625] at position 0.
        inputs["draft_sampled"].zero_()
    sampled, num_sampled = rejection_sample(
        **inputs, num_speculative_steps=num_speculative_steps, use_block_verification=True
    )
    sampled = sampled.cpu()
    num_sampled = num_sampled.cpu()
    assert ((num_sampled >= 1) & (num_sampled <= num_speculative_steps + 1)).all()
    for step in range(num_speculative_steps + 1):
        tokens = sampled[num_sampled > step, step]
        assert tokens.numel() > 0
        observed = torch.bincount(tokens, minlength=3).float() / tokens.numel()
        # Later one-hot positions have fewer observations.
        tolerance = 5.0 * (0.25 / tokens.numel()) ** 0.5
        torch.testing.assert_close(observed, target.cpu(), rtol=0, atol=tolerance)


@pytest.mark.parametrize("placeholder_step", [0, 1])
@pytest.mark.parametrize("method", ["standard", "synthetic", "block"])
@pytest.mark.parametrize("has_draft_logits", [True, False])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
@torch.inference_mode()
def test_placeholder_recovers_from_target(placeholder_step, method, has_draft_logits, temperature):
    """Every verification mode must recover a placeholder from the target."""
    torch.manual_seed(7)
    target = torch.tensor([0.1, 0.2, 0.3, 0.4], device="npu")
    inputs = _build_rejection_sample_inputs(target.log(), target.log(), 3, temperature, num_trials=4096)
    if temperature == 0.0 or not has_draft_logits:
        inputs["draft_sampled"].view(-1, 4)[:, 1:] = 3
    if not has_draft_logits:
        inputs["draft_logits"] = None
        # Ensure every request reaches the placeholder, even with one-hot
        # proposals, while keeping its recovery distribution non-degenerate.
        prefix_logits = inputs["target_logits"].view(-1, 4, 4)[:, :placeholder_step, :]
        prefix_logits.fill_(-float("inf"))
        prefix_logits[:, :, 3] = 0
    inputs["draft_sampled"].view(-1, 4)[:, placeholder_step + 1] = -1
    sampled, num_sampled = rejection_sample(
        **inputs,
        num_speculative_steps=3,
        use_block_verification=method == "block",
        synthetic_conditional_rates=torch.ones(3, device="npu") if method == "synthetic" else None,
    )
    assert (num_sampled == placeholder_step + 1).all()
    if temperature == 0.0:
        assert (sampled[:, placeholder_step] == 3).all()
    else:
        counts = torch.bincount(sampled[:, placeholder_step].cpu(), minlength=4).float()
        torch.testing.assert_close(counts / counts.sum(), target.cpu(), rtol=0, atol=0.03)


@torch.inference_mode()
def test_block_verification_extends_accepted_prefix():
    """Block verification must do more than silently run standard verification."""
    torch.manual_seed(0)
    target = torch.tensor([0.4, 0.3, 0.3], device="npu")
    draft = torch.tensor([0.5, 0.2, 0.3], device="npu")
    inputs = _build_rejection_sample_inputs(target.log(), draft.log(), 3, 1.0, num_trials=8192)
    _, standard_counts = rejection_sample(**inputs, num_speculative_steps=3)
    _, block_counts = rejection_sample(**inputs, num_speculative_steps=3, use_block_verification=True)
    # Exact mean accepted lengths for these distributions are approximately
    # 2.439 (standard) and 2.509 (block); leave room for sampling noise.
    assert block_counts.float().mean().item() > standard_counts.float().mean().item() + 0.02


@pytest.mark.parametrize("has_draft_logits", [True, False])
@torch.inference_mode()
def test_block_verification_ragged_requests_and_mapping(has_draft_logits):
    """Cover bonus-only/short requests, mixed temperatures and multiple vocab blocks."""
    lengths = [4, 1, 3, 2]
    mapping = [3, 0, 2, 1]
    vocab_size = 8193
    expected_ids = torch.tensor([8192, 1024, 3, 0], device="npu")
    state_logits = torch.full((4, vocab_size), -float("inf"), device="npu")
    state_logits[torch.tensor(mapping, device="npu"), expected_ids] = 0
    expanded_mapping = torch.repeat_interleave(torch.tensor(mapping, dtype=torch.int32), torch.tensor(lengths)).to(
        "npu"
    )
    cu = torch.tensor([0, 4, 5, 8, 10], dtype=torch.int32, device="npu")
    draft_sampled = torch.repeat_interleave(expected_ids, torch.tensor(lengths, device="npu"))
    inputs = dict(
        target_logits=state_logits[expanded_mapping.long()],
        draft_logits=state_logits[:, None, :].expand(-1, 3, -1).contiguous() if has_draft_logits else None,
        draft_sampled=draft_sampled,
        cu_num_logits=cu,
        pos=torch.arange(sum(lengths), dtype=torch.int32, device="npu"),
        idx_mapping=torch.tensor(mapping, dtype=torch.int32, device="npu"),
        expanded_idx_mapping=expanded_mapping,
        expanded_local_pos=torch.cat([torch.arange(n, dtype=torch.int32) for n in lengths]).to("npu"),
        temperature=torch.tensor([1.0, 0.6, 0.0, 1.0], device="npu"),
        seed=torch.arange(4, dtype=torch.int64, device="npu"),
    )
    sampled, num_sampled = rejection_sample(**inputs, num_speculative_steps=3, use_block_verification=True)
    assert num_sampled.cpu().tolist() == lengths
    for row, length in enumerate(lengths):
        assert (sampled[row, :length] == expected_ids[row]).all()


@pytest.mark.parametrize("has_draft_logits", [True, False])
@pytest.mark.parametrize("temperature", [0.0, 0.7])
@torch.inference_mode()
def test_block_verification_single_bonus_only_request(has_draft_logits, temperature):
    """A one-logit batch must recover its bonus without a zero-sized residual grid."""
    logits = torch.tensor([-float("inf"), 0.0, -float("inf")], device="npu")
    inputs = _build_rejection_sample_inputs(logits, logits, 3, temperature, num_trials=1)
    # Keep the persistent draft cache, but verify a request with no proposals.
    for name in ("target_logits", "draft_sampled", "pos", "expanded_idx_mapping", "expanded_local_pos"):
        inputs[name] = inputs[name][:1]
    inputs["cu_num_logits"] = torch.tensor([0, 1], dtype=torch.int32, device="npu")
    if not has_draft_logits:
        inputs["draft_logits"] = None

    sampled, num_sampled = rejection_sample(**inputs, num_speculative_steps=3, use_block_verification=True)

    assert num_sampled.cpu().tolist() == [1]
    assert sampled[:, 0].cpu().tolist() == [1]


@pytest.mark.parametrize("has_draft_logits", [True, False])
@torch.inference_mode()
def test_block_verification_keeps_greedy_results(has_draft_logits):
    torch.manual_seed(9)
    logits = torch.tensor([-2.0, 0.0, 2.0], device="npu")
    inputs = _build_rejection_sample_inputs(logits, logits, 3, 0.0, num_trials=16)
    if not has_draft_logits:
        inputs["draft_logits"] = None
    # Include all accepted, first rejected, and placeholder paths.
    drafts = inputs["draft_sampled"].view(-1, 4)
    drafts[:, 1:] = 2
    drafts[::3, 1] = 0
    drafts[1::3, 2] = -1
    expected, expected_counts = rejection_sample(**inputs, num_speculative_steps=3)
    sampled, counts = rejection_sample(**inputs, num_speculative_steps=3, use_block_verification=True)
    torch.testing.assert_close(counts, expected_counts)
    valid = torch.arange(4, device="npu")[None, :] < counts[:, None]
    torch.testing.assert_close(sampled[valid], expected[valid])
