# SPDX-License-Identifier: Apache-2.0
# Numerical test for `_resample_kernel` in
# vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils, against a plain
# PyTorch fp32 reference.
# Requires NPU and Triton-Ascend.
#
# See vllm_ascend/ops/triton/docs/resample.md for the operator spec.
#
# Regression scope: #9155 (main2main import of the MRV2 rejection sampler) and
# #13470 (probabilistic rejection sampling enabled on NPU) -- neither PR shipped
# any numerical coverage for this kernel.
#
# `_npu_gumbel_block_argmax` is a `@triton.jit` device function inlined into
# `_resample_kernel`, not a separate operator: it has no `tl.program_id` and
# cannot be launched on its own.  It is covered here only through
# `_resample_kernel`, which is the sole caller.

import gc

import pytest
import torch
import torch_npu  # noqa: F401  # registers the npu backend / torch.npu namespace
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import _resample_kernel, rejection_sample

DEVICE = "npu"

# Everything is fp32 end to end; the only slack needed is for the different
# order of the `log`/`exp` chain in the residual-logits branch.
_RTOL = 1e-5
_ATOL = 1e-5

# Production launch constants, mirrored so the tests exercise the real tiling.
RESAMPLE_BLOCK_SIZE = 1024
VOCAB_BLOCK_SIZE = 8192

# Sentinels written into the outputs before every launch, so that "the kernel
# returned early and left the slot untouched" is observable.
_ARGMAX_POISON = -777
_MAX_POISON = -12345.0


@pytest.fixture(autouse=True)
def _npu_env():
    init_device_properties_triton()
    yield
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ref_block_argmax(logits, vocab_size, block_size):
    """Plain PyTorch per-block max/argmax over fp32 logits.

    Positions beyond `vocab_size` are padded with -inf, and the returned indices
    are global vocabulary indices rather than offsets within each block.

    Returns (values [T, num_blocks] fp32, indices [T, num_blocks] int64).
    """
    num_tokens = logits.shape[0]
    num_blocks = triton.cdiv(vocab_size, block_size)
    padded = torch.full(
        (num_tokens, num_blocks * block_size),
        float("-inf"),
        dtype=torch.float32,
        device=logits.device,
    )
    padded[:, :vocab_size] = logits.float()
    padded = padded.view(num_tokens, num_blocks, block_size)
    values, idx = padded.max(dim=-1)
    offsets = torch.arange(num_blocks, device=logits.device, dtype=torch.int64) * block_size
    return values, idx.to(torch.int64) + offsets


def _assert_valid_block_outputs(argmax, local_max, vocab_size, block_size):
    """Check that every launched block writes a finite, in-range winner."""
    num_blocks = triton.cdiv(vocab_size, block_size)
    assert argmax.shape[1] == num_blocks
    assert bool(torch.isfinite(local_max).all()), "resample produced a non-finite block maximum"
    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = min(block_start + block_size, vocab_size)
        chosen = argmax[:, block_idx]
        assert bool(((chosen >= block_start) & (chosen < block_end)).all()), (
            f"block {block_idx} returned an index outside [{block_start}, {block_end})"
        )


def _new_outputs(num_reqs, num_blocks):
    """Poisoned resample outputs, so an early return is distinguishable from a write."""
    argmax = torch.full((num_reqs, num_blocks), _ARGMAX_POISON, dtype=torch.int64, device=DEVICE)
    local_max = torch.full((num_reqs, num_blocks), _MAX_POISON, dtype=torch.float32, device=DEVICE)
    return argmax, local_max


def _run_resample(
    *,
    target_logits,
    draft_logits,
    draft_sampled,
    cu_num_logits,
    expanded_idx_mapping,
    rejected_step,
    temperature,
    seeds,
    pos,
    target_lse,
    draft_lse,
    block_size=RESAMPLE_BLOCK_SIZE,
):
    num_reqs = cu_num_logits.shape[0] - 1
    vocab_size = target_logits.shape[1]
    num_blocks = triton.cdiv(vocab_size, block_size)
    has_draft_logits = draft_logits is not None
    if draft_logits is None:
        draft_logits = target_logits.new_empty(1, 1, 1)

    argmax, local_max = _new_outputs(num_reqs, num_blocks)
    _resample_kernel[(num_reqs, num_blocks)](
        argmax,
        argmax.stride(0),
        local_max,
        local_max.stride(0),
        target_logits,
        target_logits.stride(0),
        target_lse,
        draft_logits,
        draft_logits.stride(0),
        draft_logits.stride(1),
        draft_lse,
        rejected_step,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        seeds,
        pos,
        vocab_size,
        BLOCK_SIZE=block_size,
        HAS_DRAFT_LOGITS=has_draft_logits,
    )
    torch.npu.synchronize()
    return argmax, local_max


def _make_batch(num_logits_per_req, vocab_size, max_num_reqs, temps, seed=99):
    """Build a resample batch.

    `expanded_idx_mapping` deliberately maps to shuffled, non-contiguous
    request-state rows, and the number of logits differs per request, so that
    `req_idx` / `req_state_idx` / `resample_token_idx` confusions cannot pass.
    """
    torch.manual_seed(seed)
    num_reqs = len(num_logits_per_req)
    cu = [0]
    for n in num_logits_per_req:
        cu.append(cu[-1] + n)
    num_logits = cu[-1]
    cu_num_logits = torch.tensor(cu, dtype=torch.int32, device=DEVICE)

    rows = torch.randperm(max_num_reqs)[:num_reqs].to(torch.int32)
    expanded = torch.empty(num_logits, dtype=torch.int32)
    for r, n in enumerate(num_logits_per_req):
        expanded[cu[r] : cu[r + 1]] = rows[r]
    expanded_idx_mapping = expanded.to(DEVICE)

    target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=DEVICE)
    draft_sampled = torch.randint(0, vocab_size, (num_logits,), dtype=torch.int32, device=DEVICE)
    temperature = torch.zeros(max_num_reqs, dtype=torch.float32, device=DEVICE)
    for r, t in enumerate(temps):
        temperature[int(rows[r])] = t
    seeds = torch.randint(1, 2**30, (max_num_reqs,), dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_logits, dtype=torch.int64, device=DEVICE) * 3 + 11
    return {
        "cu": cu,
        "rows": rows,
        "cu_num_logits": cu_num_logits,
        "expanded_idx_mapping": expanded_idx_mapping,
        "target_logits": target_logits,
        "draft_sampled": draft_sampled,
        "temperature": temperature,
        "seeds": seeds,
        "pos": pos,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@torch.inference_mode()
def test_resample_greedy_bonus_is_plain_argmax():
    """Greedy request whose whole draft was accepted: resample the bonus token.

    `temp == 0 and is_bonus` is the one combination that does *not* take the
    early return, and it carries no noise, so the expected output is an exact
    per-block argmax of the raw target logits.  This is the path every greedy
    spec-decode step ends on.
    """
    num_logits_per_req = [4, 3, 5]
    vocab_size = 2 * RESAMPLE_BLOCK_SIZE + 137
    batch = _make_batch(num_logits_per_req, vocab_size, max_num_reqs=9, temps=[0.0, 0.0, 0.0])
    num_reqs = len(num_logits_per_req)
    # rejected_step = num_tokens - 1 => resample_token_idx == end_idx - 1 => bonus.
    rejected_step = torch.tensor([n - 1 for n in num_logits_per_req], dtype=torch.int32, device=DEVICE)
    lse = torch.zeros(num_reqs, dtype=torch.float32, device=DEVICE)

    argmax, local_max = _run_resample(
        target_logits=batch["target_logits"],
        draft_logits=None,
        draft_sampled=batch["draft_sampled"],
        cu_num_logits=batch["cu_num_logits"],
        expanded_idx_mapping=batch["expanded_idx_mapping"],
        rejected_step=rejected_step,
        temperature=batch["temperature"],
        seeds=batch["seeds"],
        pos=batch["pos"],
        target_lse=lse,
        draft_lse=lse,
    )

    bonus_rows = torch.tensor([batch["cu"][r + 1] - 1 for r in range(num_reqs)], dtype=torch.long, device=DEVICE)
    ref_val, ref_idx = _ref_block_argmax(batch["target_logits"][bonus_rows], vocab_size, RESAMPLE_BLOCK_SIZE)
    torch.testing.assert_close(local_max, ref_val, rtol=_RTOL, atol=_ATOL)
    assert torch.equal(argmax, ref_idx)
    assert int(argmax.max()) < vocab_size, "tail block resampled a padded position"


@torch.inference_mode()
def test_resample_greedy_non_bonus_returns_without_writing():
    """Greedy request with a rejected draft token: the kernel must return early.

    `_insert_resampled_kernel` skips the same `temp == 0 and not is_bonus`
    combination and reuses the target argmax already stored by the rejection
    kernel, so the resample outputs stay *uninitialised* (`new_empty`) on this
    path.  If the early return were dropped, the sampler would still work but
    the outputs would silently become live -- which is exactly what this asserts
    against, by poisoning them first.
    """
    num_logits_per_req = [4, 3]
    vocab_size = RESAMPLE_BLOCK_SIZE + 3
    batch = _make_batch(num_logits_per_req, vocab_size, max_num_reqs=6, temps=[0.0, 0.0])
    # rejected at step 0 and 1: strictly before the bonus slot.
    rejected_step = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    lse = torch.zeros(2, dtype=torch.float32, device=DEVICE)

    argmax, local_max = _run_resample(
        target_logits=batch["target_logits"],
        draft_logits=None,
        draft_sampled=batch["draft_sampled"],
        cu_num_logits=batch["cu_num_logits"],
        expanded_idx_mapping=batch["expanded_idx_mapping"],
        rejected_step=rejected_step,
        temperature=batch["temperature"],
        seeds=batch["seeds"],
        pos=batch["pos"],
        target_lse=lse,
        draft_lse=lse,
    )

    assert bool((argmax == _ARGMAX_POISON).all()), "greedy non-bonus request wrote resampled_local_argmax"
    assert bool((local_max == _MAX_POISON).all()), "greedy non-bonus request wrote resampled_local_max"


@torch.inference_mode()
def test_resample_one_hot_draft_excludes_rejected_token():
    """HAS_DRAFT_LOGITS=False: the residual is the target with the draft token knocked out.

    Two things are pinned here:
      * the token read is `draft_sampled[resample_token_idx + 1]` -- the input-id
        stream is shifted one slot ahead of the logits, and an off-by-one here
        would exclude an innocent token and keep the rejected one eligible;
      * -inf survives the noise add, so the rejected token can never come back.
    """
    num_logits_per_req = [4, 5]
    vocab_size = 2 * RESAMPLE_BLOCK_SIZE + 91
    batch = _make_batch(num_logits_per_req, vocab_size, max_num_reqs=7, temps=[1.0, 0.6])
    num_reqs = len(num_logits_per_req)
    rejected_step = torch.tensor([1, 2], dtype=torch.int32, device=DEVICE)
    lse = torch.zeros(num_reqs, dtype=torch.float32, device=DEVICE)

    # Guard the guard: make the excluded token the outright winner of its block,
    # so that "the kernel forgot to exclude it" is a guaranteed failure rather
    # than something a random draw might hide.
    resample_tokens = [batch["cu"][r] + int(rejected_step[r]) for r in range(num_reqs)]
    for r, tok in enumerate(resample_tokens):
        rejected = int(batch["draft_sampled"][tok + 1])
        batch["target_logits"][tok, rejected] = 50.0

    argmax, local_max = _run_resample(
        target_logits=batch["target_logits"],
        draft_logits=None,
        draft_sampled=batch["draft_sampled"],
        cu_num_logits=batch["cu_num_logits"],
        expanded_idx_mapping=batch["expanded_idx_mapping"],
        rejected_step=rejected_step,
        temperature=batch["temperature"],
        seeds=batch["seeds"],
        pos=batch["pos"],
        target_lse=lse,
        draft_lse=lse,
    )

    _assert_valid_block_outputs(argmax, local_max, vocab_size, RESAMPLE_BLOCK_SIZE)

    for r, tok in enumerate(resample_tokens):
        rejected = int(batch["draft_sampled"][tok + 1])
        assert rejected not in argmax[r].tolist(), "the rejected draft token was resampled"


@torch.inference_mode()
def test_resample_draft_logits_selects_positive_residual():
    """HAS_DRAFT_LOGITS=True keeps only positions where target probability wins.

    Each vocabulary block has exactly one position with `q < p`; every other
    position has `q == p` and therefore a -inf residual.  Gumbel noise cannot
    change the sole finite winner, so this checks the production kernel without
    reproducing its RNG implementation in the test.
    """
    num_logits_per_req = [4, 3]
    vocab_size = RESAMPLE_BLOCK_SIZE + 233
    num_spec_steps = 3
    max_num_reqs = 6
    batch = _make_batch(num_logits_per_req, vocab_size, max_num_reqs, temps=[1.0, 0.9])
    num_reqs = len(num_logits_per_req)
    rejected_step = torch.tensor([1, 0], dtype=torch.int32, device=DEVICE)

    batch["target_logits"].zero_()
    draft_logits = torch.zeros(max_num_reqs, num_spec_steps, vocab_size, dtype=torch.float32, device=DEVICE)
    target_lse = torch.zeros(num_reqs, dtype=torch.float32, device=DEVICE)
    draft_lse = torch.zeros(num_reqs, dtype=torch.float32, device=DEVICE)

    num_blocks = triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE)
    expected_winners = []
    for req_idx in range(num_reqs):
        req_state_idx = int(batch["rows"][req_idx])
        step = int(rejected_step[req_idx])
        request_winners = []
        for block_idx in range(num_blocks):
            block_start = block_idx * RESAMPLE_BLOCK_SIZE
            block_end = min(block_start + RESAMPLE_BLOCK_SIZE, vocab_size)
            winner = min(block_start + 17 + req_idx, block_end - 1)
            draft_logits[req_state_idx, step, winner] = -10.0
            request_winners.append(winner)
        expected_winners.append(request_winners)

    argmax, local_max = _run_resample(
        target_logits=batch["target_logits"],
        draft_logits=draft_logits,
        draft_sampled=batch["draft_sampled"],
        cu_num_logits=batch["cu_num_logits"],
        expanded_idx_mapping=batch["expanded_idx_mapping"],
        rejected_step=rejected_step,
        temperature=batch["temperature"],
        seeds=batch["seeds"],
        pos=batch["pos"],
        target_lse=target_lse,
        draft_lse=draft_lse,
    )

    expected = torch.tensor(expected_winners, dtype=torch.int64, device=DEVICE)
    assert torch.equal(argmax, expected)
    _assert_valid_block_outputs(argmax, local_max, vocab_size, RESAMPLE_BLOCK_SIZE)


@torch.inference_mode()
def test_resample_bonus_ignores_draft_logits():
    """A sampling bonus uses target logits even when draft logits are present.

    Each block has one finite target position, making the expected winner
    independent of the Gumbel draw.  Arbitrary draft logits must not affect the
    bonus branch.
    """
    num_logits_per_req = [3, 4]
    vocab_size = RESAMPLE_BLOCK_SIZE + 401
    num_spec_steps = 3
    max_num_reqs = 5
    batch = _make_batch(num_logits_per_req, vocab_size, max_num_reqs, temps=[1.0, 1.5])
    num_reqs = len(num_logits_per_req)
    rejected_step = torch.tensor([n - 1 for n in num_logits_per_req], dtype=torch.int32, device=DEVICE)
    draft_logits = torch.randn(max_num_reqs, num_spec_steps, vocab_size, dtype=torch.float32, device=DEVICE)
    lse = torch.zeros(num_reqs, dtype=torch.float32, device=DEVICE)

    num_blocks = triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE)
    expected_winners = []
    for req_idx in range(num_reqs):
        token_idx = batch["cu"][req_idx + 1] - 1
        batch["target_logits"][token_idx].fill_(float("-inf"))
        request_winners = []
        for block_idx in range(num_blocks):
            block_start = block_idx * RESAMPLE_BLOCK_SIZE
            block_end = min(block_start + RESAMPLE_BLOCK_SIZE, vocab_size)
            winner = min(block_start + 7 + req_idx, block_end - 1)
            batch["target_logits"][token_idx, winner] = 0.0
            request_winners.append(winner)
        expected_winners.append(request_winners)

    argmax, local_max = _run_resample(
        target_logits=batch["target_logits"],
        draft_logits=draft_logits,
        draft_sampled=batch["draft_sampled"],
        cu_num_logits=batch["cu_num_logits"],
        expanded_idx_mapping=batch["expanded_idx_mapping"],
        rejected_step=rejected_step,
        temperature=batch["temperature"],
        seeds=batch["seeds"],
        pos=batch["pos"],
        target_lse=lse,
        draft_lse=lse,
    )

    expected = torch.tensor(expected_winners, dtype=torch.int64, device=DEVICE)
    assert torch.equal(argmax, expected)
    _assert_valid_block_outputs(argmax, local_max, vocab_size, RESAMPLE_BLOCK_SIZE)


@torch.inference_mode()
def test_resample_mixed_batch_keeps_requests_independent():
    """One launch, greedy + sampling + bonus + rejected all mixed.

    5 requests and 3 vocab blocks -- both non-powers of two and unequal, so a
    swapped `//` / `%` in the grid mapping cannot come out right by accident.
    Greedy non-bonus rows must stay poisoned while their neighbours are written,
    which is the invariant a per-request early return is easiest to break.
    """
    num_logits_per_req = [2, 5, 3, 4, 3]
    vocab_size = 2 * RESAMPLE_BLOCK_SIZE + 17
    max_num_reqs = 13
    temps = [0.0, 0.0, 1.0, 0.8, 1.2]
    batch = _make_batch(num_logits_per_req, vocab_size, max_num_reqs, temps=temps, seed=4242)
    num_reqs = len(num_logits_per_req)
    # req0: greedy bonus, req1: greedy rejected, req2: sampling rejected,
    # req3: sampling bonus, req4: sampling rejected at step 0.
    steps = [num_logits_per_req[0] - 1, 2, 1, num_logits_per_req[3] - 1, 0]
    rejected_step = torch.tensor(steps, dtype=torch.int32, device=DEVICE)
    lse = torch.zeros(num_reqs, dtype=torch.float32, device=DEVICE)

    argmax, local_max = _run_resample(
        target_logits=batch["target_logits"],
        draft_logits=None,
        draft_sampled=batch["draft_sampled"],
        cu_num_logits=batch["cu_num_logits"],
        expanded_idx_mapping=batch["expanded_idx_mapping"],
        rejected_step=rejected_step,
        temperature=batch["temperature"],
        seeds=batch["seeds"],
        pos=batch["pos"],
        target_lse=lse,
        draft_lse=lse,
    )

    for r in range(num_reqs):
        tok = batch["cu"][r] + steps[r]
        is_bonus = tok == batch["cu"][r + 1] - 1
        temp = temps[r]
        if temp == 0.0 and not is_bonus:
            assert bool((argmax[r] == _ARGMAX_POISON).all()), f"req {r} should have returned early"
            assert bool((local_max[r] == _MAX_POISON).all()), f"req {r} should have returned early"
            continue

        if temp == 0.0:
            ref_val, ref_idx = _ref_block_argmax(
                batch["target_logits"][tok].unsqueeze(0), vocab_size, RESAMPLE_BLOCK_SIZE
            )
            torch.testing.assert_close(local_max[r : r + 1], ref_val, rtol=_RTOL, atol=_ATOL)
            assert torch.equal(argmax[r : r + 1], ref_idx)
        else:
            _assert_valid_block_outputs(argmax[r : r + 1], local_max[r : r + 1], vocab_size, RESAMPLE_BLOCK_SIZE)
            if not is_bonus:
                rejected = int(batch["draft_sampled"][tok + 1])
                assert rejected not in argmax[r].tolist(), f"req {r} resampled its rejected draft token"


@torch.inference_mode()
def test_resample_is_deterministic():
    """Two identical launches must produce bit-identical output.

    The kernel derives its randomness only from (seed, pos); if anything else
    leaked in -- program id, launch order, uninitialised memory -- rerunning the
    same batch would drift, and a seeded request would stop being reproducible.
    """
    num_logits_per_req = [4, 4]
    vocab_size = RESAMPLE_BLOCK_SIZE + 7
    batch = _make_batch(num_logits_per_req, vocab_size, max_num_reqs=6, temps=[1.0, 1.0])
    rejected_step = torch.tensor([1, 3], dtype=torch.int32, device=DEVICE)
    lse = torch.zeros(2, dtype=torch.float32, device=DEVICE)

    kwargs = dict(
        target_logits=batch["target_logits"],
        draft_logits=None,
        draft_sampled=batch["draft_sampled"],
        cu_num_logits=batch["cu_num_logits"],
        expanded_idx_mapping=batch["expanded_idx_mapping"],
        rejected_step=rejected_step,
        temperature=batch["temperature"],
        seeds=batch["seeds"],
        pos=batch["pos"],
        target_lse=lse,
        draft_lse=lse,
    )
    a1, m1 = _run_resample(**kwargs)
    a2, m2 = _run_resample(**kwargs)
    assert torch.equal(a1, a2)
    torch.testing.assert_close(m1, m2, rtol=0.0, atol=0.0)


@torch.inference_mode()
def test_resample_argmax_follows_softmax_distribution():
    """The resampled token must follow softmax(residual), not merely "some token".

    This validates the RNG from its mathematical behavior without reproducing
    the implementation: adding independent Gumbel noise and taking the argmax
    must sample from `softmax(logits)`.  It catches a constant draw, biased seed
    mixing, or a sign error in the Gumbel transform.

    Driven through the bonus branch (residual == target logits) so the expected
    distribution is the target softmax itself.  One request per draw, each with
    a distinct `pos`, since the noise is keyed on (seed, pos).
    """
    num_draws, vocab_size, max_num_reqs = 16384, 8, 1
    torch.manual_seed(7)
    row_logits = torch.tensor([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0], dtype=torch.float32)
    target_logits = row_logits.to(DEVICE).repeat(num_draws, 1).contiguous()

    # One logit per request => resample_token_idx == end_idx - 1 => bonus branch.
    cu_num_logits = torch.arange(num_draws + 1, dtype=torch.int32, device=DEVICE)
    expanded_idx_mapping = torch.zeros(num_draws, dtype=torch.int32, device=DEVICE)
    rejected_step = torch.zeros(num_draws, dtype=torch.int32, device=DEVICE)
    draft_sampled = torch.zeros(num_draws, dtype=torch.int32, device=DEVICE)
    temperature = torch.ones(max_num_reqs, dtype=torch.float32, device=DEVICE)
    seeds = torch.full((max_num_reqs,), 20260827, dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_draws, dtype=torch.int64, device=DEVICE)
    lse = torch.zeros(num_draws, dtype=torch.float32, device=DEVICE)

    argmax, _ = _run_resample(
        target_logits=target_logits,
        draft_logits=None,
        draft_sampled=draft_sampled,
        cu_num_logits=cu_num_logits,
        expanded_idx_mapping=expanded_idx_mapping,
        rejected_step=rejected_step,
        temperature=temperature,
        seeds=seeds,
        pos=pos,
        target_lse=lse,
        draft_lse=lse,
        block_size=vocab_size,
    )

    counts = torch.bincount(argmax.flatten().cpu(), minlength=vocab_size).float()
    empirical = counts / num_draws
    expected = torch.softmax(row_logits, dim=0)
    # 16384 draws => per-category std <= 0.004; 0.02 is ~5 sigma and leaves room
    # for the coarse fp32 tail of `-log(-log(u))` (see the operator doc).
    assert torch.allclose(empirical, expected, atol=0.02), (
        f"argmax frequencies {empirical.tolist()} deviate from softmax {expected.tolist()}"
    )


@torch.inference_mode()
def test_resample_draft_logits_follows_residual_distribution():
    """The draft-logits branch must sample from the normalised residual `(p - q)+`.

    `test_resample_draft_logits_selects_positive_residual` pins which positions
    survive `ratio < 1`, but with degenerate all-zero inputs it cannot see the
    value the surviving positions get: writing `log(ratio)` instead of
    `log(1 - ratio)`, or dropping a log-sum-exp, leaves the same single winner.
    Sampling frequencies do see it -- both of those mutations shift this
    distribution by more than 0.3 and 0.11 respectively, against a 0.02 tolerance.

    Two logits per request so the resample slot is not the bonus slot, which is
    what selects the draft-logits branch.
    """
    num_draws, vocab_size, num_spec_steps = 16384, 8, 1
    torch.manual_seed(11)
    target_row = torch.tensor([0.9, 0.7, 0.6, 0.5, 0.3, 0.1, -0.2, -0.6], dtype=torch.float32)
    draft_row = torch.tensor([-0.8, 1.5, -0.7, 1.3, 0.2, 1.0, -0.4, 0.6], dtype=torch.float32)

    num_logits = 2 * num_draws
    target_logits = target_row.to(DEVICE).repeat(num_logits, 1).contiguous()
    draft_logits = draft_row.to(DEVICE).view(1, 1, vocab_size).repeat(1, num_spec_steps, 1).contiguous()

    # Two logits per request, resampling step 0 => resample_token_idx < end_idx - 1.
    cu_num_logits = torch.arange(0, num_logits + 1, 2, dtype=torch.int32, device=DEVICE)
    expanded_idx_mapping = torch.zeros(num_logits, dtype=torch.int32, device=DEVICE)
    rejected_step = torch.zeros(num_draws, dtype=torch.int32, device=DEVICE)
    draft_sampled = torch.zeros(num_logits, dtype=torch.int32, device=DEVICE)
    temperature = torch.ones(1, dtype=torch.float32, device=DEVICE)
    seeds = torch.full((1,), 20260829, dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_logits, dtype=torch.int64, device=DEVICE)
    target_lse = torch.full((num_draws,), float(torch.logsumexp(target_row, dim=0)), device=DEVICE)
    draft_lse = torch.full((num_draws,), float(torch.logsumexp(draft_row, dim=0)), device=DEVICE)

    argmax, _ = _run_resample(
        target_logits=target_logits,
        draft_logits=draft_logits,
        draft_sampled=draft_sampled,
        cu_num_logits=cu_num_logits,
        expanded_idx_mapping=expanded_idx_mapping,
        rejected_step=rejected_step,
        temperature=temperature,
        seeds=seeds,
        pos=pos,
        target_lse=target_lse,
        draft_lse=draft_lse,
        block_size=vocab_size,
    )

    prob_target = torch.softmax(target_row, dim=0)
    prob_draft = torch.softmax(draft_row, dim=0)
    residual = (prob_target - prob_draft).clamp(min=0.0)
    expected = residual / residual.sum()

    # Guard the guard: the input must actually exercise both sides of `ratio < 1`.
    excluded = (residual == 0.0).nonzero().flatten().tolist()
    assert excluded and len(excluded) < vocab_size, "input no longer produces both ratio<1 and ratio>=1 tokens"

    counts = torch.bincount(argmax.flatten().cpu(), minlength=vocab_size).float()
    empirical = counts / num_draws
    assert not any(counts[i] for i in excluded), (
        f"tokens with q >= p were resampled: {[i for i in excluded if counts[i]]}"
    )
    # 16384 draws => per-category std <= 0.004; 0.02 is ~5 sigma.
    assert torch.allclose(empirical, expected, atol=0.02), (
        f"resampled frequencies {empirical.tolist()} deviate from the residual {expected.tolist()}"
    )


# ---------------------------------------------------------------------------
# End-to-end through the patched entry point
# ---------------------------------------------------------------------------


@torch.inference_mode()
def test_rejection_sample_greedy_end_to_end():
    """Full `rejection_sample` on a greedy batch, against a loop reference.

    This is the only case that runs `_resample_kernel` with the production
    launch configuration (BLOCK_SIZE=1024, grid from `cdiv(vocab, 1024)`) and
    with `_insert_resampled_kernel` downstream, so it is what proves the two
    early-return conditions in the two kernels still agree: on a rejected greedy
    token the resample output is garbage *and must not be read*, on a fully
    accepted one the bonus token comes out of it.
    """
    num_reqs = 5
    num_spec_steps = 3
    num_logits_per_req = num_spec_steps + 1
    vocab_size = 2 * VOCAB_BLOCK_SIZE + 37
    max_num_reqs = 11
    torch.manual_seed(2026)

    cu = [i * num_logits_per_req for i in range(num_reqs + 1)]
    num_logits = cu[-1]
    cu_num_logits = torch.tensor(cu, dtype=torch.int32, device=DEVICE)
    rows = torch.randperm(max_num_reqs)[:num_reqs].to(torch.int32)
    idx_mapping = rows.to(DEVICE)
    expanded_idx_mapping = rows.repeat_interleave(num_logits_per_req).to(DEVICE)
    expanded_local_pos = torch.arange(num_logits_per_req, dtype=torch.int32).repeat(num_reqs).to(DEVICE)

    target_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=DEVICE)
    draft_sampled = torch.randint(0, vocab_size, (num_logits,), dtype=torch.int32, device=DEVICE)
    # Force a spread of acceptance lengths: request r accepts exactly r draft
    # tokens (r == num_spec_steps means "everything accepted, take the bonus").
    target_argmax = target_logits.argmax(dim=1)
    for r in range(num_reqs):
        accept = min(r, num_spec_steps)
        for i in range(num_spec_steps):
            slot = cu[r] + i + 1
            if i < accept:
                draft_sampled[slot] = target_argmax[cu[r] + i]
            else:
                bad = (int(target_argmax[cu[r] + i]) + 1) % vocab_size
                draft_sampled[slot] = bad

    temperature = torch.zeros(max_num_reqs, dtype=torch.float32, device=DEVICE)
    seeds = torch.randint(1, 2**30, (max_num_reqs,), dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_logits, dtype=torch.int64, device=DEVICE) + 5

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
        seeds,
        num_spec_steps,
    )
    torch.npu.synchronize()

    argmax_cpu = target_argmax.cpu().tolist()
    draft_cpu = draft_sampled.cpu().tolist()
    for r in range(num_reqs):
        expected = []
        accepted = 0
        for i in range(num_logits_per_req - 1):
            targ = argmax_cpu[cu[r] + i]
            expected.append(targ)
            if targ != draft_cpu[cu[r] + i + 1]:
                break
            accepted += 1
        else:
            # Everything accepted: the bonus token is resampled from the last
            # logit row, which is the `_resample_kernel` greedy-bonus path.
            expected.append(argmax_cpu[cu[r + 1] - 1])
        assert int(num_sampled[r]) == accepted + 1, f"req {r}: wrong accepted length"
        assert sampled[r, : accepted + 1].cpu().tolist() == expected, f"req {r}: wrong tokens"

    # Guard the guard: the batch must contain both a rejected request (early
    # return path) and a fully accepted one (bonus resample path).
    lengths = num_sampled.cpu().tolist()
    assert min(lengths) == 1 and max(lengths) == num_logits_per_req, (
        f"batch no longer covers both resample branches: {lengths}"
    )
