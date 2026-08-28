# SPDX-License-Identifier: Apache-2.0
# Numerical test for vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils
# (`_resample_kernel` and the `_npu_gumbel_block_argmax` device function it
# calls) against a plain PyTorch fp32 reference.
# Requires NPU and Triton-Ascend.
#
# See vllm_ascend/ops/triton/docs/resample.md for the operator spec.
#
# Regression scope: #9155 (main2main import of the MRV2 rejection sampler) and
# #13470 (probabilistic rejection sampling enabled on NPU) -- neither PR shipped
# any numerical coverage for these two kernels.
#
# `_npu_gumbel_block_argmax` is a `@triton.jit` *device* function: it cannot be
# launched from host.  It is exercised two ways here:
#   * directly, through the thin probe kernel `_gumbel_probe_kernel` below,
#     which mirrors the real call site in `_resample_kernel`;
#   * indirectly, through `_resample_kernel` itself.
# The Gumbel noise it draws comes from Triton's philox, which has no PyTorch
# equivalent, so `_gumbel_noise_probe_kernel` re-draws the *same* stream and the
# reference consumes it.  That pins down everything except the RNG itself
# (temperature scaling, processed-logits store, masking, the -inf branch, the
# block-local argmax); the RNG is covered separately and statistically by
# `test_gumbel_argmax_follows_softmax_distribution`.

import gc

import pytest
import torch
import torch_npu  # noqa: F401  # registers the npu backend / torch.npu namespace
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import (
    _npu_gumbel_block_argmax,
    _resample_kernel,
    rejection_sample,
)

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
# Probe kernels (test-only harness)
# ---------------------------------------------------------------------------


@triton.jit
def _gumbel_probe_kernel(
    out_value_ptr,
    out_value_stride,
    out_idx_ptr,
    out_idx_stride,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    # 0 = no processed_logits (mirrors the real call site in _resample_kernel)
    # 1 = processed_logits, implicit column 0
    # 2 = processed_logits, column read from processed_logits_col_ptr
    PROCESSED_MODE: tl.constexpr,
):
    """Thin host-launchable wrapper around the `_npu_gumbel_block_argmax` device function.

    The three `PROCESSED_MODE` variants pass literal `None`s rather than relying
    on `None` surviving a kernel-argument boundary, so each variant compiles the
    same way the production call site does.
    """
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)

    if PROCESSED_MODE == 0:
        value, idx = _npu_gumbel_block_argmax(
            logits,
            block,
            mask,
            token_idx,
            expanded_idx_mapping_ptr,
            temp_ptr,
            seeds_ptr,
            pos_ptr,
            None,
            0,
            None,
            vocab_size,
            APPLY_TEMPERATURE=APPLY_TEMPERATURE,
        )
    elif PROCESSED_MODE == 1:
        value, idx = _npu_gumbel_block_argmax(
            logits,
            block,
            mask,
            token_idx,
            expanded_idx_mapping_ptr,
            temp_ptr,
            seeds_ptr,
            pos_ptr,
            processed_logits_ptr,
            processed_logits_stride,
            None,
            vocab_size,
            APPLY_TEMPERATURE=APPLY_TEMPERATURE,
        )
    else:
        value, idx = _npu_gumbel_block_argmax(
            logits,
            block,
            mask,
            token_idx,
            expanded_idx_mapping_ptr,
            temp_ptr,
            seeds_ptr,
            pos_ptr,
            processed_logits_ptr,
            processed_logits_stride,
            processed_logits_col_ptr,
            vocab_size,
            APPLY_TEMPERATURE=APPLY_TEMPERATURE,
        )

    tl.store(out_value_ptr + token_idx * out_value_stride + block_idx, value)
    tl.store(out_idx_ptr + token_idx * out_idx_stride + block_idx, block_idx * BLOCK_SIZE + idx)


@triton.jit
def _gumbel_noise_probe_kernel(
    noise_ptr,
    noise_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Re-draw the exact Gumbel noise `_npu_gumbel_block_argmax` uses.

    Line-for-line copy of the RNG block of the device function.  It only
    reproduces the noise; it says nothing about how the noise is combined with
    the logits, which is what the tests using it check.
    """
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    seed = tl.load(seeds_ptr + req_state_idx)
    pos = tl.load(pos_ptr + token_idx).to(tl.int32)
    gumbel_seed = tl.randint(seed, pos)
    r = tl.rand(gumbel_seed, block).to(tl.float32)
    gumbel_noise = -tl.log(-tl.log(r + 1e-20) + 1e-20)
    tl.store(noise_ptr + token_idx * noise_stride + block, gumbel_noise, mask=mask)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _draw_noise(num_tokens, vocab_size, expanded_idx_mapping, seeds, pos, block_size):
    """Materialise [num_tokens, vocab_size] of the kernel's own Gumbel noise."""
    num_blocks = triton.cdiv(vocab_size, block_size)
    noise = torch.zeros(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    _gumbel_noise_probe_kernel[(num_tokens, num_blocks)](
        noise,
        noise.stride(0),
        expanded_idx_mapping,
        seeds,
        pos,
        vocab_size,
        BLOCK_SIZE=block_size,
    )
    torch.npu.synchronize()
    return noise


def _ref_block_argmax(logits, vocab_size, block_size, noise=None):
    """Per-block max/argmax over `logits`, fp32, deliberately loop-free but
    written straight from the kernel's semantics:

      * positions >= vocab_size are -inf (the kernel's `other=-inf` load plus
        the `tl.where(mask, ...)` re-mask), so they can never win a block;
      * noise is added only when the caller says so, and -inf + noise stays -inf,
        which is what keeps excluded tokens out of the argmax;
      * the returned index is *global* (`block_idx * BLOCK_SIZE + idx`).

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
    scored = logits.float() if noise is None else logits.float() + noise.float()
    # -inf entries (masked-out / excluded tokens) must stay -inf even after the
    # noise add; float arithmetic already does that, but nan would not, so guard.
    scored = torch.where(torch.isneginf(logits.float()), logits.float(), scored)
    padded[:, :vocab_size] = scored
    padded = padded.view(num_tokens, num_blocks, block_size)
    values, idx = padded.max(dim=-1)
    offsets = torch.arange(num_blocks, device=logits.device, dtype=torch.int64) * block_size
    return values, idx.to(torch.int64) + offsets


def _assert_block_argmax_close(actual_idx, actual_val, ref_idx, ref_val, ref_scores, block_size):
    """Compare (value, index) pairs, tolerating an exact-tie index swap.

    The kernel and the reference both reduce in fp32 but not necessarily in the
    same order, so two near-equal candidates inside one block can swap.  A wrong
    index is only a real failure when the value it points at is *worse* than the
    reference maximum.
    """
    torch.testing.assert_close(actual_val.float(), ref_val.float(), rtol=_RTOL, atol=_ATOL)
    mismatch = actual_idx != ref_idx
    if not bool(mismatch.any()):
        return
    rows, blocks = torch.nonzero(mismatch, as_tuple=True)
    for r, b in zip(rows.tolist(), blocks.tolist()):
        chosen = int(actual_idx[r, b])
        assert b * block_size <= chosen < (b + 1) * block_size, (
            f"token {r} block {b}: index {chosen} escaped its own block"
        )
        assert chosen < ref_scores.shape[1], (
            f"token {r} block {b}: index {chosen} points into the padded tail past the vocabulary"
        )
        got = float(ref_scores[r, chosen])
        want = float(ref_val[r, b])
        assert abs(got - want) <= _ATOL + _RTOL * abs(want), (
            f"token {r} block {b}: kernel picked index {chosen} (score {got}) "
            f"but the reference maximum is {want} at index {int(ref_idx[r, b])}"
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


def _ref_residual_one_hot(target_logits, draft_sampled, resample_token_idx):
    """The `HAS_DRAFT_LOGITS=False` residual: the target row with one token removed.

    Mirrors the detail that is easy to get wrong -- the token read is
    `draft_sampled[resample_token_idx + 1]`, i.e. the *next* slot, because
    `draft_sampled` is the input-id stream and sits one position ahead of the
    logits it was drafted from.
    """
    out = target_logits[resample_token_idx].float().clone()
    out[int(draft_sampled[resample_token_idx + 1])] = float("-inf")
    return out


def _ref_residual_from_draft(target, draft, target_lse_val, draft_lse_val):
    """The `HAS_DRAFT_LOGITS=True` residual: `ratio < 1 ? log p + log(1 - ratio) : -inf`.

    fp32 throughout, and written in the kernel's own order (subtract the two
    logsumexps first, exponentiate the difference) rather than the algebraically
    equivalent `log(p - q)`, so the comparison is not testing a different
    formula's rounding.
    """
    target_log_probs = target.float() - target_lse_val
    draft_log_probs = draft.float() - draft_lse_val
    ratio = torch.exp(draft_log_probs - target_log_probs)
    residual = target_log_probs + torch.log(1.0 - ratio)
    return torch.where(ratio < 1.0, residual, torch.full_like(residual, float("-inf")))


# ---------------------------------------------------------------------------
# _npu_gumbel_block_argmax
# ---------------------------------------------------------------------------


def _gumbel_setup(num_tokens, vocab_size, max_num_reqs, temps, seed=1234, shuffle_rows=True):
    torch.manual_seed(seed)
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    if shuffle_rows:
        # req_state rows deliberately not equal to the token index, so mixing up
        # `token_idx` and `req_state_idx` cannot pass by accident.
        rows = torch.randperm(max_num_reqs)[:num_tokens].to(torch.int32)
    else:
        rows = torch.zeros(num_tokens, dtype=torch.int32)
    expanded_idx_mapping = rows.to(DEVICE)
    temperature = torch.zeros(max_num_reqs, dtype=torch.float32, device=DEVICE)
    for i, t in enumerate(temps):
        temperature[int(rows[i])] = t
    seeds = torch.randint(1, 2**30, (max_num_reqs,), dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE) * 7 + 3
    return logits, expanded_idx_mapping, temperature, seeds, pos


@torch.inference_mode()
def test_gumbel_greedy_is_plain_block_argmax():
    """temp == 0 disables the noise entirely -- the only fully deterministic path.

    This is the branch `_resample_kernel` takes for greedy bonus tokens, and the
    one the whole greedy spec-decode path depends on, so it is checked exactly
    rather than through the noise probe.  `vocab_size` is deliberately not a
    multiple of BLOCK_SIZE so the padded tail (`other=-inf`) is exercised.
    """
    num_tokens, vocab_size, max_num_reqs = 6, 3 * RESAMPLE_BLOCK_SIZE + 37, 11
    logits, mapping, temperature, seeds, pos = _gumbel_setup(num_tokens, vocab_size, max_num_reqs, [0.0] * num_tokens)
    num_blocks = triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE)

    values = torch.empty(num_tokens, num_blocks, dtype=torch.float32, device=DEVICE)
    idxs = torch.empty(num_tokens, num_blocks, dtype=torch.int64, device=DEVICE)
    dummy = torch.empty(1, dtype=torch.float32, device=DEVICE)
    dummy_col = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    _gumbel_probe_kernel[(num_tokens, num_blocks)](
        values,
        values.stride(0),
        idxs,
        idxs.stride(0),
        logits,
        logits.stride(0),
        mapping,
        temperature,
        seeds,
        pos,
        dummy,
        0,
        dummy_col,
        vocab_size,
        BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,
        APPLY_TEMPERATURE=False,
        PROCESSED_MODE=0,
    )
    torch.npu.synchronize()

    ref_val, ref_idx = _ref_block_argmax(logits, vocab_size, RESAMPLE_BLOCK_SIZE)
    torch.testing.assert_close(values, ref_val, rtol=_RTOL, atol=_ATOL)
    assert torch.equal(idxs, ref_idx)
    # The tail block must never point past the vocabulary.
    assert int(idxs.max()) < vocab_size


@pytest.mark.parametrize("apply_temperature", [True, False])
@torch.inference_mode()
def test_gumbel_matches_reference_with_noise(apply_temperature):
    """temp != 0: noise on, and `APPLY_TEMPERATURE` toggled on both sides.

    The `APPLY_TEMPERATURE=True` half is unreachable from `_resample_kernel`
    (which hardcodes False) but is part of the device function's contract and is
    what the upstream sampler uses, so both sides of the constexpr are pinned.
    """
    num_tokens, vocab_size, max_num_reqs = 5, 2 * RESAMPLE_BLOCK_SIZE + 11, 9
    temps = [0.5, 1.0, 2.0, 0.7, 1.3]
    logits, mapping, temperature, seeds, pos = _gumbel_setup(num_tokens, vocab_size, max_num_reqs, temps)
    num_blocks = triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE)

    values = torch.empty(num_tokens, num_blocks, dtype=torch.float32, device=DEVICE)
    idxs = torch.empty(num_tokens, num_blocks, dtype=torch.int64, device=DEVICE)
    dummy = torch.empty(1, dtype=torch.float32, device=DEVICE)
    dummy_col = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    _gumbel_probe_kernel[(num_tokens, num_blocks)](
        values,
        values.stride(0),
        idxs,
        idxs.stride(0),
        logits,
        logits.stride(0),
        mapping,
        temperature,
        seeds,
        pos,
        dummy,
        0,
        dummy_col,
        vocab_size,
        BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,
        APPLY_TEMPERATURE=apply_temperature,
        PROCESSED_MODE=0,
    )
    torch.npu.synchronize()

    noise = _draw_noise(num_tokens, vocab_size, mapping, seeds, pos, RESAMPLE_BLOCK_SIZE)
    # Guard the guard: a degenerate (constant / all-zero) noise draw would make
    # this test collapse into the greedy one above.
    assert float(noise.std()) > 0.1, "gumbel noise probe no longer produces a spread of values"

    scaled = logits.float()
    if apply_temperature:
        per_token_temp = temperature[mapping.long()].unsqueeze(1)
        scaled = scaled / per_token_temp
    ref_val, ref_idx = _ref_block_argmax(scaled, vocab_size, RESAMPLE_BLOCK_SIZE, noise=noise)

    _assert_block_argmax_close(idxs, values, ref_idx, ref_val, scaled + noise, RESAMPLE_BLOCK_SIZE)

    # Guard the guard: the noise must actually move at least one winner, else
    # this case proves nothing beyond the greedy path.
    _, greedy_idx = _ref_block_argmax(scaled, vocab_size, RESAMPLE_BLOCK_SIZE)
    assert bool((greedy_idx != ref_idx).any()), "noise no longer changes any block winner"


@pytest.mark.parametrize("processed_mode", [1, 2], ids=["implicit-col-0", "explicit-col"])
@torch.inference_mode()
def test_gumbel_stores_processed_logits(processed_mode):
    """The `processed_logits` side output: written before the noise, after the temperature.

    Both column modes are covered: `processed_logits_col_ptr is None` (column 0)
    and an explicit column, which is the branch that makes the write land at
    `req_state_idx * stride + col * vocab_size`.  Note the row index is
    `req_state_idx`, *not* `token_idx` -- getting that wrong is silent.
    """
    num_tokens, vocab_size, max_num_reqs = 4, RESAMPLE_BLOCK_SIZE + 5, 7
    num_cols = 3
    col = 2 if processed_mode == 2 else 0
    temps = [0.0, 0.5, 2.0, 1.0]
    logits, mapping, temperature, seeds, pos = _gumbel_setup(num_tokens, vocab_size, max_num_reqs, temps)
    num_blocks = triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE)

    processed = torch.full((max_num_reqs, num_cols * vocab_size), float("nan"), dtype=torch.float32, device=DEVICE)
    col_tensor = torch.tensor([col], dtype=torch.int32, device=DEVICE)
    values = torch.empty(num_tokens, num_blocks, dtype=torch.float32, device=DEVICE)
    idxs = torch.empty(num_tokens, num_blocks, dtype=torch.int64, device=DEVICE)
    _gumbel_probe_kernel[(num_tokens, num_blocks)](
        values,
        values.stride(0),
        idxs,
        idxs.stride(0),
        logits,
        logits.stride(0),
        mapping,
        temperature,
        seeds,
        pos,
        processed,
        processed.stride(0),
        col_tensor,
        vocab_size,
        BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,
        APPLY_TEMPERATURE=True,
        PROCESSED_MODE=processed_mode,
    )
    torch.npu.synchronize()

    for token_idx in range(num_tokens):
        row = int(mapping[token_idx])
        temp = float(temperature[row])
        expected = logits[token_idx].float()
        if temp != 0.0:
            expected = expected / temp
        actual = processed[row, col * vocab_size : col * vocab_size + vocab_size]
        torch.testing.assert_close(actual, expected, rtol=_RTOL, atol=_ATOL)

    # Rows/columns nobody wrote must still be untouched: the store is masked to
    # `block < vocab_size`, so no neighbouring column may be clobbered.
    written_rows = {int(r) for r in mapping}
    for row in range(max_num_reqs):
        for c in range(num_cols):
            if row in written_rows and c == col:
                continue
            chunk = processed[row, c * vocab_size : c * vocab_size + vocab_size]
            assert bool(torch.isnan(chunk).all()), f"row {row} col {c} was overwritten"


@torch.inference_mode()
def test_gumbel_argmax_follows_softmax_distribution():
    """Gumbel-max must sample proportionally to softmax(logits).

    This is the one check that does *not* reuse the kernel's own RNG, so it is
    the only thing standing between a broken philox call (wrong seed mixing,
    a constant draw, a sign error in `-log(-log(u))`) and a silently biased
    sampler.  8 categories in a single block, 16384 draws, one distinct `pos`
    per draw.
    """
    num_tokens, vocab_size, max_num_reqs = 16384, 8, 1
    torch.manual_seed(7)
    row_logits = torch.tensor([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0], dtype=torch.float32)
    logits = row_logits.to(DEVICE).repeat(num_tokens, 1).contiguous()
    mapping = torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE)
    temperature = torch.ones(max_num_reqs, dtype=torch.float32, device=DEVICE)
    seeds = torch.full((max_num_reqs,), 20260827, dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)

    values = torch.empty(num_tokens, 1, dtype=torch.float32, device=DEVICE)
    idxs = torch.empty(num_tokens, 1, dtype=torch.int64, device=DEVICE)
    dummy = torch.empty(1, dtype=torch.float32, device=DEVICE)
    dummy_col = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    _gumbel_probe_kernel[(num_tokens, 1)](
        values,
        values.stride(0),
        idxs,
        idxs.stride(0),
        logits,
        logits.stride(0),
        mapping,
        temperature,
        seeds,
        pos,
        dummy,
        0,
        dummy_col,
        vocab_size,
        BLOCK_SIZE=vocab_size,
        APPLY_TEMPERATURE=False,
        PROCESSED_MODE=0,
    )
    torch.npu.synchronize()

    counts = torch.bincount(idxs.flatten().cpu(), minlength=vocab_size).float()
    empirical = counts / num_tokens
    expected = torch.softmax(row_logits, dim=0)
    # 16384 draws => per-category std <= 0.004; 0.02 is ~5 sigma and leaves room
    # for the coarse fp32 tail of `-log(-log(u))` (see the operator doc).
    assert torch.allclose(empirical, expected, atol=0.02), (
        f"argmax frequencies {empirical.tolist()} deviate from softmax {expected.tolist()}"
    )


@torch.inference_mode()
def test_gumbel_is_deterministic_in_seed_and_pos():
    """Same (seed, pos) must give the same token; a different pos must not.

    Reproducibility across two runs of the same request is a user-visible
    property (`SamplingParams.seed`), and it is what makes the noise-probe
    oracle in the tests above legitimate.
    """
    num_tokens, vocab_size, max_num_reqs = 8, RESAMPLE_BLOCK_SIZE, 8
    logits, mapping, temperature, seeds, pos = _gumbel_setup(num_tokens, vocab_size, max_num_reqs, [1.0] * num_tokens)

    def _run(pos_tensor):
        values = torch.empty(num_tokens, 1, dtype=torch.float32, device=DEVICE)
        idxs = torch.empty(num_tokens, 1, dtype=torch.int64, device=DEVICE)
        dummy = torch.empty(1, dtype=torch.float32, device=DEVICE)
        dummy_col = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        _gumbel_probe_kernel[(num_tokens, 1)](
            values,
            values.stride(0),
            idxs,
            idxs.stride(0),
            logits,
            logits.stride(0),
            mapping,
            temperature,
            seeds,
            pos_tensor,
            dummy,
            0,
            dummy_col,
            vocab_size,
            BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,
            APPLY_TEMPERATURE=False,
            PROCESSED_MODE=0,
        )
        torch.npu.synchronize()
        return values, idxs

    v1, i1 = _run(pos)
    v2, i2 = _run(pos)
    assert torch.equal(i1, i2)
    torch.testing.assert_close(v1, v2, rtol=0.0, atol=0.0)

    v3, i3 = _run(pos + 1)
    assert bool((i3 != i1).any()), "shifting pos no longer changes the draw"


# ---------------------------------------------------------------------------
# _resample_kernel
# ---------------------------------------------------------------------------


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
        "num_logits": num_logits,
    }


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

    residual = torch.stack(
        [
            _ref_residual_one_hot(batch["target_logits"], batch["draft_sampled"], resample_tokens[r])
            for r in range(num_reqs)
        ]
    )
    token_rows = torch.tensor(resample_tokens, dtype=torch.long, device=DEVICE)
    noise = _draw_noise(
        batch["num_logits"],
        vocab_size,
        batch["expanded_idx_mapping"],
        batch["seeds"],
        batch["pos"],
        RESAMPLE_BLOCK_SIZE,
    )[token_rows]
    ref_val, ref_idx = _ref_block_argmax(residual, vocab_size, RESAMPLE_BLOCK_SIZE, noise=noise)
    _assert_block_argmax_close(argmax, local_max, ref_idx, ref_val, residual + noise, RESAMPLE_BLOCK_SIZE)

    for r, tok in enumerate(resample_tokens):
        rejected = int(batch["draft_sampled"][tok + 1])
        assert rejected not in argmax[r].tolist(), "the rejected draft token was resampled"


@torch.inference_mode()
def test_resample_draft_logits_residual_matches_reference():
    """HAS_DRAFT_LOGITS=True: the `max(0, p - q)` residual, in log space.

    Covers `residual = log p + log(1 - q/p)` where `q < p`, and the -inf fallback
    where `q >= p`.  The two logsumexp scalars come in per *request*, not per
    token, and `draft_logits` is indexed by `[req_state_idx, resample_idx]`,
    which is a different addressing scheme from every other tensor in the kernel.
    """
    num_logits_per_req = [4, 3]
    vocab_size = RESAMPLE_BLOCK_SIZE + 233
    num_spec_steps = 3
    max_num_reqs = 6
    batch = _make_batch(num_logits_per_req, vocab_size, max_num_reqs, temps=[1.0, 0.9])
    num_reqs = len(num_logits_per_req)
    rejected_step = torch.tensor([1, 0], dtype=torch.int32, device=DEVICE)

    draft_logits = torch.randn(max_num_reqs, num_spec_steps, vocab_size, dtype=torch.float32, device=DEVICE)
    target_lse = torch.logsumexp(batch["target_logits"], dim=1)[
        torch.tensor([batch["cu"][r] + int(rejected_step[r]) for r in range(num_reqs)], device=DEVICE)
    ].contiguous()
    draft_lse = torch.stack(
        [torch.logsumexp(draft_logits[int(batch["rows"][r]), int(rejected_step[r])], dim=0) for r in range(num_reqs)]
    ).contiguous()

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

    residuals = []
    for r in range(num_reqs):
        tok = batch["cu"][r] + int(rejected_step[r])
        residuals.append(
            _ref_residual_from_draft(
                batch["target_logits"][tok],
                draft_logits[int(batch["rows"][r]), int(rejected_step[r])],
                float(target_lse[r]),
                float(draft_lse[r]),
            )
        )
    residual = torch.stack(residuals)

    # Guard the guard: both sides of `ratio < 1.0` must be populated, otherwise
    # this case degenerates into the plain-argmax one.
    finite = torch.isfinite(residual)
    assert bool(finite.any()) and bool((~finite).any()), (
        "the random draft/target pair no longer produces both ratio<1 and ratio>=1 tokens"
    )

    token_rows = torch.tensor(
        [batch["cu"][r] + int(rejected_step[r]) for r in range(num_reqs)], dtype=torch.long, device=DEVICE
    )
    noise = _draw_noise(
        batch["num_logits"],
        vocab_size,
        batch["expanded_idx_mapping"],
        batch["seeds"],
        batch["pos"],
        RESAMPLE_BLOCK_SIZE,
    )[token_rows]
    ref_val, ref_idx = _ref_block_argmax(residual, vocab_size, RESAMPLE_BLOCK_SIZE, noise=noise)
    _assert_block_argmax_close(argmax, local_max, ref_idx, ref_val, residual + noise, RESAMPLE_BLOCK_SIZE)


@torch.inference_mode()
def test_resample_bonus_with_temperature_matches_reference():
    """temp != 0 and is_bonus: residual is the raw target, noise on.

    `is_bonus` short-circuits both residual branches, so the bonus token of a
    sampling request is drawn straight from the target distribution -- including
    when `HAS_DRAFT_LOGITS` is True, which is the case this pins.
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

    token_rows = torch.tensor([batch["cu"][r + 1] - 1 for r in range(num_reqs)], dtype=torch.long, device=DEVICE)
    residual = batch["target_logits"][token_rows].float()
    noise = _draw_noise(
        batch["num_logits"],
        vocab_size,
        batch["expanded_idx_mapping"],
        batch["seeds"],
        batch["pos"],
        RESAMPLE_BLOCK_SIZE,
    )[token_rows]
    ref_val, ref_idx = _ref_block_argmax(residual, vocab_size, RESAMPLE_BLOCK_SIZE, noise=noise)
    _assert_block_argmax_close(argmax, local_max, ref_idx, ref_val, residual + noise, RESAMPLE_BLOCK_SIZE)


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

    noise_all = _draw_noise(
        batch["num_logits"],
        vocab_size,
        batch["expanded_idx_mapping"],
        batch["seeds"],
        batch["pos"],
        RESAMPLE_BLOCK_SIZE,
    )
    for r in range(num_reqs):
        tok = batch["cu"][r] + steps[r]
        is_bonus = tok == batch["cu"][r + 1] - 1
        temp = temps[r]
        if temp == 0.0 and not is_bonus:
            assert bool((argmax[r] == _ARGMAX_POISON).all()), f"req {r} should have returned early"
            assert bool((local_max[r] == _MAX_POISON).all()), f"req {r} should have returned early"
            continue

        if is_bonus:
            residual = batch["target_logits"][tok].float()
        else:
            residual = _ref_residual_one_hot(batch["target_logits"], batch["draft_sampled"], tok)
        noise = None if temp == 0.0 else noise_all[tok].unsqueeze(0)
        ref_val, ref_idx = _ref_block_argmax(residual.unsqueeze(0), vocab_size, RESAMPLE_BLOCK_SIZE, noise=noise)
        scores = residual.unsqueeze(0) if noise is None else residual.unsqueeze(0) + noise
        _assert_block_argmax_close(
            argmax[r : r + 1], local_max[r : r + 1], ref_idx, ref_val, scores, RESAMPLE_BLOCK_SIZE
        )


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
