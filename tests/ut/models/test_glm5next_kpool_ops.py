# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.3-Flash kpool indexer's compress and expand steps.

Upstream fuses each of these into a Triton kernel; the Ascend versions are torch
expressions that have to reproduce the same arithmetic, because the compressed
keys they write are read back by the same top-k that the checkpoint was trained
against. Neither step touches the KV cache or attention metadata, so both are
checked here against references written independently of the implementation --
a Sylvester-constructed Hadamard matrix for the rotation, and a per-column loop
mirroring the Triton kernel for the expansion.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from vllm_ascend.models.glm5next.ops.kpool_compress import (
    FP8_E4M3_MAX,
    _normalized_hadamard,
    compress_pool,
    expand_pools_and_append_tail,
    fwht128_quant_fp8,
    kpool_compress_k,
    shared_pool_prefix,
)

HEAD_DIM = 128


def _sylvester_hadamard(dim: int) -> np.ndarray:
    """Build a Hadamard matrix by doubling, independently of SciPy."""
    matrix = np.ones((1, 1))
    while matrix.shape[0] < dim:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


def _reference_expand(
    pool_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Expand pool ids column by column, the way the fused kernel does."""
    rows, num_groups = pool_ids.shape
    topk = num_groups * pool_size

    expanded = []
    for row in range(rows):
        seq_len = int(seq_lens[row])
        offered = seq_len // pool_size
        tail_start = offered * pool_size
        tail_count = seq_len - tail_start

        columns = []
        for column in range(topk):
            pool_id = int(pool_ids[row, column // pool_size])
            # A pool the row was never offered is padding whatever its value.
            in_range = 0 <= pool_id < offered
            columns.append(pool_id * pool_size + column % pool_size if in_range else -1)
        for offset in range(pool_size - 1):
            columns.append(tail_start + offset if offset < tail_count else -1)
        expanded.append(columns)

    return torch.tensor(expanded, dtype=torch.int32).reshape(rows, topk + pool_size - 1)


def _pool_inputs(n_pools: int, pool_size: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    slot_k = torch.randn(n_pools, pool_size, HEAD_DIM, generator=generator).to(torch.bfloat16)
    slot_score = torch.randn(n_pools, pool_size, HEAD_DIM, generator=generator).to(torch.bfloat16)
    ape = torch.randn(pool_size, HEAD_DIM, generator=generator)
    return slot_k, slot_score, ape


def test_pooling_mixes_slots_independently_per_dimension() -> None:
    """The softmax runs over a pool's slots, separately for each dimension.

    Each dimension is given one clearly winning slot, and a different one, so a
    softmax taken over the wrong axis cannot reproduce the result.
    """
    pool_size = 4
    slot_k = torch.arange(pool_size * HEAD_DIM, dtype=torch.float32)
    slot_k = slot_k.reshape(1, pool_size, HEAD_DIM).to(torch.bfloat16)

    winner = torch.arange(HEAD_DIM) % pool_size
    slot_score = torch.full((1, pool_size, HEAD_DIM), -30.0)
    slot_score[0, winner, torch.arange(HEAD_DIM)] = 30.0

    pooled = compress_pool(slot_k, slot_score.to(torch.bfloat16), torch.zeros(pool_size, HEAD_DIM))

    expected = slot_k[0, winner, torch.arange(HEAD_DIM)]
    torch.testing.assert_close(pooled[0].float(), expected.float(), rtol=1e-2, atol=1e-2)


def test_position_bias_can_override_the_gate_score() -> None:
    """``ape`` is added to the gate score, so it moves the mixture too."""
    pool_size = 4
    slot_k, slot_score, _ = _pool_inputs(n_pools=2, pool_size=pool_size)

    ape = torch.full((pool_size, HEAD_DIM), -60.0)
    ape[2] = 60.0

    pooled = compress_pool(slot_k, slot_score, ape)

    torch.testing.assert_close(pooled.float(), slot_k[:, 2].float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("pool_size", [2, 4, 16])
def test_pooling_is_a_weighted_average_of_the_pooled_tokens(pool_size: int) -> None:
    """Softmax weights sum to one, so no dimension can leave the input range."""
    slot_k, slot_score, ape = _pool_inputs(n_pools=5, pool_size=pool_size)

    pooled = compress_pool(slot_k, slot_score, ape).float()

    lower = slot_k.float().amin(dim=1)
    upper = slot_k.float().amax(dim=1)
    # The bounds are exact in fp32; bfloat16 rounding of the result can step
    # just past them, by at most one bfloat16 ulp of the bound itself.
    slack = 2**-8 * upper.abs().maximum(lower.abs())
    assert torch.all(pooled >= lower - slack)
    assert torch.all(pooled <= upper + slack)


def test_hadamard_matrix_matches_the_sylvester_construction() -> None:
    """Entries are +-1/sqrt(128), so this is exact in fp32."""
    matrix = _normalized_hadamard(HEAD_DIM, torch.device("cpu"), torch.float32)

    expected = torch.from_numpy(_sylvester_hadamard(HEAD_DIM)).float() * HEAD_DIM**-0.5

    torch.testing.assert_close(matrix, expected, rtol=0, atol=0)


def test_quantization_scale_is_a_power_of_two() -> None:
    """The index-K cache is ue8m0, so the scale carries no mantissa."""
    rows = torch.randn(7, HEAD_DIM).to(torch.bfloat16)

    _, scale = fwht128_quant_fp8(rows)

    exponents = torch.log2(scale)
    torch.testing.assert_close(exponents, exponents.round(), rtol=0, atol=0)


def test_compression_round_trips_within_fp8_precision() -> None:
    """Dequantizing recovers the rotated pooled key, up to one e4m3 step."""
    slot_k, slot_score, ape = _pool_inputs(n_pools=6, pool_size=4)

    k_fp8, scale = kpool_compress_k(slot_k, slot_score, ape)

    pooled = compress_pool(slot_k, slot_score, ape)
    rotated = (pooled.float() @ _normalized_hadamard(HEAD_DIM, torch.device("cpu"), torch.float32)).to(torch.bfloat16)
    # e4m3 keeps 3 mantissa bits, so a value sits within half a step -- 2**-4
    # relative -- of a representable one once the shared scale is removed.
    torch.testing.assert_close(
        k_fp8.float() * scale,
        rotated.float(),
        rtol=2**-4,
        atol=float(scale.max()) * 2**-4,
    )


def test_compression_reports_cache_ready_shapes() -> None:
    slot_k, slot_score, ape = _pool_inputs(n_pools=3, pool_size=16)

    k_fp8, scale = kpool_compress_k(slot_k, slot_score, ape)

    assert k_fp8.shape == (3, HEAD_DIM)
    assert k_fp8.dtype == torch.float8_e4m3fn
    assert scale.shape == (3, 1)
    assert scale.dtype == torch.float32
    assert torch.all(scale.isfinite())


def test_compressing_no_pools_is_not_an_error() -> None:
    """A prefill batch can carry decode-only requests, which complete no pool."""
    pool_size = 4
    empty = torch.empty(0, pool_size, HEAD_DIM, dtype=torch.bfloat16)

    k_fp8, scale = kpool_compress_k(empty, empty, torch.zeros(pool_size, HEAD_DIM))

    assert k_fp8.shape == (0, HEAD_DIM)
    assert scale.shape == (0, 1)


def test_all_but_the_lowest_scales_stay_representable() -> None:
    """A near-zero pooled key must not drive the scale to zero or infinity."""
    pool_size = 2
    tiny = torch.zeros(1, pool_size, HEAD_DIM, dtype=torch.bfloat16)

    k_fp8, scale = kpool_compress_k(tiny, tiny, torch.zeros(pool_size, HEAD_DIM))

    assert torch.all(scale > 0)
    assert torch.all(k_fp8.float().abs() <= FP8_E4M3_MAX)
    assert not math.isnan(float(scale[0]))


@pytest.mark.parametrize("pool_size", [1, 2, 4, 16])
@pytest.mark.parametrize("num_groups", [1, 3])
def test_expansion_matches_a_per_column_reference(pool_size: int, num_groups: int) -> None:
    rows = 5
    generator = torch.Generator().manual_seed(pool_size * 100 + num_groups)
    # -1 stands for a budget slot that top-k could not fill.
    pool_ids = torch.randint(-1, 64, (rows, num_groups), generator=generator, dtype=torch.int32)
    seq_lens = torch.randint(0, 4096, (rows,), generator=generator, dtype=torch.int32)

    expanded = expand_pools_and_append_tail(pool_ids, seq_lens, pool_size)

    torch.testing.assert_close(expanded, _reference_expand(pool_ids, seq_lens, pool_size))
    assert expanded.dtype == torch.int32
    assert expanded.shape == (rows, num_groups * pool_size + pool_size - 1)


def test_unfilled_budget_slots_expand_to_padding() -> None:
    """A short history leaves budget slots empty; they must not name token 0."""
    pool_size = 4
    pool_ids = torch.tensor([[3, -1, -1]], dtype=torch.int32)

    # seq_len 16, not 12: pool 3 holds tokens 12..15, so a row that knows only
    # 12 tokens was never offered it and the expansion is right to drop it.
    expanded = expand_pools_and_append_tail(pool_ids, torch.tensor([16]), pool_size)

    torch.testing.assert_close(expanded[0, :4], torch.tensor([12, 13, 14, 15], dtype=torch.int32))
    assert torch.all(expanded[0, 4:12] == -1)


@pytest.mark.parametrize("pad", [-1, 5, 7, 1_000_000])
def test_padding_past_the_offered_prefix_is_dropped_whatever_its_value(pad: int) -> None:
    """What top-k fills the unused budget with is not part of its contract.

    A short history leaves nearly the whole budget unfilled, and taking those
    slots at face value points the attention at tokens whose KV was never
    written. Only an out-of-range fill can be recognised: a fill of 0 is
    indistinguishable from a row that genuinely picked pool 0, and costs a
    double-counted pool rather than a read of uninitialised memory.
    """
    pool_size = 4
    # seq_len 22 offers pools 0..4. The row picked pool 3; the rest is fill.
    pool_ids = torch.tensor([[3, pad, pad]], dtype=torch.int64)

    expanded = expand_pools_and_append_tail(
        pool_ids,
        torch.tensor([22]),
        pool_size,
        selectable_pools=torch.tensor([5]),
        tail_width=3,
    )

    torch.testing.assert_close(expanded[0, :4], torch.tensor([12, 13, 14, 15], dtype=torch.int32))
    assert torch.all(expanded[0, 4:12] == -1)
    torch.testing.assert_close(expanded[0, 12:], torch.tensor([20, 21, -1], dtype=torch.int32))


def test_the_appended_tail_is_the_trailing_incomplete_pool() -> None:
    """``seq_len == 22`` with pools of 4 leaves tokens 20 and 21 uncompressed."""
    pool_size = 4
    pool_ids = torch.tensor([[0]], dtype=torch.int32)

    expanded = expand_pools_and_append_tail(pool_ids, torch.tensor([22]), pool_size)

    torch.testing.assert_close(expanded[0, 4:], torch.tensor([20, 21, -1], dtype=torch.int32))


def test_a_completed_pool_leaves_no_tail() -> None:
    """``seq_len`` divisible by the pool size means every token is compressed."""
    pool_size = 4
    pool_ids = torch.tensor([[0], [1]], dtype=torch.int32)

    expanded = expand_pools_and_append_tail(pool_ids, torch.tensor([16, 16]), pool_size)

    assert torch.all(expanded[:, 4:] == -1)


def test_token_granular_pools_need_no_tail_columns() -> None:
    pool_ids = torch.tensor([[5, 9]], dtype=torch.int32)

    expanded = expand_pools_and_append_tail(pool_ids, torch.tensor([32]), pool_size=1)

    torch.testing.assert_close(expanded, torch.tensor([[5, 9]], dtype=torch.int32))


def test_the_shared_prefix_is_what_every_query_row_can_select() -> None:
    """The earliest row in a request sets the boundary for the whole request."""
    # One plain decode row, then a four-row draft-verify batch. The verify
    # batch's earliest row is at seq_len 30, whose complete pools number 7.
    prefix = shared_pool_prefix(
        seq_lens=torch.tensor([21, 33]),
        query_lens=torch.tensor([1, 4]),
        pool_size=4,
    )
    torch.testing.assert_close(prefix, torch.tensor([5, 7], dtype=torch.int32))


def test_a_shared_prefix_still_covers_every_recent_token() -> None:
    """Whatever top-k misses near the end, the tail must name explicitly.

    With a shared boundary the uncovered run is longer than one pool, so this
    is the property that keeps a draft-verify batch from dropping tokens the
    rows before it produced.
    """
    pool_size, query_len = 4, 4
    seq_lens = torch.tensor([30, 31, 32, 33])
    prefix = shared_pool_prefix(seq_lens, torch.full_like(seq_lens, query_len), pool_size)
    # No pool selected, so the tail is the only thing covering the recent run.
    pool_ids = torch.full((4, 1), -1, dtype=torch.int32)

    expanded = expand_pools_and_append_tail(
        pool_ids,
        seq_lens,
        pool_size,
        selectable_pools=prefix,
        tail_width=pool_size - 1 + query_len - 1,
    )

    for row in range(4):
        covered = {int(v) for v in expanded[row] if v >= 0}
        # Everything from the shared boundary up to this row's own last token.
        assert covered == set(range(int(prefix[row]) * pool_size, int(seq_lens[row])))


@pytest.mark.parametrize("query_len", [1, 2, 4, 5])
def test_a_shared_prefix_never_names_a_future_token(query_len: int) -> None:
    generator = torch.Generator().manual_seed(query_len)
    pool_size = 4
    rows = 16
    seq_lens = torch.randint(1, 4096, (rows,), generator=generator, dtype=torch.int32)
    prefix = shared_pool_prefix(seq_lens, torch.full_like(seq_lens, query_len), pool_size)

    # The operator only ever returns pools inside the key length it was given,
    # padding the rest, so sample the way it would answer.
    drawn = torch.randint(0, 1 << 20, (rows, 8), generator=generator, dtype=torch.int64)
    selectable = prefix.to(torch.int64).unsqueeze(-1)
    pool_ids = torch.where(selectable > 0, drawn % selectable.clamp_min(1), -1).to(torch.int32)

    expanded = expand_pools_and_append_tail(
        pool_ids,
        seq_lens,
        pool_size,
        selectable_pools=prefix,
        tail_width=pool_size - 1 + query_len - 1,
    )

    # Every id names a token this row has already seen: the selected pools end
    # at the shared boundary and the tail stops at the row's own length.
    assert torch.all(expanded < seq_lens.unsqueeze(-1))
