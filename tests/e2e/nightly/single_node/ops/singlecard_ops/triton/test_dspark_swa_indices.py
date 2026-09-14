# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the fused Triton-Ascend DSpark SWA indices kernel.

Checks the kernel against the eager reference implementation copied from
vllm_ascend.attention.dsa_v1.build_dspark_swa_indices, including the
graph-mode specifics that the integration review called out:

* capacity grid with max_num_reqs > active num_reqs (pad rows must be
  explicitly reset to -1/0 rather than left stale);
* out-variant buffers whose extent exceeds the active rows;
* non-contiguous (column-sliced) block tables: the kernel takes the row
  pitch from ``stride(0)``, not ``shape[1]``;
* int32 metadata inputs (query_start_loc / seq_lens / block_table);
* the eager-vs-triton boundary on padded rows is asserted as an
  intentional difference, not a regression.
"""

import pytest
import torch

from vllm_ascend.ops.triton.dspark_swa_indices import (
    build_dspark_swa_indices_triton,
    dspark_swa_indices_supported,
)

DEVICE = "npu"


def eager_dspark_swa_indices(
    block_table,
    num_speculative_tokens,
    window_size,
    block_size,
    query_start_loc,
    seq_lens,
    num_decode_tokens=None,
    index_width=None,
):
    """Reference implementation (mirrors dsa_v1.build_dspark_swa_indices)."""

    if index_width is None:
        min_width = int(window_size) + int(num_speculative_tokens)
        index_width = ((min_width + 127) // 128) * 128

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    prefix_lens = seq_lens - query_lens
    start_pos = (prefix_lens - int(window_size)).clamp(min=0)
    visible_lens = seq_lens - start_pos

    cols = torch.arange(index_width, device=start_pos.device)
    col_mask = cols[None, :] < visible_lens[:, None]
    pos = start_pos[:, None] + cols[None, :]
    block_nums = pos // block_size
    safe_nums = block_nums.clamp(min=0, max=int(block_table.shape[1]) - 1)
    block_offsets = pos % block_size
    block_ids = torch.gather(block_table, 1, safe_nums)
    slot_ids = (block_ids * block_size + block_offsets).to(torch.int32)
    slot_ids = slot_ids.where(col_mask, torch.full_like(slot_ids, -1))

    if num_decode_tokens is None:
        num_decode_tokens = int(query_start_loc[-1].item())
    per_token_slots = torch.repeat_interleave(slot_ids, query_lens, dim=0, output_size=num_decode_tokens).unsqueeze(1)
    per_token_lens = torch.repeat_interleave(visible_lens, query_lens, dim=0, output_size=num_decode_tokens)
    return per_token_slots, per_token_lens


def _make_inputs(
    num_reqs,
    num_speculative_tokens,
    window_size,
    block_size,
    max_seq_len,
    max_num_blocks,
    seed=0,
):
    """Uniform-query DSpark decode batch: every request drafts spec+1 tokens."""

    g = torch.Generator().manual_seed(seed)
    num_query_per_req = num_speculative_tokens + 1
    query_lens = torch.full((num_reqs,), num_query_per_req, dtype=torch.int64)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int64)
    torch.cumsum(query_lens, 0, out=query_start_loc[1:])
    seq_lens = torch.randint(low=num_query_per_req, high=max_seq_len, size=(num_reqs,), dtype=torch.int64, generator=g)
    block_table = torch.randint(low=0, high=200_000, size=(num_reqs, max_num_blocks), dtype=torch.int64, generator=g)
    return query_start_loc, seq_lens, block_table


@pytest.mark.parametrize(
    (
        "num_reqs",
        "num_speculative_tokens",
        "window_size",
        "block_size",
        "max_seq_len",
        "max_num_blocks",
    ),
    [
        pytest.param(1, 4, 944, 128, 1200, 16, id="single-req"),
        pytest.param(8, 4, 944, 128, 1200, 16, id="shape-under-review"),
        pytest.param(64, 4, 944, 128, 4096, 64, id="r64"),
        pytest.param(8, 4, 944, 128, 600, 8, id="seq-len-under-window"),
        pytest.param(4, 4, 944, 4, 300, 128, id="block-size-4"),
        pytest.param(4, 4, 944, 64, 2000, 64, id="block-size-64"),
        pytest.param(8, 1, 512, 128, 800, 8, id="spec-1"),
        pytest.param(8, 7, 512, 128, 800, 8, id="spec-7"),
        pytest.param(5, 4, 944, 128, 5000, 64, id="odd-num-reqs"),
        pytest.param(32, 4, 944, 128, 100_000, 2048, id="long-context"),
        pytest.param(16, 4, 944, 128, 1200, 6, id="block-table-narrow"),
        # window_size 8192 -> index_width 8320: not divisible by the default
        # BLOCK_W=1024. Before the cdiv fix the tail programs read past the
        # qsl/seq_lens arrays (MTE "DDR address out of range" vector-core
        # fault); the tail column block must store with a lane mask.
        pytest.param(8, 3, 8192, 128, 16_384, 128, id="window-8k-unaligned-width"),
        pytest.param(8, 4, 512, 128, 1200, 16, id="window-544-unaligned-width"),
    ],
)
def test_triton_matches_eager(
    num_reqs,
    num_speculative_tokens,
    window_size,
    block_size,
    max_seq_len,
    max_num_blocks,
):
    """Active rows must match the eager chain bit-exactly on every fuzz shape."""

    if not dspark_swa_indices_supported(block_size, max_num_blocks):
        pytest.skip("triton fast path not supported in this environment")

    query_start_loc, seq_lens, block_table = _make_inputs(
        num_reqs, num_speculative_tokens, window_size, block_size, max_seq_len, max_num_blocks
    )
    query_start_loc = query_start_loc.to(DEVICE)
    seq_lens = seq_lens.to(DEVICE)
    block_table = block_table.to(DEVICE)

    ref_slots, ref_lens = eager_dspark_swa_indices(
        block_table,
        num_speculative_tokens,
        window_size,
        block_size,
        query_start_loc,
        seq_lens,
    )
    out_slots, out_lens = build_dspark_swa_indices_triton(
        block_table,
        num_speculative_tokens,
        window_size,
        block_size,
        query_start_loc,
        seq_lens,
    )

    torch.testing.assert_close(out_slots, ref_slots)
    torch.testing.assert_close(out_lens, ref_lens)


def test_capacity_grid_padding():
    """max_num_reqs > num_reqs: pad rows are reset to (-1, 0), not stale."""

    num_reqs, num_speculative_tokens = 4, 4
    window_size, block_size, max_seq_len, max_num_blocks = 944, 128, 2000, 32
    if not dspark_swa_indices_supported(block_size, max_num_blocks):
        pytest.skip("triton fast path not supported in this environment")

    query_start_loc, seq_lens, block_table = _make_inputs(
        num_reqs, num_speculative_tokens, window_size, block_size, max_seq_len, max_num_blocks
    )
    query_start_loc = query_start_loc.to(DEVICE)
    seq_lens = seq_lens.to(DEVICE)
    block_table = block_table.to(DEVICE)

    num_query_per_req = num_speculative_tokens + 1
    T_active = num_reqs * num_query_per_req
    R_alloc = 16  # capture-size bound, deliberately > num_reqs
    T_padded = R_alloc * num_query_per_req
    index_width = ((window_size + num_speculative_tokens + 127) // 128) * 128

    # Persistent buffers pre-filled with sentinel garbage: after the kernel
    # runs, every row in [T_active, T_padded) must read -1 / 0.
    slots_buf = torch.full((T_padded, 1, index_width), 12345, dtype=torch.int32, device=DEVICE)
    lens_buf = torch.full((T_padded,), 777, dtype=torch.int64, device=DEVICE)

    # qsl/seq_lens padded to the allocation extent: padded requests carry
    # q_len == 0 (clamped cumulative series) and seq_len == 0, mirroring the
    # upstream draft metadata builder.
    qsl_alloc = torch.zeros(R_alloc + 1, dtype=torch.int64, device=DEVICE)
    qsl_alloc[: num_reqs + 1] = query_start_loc
    qsl_alloc[num_reqs + 1 :] = T_active
    seq_lens_alloc = torch.zeros(R_alloc, dtype=torch.int64, device=DEVICE)
    seq_lens_alloc[:num_reqs] = seq_lens
    bt_alloc = torch.zeros((R_alloc, max_num_blocks), dtype=torch.int64, device=DEVICE)
    bt_alloc[:num_reqs] = block_table

    out_slots, out_lens = build_dspark_swa_indices_triton(
        bt_alloc,
        num_speculative_tokens,
        window_size,
        block_size,
        qsl_alloc,
        seq_lens_alloc,
        num_decode_tokens=T_active,
        index_width=index_width,
        indices_output=slots_buf,
        lens_output=lens_buf,
        max_num_reqs=R_alloc,
    )

    # Active extent matches the eager result bit-exactly.
    ref_slots, ref_lens = eager_dspark_swa_indices(
        block_table,
        num_speculative_tokens,
        window_size,
        block_size,
        query_start_loc,
        seq_lens,
        num_decode_tokens=T_active,
        index_width=index_width,
    )
    torch.testing.assert_close(out_slots, ref_slots)
    torch.testing.assert_close(out_lens, ref_lens)

    # Padded extent is explicitly reset (the intentional superset of eager).
    assert torch.equal(slots_buf[T_active:], torch.full_like(slots_buf[T_active:], -1)), (
        "padded slot rows must be -1, found stale values"
    )
    assert torch.equal(lens_buf[T_active:], torch.zeros_like(lens_buf[T_active:])), (
        "padded lens rows must be 0, found stale values"
    )


def test_num_decode_tokens_is_authoritative():
    """num_decode_tokens drives T_active without any D2H sync."""

    num_reqs, num_speculative_tokens = 3, 4
    window_size, block_size, max_seq_len, max_num_blocks = 944, 128, 2000, 32
    if not dspark_swa_indices_supported(block_size, max_num_blocks):
        pytest.skip("triton fast path not supported in this environment")

    query_start_loc, seq_lens, block_table = _make_inputs(
        num_reqs, num_speculative_tokens, window_size, block_size, max_seq_len, max_num_blocks
    )
    query_start_loc = query_start_loc.to(DEVICE)
    seq_lens = seq_lens.to(DEVICE)
    block_table = block_table.to(DEVICE)

    T_active = num_reqs * (num_speculative_tokens + 1)
    out_slots, out_lens = build_dspark_swa_indices_triton(
        block_table,
        num_speculative_tokens,
        window_size,
        block_size,
        query_start_loc,
        seq_lens,
        num_decode_tokens=T_active,
    )
    ref_slots, ref_lens = eager_dspark_swa_indices(
        block_table,
        num_speculative_tokens,
        window_size,
        block_size,
        query_start_loc,
        seq_lens,
        num_decode_tokens=T_active,
    )
    assert out_slots.shape[0] == T_active
    torch.testing.assert_close(out_slots, ref_slots)
    torch.testing.assert_close(out_lens, ref_lens)


def test_capacity_grid_exact_sized_inputs():
    """Capacity grid over exact-sized (non-padded) input tensors is safe.

    The production caller pads qsl/seq_lens/block_table to the allocation
    extent, but a capacity grid with max_num_reqs > R must not read past
    exact-sized tensors either: programs with r >= R early-out before
    touching qsl/seq_lens/block_table (previously an MTE out-of-range
    vector-core fault on real hardware).
    """

    num_reqs, num_speculative_tokens = 4, 4
    window_size, block_size, max_seq_len, max_num_blocks = 944, 128, 2000, 32
    if not dspark_swa_indices_supported(block_size, max_num_blocks):
        pytest.skip("triton fast path not supported in this environment")

    query_start_loc, seq_lens, block_table = _make_inputs(
        num_reqs, num_speculative_tokens, window_size, block_size, max_seq_len, max_num_blocks
    )
    query_start_loc = query_start_loc.to(DEVICE)
    seq_lens = seq_lens.to(DEVICE)
    block_table = block_table.to(DEVICE)

    num_query_per_req = num_speculative_tokens + 1
    T_active = num_reqs * num_query_per_req
    R_alloc = 16  # deliberately far beyond the exact-sized inputs
    T_padded = R_alloc * num_query_per_req
    index_width = ((window_size + num_speculative_tokens + 127) // 128) * 128

    slots_buf = torch.full((T_padded, 1, index_width), 12345, dtype=torch.int32, device=DEVICE)
    lens_buf = torch.full((T_padded,), 777, dtype=torch.int64, device=DEVICE)

    out_slots, out_lens = build_dspark_swa_indices_triton(
        block_table,  # exact-sized [num_reqs, max_num_blocks]
        num_speculative_tokens,
        window_size,
        block_size,
        query_start_loc,  # exact-sized [num_reqs + 1]
        seq_lens,  # exact-sized [num_reqs]
        num_decode_tokens=T_active,
        index_width=index_width,
        indices_output=slots_buf,
        lens_output=lens_buf,
        max_num_reqs=R_alloc,
    )

    ref_slots, ref_lens = eager_dspark_swa_indices(
        block_table,
        num_speculative_tokens,
        window_size,
        block_size,
        query_start_loc,
        seq_lens,
        num_decode_tokens=T_active,
        index_width=index_width,
    )
    torch.testing.assert_close(out_slots, ref_slots)
    torch.testing.assert_close(out_lens, ref_lens)
    assert torch.equal(slots_buf[T_active:], torch.full_like(slots_buf[T_active:], -1))
    assert torch.equal(lens_buf[T_active:], torch.zeros_like(lens_buf[T_active:]))


def test_active_sized_buffer_warns_and_disables_cleanup():
    """A buffer sliced to the active rows under graph intent must warn.

    The classic eager-style call site passes ``buffer[:T_active]`` as
    indices_output; under a capacity grid (max_num_reqs > R) that silences
    the pad-row cleanup, so the wrapper must flag it instead of failing
    silently. With no graph intent (max_num_reqs is None) the same slice is
    an ordinary allocation choice and must stay quiet.
    """

    num_reqs, num_speculative_tokens = 4, 4
    window_size, block_size, max_seq_len, max_num_blocks = 944, 128, 2000, 32
    if not dspark_swa_indices_supported(block_size, max_num_blocks):
        pytest.skip("triton fast path not supported in this environment")

    query_start_loc, seq_lens, block_table = _make_inputs(
        num_reqs, num_speculative_tokens, window_size, block_size, max_seq_len, max_num_blocks
    )
    query_start_loc = query_start_loc.to(DEVICE)
    seq_lens = seq_lens.to(DEVICE)
    block_table = block_table.to(DEVICE)

    num_query_per_req = num_speculative_tokens + 1
    T_active = num_reqs * num_query_per_req
    index_width = ((window_size + num_speculative_tokens + 127) // 128) * 128

    # Buffer cut to exactly the active rows (the eager-style slice).
    slots_slice = torch.zeros((T_active, 1, index_width), dtype=torch.int32, device=DEVICE)
    lens_slice = torch.zeros((T_active,), dtype=torch.int64, device=DEVICE)

    # Graph intent + active-sized slice -> UserWarning about disabled cleanup.
    with pytest.warns(UserWarning, match="pad-row cleanup is disabled"):
        build_dspark_swa_indices_triton(
            block_table,
            num_speculative_tokens,
            window_size,
            block_size,
            query_start_loc,
            seq_lens,
            num_decode_tokens=T_active,
            index_width=index_width,
            indices_output=slots_slice,
            lens_output=lens_slice,
            max_num_reqs=16,
        )

    # No graph intent + active-sized slice -> no warning.
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        build_dspark_swa_indices_triton(
            block_table,
            num_speculative_tokens,
            window_size,
            block_size,
            query_start_loc,
            seq_lens,
            num_decode_tokens=T_active,
            index_width=index_width,
            indices_output=slots_slice,
            lens_output=lens_slice,
        )


def test_anchor_sampling_num_query_per_req():
    """num_query_per_req=spec (anchor sampling) sizes the grid correctly.

    DSpark anchor-sampling mode drafts num_speculative_tokens queries per
    request (no +1). The capacity grid and pad cleanup must follow that
    factor when it is passed explicitly.
    """

    num_reqs, num_speculative_tokens = 4, 4
    window_size, block_size, max_seq_len, max_num_blocks = 944, 128, 2000, 32
    if not dspark_swa_indices_supported(block_size, max_num_blocks):
        pytest.skip("triton fast path not supported in this environment")

    # Anchor mode: 4 queries per request, NOT spec+1 == 5.
    nqp = num_speculative_tokens
    query_start_loc, seq_lens, block_table = _make_inputs(
        num_reqs, nqp - 1, window_size, block_size, max_seq_len, max_num_blocks
    )
    # Rebuild query_start_loc for nqp-per-request uniform drafting.
    query_lens = torch.full((num_reqs,), nqp, dtype=torch.int64)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int64)
    torch.cumsum(query_lens, 0, out=query_start_loc[1:])
    query_start_loc = query_start_loc.to(DEVICE)
    seq_lens = seq_lens.to(DEVICE)
    block_table = block_table.to(DEVICE)

    T_active = num_reqs * nqp
    R_alloc = 8
    T_padded = R_alloc * nqp
    index_width = ((window_size + num_speculative_tokens + 127) // 128) * 128

    slots_buf = torch.full((T_padded, 1, index_width), 12345, dtype=torch.int32, device=DEVICE)
    lens_buf = torch.full((T_padded,), 777, dtype=torch.int64, device=DEVICE)

    out_slots, out_lens = build_dspark_swa_indices_triton(
        block_table,
        num_speculative_tokens,
        window_size,
        block_size,
        query_start_loc,
        seq_lens,
        num_decode_tokens=T_active,
        index_width=index_width,
        indices_output=slots_buf,
        lens_output=lens_buf,
        max_num_reqs=R_alloc,
        num_query_per_req=nqp,
    )

    ref_slots, ref_lens = eager_dspark_swa_indices(
        block_table,
        num_speculative_tokens,
        window_size,
        block_size,
        query_start_loc,
        seq_lens,
        num_decode_tokens=T_active,
        index_width=index_width,
    )
    torch.testing.assert_close(out_slots, ref_slots)
    torch.testing.assert_close(out_lens, ref_lens)
    # Pad extent follows nqp == spec, not spec + 1.
    assert torch.equal(slots_buf[T_active:], torch.full_like(slots_buf[T_active:], -1))
    assert torch.equal(lens_buf[T_active:], torch.zeros_like(lens_buf[T_active:]))


def test_capacity_grid_exact_allocation_boundary():
    """num_reqs == R_alloc: every grid program owns an active request.

    The capacity grid is sized by R_alloc, so num_reqs == R_alloc is the
    boundary where no pad programs exist beyond the active extent: the
    R-guard (r >= R) never trips, yet T_padded == T_active means the pad
    cleanup loop body is empty. This exercises the eligibility boundary the
    wiring layer relies on (num_reqs <= max_num_reqs_for_dspark() passing
    with equality) and pins that a full-capacity batch is bit-exact against
    eager with zero stale rows.
    """

    num_reqs, num_speculative_tokens = 8, 4
    window_size, block_size, max_seq_len, max_num_blocks = 944, 128, 2000, 32
    if not dspark_swa_indices_supported(block_size, max_num_blocks):
        pytest.skip("triton fast path not supported in this environment")

    query_start_loc, seq_lens, block_table = _make_inputs(
        num_reqs, num_speculative_tokens, window_size, block_size, max_seq_len, max_num_blocks
    )
    query_start_loc = query_start_loc.to(DEVICE)
    seq_lens = seq_lens.to(DEVICE)
    block_table = block_table.to(DEVICE)

    num_query_per_req = num_speculative_tokens + 1
    T_active = num_reqs * num_query_per_req
    R_alloc = num_reqs  # boundary: allocation capacity exactly equals demand
    T_padded = R_alloc * num_query_per_req
    index_width = ((window_size + num_speculative_tokens + 127) // 128) * 128

    slots_buf = torch.full((T_padded, 1, index_width), 12345, dtype=torch.int32, device=DEVICE)
    lens_buf = torch.full((T_padded,), 777, dtype=torch.int64, device=DEVICE)

    out_slots, out_lens = build_dspark_swa_indices_triton(
        block_table,
        num_speculative_tokens,
        window_size,
        block_size,
        query_start_loc,
        seq_lens,
        num_decode_tokens=T_active,
        index_width=index_width,
        indices_output=slots_buf,
        lens_output=lens_buf,
        max_num_reqs=R_alloc,
    )

    ref_slots, ref_lens = eager_dspark_swa_indices(
        block_table,
        num_speculative_tokens,
        window_size,
        block_size,
        query_start_loc,
        seq_lens,
        num_decode_tokens=T_active,
        index_width=index_width,
    )
    torch.testing.assert_close(out_slots, ref_slots)
    torch.testing.assert_close(out_lens, ref_lens)
    # T_padded == T_active: the whole buffer is the active extent and every
    # row was rewritten (no sentinel survives anywhere).
    assert not torch.equal(slots_buf, torch.full_like(slots_buf, 12345))
    assert torch.equal(out_slots.view(T_active, index_width), slots_buf.view(T_active, index_width))


def test_non_contiguous_block_table():
    """A column-sliced block table must resolve rows via stride(0), not B.

    Mirrors the production caller, which passes
    ``block_table_tensor[:num_reqs]`` — a row slice of the padded
    allocation whose stride(0) can differ from shape[1] when the
    underlying tensor was allocated wider. The kernel must honor the
    actual row pitch instead of assuming contiguous rows.
    """

    num_reqs, num_speculative_tokens = 8, 4
    window_size, block_size, max_seq_len, max_num_blocks = 944, 128, 2000, 32
    if not dspark_swa_indices_supported(block_size, max_num_blocks):
        pytest.skip("triton fast path not supported in this environment")

    query_start_loc, seq_lens, block_table = _make_inputs(
        num_reqs, num_speculative_tokens, window_size, block_size, max_seq_len, max_num_blocks
    )
    query_start_loc = query_start_loc.to(DEVICE)
    seq_lens = seq_lens.to(DEVICE)
    # Allocate a table wider than the visible columns and hand the kernel a
    # column slice: row pitch (32 + 16) != shape[1] (32), so a kernel that
    # indexes ``bt_ptr + r * B`` would read the wrong rows.
    wide_table = torch.zeros((num_reqs, max_num_blocks + 16), dtype=block_table.dtype, device=DEVICE)
    wide_table[:, :max_num_blocks] = block_table
    sliced_table = wide_table[:, :max_num_blocks]
    assert sliced_table.stride(0) == max_num_blocks + 16 != sliced_table.shape[1]

    ref_slots, ref_lens = eager_dspark_swa_indices(
        sliced_table,
        num_speculative_tokens,
        window_size,
        block_size,
        query_start_loc,
        seq_lens,
    )
    out_slots, out_lens = build_dspark_swa_indices_triton(
        sliced_table,
        num_speculative_tokens,
        window_size,
        block_size,
        query_start_loc,
        seq_lens,
    )
    torch.testing.assert_close(out_slots, ref_slots)
    torch.testing.assert_close(out_lens, ref_lens)


def test_int32_metadata_inputs():
    """int32 query_start_loc / seq_lens / block_table must work end-to-end.

    The kernel reads qsl/seq_lens/block_table through pointer casts, so the
    JIT must specialize on whatever dtype the runtime metadata carries
    (the warmup path compiles against the real tensors' dtypes).
    """

    num_reqs, num_speculative_tokens = 8, 4
    window_size, block_size, max_seq_len, max_num_blocks = 944, 128, 2000, 32
    if not dspark_swa_indices_supported(block_size, max_num_blocks):
        pytest.skip("triton fast path not supported in this environment")

    query_start_loc, seq_lens, block_table = _make_inputs(
        num_reqs, num_speculative_tokens, window_size, block_size, max_seq_len, max_num_blocks
    )
    query_start_loc = query_start_loc.to(torch.int32).to(DEVICE)
    seq_lens = seq_lens.to(torch.int32).to(DEVICE)
    block_table = block_table.to(torch.int32).to(DEVICE)

    ref_slots, ref_lens = eager_dspark_swa_indices(
        block_table,
        num_speculative_tokens,
        window_size,
        block_size,
        query_start_loc,
        seq_lens,
    )
    out_slots, out_lens = build_dspark_swa_indices_triton(
        block_table,
        num_speculative_tokens,
        window_size,
        block_size,
        query_start_loc,
        seq_lens,
    )
    torch.testing.assert_close(out_slots, ref_slots)
    # The eager chain's lens dtype follows seq_lens (int32 here), while the
    # triton wrapper always allocates int64 lens — a documented contract.
    # Compare values through the canonical int64 dtype.
    torch.testing.assert_close(out_lens, ref_lens.to(torch.int64))


@pytest.mark.parametrize("bad_block_size", [100, 96, 0])
def test_gate_rejects_non_pow2_block_size(bad_block_size):
    """The host-side gate must reject non-power-of-two block sizes."""

    assert not dspark_swa_indices_supported(bad_block_size, 16)


def test_gate_accepts_valid_config():
    assert dspark_swa_indices_supported(128, 64)
