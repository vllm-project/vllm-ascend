# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU L1 tests for YOCO mask/compact prefill truncation metadata."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v41.yoco import (
    YocoSeqSpec,
    apply_yoco_mask_to_window_lengths,
    build_yoco_truncation_metadata,
    clip_prefix_for_seq,
    compact_yoco_boundary,
    expand_swa_pivots,
    make_yoco_prefill_runtime,
    reference_visible_keys,
    resolve_yoco_consumer_start,
    yoco_enabled,
    yoco_swa_visibility,
)


def _mask_cases(keep_tokens: int, window: int, chunk: int) -> list[tuple[str, tuple[YocoSeqSpec, ...]]]:
    """Scene matrix mirroring DeepInfer examples/yoco L1 coverage."""
    long_n = 2 * chunk + 904
    exact = 2 * chunk + keep_tokens
    inside = 2 * chunk + keep_tokens + 60
    return [
        ("short_lt_keep", (YocoSeqSpec(64, 0, 64),)),
        ("eq_keep", (YocoSeqSpec(keep_tokens, 0, keep_tokens),)),
        ("gt_keep_single_chunk", (YocoSeqSpec(4 * window, 0, 4 * window),)),
        ("chunked_c0_all_clipped", (YocoSeqSpec(long_n, 0, chunk),)),
        ("chunked_c1_all_clipped", (YocoSeqSpec(long_n, chunk, chunk),)),
        ("chunked_c2_partial", (YocoSeqSpec(long_n, 2 * chunk, long_n - 2 * chunk),)),
        ("rect_at_chunk_start", (YocoSeqSpec(exact, 2 * chunk, keep_tokens),)),
        ("rect_inside_chunk", (YocoSeqSpec(inside, 2 * chunk, inside - 2 * chunk),)),
        ("rect_cuts_history", (YocoSeqSpec(2000, 1900, 100),)),
        (
            "mixed_batch",
            (
                YocoSeqSpec(64, 0, 64, state_id=0),
                YocoSeqSpec(long_n, 2 * chunk, long_n - 2 * chunk, state_id=1),
                YocoSeqSpec(4 * window, 0, 4 * window, state_id=2),
                YocoSeqSpec(long_n, chunk, chunk, state_id=3),
                YocoSeqSpec(keep_tokens, 0, keep_tokens, state_id=4),
            ),
        ),
    ]


def _rows_from_metadata(meta, seqs: tuple[YocoSeqSpec, ...], *, compact: bool):
    n_seq = len(seqs)
    window = meta.window_size
    pivots = meta.swa_index_pivots.tolist()
    if compact:
        cu = meta.cu_seqlens_q.tolist()
        positions = meta.positions.tolist()
        store = meta.slot_mapping_store_swa.tolist()
        seq_of_row = []
        for i in range(n_seq):
            seq_of_row += [i] * (cu[i + 1] - cu[i])
        sec2_starts = [n_seq * window + cu[i] for i in range(n_seq)]
        clip_prefixes = [seqs[i].q_len - (cu[i + 1] - cu[i]) for i in range(n_seq)]
    else:
        cu = [0]
        for spec in seqs:
            cu.append(cu[-1] + spec.q_len)
        positions = [spec.q_start + j for spec in seqs for j in range(spec.q_len)]
        store = [-1] * len(positions)  # mask mode does not rewrite store slots here
        seq_of_row = [i for i, spec in enumerate(seqs) for _ in range(spec.q_len)]
        sec2_starts = [n_seq * window + cu[i] for i in range(n_seq)]
        clip_prefixes = [0] * n_seq

    rows = []
    for r, piv in enumerate(pivots):
        i_seq = seq_of_row[r]
        spec = seqs[i_seq]
        keys = expand_swa_pivots(
            piv,
            sec1_start=i_seq * window,
            sec2_start=sec2_starts[i_seq],
            spec=spec,
            window=window,
            sec2_base_abs=spec.q_start + clip_prefixes[i_seq],
        )
        rows.append(
            {
                "seq": i_seq,
                "pos": positions[r],
                "keys": keys,
                "store_slot": store[r],
            }
        )
    return rows


@pytest.mark.parametrize("pos,q_start,j,window,rect_lo,expect_empty", [
    (10, 0, 10, 8, 0, False),
    (10, 0, 10, 8, 20, True),  # entirely before rectangle
    (50, 40, 10, 16, 45, False),
])
def test_yoco_swa_visibility_empty_outside_rect(pos, q_start, j, window, rect_lo, expect_empty):
    vis = yoco_swa_visibility(pos, q_start, j, window, rect_lo, 0)
    total = vis.sec1_len + vis.new_len
    if expect_empty:
        assert total == 0
    else:
        assert total > 0
        assert total <= window


def test_resolve_yoco_consumer_start():
    assert resolve_yoco_consumer_start(SimpleNamespace(num_hidden_layers=40)) is None
    assert not yoco_enabled(SimpleNamespace(num_hidden_layers=40, n_prefill_layer=40))
    assert resolve_yoco_consumer_start(SimpleNamespace(num_hidden_layers=40, n_prefill_layer=24)) == 24
    assert yoco_enabled(SimpleNamespace(num_hidden_layers=40, n_prefill_layer=24))


def test_make_yoco_prefill_runtime_off_only_until_wired():
    cfg = SimpleNamespace(num_hidden_layers=40, n_prefill_layer=24, sliding_window=128)
    assert make_yoco_prefill_runtime(cfg, SimpleNamespace(yoco_prefill_truncation_mode="off")) is None
    with pytest.raises(NotImplementedError, match="not executable yet"):
        make_yoco_prefill_runtime(
            cfg,
            SimpleNamespace(yoco_prefill_truncation_mode="mask", yoco_prefill_keep_tokens=140),
        )
    with pytest.raises(NotImplementedError, match="not executable yet"):
        make_yoco_prefill_runtime(
            SimpleNamespace(num_hidden_layers=40, n_prefill_layer=40),
            SimpleNamespace(yoco_prefill_truncation_mode="compact", yoco_prefill_keep_tokens=140),
        )


@pytest.mark.parametrize("keep_tokens,window,chunk", [(140, 128, 512)])
def test_mask_pivots_match_reference_visible_keys(keep_tokens, window, chunk):
    for name, seqs in _mask_cases(keep_tokens, window, chunk):
        meta = build_yoco_truncation_metadata(
            seqs,
            keep_tokens=keep_tokens,
            window_size=window,
            mode="mask",
            swa_block_size=window + 12,
            max_backtrack_tokens=12,
        )
        assert meta.query_indices is None
        assert meta.total_q == sum(s.q_len for s in seqs)
        rows = _rows_from_metadata(meta, seqs, compact=False)
        for row in rows:
            spec = seqs[row["seq"]]
            ref = reference_visible_keys(spec, row["pos"], keep_tokens, window)
            assert row["keys"] == ref, f"{name} seq{row['seq']} pos{row['pos']}"
            assert len(row["keys"]) <= keep_tokens


@pytest.mark.parametrize("keep_tokens,window,chunk", [(140, 128, 512)])
def test_compact_reduces_rows_and_matches_reference(keep_tokens, window, chunk):
    swa_block = window + 12
    for name, seqs in _mask_cases(keep_tokens, window, chunk):
        meta = build_yoco_truncation_metadata(
            seqs,
            keep_tokens=keep_tokens,
            window_size=window,
            mode="compact",
            swa_block_size=swa_block,
            sparse_compress_ratios=(1, 4),
            dense_compress_ratios=(1,),
            max_backtrack_tokens=12,
        )
        assert meta.query_indices is not None
        assert meta.total_q == sum(s.q_len - clip_prefix_for_seq(s, keep_tokens, compact=True) for s in seqs)
        assert meta.total_q <= sum(min(s.q_len, keep_tokens + 1) for s in seqs)  # placeholder may add 1
        rows = _rows_from_metadata(meta, seqs, compact=True)
        for row in rows:
            spec = seqs[row["seq"]]
            ref = reference_visible_keys(spec, row["pos"], keep_tokens, window)
            assert row["keys"] == ref, f"{name} seq{row['seq']} pos{row['pos']}"
            assert len(row["keys"]) <= window
            in_rect = row["pos"] >= spec.N - keep_tokens
            in_ring_tail = spec.q_len - (row["pos"] - spec.q_start) <= swa_block
            expect_slot = (
                spec.state_id * swa_block + row["pos"] % swa_block if in_ring_tail and in_rect else -1
            )
            assert row["store_slot"] == expect_slot, f"{name} pos{row['pos']}"
        assert meta.ks is not None and "1" in meta.ks and "4" in meta.ks
        assert meta.kv_ca_index_offset_len is not None and "1" in meta.kv_ca_index_offset_len


@pytest.mark.parametrize("keep_tokens,window,chunk", [(140, 128, 512)])
def test_mask_compact_equivalence_on_retained_suffix(keep_tokens, window, chunk):
    """On retained (seq, pos) rows, mask and compact SWA visibility must match."""
    for name, seqs in _mask_cases(keep_tokens, window, chunk):
        mask = build_yoco_truncation_metadata(
            seqs, keep_tokens=keep_tokens, window_size=window, mode="mask", max_backtrack_tokens=12
        )
        compact = build_yoco_truncation_metadata(
            seqs,
            keep_tokens=keep_tokens,
            window_size=window,
            mode="compact",
            swa_block_size=window + 12,
            max_backtrack_tokens=12,
        )
        mask_rows = {(r["seq"], r["pos"]): r["keys"] for r in _rows_from_metadata(mask, seqs, compact=False)}
        compact_rows = {(r["seq"], r["pos"]): r["keys"] for r in _rows_from_metadata(compact, seqs, compact=True)}
        # Compact rows are a subset; every compact row must match mask on that key.
        for key, keys in compact_rows.items():
            assert key in mask_rows, f"{name}: compact row {key} missing from mask"
            assert keys == mask_rows[key], f"{name}: mismatch at {key}"
        # Tokens outside the keep rectangle that still exist under mask must be empty.
        for (i_seq, pos), keys in mask_rows.items():
            if (i_seq, pos) not in compact_rows:
                # Mask keeps the full layout; clipped rows (if any under mask) shouldn't exist —
                # under mask clip_prefix is 0, so all full rows exist. Placeholder-only compact
                # rows that sit before the rectangle may still appear in compact_rows with empty keys.
                pass
            spec = seqs[i_seq]
            if pos < spec.N - keep_tokens:
                assert keys == [], f"{name}: mask row outside rect must be empty at pos={pos}"


def test_compact_yoco_boundary_gather_cp1():
    hidden = torch.arange(20, dtype=torch.float32).view(10, 2)
    qi = torch.tensor([7, 8, 9], dtype=torch.int32)
    out = compact_yoco_boundary(hidden, qi)
    assert out.tolist() == [[14, 15], [16, 17], [18, 19]]


def test_apply_yoco_mask_zeros_lengths_outside_keep_tail():
    # N=20, keep=8 → rect_lo=12. Early rows empty; retained rows clipped to pos-rect_lo+1.
    positions = torch.arange(20, dtype=torch.int32)
    lengths = torch.full((20, 1), 8, dtype=torch.int32)
    masked = apply_yoco_mask_to_window_lengths(
        positions,
        lengths,
        seq_lens=[20],
        query_start_loc=[0, 20],
        keep_tokens=8,
    )
    assert masked[:12].sum().item() == 0
    expect = torch.tensor([[min(8, p - 12 + 1)] for p in range(12, 20)], dtype=torch.int32)
    assert torch.equal(masked[12:], expect)
    assert not torch.equal(masked, lengths)  # clone, not in-place


def test_apply_yoco_mask_clips_retained_row_to_rect_lo():
    """Pivot math vs length helper must agree at the keep-tail boundary.

    N=1000, keep=140, W=128, pos=860 → visible keys = {860} (length 1), not raw W.
    """
    n, keep, window, pos = 1000, 140, 128, 860
    positions = torch.tensor([pos], dtype=torch.int32)
    lengths = torch.tensor([[window]], dtype=torch.int32)
    masked = apply_yoco_mask_to_window_lengths(
        positions,
        lengths,
        seq_lens=[n],
        query_start_loc=[0, 1],
        keep_tokens=keep,
    )
    ref = reference_visible_keys(YocoSeqSpec(n, 0, n), pos, keep, window)
    assert len(ref) == 1 and ref[0] == pos
    assert masked.item() == len(ref)


def test_build_rejects_keep_below_window_plus_backtrack():
    with pytest.raises(ValueError, match="keep_tokens"):
        build_yoco_truncation_metadata(
            [YocoSeqSpec(256, 0, 256)],
            keep_tokens=100,
            window_size=128,
            mode="mask",
            max_backtrack_tokens=12,
        )


def test_model_applies_compact_boundary():
    """Smoke: DeepseekV41Model._apply_yoco_compact_boundary gathers HC states."""
    from vllm_ascend.models.deepseek_v41.model import DeepseekV41Model
    from vllm_ascend.models.deepseek_v41.yoco import YocoPrefillRuntime, YocoTruncationMetadata

    qi = torch.tensor([2, 3], dtype=torch.int32)
    meta = YocoTruncationMetadata(
        mode="compact",
        keep_tokens=2,
        window_size=2,
        clip_prefixes=[2],
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        total_q=2,
        swa_index_pivots=torch.zeros((2, 4), dtype=torch.int32),
        query_indices=qi,
        positions=torch.tensor([2, 3], dtype=torch.int32),
        slot_mapping_store_swa=torch.tensor([-1, -1], dtype=torch.int32),
    )
    runtime = YocoPrefillRuntime(mode="compact", keep_tokens=2, consumer_start=1, metadata=meta)
    model = SimpleNamespace(yoco_prefill=runtime)
    hidden = torch.arange(16, dtype=torch.float32).view(4, 2, 2)
    pre_mix = torch.arange(8, dtype=torch.float32).view(4, 2)
    positions = torch.arange(4, dtype=torch.int64)
    input_ids = torch.arange(4, dtype=torch.int64) + 100
    h2, p2, pos2, ids2 = DeepseekV41Model._apply_yoco_compact_boundary(
        model, hidden, pre_mix, positions, input_ids
    )
    assert h2.shape[0] == 2
    assert torch.equal(h2, hidden[2:])
    assert torch.equal(p2, pre_mix[2:])
    assert torch.equal(pos2, positions[2:])
    assert torch.equal(ids2, input_ids[2:])
