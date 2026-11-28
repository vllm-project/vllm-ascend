# SPDX-License-Identifier: Apache-2.0
"""AscendC Host-Sim oracle tests for all 4 HiSparse prefill C++ operators."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.ut.distributed.kv_transfer.hisparse_prefill.conftest import load_prefill_host_sim
from vllm_ascend.hisparse.prefill_pipeline import (
    BLOCK_SIZE,
    MAX_BITMAP_WORDS,
    MAX_SUPER_BLOCKS,
    MODE_DEMAND,
    MODE_FULL,
    TOPK_WIDTH,
    _to_signed_int64,
)

ROW_BYTES = 656
MAX_HISTORY = 1_048_576


def _oracle_union_plan(
    topk_np: np.ndarray,
    histories: list[int],
    starts: list[int],
    queries: list[int],
    capacity: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    union_list = []
    status = np.zeros((len(histories), 8), dtype=np.int32)
    for idx, (h, st, q) in enumerate(zip(histories, starts, queries)):
        sub = topk_np[st : st + q]
        invalid = bool(np.any((sub < -1) | (sub >= h + q)))
        hist_vals = sub[(sub >= 0) & (sub < h)]
        uniq = np.unique(hist_vals).astype(np.int32)
        ovf = 1 if len(uniq) > capacity else 0
        err = 1 if invalid else 0
        union_list.append(uniq if (not ovf and not err) else np.empty((0,), dtype=np.int32))
        status[idx, 0] = len(uniq)
        status[idx, 1] = q
        status[idx, 2] = len(uniq) + q
        status[idx, 3] = err
        status[idx, 4] = ovf
    return union_list, status


def _oracle_remap_topk(
    topk_np: np.ndarray,
    desc_rows: list[tuple[int, int, int, int, int, int, int, int]],
    page_table: np.ndarray,
    union_lists: dict[int, np.ndarray],
    topk_row_offset: int,
) -> np.ndarray:
    total_q = sum(row[2] for row in desc_rows)
    out = np.full((total_q, TOPK_WIDTH), -1, dtype=np.int32)
    for mode, h, q, q_in_start, q_out_start, slot_idx, dst_hist_base, aux in desc_rows:
        union_rank_map = (
            {int(tok): rank for rank, tok in enumerate(union_lists[slot_idx])} if mode == MODE_DEMAND else None
        )
        for ql in range(q):
            causal = h + ql
            row_in = topk_np[topk_row_offset + q_in_start + ql]
            mapped = []
            for tok in row_in:
                itok = int(tok)
                if 0 <= itok <= causal and itok < h + q:
                    if mode == MODE_FULL:
                        dst_pg = int(page_table[slot_idx, itok // BLOCK_SIZE])
                        mapped.append(dst_pg * BLOCK_SIZE + (itok % BLOCK_SIZE))
                    elif itok < h:
                        assert union_rank_map is not None
                        mapped.append(dst_hist_base + union_rank_map[itok])
                    else:
                        mapped.append(aux + (itok - h))
            if mapped:
                out[q_out_start + ql, : len(mapped)] = np.asarray(mapped, dtype=np.int32)
    return out


def _make_union_buffers(max_reqs: int, cap: int):
    part_bits = torch.empty((MAX_HISTORY,), dtype=torch.int32)
    status_words = torch.empty((64 * 8,), dtype=torch.int32)
    union_ids = torch.full((max_reqs, cap), -7, dtype=torch.int32)
    union_status = torch.full((max_reqs, 8), -7, dtype=torch.int32)
    bitmap_buf = torch.empty((max_reqs, MAX_BITMAP_WORDS), dtype=torch.int32)
    super_prefix_buf = torch.empty((max_reqs, MAX_SUPER_BLOCKS), dtype=torch.int32)
    return [part_bits, status_words], [union_ids, union_status, bitmap_buf, super_prefix_buf]


@pytest.mark.parametrize(
    ("history", "query"),
    [
        (131_072, 256),
        (524_288, 1024),
        (1_048_576, 64),  # exact 1M boundary!
    ],
)
def test_host_sim_union_plan_and_remap_scales(history: int, query: int):
    ops, lib = load_prefill_host_sim()
    rng = np.random.default_rng(1000 + history + query)
    topk = torch.full((query, TOPK_WIDTH), -1, dtype=torch.int32)
    for ql in range(query):
        # Mix non-8-aligned valid counts across [1, 3, 7, 9, 15, ..., 2047]
        valid_hist_n = [1, 3, 7, 9, 15, 63, 255, 1999][ql % 8]
        picks = rng.integers(0, history, size=valid_hist_n, dtype=np.int32)
        topk[ql, :valid_hist_n] = torch.from_numpy(picks)
        # Add query-self tokens (causal valid and future causal-masked)
        if valid_hist_n + 2 <= TOPK_WIDTH:
            topk[ql, valid_hist_n] = history + ql
            if ql + 1 < query:
                topk[ql, valid_hist_n + 1] = history + ql + 1  # masked out by causal guard

    cap = min(history, query * TOPK_WIDTH)
    scratch, outputs = _make_union_buffers(1, cap)
    before_traps = lib.selected_kv_sim_trap_count()
    ops.hisparse_prefill_union_plan(topk, scratch, outputs, [history], [0], [query], cap)
    assert lib.selected_kv_sim_trap_count() == before_traps

    oracle_u, oracle_st = _oracle_union_plan(topk.numpy(), [history], [0], [query], cap)
    cnt = int(outputs[1][0, 0])
    assert outputs[1][0, :5].numpy().tolist() == oracle_st[0, :5].tolist()
    np.testing.assert_array_equal(outputs[0][0, :cnt].numpy(), oracle_u[0])

    q_dst_base = 16 * BLOCK_SIZE
    u_dst_base = q_dst_base + query
    staging_rows = u_dst_base + cnt + 128
    desc = torch.tensor([[MODE_DEMAND, history, query, 0, 0, 0, u_dst_base, q_dst_base]], dtype=torch.int32)
    mapped_out = torch.full((query, TOPK_WIDTH), -99, dtype=torch.int32)
    page_table = torch.zeros((1, 1), dtype=torch.int32)
    ops.hisparse_prefill_remap_topk(topk, desc, outputs[2], outputs[3], page_table, mapped_out, 0, staging_rows)
    assert lib.selected_kv_sim_trap_count() == before_traps
    oracle_mapped = _oracle_remap_topk(topk.numpy(), [tuple(desc[0].tolist())], page_table.numpy(), {0: oracle_u[0]}, 0)
    np.testing.assert_array_equal(mapped_out.numpy(), oracle_mapped)


def test_host_sim_non_8_aligned_gathermask_tail_cleanup():
    ops, lib = load_prefill_host_sim()
    history = 131_072
    valid_counts = [0, 1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 33, 127, 2047]
    query = len(valid_counts)
    topk = torch.full((query, TOPK_WIDTH), -1, dtype=torch.int32)
    for ql, vc in enumerate(valid_counts):
        if vc > 0:
            # Put valid tokens at scattered non-contiguous columns to exercise GatherMask
            cols = np.linspace(0, TOPK_WIDTH - 1, num=vc, dtype=np.int32)
            vals = np.linspace(0, history - 1, num=vc, dtype=np.int32)
            topk[ql, cols] = torch.from_numpy(vals)

    scratch, outputs = _make_union_buffers(1, 32_768)
    ops.hisparse_prefill_union_plan(topk, scratch, outputs, [history], [0], [query], 32_768)
    oracle_u, _ = _oracle_union_plan(topk.numpy(), [history], [0], [query], 32_768)

    desc = torch.tensor([[MODE_DEMAND, history, query, 0, 0, 0, 512, 256]], dtype=torch.int32)
    # Pre-fill mapped_out with garbage to ensure trailing -1 padding overwrites everything
    mapped_out = torch.full((query, TOPK_WIDTH), 123456, dtype=torch.int32)
    page_table = torch.zeros((1, 1), dtype=torch.int32)
    before_traps = lib.selected_kv_sim_trap_count()
    ops.hisparse_prefill_remap_topk(topk, desc, outputs[2], outputs[3], page_table, mapped_out, 0, 100_000)
    assert lib.selected_kv_sim_trap_count() == before_traps
    oracle_mapped = _oracle_remap_topk(topk.numpy(), [tuple(desc[0].tolist())], page_table.numpy(), {0: oracle_u[0]}, 0)
    np.testing.assert_array_equal(mapped_out.numpy(), oracle_mapped)


def test_host_sim_strided_resident_host_dva_and_heterogeneous_batch():
    ops, lib = load_prefill_host_sim()
    rng = np.random.default_rng(20261127)

    # Non-contiguous strided resident_cache: physical HMA block stride = (128 + 32) rows * 656 B
    res_num_blocks = 24
    stride_rows = BLOCK_SIZE + 32
    res_backing = torch.from_numpy(rng.integers(0, 256, size=(res_num_blocks, stride_rows, ROW_BYTES), dtype=np.uint8))
    resident_cache = res_backing[:, :BLOCK_SIZE, :]
    assert not resident_cache.is_contiguous()
    assert resident_cache.stride(0) == stride_rows * ROW_BYTES

    # Host pool for DVA reads (block 0 unused null, blocks 1..15 populated with unique bytes)
    host_num_blocks = 16
    host_pool = torch.from_numpy(rng.integers(0, 256, size=(host_num_blocks, BLOCK_SIZE, ROW_BYTES), dtype=np.uint8))
    host_dva = _to_signed_int64(int(host_pool.data_ptr()))
    host_bytes = int(host_pool.numel() * host_pool.element_size())

    # Heterogeneous batch of 3 requests:
    #   req 0: MODE_FULL (Tier 3, H=256 (2 host pages: pg0 resident=3, pg1 host=5), Q=10)
    #   req 1: MODE_DEMAND (Tier 1/2 style, H=512 (4 pages: pg0 res=7, pg1 host=9, pg2 res=11, pg3 host=13), Q=6)
    #   req 2: Tier-0 ON_CARD (MODE_FULL, H=256 (2 resident pages: pg0 res=14, pg1 res=15), Q=8)
    h0, q0 = 256, 10
    h1, q1 = 512, 6
    h2, q2 = 256, 8
    total_q = q0 + q1 + q2
    topk = torch.full((total_q, TOPK_WIDTH), -1, dtype=torch.int32)

    # Populate valid causal TopK indices for all 3 requests
    for ql in range(q0):
        topk[ql, :5] = torch.tensor([0, 127, 128, 255, h0 + ql], dtype=torch.int32)
    for ql in range(q1):
        # Touch all 4 pages of req 1 (mix of resident and host-only pages)
        topk[q0 + ql, :6] = torch.tensor([3, 131, 260, 400, 511, h1 + ql], dtype=torch.int32)
    for ql in range(q2):
        topk[q0 + q1 + ql, :4] = torch.tensor([10, 140, 250, h2 + ql], dtype=torch.int32)

    # 1. Run union_plan for the single ON_DEMAND request (req 1)
    cap = 2048
    scratch, outputs = _make_union_buffers(1, cap)
    before_traps = lib.selected_kv_sim_trap_count()
    ops.hisparse_prefill_union_plan(topk, scratch, outputs, [h1], [q0], [q1], cap)
    assert lib.selected_kv_sim_trap_count() == before_traps
    oracle_u, _ = _oracle_union_plan(topk.numpy(), [h1], [q0], [q1], cap)
    u_cnt = int(outputs[1][0, 0])
    assert u_cnt == len(oracle_u[0]) == 5

    # 2. Layout staging_pool:
    # Zone A (FULL): req0 has 3 pages (0..2), req2 has 3 pages (3..5) -> 6 pages = 768 rows
    # Zone B (DEMAND Q): q1=6 rows at [768..774)
    # Zone C (DEMAND union): u_cnt=5 rows at [774..779)
    staging_pages = 8
    staging_pool = torch.zeros((staging_pages, BLOCK_SIZE, ROW_BYTES), dtype=torch.uint8)
    page_table = torch.tensor(
        [
            [0, 1, 2],  # req 0 (slot 0)
            [3, 4, 5],  # req 2 (slot 1)
        ],
        dtype=torch.int32,
    )
    q1_dst = 6 * BLOCK_SIZE
    u1_dst = q1_dst + q1

    # 3. Copy D2D resident spans + tail/demand-Q spans via hisparse_prefill_copy_spans
    state_indices = torch.tensor([0, 1, 2], dtype=torch.int32)
    source_bt = torch.tensor(
        [
            [0, 5, 0, 0],
            [0, 9, 0, 13],
            [0, 0, 0, 0],
        ],
        dtype=torch.int32,
    )
    resident_bt = torch.tensor(
        [
            [3, 0, 4, 0],  # req 0: pg0=res 3, pg1=host(5), pg2(tail)=res 4
            [7, 0, 11, 0],  # req 1: pg0=res 7, pg1=host(9), pg2=res 11, pg3=host(13), pg4(Q)=res 12
            [14, 15, 16, 0],  # req 2: all resident (14, 15, tail=16)
        ],
        dtype=torch.int32,
    )
    # Fix req 1 page 4 (active Q rows [512..518)) in resident_bt:
    resident_bt_5 = torch.zeros((3, 5), dtype=torch.int32)
    resident_bt_5[:, :4] = resident_bt
    resident_bt_5[1, 4] = 12

    # Simulate host->staging SDMA for req0 pg1 (host_blk=5 -> dst_page=1)
    staging_pool[1].copy_(host_pool[5])

    spans = torch.tensor(
        [
            [0, 0, 0, BLOCK_SIZE, 0 * BLOCK_SIZE, 0, 0, 0],  # req0 pg0 (res 3 -> page 0)
            [0, 2, 0, q0, 2 * BLOCK_SIZE, 0, 0, 0],  # req0 tail (res 4 -> page 2)
            [2, 0, 0, BLOCK_SIZE, 3 * BLOCK_SIZE, 0, 0, 0],  # req2 pg0 (res 14 -> page 3)
            [2, 1, 0, BLOCK_SIZE, 4 * BLOCK_SIZE, 0, 0, 0],  # req2 pg1 (res 15 -> page 4)
            [2, 2, 0, q2, 5 * BLOCK_SIZE, 0, 0, 0],  # req2 tail (res 16 -> page 5)
            [1, 4, 0, q1, q1_dst, 0, 0, 0],  # req1 new Q (res 12 -> q1_dst)
        ],
        dtype=torch.int32,
    )
    ops.hisparse_prefill_copy_spans(spans, spans.shape[0], state_indices, resident_bt_5, resident_cache, staging_pool)
    assert lib.selected_kv_sim_trap_count() == before_traps

    # 4. Gather union rows for req 1 (mix of strided resident_cache & Host DVA)
    req_meta = torch.tensor([[1, h1, u1_dst, 0]], dtype=torch.int32)
    ops.hisparse_prefill_gather_union(
        outputs[0],
        outputs[1],
        req_meta,
        state_indices,
        source_bt,
        resident_bt_5,
        resident_cache,
        staging_pool,
        host_dva,
        host_bytes,
    )
    assert lib.selected_kv_sim_trap_count() == before_traps

    # 5. Heterogeneous remap_topk (FULL req0 + DEMAND req1 + Tier-0 FULL req2)
    desc = torch.tensor(
        [
            [MODE_FULL, h0, q0, 0, 0, 0, 0, 3],
            [MODE_DEMAND, h1, q1, q0, q0, 0, u1_dst, q1_dst],
            [MODE_FULL, h2, q2, q0 + q1, q0 + q1, 1, 0, 3],
        ],
        dtype=torch.int32,
    )
    mapped_out = torch.full((total_q, TOPK_WIDTH), -9, dtype=torch.int32)
    ops.hisparse_prefill_remap_topk(
        topk, desc, outputs[2], outputs[3], page_table, mapped_out, 0, staging_pages * BLOCK_SIZE
    )
    assert lib.selected_kv_sim_trap_count() == before_traps

    # Verify byte-for-byte staging_pool & mapped_out against Python oracle
    flat_staging = staging_pool.view(-1, ROW_BYTES).numpy()
    np.testing.assert_array_equal(flat_staging[0:128], resident_cache[3].numpy())
    np.testing.assert_array_equal(flat_staging[128:256], host_pool[5].numpy())
    np.testing.assert_array_equal(flat_staging[256 : 256 + q0], resident_cache[4, :q0].numpy())
    np.testing.assert_array_equal(flat_staging[384:512], resident_cache[14].numpy())
    np.testing.assert_array_equal(flat_staging[512:640], resident_cache[15].numpy())
    np.testing.assert_array_equal(flat_staging[640 : 640 + q2], resident_cache[16, :q2].numpy())
    np.testing.assert_array_equal(flat_staging[q1_dst : q1_dst + q1], resident_cache[12, :q1].numpy())

    for rank, tok in enumerate(oracle_u[0].tolist()):
        pg_col, row_off = divmod(tok, BLOCK_SIZE)
        res_blk = int(resident_bt_5[1, pg_col])
        if res_blk > 0:
            expected_row = resident_cache[res_blk, row_off].numpy()
        else:
            host_blk = int(source_bt[1, pg_col])
            expected_row = host_pool[host_blk, row_off].numpy()
        np.testing.assert_array_equal(flat_staging[u1_dst + rank], expected_row)

    oracle_mapped = _oracle_remap_topk(
        topk.numpy(), [tuple(row) for row in desc.tolist()], page_table.numpy(), {0: oracle_u[0]}, 0
    )
    np.testing.assert_array_equal(mapped_out.numpy(), oracle_mapped)


def test_host_sim_mode_full_above_1m_pages():
    ops, lib = load_prefill_host_sim()
    history = 1_500_000
    query = 128
    num_pages = (history + query + BLOCK_SIZE - 1) // BLOCK_SIZE
    assert num_pages > 8192  # Exercises expanded MODE_FULL page table limit (up to 32,768 pages)

    rng = np.random.default_rng(1500000)
    topk = torch.full((query, TOPK_WIDTH), -1, dtype=torch.int32)
    for ql in range(query):
        picks = np.array(
            [0, 127, 1_048_575, 1_048_576, 1_300_000, history - 1, history + ql],
            dtype=np.int32,
        )
        topk[ql, : len(picks)] = torch.from_numpy(picks)

    perm = rng.permutation(num_pages).astype(np.int32)
    page_table = torch.from_numpy(perm).unsqueeze(0)
    bitmap_buf = torch.zeros((1, MAX_BITMAP_WORDS), dtype=torch.int32)
    super_prefix_buf = torch.zeros((1, MAX_SUPER_BLOCKS), dtype=torch.int32)
    desc = torch.tensor([[MODE_FULL, history, query, 0, 0, 0, 0, num_pages]], dtype=torch.int32)
    mapped_out = torch.full((query, TOPK_WIDTH), -99, dtype=torch.int32)

    before_traps = lib.selected_kv_sim_trap_count()
    ops.hisparse_prefill_remap_topk(
        topk, desc, bitmap_buf, super_prefix_buf, page_table, mapped_out, 0, num_pages * BLOCK_SIZE
    )
    assert lib.selected_kv_sim_trap_count() == before_traps
    oracle = _oracle_remap_topk(topk.numpy(), [tuple(desc[0].tolist())], page_table.numpy(), {}, 0)
    np.testing.assert_array_equal(mapped_out.numpy(), oracle)


def test_host_sim_negative_topk_overflow_and_slot_trap():
    ops, lib = load_prefill_host_sim()
    history = 131_072
    query = 4

    # 1. Invalid TopK (< -1 or >= H + Q) sets status[3]=1 cleanly without Trap
    topk_bad = torch.full((query, TOPK_WIDTH), -1, dtype=torch.int32)
    topk_bad[0, 0] = -2
    topk_bad[1, 0] = history + query
    scratch, outputs = _make_union_buffers(1, 256)
    before_traps = lib.selected_kv_sim_trap_count()
    ops.hisparse_prefill_union_plan(topk_bad, scratch, outputs, [history], [0], [query], 256)
    assert lib.selected_kv_sim_trap_count() == before_traps
    assert int(outputs[1][0, 3]) == 1
    assert int(outputs[1][0, 4]) == 0

    # 2. Capacity overflow sets status[4]=1 cleanly without Trap
    topk_ovf = torch.full((query, TOPK_WIDTH), -1, dtype=torch.int32)
    topk_ovf[0, :20] = torch.arange(20, dtype=torch.int32)
    scratch2, outputs2 = _make_union_buffers(1, 8)
    ops.hisparse_prefill_union_plan(topk_ovf, scratch2, outputs2, [history], [0], [query], 8)
    assert lib.selected_kv_sim_trap_count() == before_traps
    assert int(outputs2[1][0, 0]) == 20
    assert int(outputs2[1][0, 3]) == 0
    assert int(outputs2[1][0, 4]) == 1

    # 3. Out-of-bounds slot descriptor in remap_topk triggers Trap for both MODE_DEMAND and MODE_FULL
    topk_ok = torch.full((query, TOPK_WIDTH), -1, dtype=torch.int32)
    topk_ok[:, 0] = 0
    mapped_out = torch.full((query, TOPK_WIDTH), -1, dtype=torch.int32)
    page_table = torch.zeros((1, 4), dtype=torch.int32)

    bad_demand_desc = torch.tensor([[MODE_DEMAND, history, query, 0, 0, 5, 128, 0]], dtype=torch.int32)
    with pytest.raises(RuntimeError):
        ops.hisparse_prefill_remap_topk(
            topk_ok, bad_demand_desc, outputs2[2], outputs2[3], page_table, mapped_out, 0, 4096
        )
    assert lib.selected_kv_sim_trap_count() == before_traps + 1

    bad_full_desc = torch.tensor([[MODE_FULL, 256, query, 0, 0, 5, 0, 3]], dtype=torch.int32)
    with pytest.raises(RuntimeError):
        ops.hisparse_prefill_remap_topk(
            topk_ok, bad_full_desc, outputs2[2], outputs2[3], page_table, mapped_out, 0, 4096
        )
    assert lib.selected_kv_sim_trap_count() == before_traps + 2
