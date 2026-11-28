# SPDX-License-Identifier: Apache-2.0
"""End-to-end CPU simulation of HiSparsePrefillExecutor using Host-Sim ops and ctypes SDMA."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode

from tests.ut.distributed.kv_transfer.hisparse_prefill.conftest import load_prefill_host_sim
from vllm_ascend.hisparse.prefill_pipeline import (
    BLOCK_SIZE,
    MODE_DEMAND,
    TIER_HARD_DEMAND,
    TOPK_WIDTH,
    HiSparsePrefillConfig,
    HiSparsePrefillExecutor,
    HiSparsePrefillWorkspace,
)

ROW_BYTES = 656


def _install_sim_ops(monkeypatch, ops):
    calls: dict[str, int] = {
        "union_plan": 0,
        "remap_topk": 0,
        "gather_union": 0,
        "copy_spans": 0,
        "sdma": 0,
    }

    def _swap_blocks_batch(src: torch.Tensor, dst: torch.Tensor, sizes: torch.Tensor, _direction: int, **_kw):
        calls["sdma"] += int(src.numel())
        for s_addr, d_addr, nbytes in zip(src.tolist(), dst.tolist(), sizes.tolist()):
            ctypes.memmove(int(d_addr), int(s_addr), int(nbytes))

    def _wrap(name: str, fn):
        def inner(*args, **kwargs):
            calls[name] += 1
            return fn(*args, **kwargs)

        return inner

    ns = NS(
        swap_blocks_batch=_swap_blocks_batch,
        hisparse_prefill_union_plan=_wrap("union_plan", ops.hisparse_prefill_union_plan),
        hisparse_prefill_remap_topk=_wrap("remap_topk", ops.hisparse_prefill_remap_topk),
        hisparse_prefill_gather_union=_wrap("gather_union", ops.hisparse_prefill_gather_union),
        hisparse_prefill_copy_spans=_wrap("copy_spans", ops.hisparse_prefill_copy_spans),
    )
    monkeypatch.setattr(torch.ops, "_C_ascend", ns, raising=False)
    return calls


@pytest.mark.parametrize("tight_multiwave", [False, True], ids=["single_wave", "two_wave_split"])
def test_executor_end_to_end_pipeline_with_host_sim(monkeypatch, patched_npu_env, tight_multiwave: bool):
    ops, lib = load_prefill_host_sim()
    calls = _install_sim_ops(monkeypatch, ops)
    monkeypatch.setattr(
        "vllm_ascend.hisparse.prefill_pipeline.get_forward_context",
        lambda: NS(cudagraph_runtime_mode=CUDAGraphMode.NONE),
    )
    rng = np.random.default_rng(7788 + int(tight_multiwave))

    # Build 2 layers in 1 index group (leader layer 0, follower layer 1) + 1 leader in group 1 (layer 2)
    num_layers = 3
    host_pages = 3072
    res_pages = 32
    h_tok = 40_960  # 320 pages (> min_history_tokens=32768) -> Q=256 => ratio=0.00625 <= 0.01 => Tier 1 DEMAND
    q_tok = 256
    num_reqs = 2
    sealed_pages = h_tok // BLOCK_SIZE
    needed_pages = (h_tok + q_tok) // BLOCK_SIZE

    host_caches = [
        torch.from_numpy(rng.integers(0, 256, size=(host_pages, BLOCK_SIZE, ROW_BYTES), dtype=np.uint8))
        for _ in range(num_layers)
    ]
    resident_caches = [
        torch.from_numpy(rng.integers(0, 256, size=(res_pages, BLOCK_SIZE, ROW_BYTES), dtype=np.uint8))
        for _ in range(num_layers)
    ]

    # Request 0: host blocks [1..322], Request 1: host blocks [401..722]
    # Resident pages: history pages [0..319] are on Host (res=0), active Q pages [320, 321] are on Resident!
    host_blk_lists = [
        list(range(1, 1 + needed_pages)),
        list(range(401, 401 + needed_pages)),
    ]
    res_tbl = torch.zeros((num_reqs, needed_pages), dtype=torch.int32)
    res_tbl[0, sealed_pages:] = torch.tensor([1, 2], dtype=torch.int32)
    res_tbl[1, sealed_pages:] = torch.tensor([3, 4], dtype=torch.int32)
    src_tbl = torch.tensor(host_blk_lists, dtype=torch.int32)
    state_indices = torch.tensor([0, 1], dtype=torch.int32)

    handles = []
    for l_idx in range(num_layers):
        mapping = NS(address_of=lambda t, _hc=host_caches[l_idx]: int(t.data_ptr()))
        handles.append(
            NS(
                draft_layer=False,
                all_context_pages_resident=False,
                source_block_table=src_tbl,
                block_table=res_tbl,
                runtime=NS(
                    resident_source_index=0,
                    request_state_indices=state_indices,
                    cache_ops=NS(host_mapping=mapping),
                ),
            )
        )

    worker = NS(
        cache_handles=handles,
        host_caches=host_caches,
        resident_caches=resident_caches,
        kv_cache_config=NS(host_group_ids=(0,)),
        cache_layer_names=[f"layer.{i}" for i in range(num_layers)],
        backend=NS(is_current_stream_capturing=lambda: False),
    )

    # For tight_multiwave=True, force worst_total_rows > staging_rows (82,432 > 45,000) while
    # single_need=41,216 <= 45,000, and actual union_count = 25,000 (+256 Q = 25,256) per req so
    # 2 requests (50,512) split into 2 whole-request waves.
    staging_rows = 45_000 if tight_multiwave else 2_097_152
    ws = HiSparsePrefillWorkspace(
        device=torch.device("cpu"),
        kv_dtype=torch.uint8,
        staging_rows=staging_rows,
        max_num_seqs=4,
        max_batched_tokens=1024,
        max_model_len=65_536,
    )
    ws.prefill_stream = torch.npu.Stream()
    ws.sfa_done_event = torch.npu.Event()
    ws.prefetch_ready_event = torch.npu.Event()
    cfg = HiSparsePrefillConfig(enabled=True, staging_rows=staging_rows)
    executor = HiSparsePrefillExecutor(worker, cfg, ws)

    # Construct logical topk: for single_wave 800 distinct tokens; for two_wave_split 25,000 distinct tokens
    total_q = num_reqs * q_tok
    logical_topk = torch.full((total_q, TOPK_WIDTH), -1, dtype=torch.int32)
    if tight_multiwave:
        req_hist_pool = np.arange(0, 25_000, dtype=np.int32)
        chunk_w = 100
        for r in range(num_reqs):
            for ql in range(q_tok):
                row = r * q_tok + ql
                start_i = (ql * chunk_w) % 25_000
                pick = req_hist_pool[start_i : start_i + chunk_w]
                logical_topk[row, : len(pick)] = torch.from_numpy(pick)
                logical_topk[row, len(pick)] = h_tok + ql
    else:
        for r in range(num_reqs):
            req_hist_pool = np.arange(0, h_tok, 50, dtype=np.int32)[:800]
            for ql in range(q_tok):
                row = r * q_tok + ql
                pick = req_hist_pool[(ql * 7) % 800 : (ql * 7) % 800 + 8]
                if len(pick) < 8:
                    pick = req_hist_pool[:8]
                logical_topk[row, : len(pick)] = torch.from_numpy(pick)
                logical_topk[row, len(pick)] = h_tok + ql

    sched_out = NS(
        finished_req_ids=(),
        preempted_req_ids=(),
        scheduled_new_reqs=[
            NS(req_id="r0", block_ids=[host_blk_lists[0]], num_computed_tokens=h_tok),
            NS(req_id="r1", block_ids=[host_blk_lists[1]], num_computed_tokens=h_tok),
        ],
        scheduled_cached_reqs=None,
        num_scheduled_tokens={"r0": q_tok, "r1": q_tok},
    )
    conn_meta = NS(
        residency_updates={
            "r0": NS(pages=list(range(needed_pages)), block_ids=[res_tbl[0].tolist()]),
            "r1": NS(pages=list(range(needed_pages)), block_ids=[res_tbl[1].tolist()]),
        }
    )
    attn_meta = NS(
        hisparse_for_capture=False,
        num_reqs=num_reqs,
        num_decodes=0,
        num_prefills=num_reqs,
        num_decode_tokens=0,
        num_actual_tokens=total_q,
        seq_lens_cpu=torch.tensor([h_tok + q_tok, h_tok + q_tok], dtype=torch.int32),
    )

    before_traps = lib.selected_kv_sim_trap_count()
    executor.on_pre_forward(conn_meta, sched_out, ["r0", "r1"], {"layer.0": attn_meta})
    assert executor.active_step is not None
    assert executor.active_step.needs_union_count_check is tight_multiwave
    assert all(r.tier == TIER_HARD_DEMAND and r.mode == MODE_DEMAND for r in executor.active_step.requests)

    # Group 0 has layers 0, 1; Group 1 has layer 2
    grp0 = NS(num_layers=2)
    grp1 = NS(num_layers=1)
    groups = [(grp0, 0), (grp0, 1), (grp1, 0)]

    # Verify wave outputs against exact physical KV rows gathered directly from Host + Resident
    def _ref_attention(layer_idx: int, q_slice: torch.Tensor, staged_slice: torch.Tensor, mapped_slice: torch.Tensor):
        flat_staged = staged_slice.view(-1, ROW_BYTES).to(torch.int32)
        # Deterministic byte-checksum projection per query token so any row/index mismatch fails loudly
        out = torch.zeros((q_slice.shape[0], 1, 4), dtype=torch.float32)
        for row_i in range(q_slice.shape[0]):
            valid = mapped_slice[row_i][mapped_slice[row_i] >= 0].to(torch.long)
            kv_sum = flat_staged[valid].sum(dim=0)[:4].to(torch.float32)
            out[row_i, 0] = q_slice[row_i, 0, :4] + kv_sum
        return out

    q_in = torch.arange(total_q * 4, dtype=torch.float32).view(total_q, 1, 4)
    q_rope = torch.zeros((total_q, 1, 64), dtype=torch.float32)

    outputs = []
    for l_idx, (grp, grp_idx) in enumerate(groups):
        impl = NS(
            enable_sparse_sfa_c8=True,
            hisparse_cache=handles[l_idx],
            index_group=grp,
            index_group_index=grp_idx,
            _attention_physical=lambda q, _qr, staged, mapped, *, indices_are_mapped=True, _li=l_idx: _ref_attention(
                _li, q, staged, mapped
            ),
        )
        out = executor.execute_layer_prefill(impl, q_in, q_rope, logical_topk, attn_meta)
        assert out is not None and out.shape == (total_q, 1, 4)
        outputs.append(out)
        if l_idx == 0:
            # Post-SFA on leader 0 prefetched same-group follower 1 (including DEMAND gather!)
            assert executor.prefetched_layer_idx == 1
            assert executor.prefetched_has_demand is True
        elif l_idx == 1:
            # Crossing group boundary (layer 1 -> layer 2): follower prefetch has no demand
            assert executor.prefetched_has_demand is False

    assert lib.selected_kv_sim_trap_count() == before_traps
    # Group 0 (leader 0, follower 1) + Group 1 (leader 2) => union_plan called twice (once per index group)
    assert calls["union_plan"] == 2
    expected_waves = 2 if tight_multiwave else 1
    assert calls["remap_topk"] == 2 * expected_waves

    # Check step_summary
    assert len(executor.step_summaries) == 1
    summ = executor.step_summaries[0]
    assert summ["num_waves"] == expected_waves
    assert summ["tier_counts"]["tier1_hard_demand"] == 2
    assert all(m["mode"] == "demand" and m["host_history_tokens"] == h_tok for m in summ["modes"])

    # Verify every query token's output matches direct ground-truth Host/Resident lookup!
    for l_idx, out in enumerate(outputs):
        hc = host_caches[l_idx].to(torch.int32)
        rc = resident_caches[l_idx].to(torch.int32)
        for r in range(num_reqs):
            for ql in range(q_tok):
                row_i = r * q_tok + ql
                toks = logical_topk[row_i][logical_topk[row_i] >= 0].tolist()
                expected_kv = torch.zeros((4,), dtype=torch.float32)
                for tok in toks:
                    pg, off = divmod(tok, BLOCK_SIZE)
                    if res_tbl[r, pg] > 0:
                        expected_kv += rc[int(res_tbl[r, pg]), off, :4].to(torch.float32)
                    else:
                        expected_kv += hc[int(src_tbl[r, pg]), off, :4].to(torch.float32)
                torch.testing.assert_close(out[row_i, 0], q_in[row_i, 0] + expected_kv)


def test_executor_heterogeneous_mixed_decode_piecewise_and_streams(monkeypatch, patched_npu_env):
    from vllm_ascend.hisparse.prefill_pipeline import MODE_FULL, TIER_FULL_LOAD

    ops, lib = load_prefill_host_sim()
    calls = _install_sim_ops(monkeypatch, ops)
    monkeypatch.setattr(
        "vllm_ascend.hisparse.prefill_pipeline.get_forward_context",
        lambda: NS(cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE),
    )
    rng = np.random.default_rng(20261127 + 99)

    num_layers = 3
    host_pages = 3072
    res_pages = 32
    host_caches = [
        torch.from_numpy(rng.integers(0, 256, size=(host_pages, BLOCK_SIZE, ROW_BYTES), dtype=np.uint8))
        for _ in range(num_layers)
    ]
    resident_caches = [
        torch.from_numpy(rng.integers(0, 256, size=(res_pages, BLOCK_SIZE, ROW_BYTES), dtype=np.uint8))
        for _ in range(num_layers)
    ]

    # Mixed batch: 2 decode reqs (batch_idx 0, 1) + 2 prefill reqs (batch_idx 2, 3), both with H % 128 != 0!
    #   batch_idx 2 ("pref-full"): H = 260 (2 sealed host + 4 tail rows), Q = 10 -> Tier 3 MODE_FULL SDMA
    #   batch_idx 3 ("pref-demand"): H = 40965 (320 sealed + 5 tail rows), Q = 12 -> Tier 1 MODE_DEMAND
    h_full, q_full = 260, 10
    h_dem, q_dem = 40_965, 12
    full_pages_need = (h_full + q_full + BLOCK_SIZE - 1) // BLOCK_SIZE  # 3
    dem_pages_need = (h_dem + q_dem + BLOCK_SIZE - 1) // BLOCK_SIZE  # 321
    max_cols = dem_pages_need

    host_blks_full = list(range(10, 10 + full_pages_need))
    host_blks_dem = list(range(100, 100 + dem_pages_need))

    src_tbl = torch.zeros((4, max_cols), dtype=torch.int32)
    src_tbl[2, :full_pages_need] = torch.tensor(host_blks_full, dtype=torch.int32)
    src_tbl[3, :dem_pages_need] = torch.tensor(host_blks_dem, dtype=torch.int32)

    res_tbl = torch.zeros((4, max_cols), dtype=torch.int32)
    res_tbl[0, 0] = 1
    res_tbl[1, 0] = 2
    res_tbl[2, h_full // BLOCK_SIZE] = 5  # active tail page [256..270) resident for pref-full
    res_tbl[3, h_dem // BLOCK_SIZE] = 6  # active tail page [40960..40977) resident for pref-demand
    state_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32)

    handles = []
    for l_idx in range(num_layers):
        mapping = NS(address_of=lambda t, _hc=host_caches[l_idx]: int(t.data_ptr()))
        handles.append(
            NS(
                draft_layer=False,
                all_context_pages_resident=False,
                source_block_table=src_tbl,
                block_table=res_tbl,
                runtime=NS(
                    resident_source_index=0,
                    request_state_indices=state_indices,
                    cache_ops=NS(host_mapping=mapping),
                ),
            )
        )

    worker = NS(
        cache_handles=handles,
        host_caches=host_caches,
        resident_caches=resident_caches,
        kv_cache_config=NS(host_group_ids=(0,)),
        cache_layer_names=[f"layer.{i}" for i in range(num_layers)],
        backend=NS(is_current_stream_capturing=lambda: False),
    )
    ws = HiSparsePrefillWorkspace(
        device=torch.device("cpu"),
        kv_dtype=torch.uint8,
        staging_rows=2_097_152,
        max_num_seqs=8,
        max_batched_tokens=1024,
        max_model_len=65_536,
    )
    ws.prefill_stream = torch.npu.Stream()
    ws.sfa_done_event = torch.npu.Event()
    ws.prefetch_ready_event = torch.npu.Event()
    executor = HiSparsePrefillExecutor(worker, HiSparsePrefillConfig(enabled=True), ws)

    num_decode_tokens = 2
    num_prefill_tokens = q_full + q_dem
    pad_tokens = 2
    logical_full = torch.full((num_decode_tokens + num_prefill_tokens + pad_tokens, TOPK_WIDTH), -1, dtype=torch.int32)
    # pref-full: touch both sealed host pages [0..255] AND active resident tail [256..260+ql]
    for ql in range(q_full):
        logical_full[num_decode_tokens + ql, :6] = torch.tensor([0, 127, 128, 255, 258, h_full + ql], dtype=torch.int32)
    # pref-demand: touch sealed host pages, active resident tail history [40960..40964], and new Q [h_dem..h_dem+ql]
    for ql in range(q_dem):
        logical_full[num_decode_tokens + q_full + ql, :7] = torch.tensor(
            [0, 129, 40_000, 40_959, 40_962, 40_964, h_dem + ql], dtype=torch.int32
        )

    sched_out = NS(
        finished_req_ids=(),
        preempted_req_ids=(),
        scheduled_new_reqs=[
            NS(req_id="dec-0", block_ids=[[1]], num_computed_tokens=100),
            NS(req_id="dec-1", block_ids=[[2]], num_computed_tokens=100),
            NS(req_id="pref-full", block_ids=[host_blks_full], num_computed_tokens=h_full),
            NS(req_id="pref-demand", block_ids=[host_blks_dem], num_computed_tokens=h_dem),
        ],
        scheduled_cached_reqs=None,
        num_scheduled_tokens={"dec-0": 1, "dec-1": 1, "pref-full": q_full, "pref-demand": q_dem},
    )
    conn_meta = NS(
        residency_updates={
            "dec-0": NS(pages=[0], block_ids=[[1]]),
            "dec-1": NS(pages=[0], block_ids=[[2]]),
            "pref-full": NS(pages=list(range(full_pages_need)), block_ids=[res_tbl[2, :full_pages_need].tolist()]),
            "pref-demand": NS(pages=list(range(dem_pages_need)), block_ids=[res_tbl[3, :dem_pages_need].tolist()]),
        }
    )
    attn_meta = NS(
        hisparse_for_capture=False,
        num_reqs=4,
        num_decodes=2,
        num_prefills=2,
        num_decode_tokens=num_decode_tokens,
        num_actual_tokens=num_decode_tokens + num_prefill_tokens,
        seq_lens_cpu=torch.tensor([101, 101, h_full + q_full, h_dem + q_dem], dtype=torch.int32),
    )

    before_traps = lib.selected_kv_sim_trap_count()
    executor.on_pre_forward(
        conn_meta,
        sched_out,
        ["dec-0", "dec-1", "pref-full", "pref-demand"],
        {"layer.0": attn_meta},
    )
    assert executor.active_step is not None
    r_full, r_dem = executor.active_step.requests
    assert (r_full.tier, r_full.mode) == (TIER_FULL_LOAD, MODE_FULL)
    assert (r_dem.tier, r_dem.mode) == (TIER_HARD_DEMAND, MODE_DEMAND)

    # Layer-0 early SDMA already ran on prefill_stream for pref-full's 2 contiguous sealed host pages (1 coalesced op)!
    assert executor.prefetched_layer_idx == 0
    assert calls["sdma"] == 1
    assert patched_npu_env.compute_stream.npu_stream in ws.prefill_stream.waited_streams

    q_all = torch.arange((num_decode_tokens + num_prefill_tokens + pad_tokens) * 4, dtype=torch.float32).view(-1, 1, 4)
    q_rope_all = torch.zeros((q_all.shape[0], 1, 64), dtype=torch.float32)

    def _ref_attention(q_slice: torch.Tensor, staged_slice: torch.Tensor, mapped_slice: torch.Tensor):
        flat_staged = staged_slice.view(-1, ROW_BYTES).to(torch.int32)
        out = torch.zeros((q_slice.shape[0], 1, 4), dtype=torch.float32)
        for row_i in range(q_slice.shape[0]):
            valid = mapped_slice[row_i][mapped_slice[row_i] >= 0].to(torch.long)
            out[row_i, 0] = q_slice[row_i, 0, :4] + flat_staged[valid].sum(dim=0)[:4].to(torch.float32)
        return out

    grp0 = NS(num_layers=2)
    grp1 = NS(num_layers=1)
    outputs = []
    sdma_after_l0 = None
    for l_idx, (grp, grp_idx) in enumerate(((grp0, 0), (grp0, 1), (grp1, 0))):
        impl = NS(
            enable_sparse_sfa_c8=True,
            hisparse_cache=handles[l_idx],
            index_group=grp,
            index_group_index=grp_idx,
            _attention_physical=lambda q, _qr, staged, mapped, *, indices_are_mapped=True: _ref_attention(
                q, staged, mapped
            ),
        )
        out = executor.execute_layer_prefill(
            impl,
            q_all[num_decode_tokens:],
            q_rope_all[num_decode_tokens:],
            logical_full[num_decode_tokens:],
            attn_meta,
        )
        assert out is not None and out.shape == (num_prefill_tokens, 1, 4)
        outputs.append(out)
        if l_idx == 0:
            sdma_after_l0 = calls["sdma"]

    assert lib.selected_kv_sim_trap_count() == before_traps
    # Layer 0 reused early SDMA (no duplicate L0 SDMA), then post-SFA queued Layer 1 (+1 coalesced) & Layer 2 (+1)
    assert sdma_after_l0 == 2  # 1 early L0 + 1 post-SFA L1
    assert calls["sdma"] == 3  # +1 post-SFA L2
    assert ws.sfa_done_event in ws.prefill_stream.waited_events
    assert ws.prefetch_ready_event in patched_npu_env.compute_stream.waited_events
    assert calls["union_plan"] == 2

    # Numerical check on all prefill tokens (MODE_FULL H=260 + MODE_DEMAND H=40965)
    for l_idx, out in enumerate(outputs):
        hc = host_caches[l_idx].to(torch.int32)
        rc = resident_caches[l_idx].to(torch.int32)
        for batch_r, q_cnt, q_off in ((2, q_full, 0), (3, q_dem, q_full)):
            for ql in range(q_cnt):
                pref_row = q_off + ql
                toks = logical_full[num_decode_tokens + pref_row]
                valid_toks = toks[toks >= 0].tolist()
                expected_kv = torch.zeros((4,), dtype=torch.float32)
                for tok in valid_toks:
                    pg, off = divmod(tok, BLOCK_SIZE)
                    if res_tbl[batch_r, pg] > 0:
                        expected_kv += rc[int(res_tbl[batch_r, pg]), off, :4].to(torch.float32)
                    else:
                        expected_kv += hc[int(src_tbl[batch_r, pg]), off, :4].to(torch.float32)
                torch.testing.assert_close(out[pref_row, 0], q_all[num_decode_tokens + pref_row, 0] + expected_kv)


def test_executor_resolves_target_leader_names_from_group_leaders_without_cache_layer_names(
    patched_npu_env,
    monkeypatch,
):
    from vllm_ascend.hisparse.prefill_pipeline import (
        MODE_FULL,
        TIER_FULL_LOAD,
        HiSparsePrefillConfig,
        HiSparsePrefillExecutor,
        HiSparsePrefillWorkspace,
    )

    monkeypatch.setattr(torch.ops._C_ascend, "swap_blocks_batch", lambda *_a, **_kw: None, raising=False)

    h_tok, q_tok = 256, 32
    host_blks = [1, 2, 0]
    src_tbl = torch.zeros((1, 8), dtype=torch.int32)
    src_tbl[0, :3] = torch.tensor(host_blks, dtype=torch.int32)
    res_tbl = torch.zeros((1, 8), dtype=torch.int32)
    res_tbl[0, 2] = 5
    state_indices = torch.arange(4, dtype=torch.int32)

    host_cache = torch.zeros((8, BLOCK_SIZE, 1, ROW_BYTES), dtype=torch.uint8)
    resident_cache = torch.zeros((8, BLOCK_SIZE, 1, ROW_BYTES), dtype=torch.uint8)
    mapping = NS(address_of=lambda t: int(t.data_ptr()))

    main_handle = NS(
        draft_layer=False,
        all_context_pages_resident=False,
        source_block_table=src_tbl,
        block_table=res_tbl,
        runtime=NS(
            resident_source_index=0,
            request_state_indices=state_indices,
            cache_ops=NS(host_mapping=mapping),
        ),
    )
    draft_handle = NS(
        draft_layer=True,
        all_context_pages_resident=False,
        source_block_table=src_tbl,
        block_table=res_tbl,
        runtime=NS(
            resident_source_index=0,
            request_state_indices=state_indices,
            cache_ops=NS(host_mapping=mapping),
        ),
    )

    # Stub mimicking real HiSparseConnectorWorker: has _group_leaders, NO cache_layer_names attr
    worker = NS(
        cache_handles=[main_handle, draft_handle],
        host_caches=[host_cache, host_cache],
        resident_caches=[resident_cache, resident_cache],
        kv_cache_config=NS(host_group_ids=(0,)),
        _group_leaders=(
            ("model.layers.0.self_attn.attn", main_handle),
            ("draft.layers.0.self_attn.attn", draft_handle),
        ),
        backend=NS(is_current_stream_capturing=lambda: False),
    )
    assert not hasattr(worker, "cache_layer_names")

    ws = HiSparsePrefillWorkspace(
        device=torch.device("cpu"),
        kv_dtype=torch.uint8,
        staging_rows=1024,
        max_num_seqs=4,
        max_batched_tokens=512,
        max_model_len=4096,
    )
    ws.prefill_stream = torch.npu.Stream()
    ws.sfa_done_event = torch.npu.Event()
    ws.prefetch_ready_event = torch.npu.Event()
    executor = HiSparsePrefillExecutor(worker, HiSparsePrefillConfig(enabled=True), ws)
    assert executor.target_leader_names == ("model.layers.0.self_attn.attn",)

    attn_meta = NS(
        hisparse_for_capture=False,
        num_reqs=1,
        num_decodes=0,
        num_prefills=1,
        num_decode_tokens=0,
        num_actual_tokens=q_tok,
        seq_lens_cpu=torch.tensor([h_tok + q_tok], dtype=torch.int32),
    )
    assert (
        executor._find_first_attn_metadata(
            {"draft.layers.0.self_attn.attn": object(), "model.layers.0.self_attn.attn": attn_meta}
        )
        is attn_meta
    )

    sched_out = NS(
        finished_req_ids=(),
        preempted_req_ids=(),
        scheduled_new_reqs=[
            NS(req_id="req-0", block_ids=[host_blks], num_computed_tokens=h_tok),
        ],
        scheduled_cached_reqs=None,
        num_scheduled_tokens={"req-0": q_tok},
    )
    conn_meta = NS(residency_updates={"req-0": NS(pages=[0, 1, 2], block_ids=[[0, 0, 5]])})
    executor.on_pre_forward(
        conn_meta,
        sched_out,
        ["req-0"],
        {"model.layers.0.self_attn.attn": attn_meta},
    )
    assert executor.active_step is not None
    assert (executor.active_step.requests[0].tier, executor.active_step.requests[0].mode) == (
        TIER_FULL_LOAD,
        MODE_FULL,
    )


def test_initialize_worker_dma_installs_hooks_and_binds_executor(monkeypatch):
    from types import SimpleNamespace

    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import HiSparseConnector
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker

    from vllm_ascend.hisparse.backend import initialize_worker_dma
    from vllm_ascend.hisparse.prefill_pipeline import (
        HiSparsePrefillConfig,
        HiSparsePrefillExecutor,
        HiSparsePrefillWorkspace,
    )

    for cls, attr in (
        (HiSparseConnector, "start_load_kv"),
        (HiSparseConnector, "get_transfer_results"),
        (HiSparseConnectorWorker, "_init_dma"),
        (HiSparseConnectorWorker, "finish_forward"),
        (HiSparseConnectorWorker, "reset_hot_state"),
        (HiSparseConnectorWorker, "shutdown"),
    ):
        monkeypatch.setattr(cls, attr, getattr(cls, attr))

    try:
        from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector
    except Exception:
        ActiveKVConnector = None
    if ActiveKVConnector is not None:
        monkeypatch.setattr(ActiveKVConnector, "no_forward", ActiveKVConnector.no_forward)
        monkeypatch.setattr(ActiveKVConnector, "set_disabled", ActiveKVConnector.set_disabled)

    def _fresh_start_load_kv(connector, forward_context, **kwargs):
        return None

    monkeypatch.setattr(HiSparseConnector, "start_load_kv", _fresh_start_load_kv)
    for op_name in (
        "hisparse_prefill_union_plan",
        "hisparse_prefill_remap_topk",
        "hisparse_prefill_gather_union",
        "hisparse_prefill_copy_spans",
    ):
        monkeypatch.setattr(torch.ops._C_ascend, op_name, lambda *a, **k: None, raising=False)
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "hisparse_prefill_kernels_compiled",
        lambda: True,
        raising=False,
    )

    vllm_config = SimpleNamespace(
        additional_config={"hisparse_prefill_config": {"enabled": True}},
        attention_config=SimpleNamespace(hisparse_config=SimpleNamespace()),
        cache_config=SimpleNamespace(cache_dtype="int8", block_size=128),
        model_config=SimpleNamespace(max_model_len=128),
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=128),
        kernel_config=SimpleNamespace(enable_jit_warmup=False),
    )
    cfg = HiSparsePrefillConfig.from_vllm_config(vllm_config)
    workspace = HiSparsePrefillWorkspace(
        device=torch.device("cpu"),
        kv_dtype=torch.int8,
        staging_rows=cfg.staging_rows,
        max_num_seqs=2,
        max_batched_tokens=128,
        max_model_len=128,
    )
    handle = SimpleNamespace(
        draft_layer=False,
        slot_mapping=torch.empty((0,), dtype=torch.int32),
        runtime=SimpleNamespace(
            device=SimpleNamespace(type="npu"),
            kv_dtype=torch.int8,
            resident_source_index=0,
            _ascend_prefill_workspace=(cfg, workspace),
        ),
    )
    worker = SimpleNamespace(
        is_host_writer=True,
        _slot_mapping_staging=None,
        backend=SimpleNamespace(is_current_stream_capturing=lambda: False),
        kernel_block_size=0,
        leader_runtimes=(),
        vllm_config=vllm_config,
        cache_handles=[handle],
        host_caches=[torch.empty((2, 16, 8), dtype=torch.int8)],
        resident_caches=[torch.empty((2, 16, 8), dtype=torch.int8)],
        kv_cache_config=SimpleNamespace(host_group_ids=(0,)),
        _group_leaders=[("model.layers.0", handle)],
        cache_layer_names=["model.layers.0"],
    )

    initialize_worker_dma(worker)

    assert getattr(HiSparseConnector.start_load_kv, "_ascend_prefill_hook", False) is True
    assert isinstance(worker._ascend_prefill_executor, HiSparsePrefillExecutor)
    assert worker._ascend_prefill_executor.workspace is workspace
    assert worker._ascend_recycled_invalidator is None


def test_submit_sdma_entries_shared_host_follower_uses_backend_dma(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from vllm_ascend.hisparse.prefill_pipeline import (
        HiSparsePrefillConfig,
        HiSparsePrefillExecutor,
        HiSparsePrefillWorkspace,
        SDMAPageEntry,
    )

    events: list[tuple[object, ...]] = []
    unexpected_ops_calls: list[tuple[object, ...]] = []
    mapping = object()

    @contextmanager
    def _host_to_device_dma(reg):
        events.append(("enter", reg))
        try:
            yield
        finally:
            events.append(("exit", reg))

    def _swap_blocks_batch(src_t, dst_t, sizes_t):
        events.append(("swap", src_t.clone(), dst_t.clone(), sizes_t.clone()))

    monkeypatch.setattr(
        torch.ops._C_ascend,
        "swap_blocks_batch",
        lambda *a, **k: unexpected_ops_calls.append((a, k)),
        raising=False,
    )

    vllm_config = SimpleNamespace(
        additional_config={"hisparse_prefill_config": {"enabled": True}},
        attention_config=SimpleNamespace(hisparse_config=SimpleNamespace()),
        cache_config=SimpleNamespace(cache_dtype="int8", block_size=128),
        model_config=SimpleNamespace(max_model_len=128),
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=128),
    )
    cfg = HiSparsePrefillConfig.from_vllm_config(vllm_config)
    workspace = HiSparsePrefillWorkspace(
        device=torch.device("cpu"),
        kv_dtype=torch.int8,
        staging_rows=cfg.staging_rows,
        max_num_seqs=2,
        max_batched_tokens=128,
        max_model_len=128,
    )
    handle = SimpleNamespace(
        draft_layer=False,
        runtime=SimpleNamespace(device=torch.device("cpu"), kv_dtype=torch.int8, resident_source_index=0),
    )
    worker = SimpleNamespace(
        shared_host_region=SimpleNamespace(base_tensor=SimpleNamespace(_hisparse_registration=mapping)),
        is_host_writer=False,
        backend=SimpleNamespace(
            is_current_stream_capturing=lambda: False,
            host_to_device_dma=_host_to_device_dma,
            swap_blocks_batch=_swap_blocks_batch,
        ),
        cache_handles=[handle],
        host_caches=[torch.empty((4, 128, 656), dtype=torch.uint8)],
        resident_caches=[torch.empty((4, 128, 656), dtype=torch.uint8)],
        kv_cache_config=SimpleNamespace(host_group_ids=(0,)),
        _group_leaders=[("model.layers.0", handle)],
        cache_layer_names=["model.layers.0"],
    )
    executor = HiSparsePrefillExecutor(worker, cfg, workspace)

    entries = (SDMAPageEntry(host_block_id=1, dst_page=2),)
    assert executor._submit_sdma_entries(0, entries) is True
    assert unexpected_ops_calls == []
    assert [e[0] for e in events] == ["enter", "swap", "exit"]
    assert events[0][1] is mapping
    assert events[2][1] is mapping
    _, src_t, dst_t, sizes_t = events[1]
    assert tuple(src_t.shape) == (1,)
    assert tuple(dst_t.shape) == (1,)
    assert tuple(sizes_t.shape) == (1,)
