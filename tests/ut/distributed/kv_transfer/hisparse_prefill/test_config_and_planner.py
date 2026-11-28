# SPDX-License-Identifier: Apache-2.0
"""Unit tests for HiSparsePrefillConfig, 3-tier classification, wave packing, and shadow tracker."""

from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_ascend import ascend_config as ac
from vllm_ascend.hisparse.prefill_pipeline import (
    BLOCK_SIZE,
    MODE_DEMAND,
    MODE_FULL,
    TIER_ELASTIC,
    TIER_FULL_LOAD,
    TIER_HARD_DEMAND,
    TIER_ON_CARD,
    HiSparsePrefillConfig,
    HiSparsePrefillExecutor,
    HiSparseResidencyShadowTracker,
    PrefillRequestSpec,
    SDMAPageEntry,
    all_prefill_pages_resident,
    classify_prefill_requests,
    coalesce_sdma_descriptors,
    compute_staging_floor_bytes,
    compute_staging_floor_rows,
    plan_whole_request_waves,
)


def _make_cfg_ns(
    *,
    max_model_len: int = 524_288,
    hisparse_prefill_config=None,
    has_hisparse: bool = True,
    cache_dtype: str = "int8",
    block_size: int = 128,
    pp: int = 1,
    pcp: int = 1,
    dcp: int = 1,
    index_topk: int = 2048,
):
    additional = {}
    if hisparse_prefill_config is not None:
        additional["hisparse_prefill_config"] = hisparse_prefill_config
    return NS(
        additional_config=additional,
        model_config=NS(
            max_model_len=max_model_len,
            enforce_eager=True,
            is_deepseek_mla=True,
            hf_text_config=NS(model_type="glm_moe_dsa", index_topk=index_topk),
            hf_config=NS(model_type="glm_moe_dsa", index_topk=index_topk),
        ),
        attention_config=NS(
            hisparse_config=object() if has_hisparse else None,
            indexer_kv_dtype="int8",
        ),
        cache_config=NS(
            cache_dtype=cache_dtype,
            block_size=block_size,
        ),
        parallel_config=NS(
            pipeline_parallel_size=pp,
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=dcp,
            enable_batch_sharded_sampling=False,
        ),
        kv_transfer_config=None,
        quant_config=None,
        compilation_config=NS(cudagraph_mode=None),
    )


def _make_req(
    idx: int,
    h: int,
    q: int,
    *,
    host_miss_pages: int | None = None,
    host_blocks: tuple[int, ...] | None = None,
    q_batch_start: int = 0,
) -> PrefillRequestSpec:
    needed = (h + q + BLOCK_SIZE - 1) // BLOCK_SIZE
    sealed = h // BLOCK_SIZE
    if host_miss_pages is None:
        host_miss_pages = sealed
    if host_blocks is None:
        host_blocks = tuple(100_000 * (idx + 1) + p for p in range(needed))
    res = np.ones((1, needed), dtype=np.int32)
    if host_miss_pages > 0:
        res[0, :host_miss_pages] = 0
    return PrefillRequestSpec(
        prefill_idx=idx,
        batch_req_idx=idx,
        req_id=f"req-{idx}",
        history_tokens=h,
        query_tokens=q,
        q_batch_start=q_batch_start,
        host_blocks=host_blocks,
        residency_by_group=res,
        host_miss_pages=host_miss_pages,
        host_history_tokens=host_miss_pages * BLOCK_SIZE,
    )


def test_config_staging_floor_and_init_ascend_config(monkeypatch):
    # 1. Hard floor <= 1M -> 2,097,152 rows = 1,375,731,712 B (1,312 MiB)
    for mml in (131_072, 524_288, 1_048_576):
        assert compute_staging_floor_rows(mml) == 2_097_152
        assert compute_staging_floor_bytes(mml) == 1_375_731_712
        cfg = HiSparsePrefillConfig.from_vllm_config(
            _make_cfg_ns(max_model_len=mml, hisparse_prefill_config={"enabled": True})
        )
        assert cfg.enabled is True
        assert cfg.staging_rows == 2_097_152

    # > 1M scales to round_up(2 * max_model_len, 128); passing raw 2 * max_model_len succeeds and rounds up
    assert compute_staging_floor_rows(1_500_000) == 3_000_064
    cfg_large = HiSparsePrefillConfig.from_vllm_config(
        _make_cfg_ns(max_model_len=1_500_000, hisparse_prefill_config={"enabled": True})
    )
    assert cfg_large.staging_rows == 3_000_064
    cfg_exact_large = HiSparsePrefillConfig.from_vllm_config(
        _make_cfg_ns(
            max_model_len=1_500_000,
            hisparse_prefill_config={"enabled": True, "staging_rows": 3_000_000},
        )
    )
    assert cfg_exact_large.staging_rows == 3_000_064

    # 2. Foolproof rejections
    with pytest.raises(ValueError, match="hard floor"):
        HiSparsePrefillConfig.from_vllm_config(
            _make_cfg_ns(
                max_model_len=1_500_000,
                hisparse_prefill_config={"enabled": True, "staging_rows": 2_999_999},
            )
        )
    with pytest.raises(ValueError, match="index_topk == 2048"):
        HiSparsePrefillConfig.from_vllm_config(_make_cfg_ns(hisparse_prefill_config={"enabled": True}, index_topk=1024))
    with pytest.raises(ValueError, match="hard floor"):
        HiSparsePrefillConfig.from_vllm_config(
            _make_cfg_ns(hisparse_prefill_config={"enabled": True, "staging_rows": 2_097_151})
        )
    with pytest.raises(ValueError, match="hard floor"):
        HiSparsePrefillConfig.from_vllm_config(
            _make_cfg_ns(hisparse_prefill_config={"enabled": True, "staging_bytes": 1_375_731_711})
        )
    with pytest.raises(ValueError, match="at most one"):
        HiSparsePrefillConfig.from_vllm_config(
            _make_cfg_ns(
                hisparse_prefill_config={
                    "enabled": True,
                    "staging_rows": 2_097_152,
                    "staging_bytes": 1_375_731_712,
                }
            )
        )
    with pytest.raises(ValueError, match="Unknown keys"):
        HiSparsePrefillConfig.from_vllm_config(_make_cfg_ns(hisparse_prefill_config={"enabled": True, "bad_key": 1}))
    for bad_ratios in (
        {"tier1_max_ratio": 0.06, "tier2_max_ratio": 0.05},
        {"tier1_max_ratio": -0.01},
        {"tier2_max_ratio": 1.1},
    ):
        with pytest.raises(ValueError, match="tier1_max_ratio"):
            HiSparsePrefillConfig.from_vllm_config(
                _make_cfg_ns(hisparse_prefill_config={"enabled": True, **bad_ratios})
            )
    for par_kw in ({"pp": 2}, {"pcp": 2}, {"dcp": 2}):
        with pytest.raises(ValueError, match="PP == 1, PCP == 1, and DCP == 1"):
            HiSparsePrefillConfig.from_vllm_config(_make_cfg_ns(hisparse_prefill_config={"enabled": True}, **par_kw))
    for bad_dtype in ("bfloat16", "auto"):
        with pytest.raises(ValueError, match="C8 cache_dtype"):
            HiSparsePrefillConfig.from_vllm_config(
                _make_cfg_ns(hisparse_prefill_config={"enabled": True}, cache_dtype=bad_dtype)
            )
    with pytest.raises(ValueError, match="hisparse_config"):
        HiSparsePrefillConfig.from_vllm_config(
            _make_cfg_ns(hisparse_prefill_config={"enabled": True}, has_hisparse=False)
        )
    with pytest.raises(ValueError, match="block_size == 128"):
        HiSparsePrefillConfig.from_vllm_config(_make_cfg_ns(hisparse_prefill_config={"enabled": True}, block_size=64))

    # 3. init_ascend_config accepts additional_config["hisparse_prefill_config"] without extra="forbid" error
    monkeypatch.setattr(ac, "_ASCEND_CONFIG", None)
    monkeypatch.setattr(ac, "_INIT_VLLM_CONFIG", None)
    monkeypatch.setattr("vllm_ascend.utils.model_uses_sfa_sparse", lambda _mc: True)
    monkeypatch.setattr("vllm_ascend.utils.model_uses_kpool_indexer", lambda _mc: False)
    monkeypatch.setattr("vllm_ascend.utils.is_hif8_c8_kv_quant", lambda _vc: False)
    monkeypatch.setattr("vllm.compilation.breakable_cudagraph.is_breakable_cudagraph_enabled", lambda _cc: False)
    vc = _make_cfg_ns(hisparse_prefill_config={"enabled": True})
    asc_cfg = ac.init_ascend_config(vc)
    assert asc_cfg.enable_sparse_sfa_c8 is True
    assert asc_cfg.enable_sparse_li_c8 is True


def test_three_tier_classification_and_tier2_downgrade():
    cfg_fit = HiSparsePrefillConfig(enabled=True, staging_rows=2_097_152)

    # All 6 benchmark scenarios (H in {128K, 256K, 512K} x Q in {256, 1024}) classify as Tier 1 DEMAND
    for h in (131_072, 262_144, 524_288):
        for q in (256, 1024):
            r = _make_req(0, h, q)
            all_res, worst, needs_chk = classify_prefill_requests([r], cfg_fit)
            assert all_res is False
            assert r.tier == TIER_HARD_DEMAND and r.mode == MODE_DEMAND
            assert worst == q + min(h, q * 2048)
            assert needs_chk is False

    # Tier 0, Tier 2 fitting -> FULL, Tier 3 -> FULL
    r_t0 = _make_req(0, 131_072, 256, host_miss_pages=0)
    r_t2_small = _make_req(1, 131_072, 2048)  # 2048 / 131072 = 1.5625% -> Tier 2
    r_t2_large = _make_req(2, 131_072, 4096)  # 4096 / 131072 = 3.125% -> Tier 2
    r_t3_short = _make_req(3, 32_768, 256)  # H_host == 32768 (not > 32768) -> Tier 3
    r_t3_high = _make_req(4, 131_072, 8192)  # 8192 / 131072 = 6.25% -> Tier 3
    all_res, _, _ = classify_prefill_requests([r_t0, r_t2_small, r_t2_large, r_t3_short, r_t3_high], cfg_fit)
    assert all_res is False
    assert (r_t0.tier, r_t0.mode) == (TIER_ON_CARD, MODE_FULL)
    assert (r_t2_small.tier, r_t2_small.mode, r_t2_small.downgraded_from_tier2) == (TIER_ELASTIC, MODE_FULL, False)
    assert (r_t2_large.tier, r_t2_large.mode, r_t2_large.downgraded_from_tier2) == (TIER_ELASTIC, MODE_FULL, False)
    assert (r_t3_short.tier, r_t3_short.mode) == (TIER_FULL_LOAD, MODE_FULL)
    assert (r_t3_high.tier, r_t3_high.mode) == (TIER_FULL_LOAD, MODE_FULL)

    # Under staging pressure, Tier 2 downgrades smallest Q/H_host first; Tier 3 NEVER downgrades
    tight_cfg = HiSparsePrefillConfig(enabled=True, staging_rows=350_000)
    p_t2_a = _make_req(0, 131_072, 2048)  # ratio 0.015625
    p_t2_b = _make_req(1, 131_072, 4096)  # ratio 0.03125
    p_t3 = _make_req(2, 131_072, 8192)  # ratio 0.0625 -> Tier 3
    classify_prefill_requests([p_t2_a, p_t2_b, p_t3], tight_cfg)
    assert p_t2_a.mode == MODE_DEMAND and p_t2_a.downgraded_from_tier2 is True
    assert p_t2_b.mode == MODE_FULL and p_t2_b.downgraded_from_tier2 is False
    assert p_t3.tier == TIER_FULL_LOAD and p_t3.mode == MODE_FULL and p_t3.downgraded_from_tier2 is False


def test_sdma_coalescing_shared_prefix_dedup_and_whole_request_waves():
    # 1. SDMA contiguous coalescing
    page_bytes = BLOCK_SIZE * 656
    entries = (
        SDMAPageEntry(10, 0),
        SDMAPageEntry(11, 1),
        SDMAPageEntry(12, 2),
        SDMAPageEntry(15, 3),  # host_block gap
        SDMAPageEntry(16, 5),  # dst_page gap
    )
    src, dst, sizes = coalesce_sdma_descriptors(1_000_000, 2_000_000, entries, page_bytes=page_bytes)
    assert src.tolist() == [1_000_000 + 10 * page_bytes, 1_000_000 + 15 * page_bytes, 1_000_000 + 16 * page_bytes]
    assert dst.tolist() == [2_000_000 + 0 * page_bytes, 2_000_000 + 3 * page_bytes, 2_000_000 + 5 * page_bytes]
    assert sizes.tolist() == [3 * page_bytes, page_bytes, page_bytes]

    # 2. Shared prefix host_block_ids generate single H2D SDMA & shared staged_page_table slots
    shared_prefix = (101, 102, 103, 104)
    r0 = _make_req(0, 512, 128, host_miss_pages=4, host_blocks=(*shared_prefix, 201), q_batch_start=0)
    r1 = _make_req(1, 512, 128, host_miss_pages=4, host_blocks=(*shared_prefix, 202), q_batch_start=128)
    r0.mode = MODE_FULL
    r1.mode = MODE_FULL
    waves = plan_whole_request_waves([r0, r1], num_resident_groups=1, staging_rows=4096, union_capacities={})
    assert len(waves) == 1
    w0 = waves[0]
    assert [e.host_block_id for e in w0.sdma_entries_by_group[0]] == [101, 102, 103, 104]
    assert w0.staged_page_table[0, :4].tolist() == w0.staged_page_table[1, :4].tolist()
    assert w0.staged_page_table[0, 4] != w0.staged_page_table[1, 4]

    # 3. Two-beat overflow guard & whole-request wave splitting (never splits Q)
    cfg = HiSparsePrefillConfig(enabled=True, staging_rows=2_097_152)
    reqs = [
        _make_req(i, 524_288, 512, q_batch_start=i * 512)
        for i in range(4)  # 4 * worst(524288) = 2,099,200 > 2,097,152
    ]
    all_res, worst_rows, needs_chk = classify_prefill_requests(reqs, cfg)
    assert all_res is False
    assert worst_rows > cfg.staging_rows and needs_chk is True

    # If actual union counts fit in staging_rows -> 1 wave
    single = plan_whole_request_waves(reqs, 1, cfg.staging_rows, {r.demand_slot: 100_000 for r in reqs})
    assert len(single) == 1 and (single[0].req_start, single[0].req_end) == (0, 4)

    # If actual union counts exceed staging_rows -> split into whole-request waves without cutting Q
    tight_waves = plan_whole_request_waves(reqs, 1, cfg.staging_rows, {r.demand_slot: 524_288 for r in reqs})
    assert len(tight_waves) == 2
    assert (tight_waves[0].req_start, tight_waves[0].req_end) == (0, 3)
    assert (tight_waves[0].q_start, tight_waves[0].q_end) == (0, 1536)
    assert (tight_waves[1].req_start, tight_waves[1].req_end) == (3, 4)
    assert (tight_waves[1].q_start, tight_waves[1].q_end) == (1536, 2048)


def test_shadow_tracker_and_mixed_decode_resident_prefill_fix(patched_npu_env):
    # 1. HiSparseResidencyShadowTracker multi-step lifecycle
    tracker = HiSparseResidencyShadowTracker(source_group_id=0, num_resident_groups=2)
    sched1 = NS(
        finished_req_ids=(),
        preempted_req_ids=(),
        scheduled_new_reqs=[
            NS(req_id="dec-0", block_ids=[[10, 11]], num_computed_tokens=256),
            NS(req_id="pre-0", block_ids=[[20, 21, 22]], num_computed_tokens=256),
        ],
        scheduled_cached_reqs=None,
    )
    meta1 = NS(
        residency_updates={
            "dec-0": NS(pages=[0, 1], block_ids=[[0, 5], [0, 6]]),  # decode has H_host > 0
            "pre-0": NS(pages=[0, 1, 2], block_ids=[[7, 8, 9], [17, 18, 19]]),  # prefill 100% resident
        }
    )
    tracker.update_from_step(sched1, meta1)
    assert tracker.host_blocks_by_req["pre-0"] == [20, 21, 22]
    assert tracker.residency_by_req["pre-0"][:, :3].tolist() == [[7, 8, 9], [17, 18, 19]]

    # Cached request extend + resumed overwrite + drop on finish
    sched2 = NS(
        finished_req_ids=["dec-0"],
        preempted_req_ids=(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=NS(
            req_ids=["pre-0"],
            num_computed_tokens=[384],
            new_block_ids=[[[23]]],
            resumed_req_ids=(),
        ),
    )
    tracker.update_from_step(sched2, NS(residency_updates={"pre-0": NS(pages=[3], block_ids=[[10], [20]])}))
    assert "dec-0" not in tracker.host_blocks_by_req
    assert tracker.host_blocks_by_req["pre-0"] == [20, 21, 22, 23]
    assert tracker.residency_by_req["pre-0"][:, 3].tolist() == [10, 20]

    sched3 = NS(
        finished_req_ids=(),
        preempted_req_ids=(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=NS(
            req_ids=["pre-0"],
            num_computed_tokens=[128],
            new_block_ids=[[[30, 31]]],
            resumed_req_ids={"pre-0"},
        ),
    )
    tracker.update_from_step(sched3, NS(residency_updates={"pre-0": NS(pages=[0, 1], block_ids=[[1, 2], [3, 4]])}))
    assert tracker.host_blocks_by_req["pre-0"] == [30, 31]

    # 2. Mixed Decode (all_context_pages_resident=False) + 100%-Resident Prefill -> all_prefill_pages_resident=True
    handle = NS(
        draft_layer=False,
        all_context_pages_resident=False,
        runtime=NS(resident_source_index=0),
    )
    handle2 = NS(
        draft_layer=False,
        all_context_pages_resident=False,
        runtime=NS(resident_source_index=1),
    )
    worker = NS(
        cache_handles=[handle, handle2],
        host_caches=[
            torch.empty((8, 128, 656), dtype=torch.int8),
            torch.empty((8, 128, 656), dtype=torch.int8),
        ],
        resident_caches=[
            torch.empty((32, 128, 656), dtype=torch.int8),
            torch.empty((32, 128, 656), dtype=torch.int8),
        ],
        kv_cache_config=NS(host_group_ids=(0,)),
        cache_layer_names=["layer0", "layer1"],
        backend=NS(is_current_stream_capturing=lambda: False),
    )
    from vllm_ascend.hisparse.prefill_pipeline import HiSparsePrefillWorkspace

    ws = HiSparsePrefillWorkspace(
        device=torch.device("cpu"),
        kv_dtype=torch.int8,
        staging_rows=2048,
        max_num_seqs=8,
        max_batched_tokens=1024,
        max_model_len=2048,
    )
    executor = HiSparsePrefillExecutor(worker, HiSparsePrefillConfig(enabled=True, staging_rows=2_097_152), ws)
    attn_meta = NS(
        hisparse_for_capture=False,
        num_reqs=2,
        num_decodes=1,
        num_prefills=1,
        num_decode_tokens=1,
        num_actual_tokens=129,
        seq_lens_cpu=torch.tensor([256, 384], dtype=torch.int32),
    )
    executor.on_pre_forward(
        meta1,
        NS(
            finished_req_ids=(),
            preempted_req_ids=(),
            scheduled_new_reqs=sched1.scheduled_new_reqs,
            scheduled_cached_reqs=None,
            num_scheduled_tokens={"dec-0": 1, "pre-0": 128},
        ),
        ["dec-0", "pre-0"],
        {"layer0": attn_meta},
    )
    assert handle.all_context_pages_resident is False
    assert all_prefill_pages_resident(handle) is True

    # If the prefill request has H_host > 0, all_prefill_pages_resident returns False
    meta_miss = NS(
        residency_updates={
            "pre-0": NS(pages=[0, 1, 2], block_ids=[[0, 8, 9], [0, 18, 19]]),
        }
    )
    executor.on_pre_forward(
        meta_miss,
        NS(
            finished_req_ids=(),
            preempted_req_ids=(),
            scheduled_new_reqs=[NS(req_id="pre-0", block_ids=[[20, 21, 22]], num_computed_tokens=256)],
            scheduled_cached_reqs=None,
            num_scheduled_tokens={"pre-0": 128},
        ),
        ["pre-0"],
        {
            "layer0": NS(
                hisparse_for_capture=False,
                num_reqs=1,
                num_decodes=0,
                num_prefills=1,
                num_decode_tokens=0,
                num_actual_tokens=128,
                seq_lens_cpu=torch.tensor([384], dtype=torch.int32),
            )
        },
    )
    assert all_prefill_pages_resident(handle) is False


def test_mrv2_preemption_no_forward_finish_forward_and_set_disabled(monkeypatch, patched_npu_env):
    from vllm.distributed.kv_transfer import kv_transfer_state
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
        HiSparseConnector,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
        HiSparseConnectorWorker,
    )
    from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

    from vllm_ascend.hisparse.prefill_pipeline import (
        HiSparsePrefillWorkspace,
        PrefillStepPlan,
        install_hisparse_prefill_hooks,
    )

    handle = NS(draft_layer=False, all_context_pages_resident=True, runtime=NS(resident_source_index=0))
    worker = NS(
        cache_handles=[handle],
        host_caches=[torch.empty((8, 128, 656), dtype=torch.int8)],
        resident_caches=[torch.empty((8, 128, 656), dtype=torch.int8)],
        kv_cache_config=NS(host_group_ids=(0,)),
        cache_layer_names=["layer0"],
        backend=NS(is_current_stream_capturing=lambda: False),
    )
    ws = HiSparsePrefillWorkspace(
        device=torch.device("cpu"),
        kv_dtype=torch.int8,
        staging_rows=2048,
        max_num_seqs=8,
        max_batched_tokens=1024,
        max_model_len=2048,
    )
    ws.prefill_stream = torch.npu.Stream()
    ws.sfa_done_event = torch.npu.Event()
    ws.prefetch_ready_event = torch.npu.Event()
    executor = HiSparsePrefillExecutor(worker, HiSparsePrefillConfig(enabled=True, staging_rows=2_097_152), ws)

    # 1. MRV2 preemption flow: preempted_req_ids drops req-p, then re-admission populates fresh blocks
    executor.on_pre_forward(
        NS(residency_updates={"req-p": NS(pages=[0, 1], block_ids=[[5, 6]])}),
        NS(
            finished_req_ids=(),
            preempted_req_ids=(),
            scheduled_new_reqs=[NS(req_id="req-p", block_ids=[[10, 11]], num_computed_tokens=256)],
            scheduled_cached_reqs=None,
            num_scheduled_tokens={"req-p": 128},
        ),
        ["req-p"],
        {},
    )
    assert executor.tracker.host_blocks_by_req["req-p"] == [10, 11]

    executor.on_pre_forward(
        NS(residency_updates={}),
        NS(
            finished_req_ids=(),
            preempted_req_ids={"req-p"},
            scheduled_new_reqs=[],
            scheduled_cached_reqs=None,
            num_scheduled_tokens={},
        ),
        [],
        {},
    )
    assert "req-p" not in executor.tracker.host_blocks_by_req
    assert "req-p" not in executor.tracker.residency_by_req

    executor.on_pre_forward(
        NS(residency_updates={"req-p": NS(pages=[0], block_ids=[[9]])}),
        NS(
            finished_req_ids=(),
            preempted_req_ids=(),
            scheduled_new_reqs=[NS(req_id="req-p", block_ids=[[42]], num_computed_tokens=128)],
            scheduled_cached_reqs=None,
            num_scheduled_tokens={"req-p": 64},
        ),
        ["req-p"],
        {},
    )
    assert executor.tracker.host_blocks_by_req["req-p"] == [42]
    assert executor.tracker.residency_by_req["req-p"].tolist() == [[9]]

    # 2. Direct finish_forward hook clears pending state and waits prefetch_ready_event
    executor.active_step = PrefillStepPlan(
        step_id=9,
        all_prefill_resident=False,
        requests=[],
        demand_requests=[],
        staging_rows=2048,
        num_resident_groups=1,
        worst_total_rows=0,
        needs_union_count_check=False,
        waves=[],
    )
    executor.prefetched_layer_idx = 0
    executor.prefetched_sdma_pages = {(10, 0)}
    executor.prefetched_has_demand = True
    handle._ascend_all_prefill_pages_resident = False

    install_hisparse_prefill_hooks()
    cw = HiSparseConnectorWorker.__new__(HiSparseConnectorWorker)
    cw._ascend_prefill_executor = executor
    monkeypatch.setattr(HiSparseConnectorWorker, "finish_forward", lambda _self: None, raising=False)

    # Direct finish_forward verification via executor.on_finish_forward and ActiveKVConnector
    prev_waits = len(patched_npu_env.compute_stream.waited_events)
    executor.on_finish_forward()
    assert len(patched_npu_env.compute_stream.waited_events) == prev_waits + 1
    assert patched_npu_env.compute_stream.waited_events[-1] is ws.prefetch_ready_event
    assert executor.active_step is None
    assert executor.prefetched_layer_idx is None
    assert executor.prefetched_sdma_pages == set()
    assert executor.prefetched_has_demand is False
    assert handle._ascend_all_prefill_pages_resident is None

    # 3. Real ActiveKVConnector.no_forward (0-token step) and set_disabled(True) hooks
    executor.tracker.host_blocks_by_req["req-fin"] = [100]
    executor.tracker.host_blocks_by_req["req-pre"] = [101]
    executor.prefetched_layer_idx = 1

    conn = HiSparseConnector.__new__(HiSparseConnector)
    conn.connector_worker = cw
    active_conn = ActiveKVConnector.__new__(ActiveKVConnector)
    active_conn.kv_connector = conn
    active_conn._disabled = False
    active_conn._pending_load_kwargs = None
    monkeypatch.setattr(active_conn, "pre_forward", lambda *_a, **_kw: None)
    monkeypatch.setattr(active_conn, "finish_forward", lambda: None)
    monkeypatch.setattr(active_conn, "post_forward", lambda _f: None)
    monkeypatch.setattr(kv_transfer_state, "_KV_CONNECTOR_AGENT", conn, raising=False)

    waits_before_no_fwd = len(patched_npu_env.compute_stream.waited_events)
    active_conn.no_forward(NS(finished_req_ids={"req-fin"}, preempted_req_ids={"req-pre"}))
    assert "req-fin" not in executor.tracker.host_blocks_by_req
    assert "req-pre" not in executor.tracker.host_blocks_by_req
    assert executor.prefetched_layer_idx is None
    assert len(patched_npu_env.compute_stream.waited_events) == waits_before_no_fwd + 1

    executor.prefetched_layer_idx = 2
    waits_before_disable = len(patched_npu_env.compute_stream.waited_events)
    active_conn.set_disabled(True)
    assert executor.prefetched_layer_idx is None
    assert len(patched_npu_env.compute_stream.waited_events) == waits_before_disable + 1
