# SPDX-License-Identifier: Apache-2.0
import ctypes
import queue
import sys
import time
import types
from collections import Counter, deque
from typing import Any
from unittest.mock import create_autospec

import vllm.envs as envs_vllm
from vllm.v1.worker.gpu import kv_connector as gpu_kvc

from tests.ut.distributed.kv_transfer.selected_kv_prefill import conftest as ct
from vllm_ascend import platform as ascend_platform
from vllm_ascend.ascend_config import SelectedKVConfig
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import ascend_store_connector as asc
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import attention_fence as af
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import pool_worker, selected_pool_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import selected_kv_plan as plan
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import selected_kv_runtime as runtime
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
    LayerBatchBuilder,
)

GVA, kv_cfg, R, pytest, np, torch, ns = ct.HI_GVA, ct.kv_cfg, ct.raises, ct.pytest, ct.np, ct.torch, ct.ns
M, threading, AscendSFASelectedKVImpl = ct.MagicMock, ct.threading, runtime.AscendSFASelectedKVImpl
_tk = lambda n=4: torch.zeros((n, 1, 2048), dtype=torch.int32)
_attn = lambda n, r=1: ns(num_actual_tokens=n, block_size=128, block_table=torch.ones((r, 300), dtype=torch.int32))
_sel = lambda rid="r0", h=16384, q=4, p=128: ct.make_req(rid, h, q, list(range(1, p + 2)), [GVA] * p)
_nat = lambda rid="nat", q=2: ct.make_req(rid, 128, q, [130, 131], [GVA + 10, GVA + 12])
_ops = ("selected_kv_union_plan", "selected_kv_inplace_copy", "selected_kv_li_gather")


def _push(dst: list[Any], val: Any, ret: Any = 0) -> Any:
    dst.append(val)
    return ret


def _setup_rt(monkeypatch, *, layers=2, with_li=False, li_layers=None, reuse=False, **rt_kw):
    exporter, pages = rt_kw.pop("exporter", None), rt_kw.pop("pages", None)
    owners, layer_bytes = rt_kw.pop("owners", None), rt_kw.pop("layer_bytes", None)
    li_set = set(range(layers)) if li_layers is None else set(li_layers)
    pg, n_s = pages or (16384 // kv_cfg.PAGE_ROWS), 2 if reuse else layers
    z = lambda s, dt: [torch.zeros(s, dtype=dt) for _ in range(n_s)]
    sm, sk, ss = z((8, 128, 1, 656), torch.int8), z((8, 128, 128), torch.int8), z((8, 128), torch.float16)
    caches, st = {f"l.{i}": sm[i % n_s] for i in range(layers)}, layer_bytes or (100000 if reuse else 128 * 656)
    load_s = ct.make_shared(np.arange(1, pg + 2, dtype=np.int64), np.full(pg + 1, GVA, dtype=np.int64))
    w = ct.make_worker_stub(8 + pg, layers, load_s, ct.make_shared(np.empty(0, np.int64), np.empty(0, np.int64)))
    w.kv_caches, comps = caches, tuple(plan.MainComponent(f"l.{i}", i, i * st) for i in range(layers))
    d0, d1 = (plan.MAIN_PAGE_BYTES, st - kv_cfg.INDEXER_SCALE_PAGE_BYTES) if layer_bytes else (83968, 100352)
    offs = ((i * st + d0, i * st + d1) if reuse else (0, 16384) for i in range(layers))
    idx = tuple(
        plan.IndexerComponent(o, (sk[i % n_s], ss[i % n_s])) if (with_li and i in li_set) else None
        for i, o in enumerate(offs)
    )
    monkeypatch.setattr(runtime, "discover_components", lambda _w: (comps, idx))
    monkeypatch.setattr("torch.ops._C_ascend", ns(**{n: M() for n in _ops}), raising=False)
    kw: dict[str, Any] = dict(device="cpu", max_num_batched_tokens=256, max_num_seqs=16, min_history_tokens=16383)
    kw |= dict(max_model_len=kv_cfg.MAX_HISTORY_TOKENS, main_caches=caches)
    kw |= dict(owner_by_layer=owners or {f"l.{i}": "l.0" for i in range(layers)})
    kw |= dict(exporter=exporter or (lambda g, *_a: g + np.uint64(0x1000)))
    return runtime.SelectedKVRuntime(w, **(kw | rt_kw)), w, sm if reuse else caches, load_s


def _batch(*reqs, rt=None, order=None):
    reqs_l = list(reqs) or [_sel()]
    hist = [r.load_spec.kvpool_cached_tokens for r in reqs_l]
    idx = list(order) if order is not None else list(range(len(reqs_l)))
    ln = [r.target_token_len - h for r, h in zip(reqs_l, hist)]
    sched = ns(
        kv_connector_metadata=ns(requests=reqs_l), num_scheduled_tokens={r.req_id: n for r, n in zip(reqs_l, ln)}
    )
    qs, hc = np.cumsum([0, *[ln[i] for i in idx]], dtype=np.int32), np.array([hist[i] for i in idx], np.int32)
    inp = ns(req_ids=[reqs_l[i].req_id for i in idx], query_start_loc_np=qs, num_computed_tokens_np=hc)
    if rt is None:
        return sched, inp
    rt.begin_step()
    rt._leased_sizes[GVA] = 1 << 20
    return rt.bind(sched, inp)


def _run_layers(rt, caches, topk, n_toks: int, n_rows: int = 1, layers: int = 2, *, skip_last=False, finish=False):
    attn = _attn(n_toks, n_rows)
    for i in range(layers):
        _ = (skip_last and i == layers - 1) or rt.on_native_layer_ready(i)
        _ = rt.load_layer(f"l.{i}", topk, (caches[f"l.{i}"],), attn), i + 1 < layers and rt.prefetch_next(f"l.{i}")
    return (rt.finish() if finish else None) or attn


def _pool(orig_w=None, rt=None, rm=None, **kw):
    p = object.__new__(selected_pool_worker.SelectedKVPoolWorker)
    d = dict(selected_kv_runtime=rt, async_scheduling=True, _deferred_leases=deque(), _lease_refs=Counter())
    d |= dict(_held_removes=Counter(), _step_load_keys=set(), kv_send_thread=None, kv_recv_thread=None, _closed=False)
    d |= dict(use_layerwise=True, use_layerwise_transfer=True, use_kvpp=False, num_kv_cache_groups=1, use_eagle=False)
    d |= dict(cacheable_group_ids={0}, grouped_block_size=[128], hash_block_size=128, layerwise_offload=True)
    d |= dict(layerwise_backend=ns(fence_drains_recv=lambda: True, send_fence_backlog=lambda: 0))
    d |= dict(cache_coordinator=None, kv_role="kv_producer", _previous_save_batch=None)
    d |= dict(_save_disabled=threading.Event())
    d |= dict(_masked_runs_cache={}, use_block_key_layerwise=0, block_key_hybrid=0, next_layer_to_submit=0)
    d |= dict(external_slot_release_waiter=None, independent_layers=[], physical_layer_to_group_layers={})
    d |= dict(
        _snapshot_keys={},
        _failed_requests=set(),
        _retired_store_req_ids=set(),
        _put_started_keys=set(),
        _layer_load_aborted=threading.Event(),
        has_shared_kv=False,
        _global_to_local_layer={},
    )
    d |= dict(_dflash_draft_start_layer=None, _dflash_save_pending=False, _dspark_draft_layer_names=set())
    d |= dict(dflash_draft_layer_names=frozenset(), layerwise_key_layer_offset=0)
    d |= dict(layerwise_key_layers=kw.get("num_layers", getattr(orig_w, "num_layers", 4)))
    d |= dict(consumer_is_to_put=False, save_decode_cache=False, put_step=1, tp_size=1, tp_rank=0, dcp_size=1)
    d |= dict(
        group_uses_align_state=[False],
        _reused_layers_by_group={0: set(range(kw.get("num_layers", getattr(orig_w, "num_layers", 4))))},
    )
    z = lambda *_a, **_k: None
    d |= dict(_compute_reachable_load_masks=z, _drain_deferred_last_save=z)
    d |= dict(_compute_reachable_store_masks=z, _process_save_for_layer_batch=z, _alloc_gvas_for_save=z)
    d |= dict(_build_shared_save_data=z)
    d |= dict(_make_layerwise_full_key=lambda gid, bh: f"kl_{gid}_{bh[:4]}")
    p.__dict__.update((orig_w.__dict__ if orig_w else {}) | d | kw)
    p._data_plane = selected_pool_worker.SelectedGvaDataPlane(p)
    p.m_store = kw.get("m_store") or ns(batch_remove_lease=rm or M(return_value=0), close=M())

    def _set_waiter(w: Any) -> bool:
        p.external_slot_release_waiter = w
        return True

    if not isinstance(p.kv_recv_thread, KVCacheStoreLayerRecvingThread):
        p.set_external_slot_release_waiter = _set_waiter
    return p


def _setup_start_env(monkeypatch, sm=None, *, with_device_patches=False):
    from vllm_ascend.device.hardware import AscendDeviceType

    names, mp, cls = (
        [f"model.layers.{i}.self_attn.attn" for i in range(4)],
        monkeypatch.setattr,
        AscendSFASelectedKVImpl,
    )
    specs, kcfg = ct.make_specs_and_cfg(4, (0, 2))[2:]
    vcfg_p = ct.make_vllm_config(
        **{
            "model_config.max_model_len": 16384,
            "scheduler_config.max_num_batched_tokens": 256,
            "scheduler_config.max_num_seqs": 16,
        }
    )
    ly = plan.probe_reuse_layout(vcfg_p, specs, num_blocks=8)
    raw = torch.zeros(ly.kv_cache_tensors[0].size + kv_cfg.NATIVE_ALIGNMENT, dtype=torch.int8)
    m_off, li_k, li_s = 8 * 83968, 8 * 16384, 8 * 256
    sm = sm or [raw[i * m_off : (i + 1) * m_off].view(8, 128, 1, 656) for i in range(2)]
    li_b = 2 * m_off
    li_p = [
        (
            raw[li_b + s * (li_k + li_s) : li_b + s * (li_k + li_s) + li_k].view(8, 128, 1, 128),
            raw[li_b + s * (li_k + li_s) + li_k : li_b + (s + 1) * (li_k + li_s)]
            .view(torch.float16)
            .view(8, 128, 1, 1),
        )
        for s in range(2)
    ]
    ctx: dict[str, Any] = {}
    for i, n in enumerate(names):
        im = cls.__new__(cls)
        im.kv_lora_rank, im.qk_rope_head_dim = 512, 64
        im.skip_topk, im.indexer = bool(i % 2), None if i % 2 else ns(topk_output_width=2048)
        ctx[n] = ns(kv_cache=(sm[i % 2],), impl=im)
        ctx[f"model.layers.{i}.self_attn.indexer.k_cache"] = ns(
            kv_cache=li_p[i // 2] if i in (0, 2) else [torch.tensor([], dtype=torch.int8)]
        )
    comps = tuple(plan.MainComponent(n, i, i * 100000) for i, n in enumerate(names))
    fb = plan.SelectedWorkspaceSpec.from_bounds(256, 16, 16384).fixed_bytes
    tot_b = raw.nbytes
    avail = tot_b + fb + 8 * kv_cfg.OBJECT_BASE_BYTES_PER_BLOCK + kv_cfg.NATIVE_ALIGNMENT + 4096
    pw = ns(
        selected_kv_runtime=None,
        _layerwise_reuse_layout=ly,
        kv_cache_config=kcfg,
        num_layers=4,
        _as_cache_tuple=lambda v: v if isinstance(v, (tuple, list)) else (v,),
        num_blocks=8,
    )
    pw.attach_selected_runtime = lambda r: setattr(pw, "selected_kv_runtime", r)
    v_spy, orig_e = M(), torch.empty
    mp(plan, "get_ascend_device_type", lambda: AscendDeviceType.A3)
    mp(runtime, "get_kv_transfer_group", lambda: ns(connector_worker=pw))
    mp(runtime, "discover_components", lambda _w: (comps, (None,) * 4))
    mp("torch.ops._C_ascend", ns(**{n: M() for n in _ops}), raising=False)
    mp(runtime.pl, "load_default_exporter", lambda _w=None: (lambda g, *_a: g + np.uint64(0x1000)))
    mp(runtime.SelectedKVRuntime, "_verify_kernels", v_spy)
    if with_device_patches:
        mp(runtime.torch, "empty", lambda s, **kw: orig_e(s, **{k: v for k, v in kw.items() if k != "device"}))
        mp(runtime.torch, "device", lambda d: ns(type="npu" if d == "npu" else "cpu"))
    acfg = ns(
        selected_kv_config=SelectedKVConfig(enabled=True),
        enable_sparse_sfa_c8=True,
        enable_sparse_li_c8=True,
        enable_sparse_li_c4=False,
        kvpp_config=ns(size=1),
        sparse_kv_offload_config=ns(enabled=False),
        xlite_graph_config=ns(enabled=False),
    )
    ws_r = ns(
        model_config=vcfg_p.model_config,
        parallel_config=vcfg_p.parallel_config,
        vllm_config=vcfg_p,
        compilation_config=ns(static_forward_context=ctx),
    )
    ws_w = ns(
        ascend_config=acfg,
        vllm_config=vcfg_p,
        cache_config=vcfg_p.cache_config,
        model_runner=ws_r,
        available_kv_cache_memory_bytes=avail,
        get_kv_cache_spec=lambda: specs,
    )
    plan.reserve_selected_kv_workspace(ws_w)
    budget = ws_r.selected_kv_budget
    nc = {f"{a}_config": None for a in ("cache", "observability", "compilation", "offload", "attention", "kv_transfer")}
    ov = {"model_config.max_model_len": 16384, "scheduler_config.max_num_batched_tokens": 256}
    ov |= {"scheduler_config.max_num_seqs": 16, "parallel_config.enable_eplb": False}
    vcfg = ct.make_vllm_config(**(ov | nc))

    def mk_runner(scfg, dev="cpu"):
        pw.selected_kv_runtime = None
        rk = dict(device=dev, ascend_config=ns(selected_kv_config=scfg), selected_kv_budget=budget)
        rk |= dict(compilation_config=ns(static_forward_context=ctx), vllm_config=vcfg, kv_connector=M())
        return ns(**rk)

    return names, ctx, budget, tot_b, pw, vcfg, v_spy, mk_runner


def _wait_t(ignore_stop: bool = False):
    ev, ent = threading.Event(), threading.Event()
    q: queue.Queue[Any] = queue.Queue()

    def _run():
        ent.set()
        ev.wait(5.0)
        q.task_done()

    t: Any = threading.Thread(target=_run, daemon=True)
    q.put(object())
    t.entered, t.request_queue, t._ev = ent, q, ev
    t.raise_if_failed, t.stop = (lambda: None), (lambda: None if ignore_stop else ev.set())
    return t


def test_precommit_mixed_li_and_permutations(patched_npu_env, monkeypatch):
    boom = M(side_effect=RuntimeError("non-contiguous HyBM slab boundary"))
    rt, w, caches, ls_rt = _setup_rt(monkeypatch, exporter=boom, pages=256)
    t0, (sched, inp) = w.layer_load_tasks[0][0], _batch()
    oids, li = t0.shared_block_data.block_ids_arr.copy(), (lambda a, s=1 << 20: ns(gva_list=lambda: a, size=lambda: s))
    cs = (([0, GVA], 1 << 20, {"r0": 4}), ([GVA], 64, {"r0": 4}), ([GVA], 1 << 20, {"bad": 4}))
    for i, (addrs, sz, toks) in enumerate(cs):
        _ = rt.begin_step(), rt.record_leased_infos(["k"], [li(addrs, sz)])
        assert not rt.bind(ns(kv_connector_metadata=sched.kv_connector_metadata, num_scheduled_tokens=toks), inp)
        assert (rt.step, rt._leased_sizes, w.layer_load_tasks[0][0]) == (None, {}, t0) and (i or boom.call_count == 1)
        rt.exporter = lambda gvas, *_a: gvas + np.uint64(0x1000)
    rt.record_leased_infos(["k0", "k1"], [li([GVA])])
    assert not rt._lease_invalid and rt.begin_step() is None and (t0.shared_block_data.block_ids_arr == oids).all()
    rt.record_leased_infos(["k"], [li([0, GVA, plan.MEMCACHE_SSD_HIT_GVA])])
    assert rt._leased_sizes == {GVA: 1 << 20} and rt.record_leased_infos(["k0", "k1"], [li([GVA])]) is None
    assert rt._lease_invalid and rt.bind(sched, inp) is False
    m_args = (_nat("nat", 3), _sel("sa", 16384, 5, 128), _sel("sb", 32768, 2, 256))
    assert _batch(*m_args, rt=rt, order=[2, 0, 1])
    R(ValueError, "block_table", rt.load_layer, "l.0", _tk(10), (caches["l.0"],), _attn(10, 1))
    _ = rt.finish(failed=True), rt.take_step_fence()
    w.layer_load_tasks = [[ct.make_task(j, ls_rt, ["r"])] for j in range(2)]
    assert not rt._aborted and _batch(*m_args, rt=rt, order=[2, 0, 1])
    _run_layers(rt, caches, _tk(10), 10, 3, finish=True)
    up, ic = torch.ops._C_ascend.selected_kv_union_plan, torch.ops._C_ascend.selected_kv_inplace_copy
    assert up.call_args.args[3:] == ([32768, 16384], [0, 5], [2, 5], rt.capacity)
    assert ic.call_args.args[3].tolist() == [[0, 32768, 0, 2], [2, 16384, 5, 5]]
    prev = patched_npu_env.sync.call_count
    rt.begin_step()
    assert rt._completed_fence is None and patched_npu_env.sync.call_count > prev
    assert rt.close() is None
    rc = _setup_rt(monkeypatch, max_model_len=16384, pages=256)[0]
    assert (rc.capacity, _batch(_sel("rc", 32768, 16, 256), rt=rc), rc.step) == (16384, False, None)
    assert rc.close() is None


def test_connector_pool_worker_leases_and_refcount(patched_npu_env, monkeypatch):
    monkeypatch.setattr(runtime.torch.npu, "Event", lambda: ct.FakeEvent(ready=False), raising=False)
    rt, w, caches, orig_s = _setup_rt(monkeypatch)
    inner, fences, slots = create_autospec(gpu_kvc.KVConnector, instance=True), [], []
    conn = runtime.SelectedKVConnector(inner, rt)
    for i in range(3):
        w.layer_load_tasks = [[ct.make_task(j, orig_s, ["r"])] for j in range(2)]
        assert _batch(_sel(f"r{i}"), rt=rt)
        slots.append(rt.step.slot)
        _run_layers(rt, caches, _tk(4), 4)
        conn.finish_forward()
        ok, ev = rt.take_step_fence()
        assert ok and isinstance(ev, ct.FakeEvent) and (i >= 2 or ev.synchronized == 0)
        fences.append(ev)
    assert (slots, fences[0].synchronized, fences[1].synchronized) == ([0, 1, 0], 1, 0)
    conn.post_forward({"r0"}, defer_finalize=True)
    conn.finalize_step()
    assert rt.close() is None
    inner.post_forward.assert_called_once_with({"r0"}, defer_finalize=True)
    inner.finalize_step.assert_called_once()

    # Exercise real SelectedGvaDataPlane._prepare_load_gvas_for_request -> SelectedKVPoolWorker._record_leased_infos.
    rm, calls, lw, mp = M(return_value=0), [], pool_worker.LayerwiseKVPoolWorker, monkeypatch.setattr
    inst = _pool(rm=rm, current_layer=0, num_layers=6, layer_load_tasks=[[], [1], []])
    inst._drain_attention_transfers = (dr := M())
    rm_ks = lambda s: s._step_load_keys and s.m_store.batch_remove_lease(list(s._step_load_keys))
    mp(lw, "_submit_ready_layer_loads", lambda s: setattr(s, "next_layer_to_submit", len(s.layer_load_tasks)))
    mp(lw, "get_finished", lambda s, _f, _m: (rm_ks(s), s._step_load_keys.clear(), (set(), set()))[2])
    mp(lw, "shutdown", lambda _s: calls.append("base.shutdown"))

    def_l, rt_s = [], ns(step=ns(lock=threading.RLock()), on_slot_released=lambda _l: None, next_fence=(True, None))
    rt_s.record_leased_infos, rt_s.take_step_fence = M(), (lambda: rt_s.next_fence)
    rt_s.defer_indexer, rt_s.close = (lambda _s, i, c: def_l.append((i, c))), (lambda: calls.append("rt.close"))
    inst.attach_selected_runtime(rt_s)
    inst._submit_ready_layer_loads()
    info = ns(gva_list=lambda: [GVA], size=lambda: 8192)
    req_real = ct.make_req("r_real", 128, 4, [10], [0], skip_tokens=128)
    req_real.block_hashes, req_real.block_ids_np = [b"\xab" * 32], np.array([10], np.int64)
    inst.m_store.batch_get_key_info = M(return_value=[info])
    inst.m_store.batch_add_lease = M(return_value=[0])
    assert def_l == [(1, 0), (2, 0)]
    assert inst._data_plane._prepare_load_gvas_for_request([req_real]) is None
    assert rt_s.record_leased_infos.call_args.args == (["kl_0_abab"], [info])

    # None or length-mismatched leased_infos must invalidate leases on a real runtime and block bind().
    rt_inv, w_inv, _, _ = _setup_rt(monkeypatch)
    b_get = M(side_effect=[[info], None, [info], []])
    w_inv.m_store = ns(batch_get_key_info=b_get, batch_add_lease=M(return_value=[0]))
    pw_inv, (sched_i, inp_i) = _pool(w_inv, rt_inv, m_store=w_inv.m_store), _batch()
    for _ in ("none_infos", "short_infos"):
        rt_inv.begin_step()
        assert pw_inv._data_plane._prepare_load_gvas_for_request([req_real]) is False
        assert rt_inv._lease_invalid and rt_inv.bind(sched_i, inp_i) is False
    rt_inv.close()

    def gf(fn, ks, ready=None):
        if ready is not None:
            ready.ready = True
        rt_s.next_fence, inst._step_load_keys = fn, set(ks)
        inst.get_finished(set(), M())

    evs = [ct.FakeEvent(ready=False) for _ in range(3)]
    inst.async_scheduling = False
    gf((True, evs[0]), {"ks"})
    inst.async_scheduling = True
    gf((True, evs[1]), {"k_sh"})
    gf((True, evs[2]), {"k_sh"})
    assert (evs[0].synchronized, evs[1].synchronized, evs[2].synchronized, len(inst._deferred_leases)) == (1, 1, 0, 1)
    dr.reset_mock()
    inst.kv_recv_thread = ns(request_queue=ns(unfinished_tasks=1))
    gf((True, None), {"k_sh", "k_o2"})
    dr.assert_called_once_with(full=True)
    dr.reset_mock()
    gf((False, None), {"k_ab"}, ready=evs[2])
    dr.assert_not_called()

    # Clean shutdown drains one pending deferred k_shut plus active _step_load_keys k_shut (count == 2).
    inst.kv_recv_thread = None
    gf((True, ct.FakeEvent(ready=False)), {"k_shut"})
    inst._step_load_keys = {"k_shut"}
    inst.shutdown()
    rem = [k for c in rm.call_args_list for k in c.args[0]]
    assert [rem.count(k) for k in ("ks", "k_o2", "k_sh", "k_ab", "k_shut")] == [1, 1, 3, 1, 2]
    assert calls[-2:] == ["rt.close", "base.shutdown"] and not inst._lease_refs and not inst._held_removes

    # Store failure during shutdown remove is swallowed cleanly after rt.close -> base.shutdown.
    calls.clear()
    inst_err = _pool(rt=rt_s, m_store=ns(batch_remove_lease=M(side_effect=RuntimeError("store down")), close=M()))
    inst_err._step_load_keys = {"k_err"}
    inst_err.shutdown()
    assert calls[-2:] == ["rt.close", "base.shutdown"] and not inst_err._lease_refs and not inst_err._held_removes


def test_connector_pre_forward_end_to_end_and_abort_paths(patched_npu_env, monkeypatch):
    rt, ow, _, _ = _setup_rt(monkeypatch, layers=2, pages=128)
    rt.exporter = (exporter_spy := M(side_effect=lambda g, *_a: g + np.uint64(0x1000)))
    info = ns(gva_list=lambda: [GVA], size=lambda: 1 << 20)
    m_store = ns(batch_get_key_info=M(return_value=[info] * 128), batch_add_lease=M(return_value=[0] * 128))
    m_store.batch_remove_lease, m_store.close = M(return_value=0), M()
    recv_stub = object.__new__(pool_worker.KVCacheStoreLayerRecvingThread)
    recv_stub.load_disabled = threading.Event()
    recv_stub.request_queue = queue.Queue()
    recv_stub.raise_if_failed = lambda: None
    recv_stub.stop = lambda: None
    recv_stub.is_alive = lambda: False
    recv_stub.build_shared_data = lambda _t: ct.make_shared(np.arange(1, 129, dtype=np.int64), np.full(128, GVA))
    pw = _pool(ow, rt, m_store=m_store, num_layers=2, kv_recv_thread=recv_stub)
    pw._make_layerwise_full_key = lambda g, b: f"k_{g}_{b[:4]}"
    rt.worker = pw
    asc_conn: Any = object.__new__(asc.AscendStoreConnector)
    asc_conn.__dict__ |= dict(use_layerwise=True, connector_worker=pw, _loaded_layer=None, _connector_metadata=None)
    nil = ("get_block_ids_with_load_errors", "get_kv_connector_stats", "get_kv_connector_kv_cache_events")
    m0 = ("register_kv_caches", "set_host_xfer_buffer_ops", "finish_forward", "wait_for_save")
    for k in (*(m1 := (*nil, "build_connector_worker_meta")), *m0, "clear_connector_metadata", "reset_capture_state"):
        setattr(asc_conn, k, M(return_value=None if k in m1 else M()))
    tr_ok = lambda *_: ns(finished_sending=set(), finished_recving=set(), failed_recving=set())
    asc_conn.get_transfer_results = tr_ok
    patches = (("get_kv_transfer_group", lambda: asc_conn), ("is_forward_context_available", lambda: True))
    for attr, val in (*patches, ("get_forward_context", lambda: None)):
        monkeypatch.setattr(gpu_kvc, attr, val)
    monkeypatch.setattr(gpu_kvc.kv_transfer_state, "_KV_CONNECTOR_AGENT", None)
    conn = runtime.SelectedKVConnector((active := gpu_kvc.ActiveKVConnector(ct.make_vllm_config(), ow.kv_caches)), rt)
    req = _sel("r_e2e", h=16384, q=4, p=128)
    req.block_hashes, req.block_ids_np = [bytes([i]) * 32 for i in range(128)], np.arange(1, 130, dtype=np.int64)
    s_kw = dict(kv_connector_metadata=ns(requests=[req], has_sync_kv_loads=False), num_scheduled_tokens={"r_e2e": 4})
    s_kw |= dict(has_sync_kv_loads=False, finished_req_ids=set())
    i_kw = dict(req_ids=["r_e2e"], query_start_loc_np=np.array([0, 4], np.int32))
    i_kw |= dict(num_computed_tokens_np=np.array([16384], np.int32), idx_mapping=np.array([0], np.int32), num_tokens=4)
    sched, inp = ns(**s_kw), ns(**i_kw)
    # Full pre_forward chain keeps _accepting_leases open through bind_connector_metadata, then commits step.
    conn.pre_forward(sched, inp)
    assert rt.step is not None and not rt._accepting_leases and exporter_spy.call_count == 1
    assert rt.step.l01_native_sdma == (False, False) and pw.layer_load_tasks[0] == []
    t_l1 = pw.layer_load_tasks[1]
    assert len(t_l1) == 1 and t_l1[0].block_ranges == [] and t_l1[0].shared_block_data is None

    conn.no_forward(sched)
    assert rt.step is None and not rt._aborted and not rt._accepting_leases and rt._leased_sizes == {}
    rt.begin_step()
    rt._leased_sizes[GVA] = 8192
    conn.reset_capture_state()
    assert not rt._accepting_leases and rt._leased_sizes == {} and asc_conn.reset_capture_state.called

    monkeypatch.setattr(active, "pre_forward", M(side_effect=RuntimeError("pre boom")))
    R(RuntimeError, "pre boom", conn.pre_forward, sched, inp)
    assert not rt._accepting_leases and rt._leased_sizes == {}
    rt.begin_step()
    rt._leased_sizes[GVA] = 4096
    conn.set_disabled(True)
    assert not rt._accepting_leases and rt._leased_sizes == {} and active._disabled is True
    rt.close()


def test_abort_and_shutdown_thread_watchdog_and_retry(patched_npu_env, monkeypatch):
    order: list[Any] = []
    topk4 = _tk(4)
    patched_npu_env.sync.side_effect = lambda: order.append("sync")
    for is_shut in (False, True):
        order.clear()
        rt_x, ow, c_x, _ = _setup_rt(monkeypatch)
        recv_x = _wait_t()
        tripped: list[int] = []
        rm_fn = lambda ks, s=is_shut: _push(order, ("lease", list(ks))) if s else 0
        close_fn = lambda s=is_shut: _push(order, "close", None) if s else None
        ms = ns(batch_remove_lease=rm_fn, close=close_fn)
        bk = ns(fence_drains_recv=lambda: True, send_fence_backlog=lambda: 0)
        pool_x = _pool(ow, rt_x, _step_load_keys={"k_step"}, kv_recv_thread=recv_x, layerwise_backend=bk, m_store=ms)
        pool_x._check_hybrid_load_errors, rt_x.worker = (lambda: None), pool_x
        recv_x.start()
        wd = threading.Timer(2.0, lambda r=recv_x, t=tripped: (r.stop(), _push(t, 1)))
        wd.daemon = True
        try:
            assert recv_x.entered.wait(timeout=2.0) and _batch(rt=rt_x)
            rt_x.load_layer("l.0", topk4, (c_x["l.0"],), _attn(4))
            pool_x.next_layer_to_submit = 3
            t0 = time.monotonic()
            rt_x.finish(failed=True)
            wd.start()
            if is_shut:
                pool_x.shutdown()
                assert not recv_x.is_alive() and pool_x._closed and order == ["sync", ("lease", ["k_step"]), "close"]
            else:
                assert (rt_x.take_step_fence(), recv_x.is_alive(), rt_x._poisoned) == ((False, None), False, True)
                R(RuntimeError, "poisoned", rt_x.begin_step)
            assert time.monotonic() - t0 < 5.0 and not tripped
        finally:
            wd.cancel()
            recv_x.stop()
            recv_x.join(2.0)
    rt2, w2, _, _ = _setup_rt(monkeypatch)
    w2._drain_attention_transfers, rt2._aborted = M(side_effect=[RuntimeError("drain fail"), None]), True
    R(RuntimeError, "drain fail", rt2.take_step_fence)
    assert rt2._aborted and rt2.take_step_fence() == (False, None) and not rt2._aborted
    assert w2._drain_attention_transfers.call_count == 2 and rt2.close() is None
    bs, br = ns(batch_remove_lease=M(return_value=0), close=M()), ns(close=M(side_effect=RuntimeError("close boom")))
    R(RuntimeError, "close boom", runtime.stop_selected_kv, runner := ns(_selected_kv_runtime=br))
    df, lr = deque([(ct.FakeEvent(), ["k_def"])]), Counter({"k_def": 1})
    p3 = _pool(rt=br, m_store=bs, _deferred_leases=df, _lease_refs=lr, _step_load_keys={"k_l"})
    R(RuntimeError, "close boom", p3.shutdown)
    assert runner._selected_kv_runtime is br and (bs.batch_remove_lease.call_count, p3._closed) == (0, False)
    stubborn = _wait_t(ignore_stop=True)
    stubborn.start()
    assert stubborn.entered.wait(timeout=2.0)
    try:
        prev_z, (rt_z, w_z, c_z, _) = patched_npu_env.sync.call_count, _setup_rt(monkeypatch)
        w_z.kv_recv_thread, stop_w = stubborn, runtime.SelectedKVRuntime._stop_worker_threads
        rt_z._stop_worker_threads = lambda *, timeout=0.05: stop_w(rt_z, timeout=timeout)
        assert _batch(rt=rt_z) and rt_z.load_layer("l.0", topk4, (c_z["l.0"],), _attn(4)) is None
        w_z.next_layer_to_submit = 2
        rt_z.finish(failed=True)
        t0 = time.monotonic()
        R(RuntimeError, "KV transfer thread did not exit", rt_z.take_step_fence)
        assert time.monotonic() - t0 < 1.0 and (rt_z._poisoned, patched_npu_env.sync.call_count) == (True, prev_z)
        pool_z = _pool(kv_recv_thread=stubborn, _step_load_keys={"k_z"}, m_store=bs)
        R(RuntimeError, "KV transfer thread did not exit", pool_z.shutdown, join_timeout=0.05)
        assert (bs.batch_remove_lease.call_count, bs.close.call_count) == (0, 0)
    finally:
        stubborn._ev.set()
        stubborn.join(2.0)


def test_four_layer_colocated_reuse(patched_npu_env, monkeypatch):
    rt, ow, sm, _ = _setup_rt(monkeypatch, layers=4, with_li=True, pages=128, reuse=True)
    caches, pg = ow.kv_caches, 128
    sub: list[Any] = []
    emp = ct.make_shared(np.empty(0, np.int64), np.empty(0, np.int64))
    l_mix = ct.make_shared(np.array([*range(1, pg + 1), 130, 131], np.int64), np.full(pg + 2, GVA, np.int64))
    lt = lambda sh, rq: [[ct.make_task(i, sh, rq)] for i in range(4)]
    p_kw = dict(use_layerwise_transfer=True, num_layers=4, num_prefetch_layers=2, prefetch_layer_map={2: (0,), 3: (1,)})
    p_kw |= dict(layer_load_tasks=lt(l_mix, ["sel", "nat"]), layer_save_tasks=lt(emp, ["sel"]), current_layer=0)
    pool: Any = _pool(ow, rt, kv_recv_thread=ns(add_request=sub.append), **p_kw)
    rt.worker = pool
    pool.attach_selected_runtime(rt)
    t6, a6 = _tk(6), _attn(6, 2)
    ld = lambda i: (rt.on_native_layer_ready(i), rt.load_layer(f"l.{i}", t6, (caches[f"l.{i}"],), a6))
    assert _batch(_sel("sel"), _nat(), rt=rt) and af.reset_attention_gates()
    pool._submit_ready_layer_loads()
    assert (pool.next_layer_to_submit, [t.layer_id for t in sub]) == (3, [0, 1, 2])
    assert bool(sub[1].load_gate) and rt.step.indexer_dep == {1: 0, 2: 0}
    _ = rt.on_slot_released(0), rt.on_native_layer_ready(0), af.record_attention_compute_start()
    assert sub[1].load_gate.wait(timeout=1.0) and rt.on_slot_released(1) is None
    assert 1 in rt.step.pending_indexer and rt.load_layer("l.0", t6, (caches["l.0"],), a6) is None
    assert 1 in rt.step.indexer_events and 1 not in rt.step.pending_indexer and rt.prefetch_next("l.0") is None
    pool.current_layer = 1
    _ = pool._submit_ready_layer_loads(), ld(1), rt.prefetch_next("l.1")
    assert 2 in rt.step.pending_main and rt.step.prefetched["l.2"] is None
    assert rt.on_slot_released(2) is None and 2 not in rt.step.pending_main and rt.step.prefetched["l.2"] is not None
    _ = ld(2), rt.prefetch_next("l.2"), rt.on_slot_released(3), ld(3), rt.finish(), rt.take_step_fence()
    # Pure selected-only 4-layer step completes cleanly, then missing-LI step fails at finish().
    pool.layer_load_tasks = lt(ct.make_shared(l_mix.block_ids_arr[:pg], l_mix.block_gvas_arr[:pg]), ["sel"])
    pool.current_layer = pool.next_layer_to_submit = 0
    assert _batch(_sel("sel"), rt=rt)
    pool._submit_ready_layer_loads()
    _ = [rt.on_slot_released(i) for i in range(4)], _run_layers(rt, caches, _tk(4), 4, 1, 4, finish=True)
    ok_f, ev_f = rt.take_step_fence()
    assert ok_f and ev_f is not None and rt.step is None

    # Verify num_prefetch_layers=1 schedules [0, 1] with indexer_dep {1: 0} (full step covered in recving_thread[1]).
    sub1: list[Any] = []
    p1_kw = p_kw | dict(num_prefetch_layers=1, layer_load_tasks=lt(l_mix, ["sel"]))
    pool1: Any = _pool(ow, rt, kv_recv_thread=ns(add_request=sub1.append), **p1_kw)
    pool1.selected_kv_runtime, rt.worker = rt, pool1
    assert _batch(_sel("sel"), rt=rt)
    pool1._submit_ready_layer_loads()
    assert [t.layer_id for t in sub1] == [0, 1] and rt.step.indexer_dep == {1: 0}
    pool1.kv_recv_thread = None
    rt.abort_step()
    rt.worker = pool

    pool.layer_load_tasks = lt(ct.make_shared(l_mix.block_ids_arr[:pg], l_mix.block_gvas_arr[:pg]), ["sel"])
    pool.current_layer = pool.next_layer_to_submit = 0
    assert _batch(_sel("sel"), rt=rt)
    pool._submit_ready_layer_loads()
    for i in range(4):
        rt.on_slot_released(i)
    _run_layers(rt, caches, _tk(4), 4, 1, 4, skip_last=True)
    # Test host-sim kernel self-test and corruption guards in an isolated runtime setup.
    step = rt.step
    R(RuntimeError, "not joined by every consumer", rt.finish)
    assert step.failed and step.pending_main == {}
    pool.kv_recv_thread = None
    rt.close()


def test_host_sim_selftest_and_corruption_guards(patched_npu_env, monkeypatch):
    rt, ow, sm, _ = _setup_rt(monkeypatch, layers=4, with_li=True, pages=128, reuse=True)
    sim_ops, sim_lib = ct.load_host_sim()
    sim_lib.selected_kv_sim_reset()
    try:
        src_t, main_ref, _, _ = runtime.build_selftest_source()
        nb = len(src_t)
        cb_t = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64))
        id_cb = cb_t(lambda g, _m, o: o.__setitem__(0, g) or 0)
        p_t = torch.tensor([src_t.data_ptr()], dtype=torch.uint64)
        sim_lib.selected_kv_sim_set_mapper(ctypes.cast(id_cb, ctypes.c_void_p).value)
        base_va = int(sim_ops.selected_kv_export_bases(p_t, torch.tensor([nb], dtype=torch.uint64), nb)[0].item())
        assert rt._verify_kernels(ops=sim_ops, base_va=base_va) is None
        assert sim_lib.selected_kv_sim_trap_count() == 0

        # Mutate Host-Sim outputs/offsets to verify self-test catches wrong source row, dirty row, or bad LI offset.
        b_ops = {n: getattr(sim_ops, n) for n in _ops}

        def _bad_copy(fn):
            return lambda *a: (sim_ops.selected_kv_inplace_copy(*a), fn(a[5]))

        ic_op, li_op, st_fail = _ops[1], _ops[2], "self-test failed"
        f_r, f_d = (
            _bad_copy(lambda c: c[0, 3, 0].copy_(main_ref[0])),
            _bad_copy(lambda c: c.__setitem__((0, 2, 0, 0), 1)),
        )
        f_li = lambda b, k, s, d, _s: sim_ops.selected_kv_li_gather(b, k, s, 0, d)
        for op_name, fn, pat in ((ic_op, f_r, st_fail), (ic_op, f_d, st_fail), (li_op, f_li, st_fail)):
            R(RuntimeError, pat, rt._verify_kernels, ops=ns(**{**b_ops, op_name: fn}), base_va=base_va)
        R(
            RuntimeError,
            "Invalid original-page",
            rt._verify_kernels,
            ops=ns(**{**b_ops, _ops[0]: lambda *_a, **_k: None}),
            base_va=base_va,
        )
    finally:
        sim_lib.selected_kv_sim_reset()
        rt.close()


def test_start_selected_kv_timeout_warnings(patched_npu_env, monkeypatch):
    names, ctx, budget, tot_b, pw, vcfg, v_spy, mk_runner = _setup_start_env(monkeypatch, with_device_patches=True)
    warn_spy = M()
    monkeypatch.setattr(runtime.logger, "warning", warn_spy)
    runner = mk_runner(SelectedKVConfig(enabled=True))
    key, snap = "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", dict(vars(envs_vllm))
    monkeypatch.setenv("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "5")
    try:
        cs = [("240", "mp", 1, 3005.0, "is >= 300s"), ("10", "uni", 0, 15.0, "on uni"), ("240", "mp", 0, None, None)]
        for i, (val, bk, fix, exp_s, sub) in enumerate(cs):
            vars(envs_vllm).pop(key, None)
            monkeypatch.setenv(key, val)
            warn_spy.reset_mock()
            vcfg.parallel_config.distributed_executor_backend = bk
            if fix:
                ascend_platform._fix_incompatible_config(vcfg)
            pw.selected_kv_runtime, runner.kv_connector, runner._selected_kv_runtime = None, M(), None
            runner.device = "cpu" if i else "npu"
            runtime.start_selected_kv(runner, pw.kv_cache_config)
            assert isinstance(runner.kv_connector, runtime.SelectedKVConnector) and v_spy.call_count == 1
            assert all(ctx[n].impl.selected_kv_runtime is runner._selected_kv_runtime for n in names)
            assert (exp_s is None) == (not warn_spy.called)
            if a := warn_spy.call_args.args if exp_s is not None else None:
                assert "MemCache read leases" in a[0] and a[1] == pytest.approx(exp_s) and sub in a[3]
            runtime.stop_selected_kv(runner)
            assert runner._selected_kv_runtime is None
        assert tot_b * 2 > budget.native_budget_bytes
        pw.selected_kv_runtime, runner.kv_connector, runner._selected_kv_runtime = None, M(), None
        ctx["extra_unshared"] = ns(kv_cache=torch.empty(budget.available_bytes, dtype=torch.int8))
        R(ValueError, "exceeds the physical memory budget", runtime.start_selected_kv, runner, pw.kv_cache_config)
        ctx.pop("extra_unshared", None)
    finally:
        runtime.stop_selected_kv(runner)
        vars(envs_vllm).clear()
        vars(envs_vllm).update(snap)


def _make_handoff_env(monkeypatch, evs, quiesced):
    mp = monkeypatch.setattr
    mp(pool_worker.BaseKVPoolWorker, "get_finished", lambda _s, _f, _m: (set(), set()))
    mp(torch.npu, "synchronize", lambda: (quiesced.__setitem__(0, True), evs.append("sync")), raising=False)
    rm = lambda ks: evs.append(("rm", list(ks), quiesced[0])) or 0

    def make(a_s=True, keys=None, run=False):
        quiesced[0] = False
        evs.clear()
        rt, ow, caches, _ = _setup_rt(monkeypatch)
        p: Any = _pool(ow, rt, rm, async_scheduling=a_s, _step_load_keys=set(keys or ()))
        rt.worker = p
        if run:
            assert _batch(rt=rt)
            _run_layers(rt, caches, _tk(4), 4, finish=True)
        return rt, p

    return make


def test_ddr_handoff_failure_matrix(patched_npu_env, monkeypatch):
    evs: list[Any] = []
    quiesced = [False]
    make = _make_handoff_env(monkeypatch, evs, quiesced)
    # Cover full 2x2: (async_scheduling -> query vs sync -> synchronize) x (next caller: begin_step vs get_finished).
    for async_sched in (True, False):
        for next_caller in ("begin_step", "get_finished"):
            rt, p = make(async_sched, {"k_handoff"}, run=True)
            ddr = rt._completed_fence[0].ddr_done
            ddr.query = M(side_effect=RuntimeError("ddr query boom")) if async_sched else M(return_value=0)
            ddr.synchronize = M(side_effect=RuntimeError("ddr sync boom"))
            R(RuntimeError, "ddr (query|sync) boom", p.get_finished, set(), M())
            assert (evs, p._step_load_keys, rt._aborted, rt._poisoned) == ([], {"k_handoff"}, True, True)
            assert not rt._accepting_leases and not rt._leased_sizes

            quiesced[0] = False
            evs.clear()
            if next_caller == "begin_step":
                R(RuntimeError, "poisoned", rt.begin_step)
                assert evs == ["sync"] and not rt._accepting_leases and not rt._aborted
                evs.clear()
                p.kv_recv_thread, p._drain_attention_transfers = ns(request_queue=ns(unfinished_tasks=1)), (ds := M())
                p.get_finished(set(), M())
                ds.assert_not_called()
                p.kv_recv_thread = None
                assert evs == [("rm", ["k_handoff"], True)]
            else:
                p.get_finished(set(), M())
                assert evs == ["sync", ("rm", ["k_handoff"], True)]
            assert not p._step_load_keys and rt.close() is None


def test_reap_drain_and_ring_fence_isolation(patched_npu_env, monkeypatch):
    evs: list[Any] = []
    make = _make_handoff_env(monkeypatch, evs, [False])

    rt, p = make(True)
    bad_ev = ct.FakeEvent(ready=False)
    monkeypatch.setattr(bad_ev, "query", M(side_effect=RuntimeError("reap boom")))
    p._deferred_leases.append((bad_ev, ["k_reap"]))
    p._lease_refs["k_reap"] = 1
    R(RuntimeError, "reap boom", p.get_finished, set(), M())
    assert (rt._aborted, rt._poisoned, len(p._deferred_leases), evs) == (True, True, 1, []) and rt.close() is None

    rt, p = make(True, {"k_drain"}, run=True)
    p.kv_recv_thread = ns(request_queue=ns(unfinished_tasks=1))
    p._drain_attention_transfers = M(side_effect=RuntimeError("async drain boom"))
    rt._completed_fence[0].ddr_done.query = M(return_value=False)
    evs.clear()
    R(RuntimeError, "async drain boom", p.get_finished, set(), M())
    assert (rt._aborted, rt._poisoned, p._step_load_keys, evs) == (True, True, {"k_drain"}, [])
    p.__dict__.pop("_drain_attention_transfers", None)
    p.kv_recv_thread = None
    rt.close()

    rt, p = make(True)
    rt._ring[0].ddr_fence = ns(synchronize=M(side_effect=RuntimeError("ring boom")))
    R(RuntimeError, "ring boom", _batch, rt=rt)
    assert (rt._aborted, rt._poisoned) == (True, True) and rt.close() is None


def test_followup_nonselected_failures_and_abort_lease_reclaim(patched_npu_env, monkeypatch):
    # Non-selected follow-up step: prev.synchronize()/query() errors in on_slot_released(0) fail handoff.
    for has_task, attr, msg in ((True, "synchronize", "prev sync boom"), (False, "query", "prev query boom")):
        rt, w, caches, _ = _setup_rt(monkeypatch)
        assert _batch(rt=rt)
        _run_layers(rt, caches, _tk(4), 4, finish=True)
        ok, _ = rt.take_step_fence()
        assert ok and rt._prev_compute_done is not None and rt.step is None
        w.layer_load_tasks = [[object()] if has_task else [], []]
        setattr(rt._prev_compute_done, attr, M(side_effect=RuntimeError(msg)))
        R(RuntimeError, msg, rt.on_slot_released, 0)
        assert rt._aborted and rt._poisoned and rt.close() is None

    # Non-selected follow-up in get_finished: shared key drain failure fails handoff and keeps _step_load_keys.
    rm_s = lambda s: s._step_load_keys and s.m_store.batch_remove_lease(list(s._step_load_keys))
    gf_s: Any = lambda s, _f, _m: (rm_s(s), s._step_load_keys.clear(), (set(), set()))[2]
    monkeypatch.setattr(pool_worker.LayerwiseKVPoolWorker, "get_finished", gf_s)
    rt_s, ow_s, _, _ = _setup_rt(monkeypatch)
    ps_k = dict(rm=M(return_value=0), _lease_refs=Counter({"k_sh": 1}), _step_load_keys={"k_sh"})
    p_s = _pool(ow_s, rt_s, **ps_k, kv_recv_thread=ns(request_queue=ns(unfinished_tasks=1)))
    rt_s.worker = p_s
    p_s._drain_attention_transfers = M(side_effect=RuntimeError("shared drain boom"))
    R(RuntimeError, "shared drain boom", p_s.get_finished, set(), M())
    assert rt_s._aborted and rt_s._poisoned and p_s._step_load_keys == {"k_sh"}
    p_s._drain_attention_transfers.side_effect, p_s.kv_recv_thread = None, None
    assert rt_s.close() is None

    # Non-poisoned mid-step abort before post_forward leaves _step_load_keys={"k_retry"};
    # reclaiming before retry avoids set folding so "k_retry" is removed 1:1 (count == 2).
    removed: list[str] = []

    def _rm_rec(ks):
        removed.extend(ks)
        return 0

    for mode in ("pre_forward_boom", "wait_for_previous_save"):
        removed.clear()
        rt_r, ow_r, caches_r, _ = _setup_rt(monkeypatch)
        inner_r = create_autospec(gpu_kvc.KVConnector, instance=True)
        pw_r: Any = _pool(ow_r, rt_r, rm=_rm_rec, _previous_save_batch=None)
        pw_r._drain_attention_transfers, rt_r.worker = M(), pw_r
        conn_r = runtime.SelectedKVConnector(inner_r, rt_r)
        if mode == "pre_forward_boom":

            def inject_boom(*_a, pw=pw_r, **_k):
                pw._step_load_keys.add("k_retry")
                raise RuntimeError("pre_forward mid boom")

            inner_r.pre_forward.side_effect = inject_boom
            R(RuntimeError, "pre_forward mid boom", conn_r.pre_forward, ns(kv_connector_metadata=ns()), None)
        else:
            rt_r._aborted = True
            pw_r._step_load_keys.add("k_retry")
            pw_r.wait_for_previous_save()
        pw_r._drain_attention_transfers.assert_called_with(full=True)
        assert not rt_r._poisoned and not pw_r._step_load_keys and removed == ["k_retry"]
        inner_r.pre_forward.side_effect = None
        pw_r._step_load_keys.add("k_retry")
        assert _batch(rt=rt_r)
        _run_layers(rt_r, caches_r, _tk(4), 4, finish=True)
        pw_r.get_finished(set(), M())
        assert removed.count("k_retry") == 2 and rt_r.close() is None


def _make_recving_env(
    monkeypatch,
    prefetch: int,
    *,
    layers: int = 4,
    li_layers: set[int] | None = None,
    prefetch_layer_map: dict[int, tuple[int, ...]] | None = None,
):
    names, pr, req = [f"l.{i}" for i in range(layers)], kv_cfg.PAGE_ROWS, _sel()
    hist = req.load_spec.kvpool_cached_tokens
    pages, q = hist // pr, req.target_token_len - hist
    l_bytes = plan.MAIN_PAGE_BYTES + kv_cfg.INDEXER_DATA_PAGE_BYTES + kv_cfg.INDEXER_SCALE_PAGE_BYTES
    rt_k: dict[str, Any] = dict(layers=layers, with_li=True, li_layers=li_layers, pages=pages, reuse=True)
    rt_k |= dict(max_num_batched_tokens=q, max_num_seqs=1, max_model_len=hist)
    rt_k |= dict(owners={names[i]: names[i - (i % 2)] for i in range(layers)}, layer_bytes=l_bytes)
    rt, ow, _, _ = _setup_rt(monkeypatch, **rt_k)
    sh_r = ct.make_shared(np.arange(1, pages + 1), np.full(pages, GVA))
    ow.layer_load_tasks = [[ct.make_task(i, sh_r, ["r0"])] for i in range(layers)]
    load_ev, save_ev = ([threading.Event() for _ in names] for _ in range(2))
    save_fences = [ct.FakeEvent() for _ in names]
    copy_spy = M(side_effect=AssertionError("selected-only step must not call native batch_copy"))
    store = ns(store=ns(batch_copy=copy_spy))
    ev_k: dict[str, Any] = dict(layer_load_finished_events=load_ev, layer_save_finished_events=save_ev)
    ev_k |= dict(sync_save_events=save_fences, num_layers=layers)
    r_k = dict(m_store=store, token_database=ns(group_block_len=[[plan.MAIN_PAGE_BYTES]]), block_size=pr, tp_rank=0)
    r_k |= dict(tp_size=1, dcp_size=1, page_size_bytes=plan.MAIN_PAGE_BYTES, ready_event=threading.Event())
    r_k |= dict(h2d_stagger_us=0, max_transfer_blocks=0, max_transfer_bytes=0)
    r_k |= dict(group_builders=[object.__new__(LayerBatchBuilder)])
    recv = KVCacheStoreLayerRecvingThread(**(ev_k | r_k))
    monkeypatch.setattr(recv, "_wait_for_event", lambda ev, _fc=None: ev.wait(timeout=0) or _fail_wait())
    pf_map = prefetch_layer_map if prefetch_layer_map is not None else {i: (i - 2,) for i in range(2, layers)}
    pw_k: dict[str, Any] = dict(m_store=store, current_layer=0, num_prefetch_layers=prefetch, prefetch_layer_map=pf_map)
    pw_k |= dict(kv_recv_thread=recv, kv_send_thread=ns(request_queue=queue.Queue(), raise_if_failed=M()))
    pw_k |= dict(layer_save_tasks=[[] for _ in names], _attention_saved_layers=set())
    pw = _pool(ow, **(ev_k | pw_k))
    monkeypatch.setattr(pool_worker, "_LW_DEFER_LAST_SAVE", False)
    trace: list[Any] = []
    by_m = {c.object_offset: i for i, c in enumerate(rt.components)}
    by_i = {x.offsets[0]: i for i, x in enumerate(rt.indexers) if x is not None}
    ops_d = dict(selected_kv_union_plan=M(side_effect=lambda tk, *_: trace.append(("union", int(tk[0, 0, 0])))))
    ops_d |= dict(selected_kv_inplace_copy=M(side_effect=lambda *a: trace.append(("main", by_m[a[-1]]))))
    ops_d |= dict(selected_kv_li_gather=M(side_effect=lambda *a: trace.append(("li", by_i[a[-2]]))))
    monkeypatch.setattr("torch.ops._C_ascend", ns(**ops_d), raising=False)
    rt.worker = pw
    pw.attach_selected_runtime(rt)
    out: dict[str, Any] = dict(names=names, pr=pr, q=q, pages=pages, req=req, load_ev=load_ev, save_ev=save_ev)
    return ns(**(out | dict(sf=save_fences, copy=copy_spy, recv=recv, pw=pw, rt=rt, trace=trace)))


def _drive_recving_step(e, monkeypatch, li_layers, *, prefetch=2, check_dup=False, waited=None):
    rt, pw, recv, trace = e.rt, e.pw, e.recv, e.trace
    sched, inp = _batch(e.req)
    lid_m, rec = rt.layer_ids, af.record_attention_compute_start
    attn_fn = lambda i, *_, **_kw: (trace.append(("attention", lid_m[i.layer_name])), rec())
    monkeypatch.setattr(AscendSFAImpl, "_execute_sparse_flash_attention_process", attn_fn)
    bt = torch.arange(1, e.pages + 2, dtype=torch.int32).view(1, -1)
    meta = ns(num_actual_tokens=e.q, block_size=e.pr, block_table=bt)
    rt.begin_step()
    rt.record_leased_infos(["hist"], [ns(gva_list=lambda: [GVA], size=lambda: 1 << 20)])
    exp_t = [0, 1] + [0] * (len(e.names) - 2)
    assert rt.bind(sched, inp) and [len(t) for t in pw.layer_load_tasks] == exp_t
    try:
        for lid, name in enumerate(e.names):
            assert lid == 0 or e.load_ev[lid].is_set()
            pw.wait_for_layer_load()
            if lid == 0:
                assert recv.request_queue.qsize() == prefetch and (s1 := recv.request_queue.get_nowait()).layer_id == 1
                recv._handle_request(s1)
                assert ("li", 1) not in trace
                if check_dup:
                    assert 1 not in rt.step.native_ready
                    R(RuntimeError, "LI load was already requested", rt.on_slot_released, 1)
            impl = object.__new__(AscendSFASelectedKVImpl)
            impl.selected_kv_runtime, impl.layer_name = rt, name
            tk = torch.full((e.q, 1, kv_cfg.TOPK_WIDTH), lid, dtype=torch.int32)
            if lid == 0:
                R(
                    ValueError,
                    "overriding attn_metadata.block_table",
                    impl._execute_sparse_flash_attention_process,
                    None,
                    None,
                    (rt.main_caches[name],),
                    tk,
                    meta,
                    None,
                    None,
                    block_table=bt.clone(),
                )
            impl._execute_sparse_flash_attention_process(None, None, (rt.main_caches[name],), tk, meta, None, None)
            pw.save_kv_layer(None)
            if 1 <= lid <= len(e.names) - 2:
                task = recv.request_queue.get_nowait()
                assert task.layer_id == lid + 1
                if check_dup and lid == 5:
                    assert (
                        task.wait_for_save_layers == (1, 4) and 1 in rt.step.native_ready and 4 in rt.step.native_ready
                    )
                    assert waited is not None
                    waited.clear()
                recv._handle_request(task)
                if check_dup and lid == 5:
                    assert waited == [e.save_ev[1], e.save_ev[4]] and ("li", 6) in trace
        ti = trace.index
        assert all(max(ti(("main", i)), ti(("li", i))) < ti(("attention", i)) for i in sorted(li_layers))
        assert all(ti(("main", i)) < ti(("attention", i)) for i in range(len(e.names)) if i not in li_layers)
        assert recv.request_queue.unfinished_tasks == 0 and not e.copy.called
        rt.finish()
        ok, ddr_fence = rt.take_step_fence()
        assert ok and ddr_fence.recorded == 1
    finally:
        pw.kv_send_thread = None
        rt.close()


@pytest.mark.parametrize("prefetch", [1, 2])
def test_recving_thread_selected_only_two_owner_groups_and_reuse(patched_npu_env, monkeypatch, prefetch):
    e = _make_recving_env(monkeypatch, prefetch)
    _drive_recving_step(e, monkeypatch, {0, 1, 2, 3}, prefetch=prefetch)
    ti = e.trace.index
    assert [lid for op, lid in e.trace if op == "union"] == [0, 2]
    assert [lid for op, lid in e.trace if op == "main"] == [0, 1, 2, 3]
    assert ti(("main", 0)) < ti(("li", 1)) and [f.synchronized for f in e.sf] == [1, 1, 0, 0]


def _fail_wait():
    raise AssertionError("receiver reached slot before producer release")


def test_recving_thread_multi_source_reuse_and_duplicate_slot_release(patched_npu_env, monkeypatch):
    li_layers = {0, 1, 2, 6}
    specs = ct.make_specs_and_cfg(7, sorted(li_layers))[2]
    reuse = plan.probe_reuse_layout(
        ct.make_vllm_config(**{"model_config.get_total_num_hidden_layers": lambda: 7}), specs, num_blocks=8
    )
    assert reuse.prefetch_layer_map[6] == (1, 4)

    e = _make_recving_env(
        monkeypatch, prefetch=2, layers=7, li_layers=li_layers, prefetch_layer_map=reuse.prefetch_layer_map
    )
    waited: list[Any] = []

    def _spy_wait(ev: Any, _fc: Any = None) -> bool:
        waited.append(ev)
        return ev.wait(timeout=0) or _fail_wait()

    monkeypatch.setattr(e.recv, "_wait_for_event", _spy_wait)
    _drive_recving_step(e, monkeypatch, li_layers, prefetch=2, check_dup=True, waited=waited)
    assert [f.synchronized for f in e.sf] == [1, 2, 1, 1, 1, 0, 0]


def test_runtime_selection_thresholds_and_start_forwarding(patched_npu_env, monkeypatch):
    for bad in (-0.1, 1.1, True, float("nan")):
        R(ValueError, "max_query_ratio", _setup_rt, monkeypatch, max_query_ratio=bad)
    for bad_mh in (-1, 1048577, True, 32768.0, "32768"):
        R(ValueError, "min_history_tokens", _setup_rt, monkeypatch, min_history_tokens=bad_mh)
    rt_zero, w_zero, _, _ = _setup_rt(monkeypatch, max_model_len=32768, max_query_ratio=0.0)
    t_zero = w_zero.layer_load_tasks[0][0]
    assert not _batch(_sel("r_zero", h=16384, q=4, p=128), rt=rt_zero)
    assert w_zero.layer_load_tasks[0][0] is t_zero and rt_zero.step is None and rt_zero.close() is None
    req_hi = ct.make_req("r_hi", 32896, 4096, list(range(1, 290)), [GVA] * 257)
    kw_hi: dict[str, Any] = dict(max_num_batched_tokens=8192, max_model_len=65536)
    rt_b, w_b, _, ls_d = _setup_rt(monkeypatch, pages=600, **kw_hi)
    # Construct SelectedKVRuntime directly with omitted thresholds to test real defaults (0.015, 32768).
    dk = kw_hi | dict(device="cpu", max_num_seqs=16, owner_by_layer=rt_b.owner_by_layer)
    rt_def = runtime.SelectedKVRuntime(w_b, **(dk | dict(main_caches=rt_b.main_caches, exporter=rt_b.exporter)))
    assert (rt_def.max_query_ratio, rt_def.min_history_tokens) == (0.015, 32768)
    assert not _batch(req_hi, rt=rt_def) and not _batch(_sel("r_32k", h=32768, q=4, p=256), rt=rt_def)
    r_tail = ct.make_req("r_32k1", 32769, 4, list(range(1, 258)), [*([GVA] * 256), 0], partial=[GVA])
    for req_ok in (r_tail, _sel("r_32k_p", h=32896, q=4, p=257)):
        w_b.layer_load_tasks = [[ct.make_task(i, ls_d, ["r"])] for i in range(2)]
        assert _batch(req_ok, rt=rt_def)
        _run_layers(rt_def, rt_def.main_caches, _tk(4), 4, 1, 2, finish=True)
    r_nat32k = ct.make_req("r_nat32k", 32768, 4, list(range(1, 258)), [GVA] * 256)
    r_dem = ct.make_req("r_dem", 32896, 4, list(range(300, 558)), [GVA] * 257)
    w_b.layer_load_tasks = [[ct.make_task(i, ls_d, ["r_nat32k", "r_dem"])] for i in range(2)]
    assert _batch(r_nat32k, r_dem, rt=rt_def) and rt_def.step is not None and rt_def.step.histories == [32896]
    _run_layers(rt_def, rt_def.main_caches, _tk(8), 8, 2, 2, finish=True)
    for r in (rt_b, rt_def):
        r.close()
    rt_z0 = _setup_rt(monkeypatch, pages=4, min_history_tokens=0, max_query_ratio=0.5)[0]
    assert not _batch(ct.make_req("r_h64", 64, 4, [1], [0], partial=[GVA]), rt=rt_z0)
    assert _batch(ct.make_req("r_h128", 128, 4, [1, 2], [GVA]), rt=rt_z0) and rt_z0.close() is None
    rt_hi, _, c_hi, _ = _setup_rt(monkeypatch, pages=289, **kw_hi, max_query_ratio=0.3, min_history_tokens=32768)
    assert _batch(req_hi, rt=rt_hi) is True
    _run_layers(rt_hi, c_hi, _tk(4096), 4096, 1, 2, finish=True)
    rt_hi.close()
    for pg_n, mh_v, m_len, h_ok in ((513, 65536, 65668, 65664), (8193, 1048576, 1048576, 0)):
        rt_m = _setup_rt(monkeypatch, pages=pg_n, max_model_len=m_len, min_history_tokens=mh_v)[0]
        assert not _batch(_sel("r_m0", h=mh_v, q=4, p=pg_n - 1), rt=rt_m)
        assert (not h_ok or _batch(_sel("r_m1", h=h_ok, q=4, p=pg_n), rt=rt_m)) and rt_m.close() is None
    _, _, _, _, pw, vcfg, _, mk_runner = _setup_start_env(monkeypatch)
    vcfg.parallel_config.distributed_executor_backend = "mp"
    sc1 = ((SelectedKVConfig(0.25), (0.25, 32768)), (SelectedKVConfig(0.2, 8192), (0.2, 8192)))
    sc2 = (
        (SelectedKVConfig(min_history_tokens=65536), (0.015, 65536)),
        (SelectedKVConfig(0.0, 0), (0.0, 0)),
    )
    for scfg, exp in (*sc1, *sc2, (ns(enabled=True), (0.015, 32768))):
        runner = mk_runner(scfg)
        runtime.start_selected_kv(runner, pw.kv_cache_config)
        s_rt = runner._selected_kv_runtime
        assert (s_rt.max_query_ratio, s_rt.min_history_tokens) == exp
        runtime.stop_selected_kv(runner)


def test_load_layer_block_table_alias_and_mismatch_guards(patched_npu_env, monkeypatch):
    rt, w, caches, orig_s = _setup_rt(monkeypatch, layers=2)
    bt, tk = torch.ones((1, 300), dtype=torch.int32), _tk(4)
    bt0, bt1 = bt[:], bt[:]
    assert bt0 is not bt1 and _batch(rt=rt)
    _ = rt.on_native_layer_ready(0), rt.on_native_layer_ready(1)
    rt.load_layer("l.0", tk, (caches["l.0"],), ns(num_actual_tokens=4, block_size=128, block_table=bt0))
    rt.prefetch_next("l.0")
    rt.load_layer("l.1", tk, (caches["l.1"],), ns(num_actual_tokens=4, block_size=128, block_table=bt1))
    _ = rt.finish(), rt.take_step_fence()
    for bad_meta in (
        ns(num_actual_tokens=4, block_size=128, block_table=bt[:, 1:]),
        ns(num_actual_tokens=4, block_size=128, block_table=bt.clone()),
        ns(num_actual_tokens=3, block_size=128, block_table=bt[:]),
    ):
        w.layer_load_tasks = [[ct.make_task(j, orig_s, ["r"])] for j in range(2)]
        assert _batch(rt=rt)
        _ = rt.on_native_layer_ready(0), rt.on_native_layer_ready(1)
        rt.load_layer("l.0", tk, (caches["l.0"],), ns(num_actual_tokens=4, block_size=128, block_table=bt[:]))
        rt.prefetch_next("l.0")
        R(ValueError, "stable block_table across layers", rt.load_layer, "l.1", tk, (caches["l.1"],), bad_meta)
        _ = rt.finish(failed=True), rt.take_step_fence()
    assert rt.close() is None


def test_patched_npu_env_tolerates_stubbed_kv_pool(monkeypatch, request):
    from vllm_ascend.distributed import kv_transfer as kv_transfer_mod

    stub_pool = types.ModuleType("vllm_ascend.distributed.kv_transfer.kv_pool")
    monkeypatch.setitem(sys.modules, "vllm_ascend.distributed.kv_transfer.kv_pool", stub_pool)
    monkeypatch.setattr(kv_transfer_mod, "kv_pool", stub_pool, raising=False)
    env = request.getfixturevalue("patched_npu_env")
    assert not hasattr(stub_pool, "ascend_store") and env.compute_stream.npu_stream == 101
    assert af._attention_load_gate is None and af._attention_save_gate is None
