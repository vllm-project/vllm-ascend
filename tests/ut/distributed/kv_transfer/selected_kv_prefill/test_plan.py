# SPDX-License-Identifier: Apache-2.0
import contextlib
import subprocess
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import vllm.envs as envs_vllm
from pydantic import ValidationError

from tests.ut.distributed.kv_transfer.selected_kv_prefill import conftest as ct
from vllm_ascend import ascend_config as cfg
from vllm_ascend import utils as ascend_utils
from vllm_ascend.attention import sfa_v1
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import ascend_store_connector as asc
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import pool_worker as pw
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import selected_kv_plan as plan
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import selected_kv_runtime as rt_mod
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import selected_pool_worker as spw

R, np, torch, ns, vc, msc = ct.raises, ct.np, ct.torch, ct.ns, ct.make_vllm_config, ct.make_specs_and_cfg


def test_disabled_validation_classification_geometry_and_trim(monkeypatch):
    py = (
        "import sys, vllm_ascend.ascend_config; "
        "assert 'vllm.config' not in sys.modules and not any('kv_transfer' in k for k in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", py], check=True)
    off, s, mp = cfg.SelectedKVConfig(enabled=False), object(), monkeypatch.setattr
    ac = ns(ascend_config=ns(selected_kv_config=off, sparse_kv_offload_config=ns(enabled=False)))
    off.validate_deployment(vc(model_config=None))
    rt_mod.start_selected_kv(ac, ns())
    rt_mod.stop_selected_kv(ac)
    assert plan.reserve_selected_kv_workspace(ac, s) is s
    for attr_k, attr_v in (("get_ascend_config", lambda: ac.ascend_config), ("get_current_vllm_config", vc)):
        mp(sfa_v1, attr_k, attr_v)
    mp(sfa_v1, "enable_sfa_dcp_replicated_indexer", lambda _c: False)
    assert sfa_v1.AscendSFABackend.get_impl_cls() is sfa_v1.AscendSFAImpl
    ac.ascend_config.selected_kv_config = (skc := cfg.SelectedKVConfig(enabled=True))
    assert sfa_v1.AscendSFABackend.get_impl_cls() is rt_mod.AscendSFASelectedKVImpl
    val = skc.validate_deployment
    val(vc(**{"scheduler_config.async_scheduling": True}))
    ex = (c := vc()).kv_transfer_config.kv_connector_extra_config
    ex["layerwise_num_shared_buffers"] = 4
    R(ValueError, "== 2", val, c)
    ex["layerwise_num_shared_buffers"] = 2
    for pfs, ok in (((1, 2, None), True), ((0, 3, True), False)):
        for pf in pfs:
            ex["layerwise_prefetch_layers"] = pf
            assert val(c) is None if ok else bool(R(ValueError, "prefetch 1..2", val, c) or True)
    R(ValueError, "sparse_kv_offload", val, vc(), ns(sparse_kv_offload_config=ns(enabled=True)))
    R(ValueError, "hisparse_config", val, vc(**{"attention_config.hisparse_config": object()}))
    R(ValueError, "KVPP", val, vc(), ns(sparse_kv_offload_config=ns(enabled=False), kvpp_config=ns(size=2)))
    xl_on = ns(sparse_kv_offload_config=ns(enabled=False), kvpp_config=ns(size=1), xlite_graph_config=ns(enabled=True))
    R(ValueError, "xlite_graph_config", val, vc(), xl_on)
    for bad_cd in ("fp8", "auto"):
        R(ValueError, "cache_dtype='int8'", val, vc(**{"cache_config.cache_dtype": bad_cd}))
    R(ValueError, "indexer_kv_dtype='int8'", val, vc(**{"attention_config.indexer_kv_dtype": "fp8"}))
    val(vc(**{"cache_config.num_gpu_blocks_override": 8}))
    # Startup-plan gating lives in worker.py; deployment validation permits VLLM_ENABLE_STARTUP_PLAN=True.
    monkeypatch.delitem(vars(envs_vllm), (ek := "VLLM_ENABLE_STARTUP_PLAN"), raising=False)
    monkeypatch.setenv(ek, "1")
    assert envs_vllm.VLLM_ENABLE_STARTUP_PLAN is True and ek not in vars(envs_vllm) and val(vc()) is None
    assert cfg.selected_kv_enabled(ac.ascend_config) and not cfg.selected_kv_requested(ns(additional_config={}))
    spec = plan.SelectedWorkspaceSpec.from_bounds(8192, 256, 200000)
    budget = plan.SelectedKVBudget.reserve(10 * 1024**3, 400_000, spec.fixed_bytes, 16, 8 * 1024**2)
    assert spec.union_capacity % 128 == 0 and spec.scratch_shapes == plan.SCRATCH_SHAPES and budget.validate(1) is None
    R(ValueError, "exceeds", budget.validate, budget.maximum_blocks + 1)
    ms, ls, specs, _ = msc(4, (0, 2))
    specs1 = msc(4, (0,))[2]
    nv = ns(model_config=ns(get_num_layers=lambda _: 4), parallel_config=ns(), vllm_config=vc())
    names, page_bytes, pad = plan.native_geometry(nv, specs)
    assert (len(names), page_bytes, pad) == (4, 2 * ms.page_size_bytes + 2 * ls.page_size_bytes, plan.NATIVE_ALIGNMENT)
    assert plan.native_geometry(nv, specs1)[1:] == (2 * ms.page_size_bytes + ls.page_size_bytes, plan.NATIVE_ALIGNMENT)
    bad_sp = (
        (msc(4, (0,), m_dtype=torch.float8_e4m3fn)[2], "int8 packed C8 main"),
        (msc(4, (0,), l_dtype=torch.float8_e4m3fn, s_dtype=torch.float32)[2], "int8 C8 128-row indexer"),
    )
    for sp_bad, pat in bad_sp:
        R(ValueError, pat, plan.ordered_layer_specs, plan.probe_reuse_layout(vc(), sp_bad), sp_bad, 4)

    from vllm_ascend.device.hardware import AscendDeviceType

    mp(plan, "get_ascend_device_type", lambda: AscendDeviceType.A3)
    kw_m = dict(kv_lora_rank=512, qk_rope_head_dim=64)
    ctx_ok = {
        n: ns(impl=ns(**kw_m, skip_topk=bool(i % 2), indexer=None if i % 2 else ns(topk_output_width=2048)))
        for i, n in enumerate(names)
    }

    def _mk_ws(sp, ac_ov=None, **ov):
        v = vc(**ov)
        rn = ns(model_config=v.model_config, parallel_config=v.parallel_config, vllm_config=v)
        rn.compilation_config = ns(static_forward_context=ctx_ok)
        ac_d = dict(
            selected_kv_config=skc, enable_sparse_sfa_c8=True, enable_sparse_li_c8=True, enable_sparse_li_c4=False
        )
        ac_d |= dict(
            kvpp_config=ns(size=1), sparse_kv_offload_config=ns(enabled=False), xlite_graph_config=ns(enabled=False)
        )
        return ns(
            ascend_config=ns(**(ac_d | (ac_ov or {}))),
            vllm_config=v,
            cache_config=v.cache_config,
            model_runner=rn,
            available_kv_cache_memory_bytes=10 * 1024**3,
            get_kv_cache_spec=lambda: sp,
        )

    bad_ws: tuple[tuple[str, dict[str, Any] | None, dict[str, Any]], ...] = (
        ("HiSparse", None, {"attention_config.hisparse_config": object()}),
        ("KVPP", {"kvpp_config": ns(size=2)}, {}),
        ("Xlite", {"xlite_graph_config": ns(enabled=True)}, {}),
        ("int8 C8", None, {"cache_config.cache_dtype": "fp8"}),
        ("int8 C8", None, {"cache_config.cache_dtype": "auto"}),
        ("int8 C8", None, {"attention_config.indexer_kv_dtype": "fp8"}),
        ("exceeds the selected-KV workspace reservation", None, {"cache_config.num_gpu_blocks_override": 999999}),
    )
    for pat, ac_ov, v_ov in bad_ws:
        R(ValueError, pat, plan.reserve_selected_kv_workspace, _mk_ws(specs, ac_ov, **v_ov))
    plan.reserve_selected_kv_workspace(_mk_ws(specs, None, **{"cache_config.num_gpu_blocks_override": 8}))
    for sp, exp_pb in (
        (specs, 2 * ms.page_size_bytes + 2 * ls.page_size_bytes),
        (specs1, 2 * ms.page_size_bytes + ls.page_size_bytes),
    ):
        w_res = _mk_ws(sp)
        nat_b = plan.reserve_selected_kv_workspace(w_res)
        b_res = w_res.model_runner.selected_kv_budget
        assert (nat_b, b_res.native_page_bytes, b_res.native_padding_bytes) == (
            b_res.native_budget_bytes,
            exp_pb,
            plan.NATIVE_ALIGNMENT,
        )
    cp = tuple(plan.MainComponent(f"l.{i}", i, i * 83968) for i in range(4))
    sr, req, sh, trim, HI = plan.SelectedRequest, ct.make_req, ct.make_shared, plan.trim_load_plan, ct.HI_GVA
    fg, tg, a = [HI + 2 * i for i in range(511)], HI + 9998, np.array
    ls_s = sh(np.arange(1, 513, dtype=np.int64), a([*fg, tg], np.uint64))
    w = ct.make_worker_stub(600, 4, ls_s, sh(a([2, 999], np.int64), a([0, -1], np.int64)))
    r0 = req("r0", 65535, 1, list(range(1, 513)), [*fg, 0], skip_tokens=65536, partial=[tg])
    r1, rm = req("r1", 128, 4, [1, 520], [HI, HI + 100]), req("miss", 256, 128, [1, 2, 520], [HI, HI + 9999, HI + 4])
    pl = (sr("r0", 65535, 1, True), sr("r1", 128, 4, False), sr("miss", 256, 128, True))
    tr = trim([r0, r1, rm], pl, ld := {g: 16384 for g in (*fg, tg)}, 8192, cp, w)
    assert tr and (list(tr.ids), int(tr.gvas[0])) == (list(range(3, 512)), HI + 4) and tr.selected_req_ids == {"r0"}
    orig_t, (cand, l01) = w.layer_load_tasks[0][0], plan.build_trimmed_load_tasks(w.layer_load_tasks, tr.replacements)
    assert l01 == (True, True) and w.layer_load_tasks[0][0] is orig_t and orig_t.shared_block_data is ls_s
    assert cand[0][0] is not orig_t and list(cand[0][0].shared_block_data.block_ids_arr) == [1, 2, 512]
    w2 = ct.make_worker_stub(64, 4, sh(a([20, 21], np.int64), a([HI, HI + 2], np.uint64)), sh(e := a([], np.int64), e))
    t2 = trim([req("r0", 256, 4, [20, 21, 22], [HI, HI + 2, HI + 4])], (sr("r0", 256, 4, True),), ld, 8192, cp, w2)
    c2, l01_2 = plan.build_trimmed_load_tasks(w2.layer_load_tasks, t2.replacements)
    assert (l01_2, c2[0], c2[2], c2[3]) == ((False, False), [], [], []) and c2[1][0].block_ranges == []


def _mk_pkg(monkeypatch, root, name, *, lib=False, extra_files=(), **attrs):
    pkg = root / name
    (pkg / "lib" if lib else pkg).mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    if lib:
        (pkg / "lib" / "libmf_hybm_core.so").write_text("")
    for rel in extra_files:
        (pkg / rel).write_text("")
    mod = types.ModuleType(name)
    mod.__file__, mod.__path__ = str(pkg / "__init__.py"), [str(pkg)]
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return pkg, mod


def test_exporter_discover_guards_and_four_layer_reuse(monkeypatch, tmp_path):
    seen, mp = [], monkeypatch.setattr
    monkeypatch.delitem(sys.modules, "memfabric_hybrid", raising=False)
    mp(ascend_utils, "enable_custom_op", lambda: None, raising=False)
    export_spy = MagicMock(side_effect=lambda g, s, sp: torch.from_numpy(g.numpy() + sp))
    ops_ns = ns(selected_kv_bind_hybm=lambda p: seen.append(list(p)), selected_kv_export_bases=export_spy)
    mp("torch.ops._C_ascend", ops_ns, raising=False)
    _mk_pkg(monkeypatch, tmp_path, "memcache_hybrid", lib=True)
    _mk_pkg(monkeypatch, tmp_path, "custom_hybm", lib=True, extra_files=("_ext.so",))
    ext_so = str((tmp_path / "custom_hybm" / "_ext.so").resolve())
    monkeypatch.setitem(sys.modules, "custom_hybm._ext", ns(__file__=str(tmp_path / "custom_hybm" / "_ext.so")))
    sos = [str((tmp_path / n / "lib/libmf_hybm_core.so").resolve()) for n in ("memcache_hybrid", "custom_hybm")]
    store = type("Store", (), {"__module__": "custom_hybm.client"})()
    exp0 = plan.load_default_exporter()
    assert callable(exp0) and len(seen) == 1 and sos[0] in seen[0] and all("custom_hybm" not in p for p in seen[0])
    out0 = exp0(np.array([10], np.uint64), np.array([64], np.uint64), 32)
    assert out0.tolist() == [42] and len(export_spy.call_args.args) == 3
    R(RuntimeError, "initialized MemcacheBackend store", plan.load_default_exporter, ns(m_store=None))
    R(RuntimeError, "initialized MemcacheBackend store", plan.load_default_exporter, ns(m_store=ns(store=None)))
    exp1 = plan.load_default_exporter(ns(m_store=ns(store=store)))
    assert callable(exp1) and len(seen) == 2 and ext_so in seen[1][:2] and sos[1] in seen[1]
    assert seen[1].index(ext_so) < seen[1].index(sos[1]) and all("memcache_hybrid" not in p for p in seen[1])

    (real_src := tmp_path / "real_src").mkdir()
    (real_src / "__init__.py").write_text("")
    (sym_pkg := tmp_path / "site_packages" / "sym_hybm").joinpath("lib").mkdir(parents=True)
    (sym_pkg / "__init__.py").symlink_to(real_src / "__init__.py")
    (sym_lib := sym_pkg / "lib" / "libmf_hybm_core.so").write_text("")
    monkeypatch.setitem(sys.modules, "sym_hybm", ns(__file__=str(sym_pkg / "__init__.py"), bad=ns(__module__=None)))
    sym_store: Any = type("Store", (), {"__module__": "sym_hybm.client"})()
    sym_store.arr = np.zeros(1)
    assert callable(plan.load_default_exporter(ns(m_store=ns(store=sym_store)))) and str(sym_lib) in seen[-1]
    assert all("sym_hybm" in p or "real_src" in p for p in seen[-1])
    R(RuntimeError, "already initialized native HyBM SDK", plan.load_default_exporter, ns(m_store=ns(store=object())))
    tp_root = tmp_path / "tp_pymmc"
    mf_pkg, mf_mod = _mk_pkg(monkeypatch, tp_root, "memfabric_hybrid", lib=True)
    tp_mc = ("_pymmc.cpython-312-aarch64-linux-gnu.so",)
    mc_pkg, _ = _mk_pkg(monkeypatch, tp_root, "memcache_hybrid", extra_files=tp_mc, memfabric_hybrid=mf_mod)
    _mk_pkg(monkeypatch, tp_root, "unrelated_hybm", lib=True)
    pymmc_mod = types.ModuleType("_pymmc")
    pymmc_mod.__file__ = str(pymmc_so := mc_pkg / tp_mc[0])
    monkeypatch.setitem(sys.modules, "_pymmc", pymmc_mod)
    pymmc_store = type("DistributedObjectStore", (), {"__module__": "_pymmc", "__slots__": ()})()
    exp_p = plan.load_default_exporter(ns(m_store=ns(store=pymmc_store)))
    assert not hasattr(pymmc_store, "__dict__") and callable(exp_p)
    assert seen[-1].index(str(pymmc_so)) < seen[-1].index(str(mf_pkg / "lib" / "libmf_hybm_core.so"))
    assert all("unrelated_hybm" not in p for p in seen[-1])
    for m in ("memcache_hybrid", "memfabric_hybrid"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    R(RuntimeError, "already initialized native HyBM SDK", plan.load_default_exporter)


def test_discover_components_and_topk_owners(monkeypatch):
    names = [f"model.layers.{i}.self_attn.attn" for i in range(4)]
    b_kw = dict(use_layerwise_transfer=True, layerwise_offload=True, use_block_key_layerwise=False, load_async=False)
    base: dict[str, Any] = b_kw | dict(num_kv_cache_groups=1, grouped_block_size=[128], pp_size=1, dcp_size=1)
    base["kv_recv_thread"] = object()
    R(ValueError, "synchronous native GVA", plan.discover_components, ns(**base, _invalid_block_ids=set()))
    kw_m, ix = dict(kv_lora_rank=512, qk_rope_head_dim=64), ns(topk_output_width=2048)
    mk = lambda i: ns(impl=ns(**kw_m, skip_topk=bool(i % 2), indexer=None if i % 2 else ix))
    rn = ns(compilation_config=ns(static_forward_context={names[i]: mk(i) for i in range(4)}))
    assert plan.topk_owners(rn, names) == {names[i]: names[i - (i % 2)] for i in range(4)}
    impl = rn.compilation_config.static_forward_context[names[0]].impl
    impl.skip_topk = True
    R(ValueError, "earlier TopK owner", plan.topk_owners, rn, names)
    impl.skip_topk, impl.indexer.topk_output_width = False, 1024
    R(ValueError, "2048 indices", plan.topk_owners, rn, names)

    buf_a, buf_b = torch.zeros((4, 2048), dtype=torch.int32), torch.zeros((4, 2048), dtype=torch.int32)
    ix_ok = ns(topk_output_width=2048)
    mk_buf = lambda s, b, c=True: ns(
        impl=ns(**kw_m, skip_topk=s, indexer=None if s else ix_ok, topk_indices_buffer=b, use_index_cache=c)
    )
    mk_rn = lambda specs: ns(compilation_config=ns(static_forward_context=dict(zip(names, specs))))
    assert plan.topk_owners(mk_rn([mk_buf(bool(i), buf_a) for i in range(4)]), names) == {n: names[0] for n in names}
    assert plan.topk_owners(mk_rn([mk_buf(bool(i % 2), buf_a if i < 2 else buf_b) for i in range(4)]), names) == {
        names[i]: names[i - (i % 2)] for i in range(4)
    }

    fa, fb, ta, tb = (False, buf_a), (False, buf_b), (True, buf_a), (True, buf_b)
    bad_specs: list[list[Any]] = [
        [fa, fb, ta, tb],
        [fa, (True, None), fb, tb],
        [(False, None), ta, fb, tb],
        [(False, buf_a, False), ta, fb, tb],
    ]
    for row in bad_specs:
        R(ValueError, "must share topk_indices_buffer", plan.topk_owners, mk_rn([mk_buf(*a) for a in row]), names)
    specs, kcfg = msc(4)[2:]
    z = lambda s, dt: [torch.zeros(s, dtype=dt) for _ in range(2)]
    sm, sk, ss = z((8, 128, 1, 656), torch.int8), z((8, 128, 1, 128), torch.int8), z((8, 128, 1, 1), torch.float16)
    caches: dict[str, Any] = {names[i]: sm[i % 2] for i in range(4)}
    caches |= {f"model.layers.{i}.self_attn.indexer.k_cache": (sk[i % 2], ss[i % 2]) for i in range(4)}
    la = np.array([83968, 16384, 256] * 4, np.int64)
    ad = np.array([t.data_ptr() for i in range(4) for t in (sm[i % 2], sk[i % 2], ss[i % 2])], np.int64)
    b = ns(_block_len_np=la, _block_stride_np=la, layer_byte_offset=4096, _kv_caches_base_addr_np=ad)
    b._layer_cache_entry_offsets_np = np.arange(0, 15, 3, dtype=np.int64)
    w_ok = ns(**base, _invalid_block_ids=set(), _invalid_block_ids_lock=object(), num_layers=4, num_blocks=8)
    w_ok.kv_caches, w_ok.kv_recv_thread, w_ok.kv_cache_config = caches, ns(group_builders=[b]), kcfg
    w_ok._layerwise_reuse_layout = plan.probe_reuse_layout(vc(), specs, num_blocks=8)
    w_ok._as_cache_tuple = lambda v: v if isinstance(v, tuple) else (v,)
    idx_c: Any
    mains, idx_c = plan.discover_components(w_ok)
    assert idx_c[0].caches == idx_c[2].caches and idx_c[1].caches == idx_c[3].caches
    assert idx_c[0].caches[0] is not idx_c[1].caches[0] and idx_c[0].offsets == (88064, 104448)
    assert [m.object_offset for m in mains] == [4096, 104704, 205312, 305920]
    s2, k2 = msc(2, ())[2:]
    r2 = plan.probe_reuse_layout(vc(**{"model_config.get_total_num_hidden_layers": lambda: 2}), s2, num_blocks=8)
    w2_ok = ns(**(w_ok.__dict__ | {"_layerwise_reuse_layout": r2, "kv_cache_config": k2, "num_layers": 2}))
    R(ValueError, "two native reuse slots", plan.discover_components, w2_ok)


def test_worker_startup_plan_gating(monkeypatch):
    import importlib
    import importlib.util

    if "acl" not in sys.modules and importlib.util.find_spec("acl") is None:
        acl_rt = types.ModuleType("acl.rt")
        acl_rt.memcpy = MagicMock()  # type: ignore[attr-defined]
        acl_mod = types.ModuleType("acl")
        acl_mod.rt = acl_rt  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "acl", acl_mod)
        monkeypatch.setitem(sys.modules, "acl.rt", acl_rt)

    from vllm_ascend.worker import worker as worker_mod
    from vllm_ascend.worker.v2 import model_runner as mr_mod

    mp = monkeypatch.setattr
    apply_spy, save_spy = MagicMock(), MagicMock()
    w_p: tuple[tuple[str, Any], ...] = (
        ("maybe_apply_startup_plan", apply_spy),
        ("maybe_save_startup_plan", save_spy),
        ("set_current_vllm_config", contextlib.nullcontext),
    )
    for attr, fn in (*w_p, ("get_current_hardware_profile", lambda: ns(supports=lambda *_: False))):
        mp(worker_mod, attr, fn)
    mp(worker_mod, "set_random_seed", lambda *_: None)
    mp(importlib.import_module("vllm_ascend.model_executor.warmup.kernel_warmup"), "kernel_warmup", lambda *_: None)
    for k in ("memory_reserved", "memory_allocated"):
        mp(torch.npu, k, lambda: 0, raising=False)

    # Real NPUWorker instances do not carry self.ascend_config; drive gating via get_ascend_config().
    w: Any = object.__new__(worker_mod.NPUWorker)
    comp = ns(compile_sizes=[], cudagraph_mode=worker_mod.CUDAGraphMode.NONE, compilation_time=0.0)
    w.vllm_config = ns(compilation_config=comp, attention_config=ns(hisparse_config=None, indexer_kv_dtype="int8"))
    w.parallel_config = ns(data_parallel_size=1)
    w.model_config = ns(enforce_eager=True, seed=0)
    w.cache_config = ns(kv_cache_memory_bytes=1 << 30, gpu_memory_utilization=0.9, cache_dtype="int8")
    w.model_runner = ns(profile_run=lambda **_k: None, model_memory_usage=1 << 28)
    w.init_snapshot = ns(free_memory=8 << 30, total_memory=16 << 30)
    w.requested_memory, w.available_kv_cache_memory_bytes = 4 << 30, 2 << 30
    w.peak_activation_memory, w.non_torch_memory = 1 << 27, 1 << 27
    id_fn = lambda b: b
    w._scale_kv_cache_memory_for_multi_group = w._apply_kvpp_memory_budget = id_fn
    w._apply_kv_offload_decode_memory_constraints = id_fn

    on_cfg, off_cfg = (
        ns(selected_kv_config=cfg.SelectedKVConfig(enabled=e), enable_cpu_binding=False) for e in (True, False)
    )

    def raise_uninit():
        raise RuntimeError("AscendConfig not initialized")

    mp(worker_mod, "get_ascend_config", lambda: on_cfg)
    w.determine_available_memory()
    apply_spy.assert_not_called()

    for getter in (lambda: off_cfg, raise_uninit, MagicMock):
        apply_spy.reset_mock()
        mp(worker_mod, "get_ascend_config", getter)
        w.determine_available_memory()
        apply_spy.assert_called_once_with(w)

    w.cache_config.kv_cache_memory_bytes = None
    mp(worker_mod, "get_ascend_config", lambda: on_cfg)
    w.compile_or_warm_up_model()
    save_spy.assert_not_called()
    mp(worker_mod, "get_ascend_config", lambda: off_cfg)
    w.compile_or_warm_up_model()
    assert save_spy.call_count == 1 and save_spy.call_args.args[0] is w and save_spy.call_args.args[1] > 0

    # Disabled-mode lazy gating across sfa_v1, MRV2, and reserve_selected_kv_workspace tolerates MagicMock & bare ns().
    mp(sfa_v1, "get_current_vllm_config", vc)
    mp(sfa_v1, "enable_sfa_dcp_replicated_indexer", lambda _c: False)
    bad_sfa = (MagicMock(sparse_kv_offload_config=ns(enabled=False)), ns(sparse_kv_offload_config=ns(enabled=False)))
    for bad_c in bad_sfa:
        mp(sfa_v1, "get_ascend_config", lambda c=bad_c: c)
        assert sfa_v1.AscendSFABackend.get_impl_cls() is sfa_v1.AscendSFAImpl

    mr = object.__new__(mr_mod.NPUModelRunner)
    assert mr_mod.selected_kv_enabled(getattr(mr, "ascend_config", None)) is False
    for bad_cfg in (MagicMock(), ns()):
        mr.ascend_config = bad_cfg
        assert not mr_mod.selected_kv_enabled(getattr(mr, "ascend_config", None)) and not cfg.selected_kv_enabled(
            bad_cfg
        )
    assert not cfg.selected_kv_enabled(None, raise_uninit)

    for getter in (MagicMock, ns, raise_uninit):
        mp(plan, "get_ascend_config", getter)
        assert plan.reserve_selected_kv_workspace(ns(available_kv_cache_memory_bytes=12345)) == 12345


def test_max_query_ratio_and_1m_history_config(monkeypatch):
    from vllm_ascend.ascend_config import AscendConfig, SparseKVOffloadConfig

    assert (plan.DEFAULT_MIN_HISTORY_TOKENS, plan.MAX_HISTORY_TOKENS, plan.BITMAP_INTS) == (32768, 1048576, 32768)
    spec_1m = plan.SelectedWorkspaceSpec.from_bounds(8192, 256, 1048576)
    assert spec_1m.union_capacity == 1048576 and spec_1m.scratch_shapes == ((1048576,), (32768,), (64, 8))
    assert cfg.SelectedKVConfig(0.2) == cfg.SelectedKVConfig(max_query_ratio=0.2)
    ac_cls: Any = AscendConfig
    mk_ac = lambda **kw: ac_cls(sparse_kv_offload_config=SparseKVOffloadConfig(), **kw).selected_kv_config

    monkeypatch.setattr(pw.LayerwiseKVPoolWorker, "__init__", lambda *_: None)
    S, L, mh0 = spw.SelectedKVPoolWorker, pw.LayerwiseKVPoolWorker, 32768
    kr, kh = "max_query_ratio", "min_history_tokens"
    c_on = [(r, True, float(r), mh0, S) for r in (0.015, 0, 0.0, 0.2, 1, 1.0)]
    c_off = [(r, False, None, None, L) for r in (None, {}, {"enabled": False}, {kr: None, kh: None})]
    c_d: list[Any] = [({"enabled": True}, 0.015, mh0), ({kr: 0.25}, 0.25, mh0), ({kh: 0}, 0.015, 0)]
    c_d += [({kh: 65536}, 0.015, 65536), ({"enabled": True, kr: 0.25}, 0.25, mh0), ({kr: 0.25, kh: 16384}, 0.25, 16384)]
    for raw, exp_en, exp_mr, exp_mh, w_cls in (*c_on, *c_off, *((d, True, mr, mh, S) for d, mr, mh in c_d)):
        c, v_raw = mk_ac(selected_kv_config=raw), vc(additional_config={"selected_kv_config": raw})
        got = (c.enabled, c.max_query_ratio, c.min_history_tokens, cfg.selected_kv_requested(v_raw))
        assert got == (exp_en, exp_mr, exp_mh, exp_en) and type(asc.create_kv_pool_worker(v_raw, True)) is w_cls
    assert mk_ac() == cfg.SelectedKVConfig(enabled=False, max_query_ratio=None, min_history_tokens=None)
    assert not cfg.selected_kv_requested(vc(additional_config={})) and type(asc.create_kv_pool_worker(vc(), True)) is L

    b_prim = (-0.01, 1.01, 10**400, -(10**400), float("nan"), float("inf"), True, "0.015")
    b_mh, b_mr = tuple({kh: v} for v in (-1, 1048577, True, 32768.0, "32768")), tuple({kr: v} for v in (True, "0.2"))
    b_mx: tuple[Any, ...] = ({"enabled": "1"}, {"enabled": False, kr: 0.2}, {"enabled": False, kh: 32768}, {"x": 1})
    b_dict = (*b_mh, *b_mr, *b_mx, {kr: 0.2, kh: -1}, {kr: -0.1, kh: 32768})
    for bad in (*b_prim, *b_dict):
        R(ValidationError, None, mk_ac, selected_kv_config=bad)

    bad_raw: tuple[Any, ...] = (10**400, -(10**400), {kr: 0.2, "x": 1}, {kr: 0.2, "enabled": "false"})
    for r_b in (*bad_raw, {kr: 0.2, "enabled": 1}, *b_dict):
        assert cfg.selected_kv_requested(vc(additional_config={"selected_kv_config": r_b})) is False

    lo, hi = plan.DEFAULT_MIN_HISTORY_TOKENS, plan.MAX_HISTORY_TOKENS
    p1 = ((lo, 1, 0.015, lo, False), (lo + 1, 1, 0.015, lo, True), (39400, 600, 0.015, lo, False))
    p2 = ((39401, 599, 0.015, lo, True), (hi, 1, 0.015, lo, True), (hi + 1, 1, 0.015, lo, False))
    p3 = ((lo + 1, 1, 0.0, lo, False), (54000, 6000, 0.1, lo, False), (54001, 5999, 0.1, lo, True))
    p4 = ((0, 1, 1.0, 0, False), (1, 1, 1.0, 0, True), (16384, 1, 0.015, 16384, False), (16385, 1, 0.015, 16384, True))
    p5 = ((1048576, 1, 1.0, 1048576, False), (16384, 8192, 1.0, 8192, True), (1048576, 8192, 1.0, 8192, True))
    for h, q, r, mh, exp in (*p1, *p2, *p3, *p4, *p5):
        assert plan.classify_request("r", h + q, q, r, mh).selected is exp
