# SPDX-License-Identifier: Apache-2.0
import ctypes
import shutil
import subprocess
from typing import Any

import pytest

from tests.ut.distributed.kv_transfer.selected_kv_prefill import conftest as ct
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.selected_kv_plan import SCRATCH_SHAPES


class GuardedArray:
    def __init__(self, shape, dtype, fill=0):
        self.storage = np.full(int(np.prod(shape)) + 16, 0x5A, dtype=dtype)
        self.value = self.storage[8:-8].reshape(shape)
        self.value.fill(fill)
        self.addr, self.tensor = int(self.value.ctypes.data), torch.from_numpy(self.value)

    @staticmethod
    def assert_all(*items) -> None:
        for g in items:
            assert (g.storage[:8] == (r := np.full(8, 0x5A, g.storage.dtype))).all() and (g.storage[-8:] == r).all()


MAPPER = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64))
GA, np, torch = GuardedArray, ct.np, ct.torch
ae, u64, i8, f16 = np.testing.assert_array_equal, torch.uint64, torch.int8, torch.float16
_z = lambda *s, dt=torch.int32, d="cpu": torch.zeros(s, dtype=dt, device=d)
_r = lambda pat, fn, *a: ct.raises(RuntimeError, pat, fn, *a)


@pytest.fixture(scope="module")
def sim():
    return ct.load_host_sim()


def _args(d="cpu"):
    st = torch.tensor([[0, 1, 1, 0, 0, 0, 0, 0]] * 2, dtype=torch.int32, device=d)
    sc, outs = [_z(*s, d=d) for s in SCRATCH_SHAPES], [_z(2, 1024, d=d), _z(2, 8, d=d)]
    u = (_z(4, 1, 2048, d=d), sc, outs, [16384], [0], [4], 1024)
    mt, tbl = torch.tensor([[0, 128, 0, 1], [1, 128, 1, 1]], dtype=torch.int32, device=d), _z(4, 8, d=d)[:, :6] + 1
    c = (_z(2, 256, d=d), st, _z(8, dt=u64, d=d), mt, tbl, _z(8, 128, 1, 656, dt=i8, d=d), 4096)
    return u, c, (_z(4, dt=u64, d=d), _z(4, 128, 128, dt=i8, d=d), _z(4, 128, dt=f16, d=d), 0, 16384)


def _mk_mapper(seen, fn):
    def _cb(g, mt, ov):
        seen.append(fn[0](g, mt) if isinstance(fn, tuple) else mt)
        ov[0] = fn[1](g) if isinstance(fn, tuple) else g
        return 0

    return MAPPER(_cb)


def _pool_bases(ops, lib, npages: int, stride: int, seed: int, pages: set[int], span: int):
    pool, ord_p = GA((npages, stride), np.uint8), sorted(pages)
    seen: list[int] = []
    pool.value[:] = np.random.default_rng(seed).integers(1, 250, size=pool.value.shape, dtype=np.uint8)
    cb = _mk_mapper(seen, None)
    gvas = torch.tensor([pool.addr + p * stride for p in ord_p], dtype=u64)
    sz = torch.full((len(pages),), stride, dtype=u64)
    bases = _z(npages, dt=u64)
    lib.selected_kv_sim_set_mapper(ctypes.cast(cb, ctypes.c_void_p).value)
    try:
        bases.numpy()[ord_p] = ops.selected_kv_export_bases(gvas, sz, span).numpy()
    finally:
        lib.selected_kv_sim_set_mapper(None)
    assert set(seen) == {1}
    return pool, bases


def _run_union(ops, topk_arr: np.ndarray, h: list[int], s: list[int], q: list[int], cap: int):
    tk, ids, st = GA(topk_arr.shape, np.int32), GA((len(h), cap), np.int32, -7), GA((len(h), 8), np.int32)
    tk.value[:], sc = topk_arr, [torch.empty(sh, dtype=torch.int32) for sh in SCRATCH_SHAPES]
    ops.selected_kv_union_plan(tk.tensor, sc, [ids.tensor, st.tensor], h, s, q, cap)
    return (GA.assert_all(tk, ids, st), ids.value.copy(), st.value.copy())[1:]


def _copy_run(ops, pool, bases, cache, tbl_rows, sels, omitted, cap, h, q, moff, rb):
    reqs, cols, rows = len(sels), tbl_rows.shape[1], 128
    ids, st = GA((reqs, cap), np.int32, -1), GA((reqs, 8), np.int32)
    meta, tbl = GA((reqs, 4), np.int32), GA((reqs, cols + 1), np.int32)
    tbl.value[:, :cols], exp = tbl_rows, cache.value.copy()
    for r_i, sel in enumerate(sels):
        ids.value[r_i, : len(sel)], st.value[r_i, :2], meta.value[r_i] = sel, [len(sel), q], [r_i, h, r_i * q, q]
        for tok in sel:
            if (p := int(tbl.value[r_i, tok // rows])) in omitted and (r := tok % rows) >= 0:
                exp[p, r, 0, :] = pool.value[p, moff + r * rb : moff + (r + 1) * rb].view(np.int8)
    ops.selected_kv_inplace_copy(ids.tensor, st.tensor, bases, meta.tensor, tbl.tensor[:, :cols], cache.tensor, moff)
    _ = ae(cache.value, exp), GA.assert_all(pool, cache, tbl, ids, st, meta)


def test_ops_contracts_exporter_and_union_plan(sim):
    ops, lib = sim
    idx: Any = np.arange(2048, dtype=np.int32)
    lib.selected_kv_sim_reset()
    for dev in ("cpu", "meta"):
        u, c, li = _args(dev)
        ops.selected_kv_union_plan(*u)
        ops.selected_kv_inplace_copy(*c)
        ops.selected_kv_li_gather(*li)
        assert ops.selected_kv_export_bases(_z(0, dt=u64, d=dev), _z(0, dt=u64, d=dev), 4096).device.type == dev
    try:
        mem_types: list[int] = []
        split = [False]
        va_fn = lambda g: g + 0x50000000 + (0x1000 if split[0] and g == 0x100000 + (2 << 20) else 0)
        fk = _mk_mapper(mem_types, (lambda _g, mt: mt, va_fn))
        fp, span, ev = ctypes.cast(fk, ctypes.c_void_p).value, 5 << 20, ops.selected_kv_export_bases
        gvas, sz = torch.tensor([0x100000, 0x900000], dtype=u64), torch.tensor([span + 4096] * 2, dtype=u64)
        _r("requires an initialized HyBM mapper", ev, gvas, sz, span)
        ct.raises((RuntimeError, TypeError), None, ev, gvas, sz, span, 0)
        lib.selected_kv_sim_set_mapper(fp)
        assert ev(gvas, sz, span).tolist() == [0x50100000, 0x50900000] and set(mem_types) == {1}
        _r("exceeds", ev, gvas, torch.tensor([span - 1, span + 4096], dtype=u64), span)
        split[0] = True
        _r("non-contiguous", ev, gvas[:1], torch.tensor([span], dtype=u64), span)

        # Repeated GVA deduplicates HyBM probes within one call while still validating every size[i].
        split[0] = False
        seen_gvas: list[int] = []
        dedup_cb = _mk_mapper(seen_gvas, (lambda g, _mt: g, va_fn))
        sm_span = 1 << 20
        lib.selected_kv_sim_set_mapper(ctypes.cast(dedup_cb, ctypes.c_void_p).value)
        dup_gvas = torch.tensor([0x100000, 0x900000, 0x100000], dtype=u64)
        dup_sz = torch.tensor([sm_span + 4096] * 3, dtype=u64)
        assert ev(dup_gvas, dup_sz, sm_span).tolist() == [0x50100000, 0x50900000, 0x50100000]
        assert len(seen_gvas) == 4  # 2 probes per unique GVA (base and base + span - 1)
        bad_sz = torch.tensor([sm_span + 4096, sm_span - 1], dtype=u64)
        for pair in (dup_gvas[:2], torch.tensor([0x100000, 0x100000], dtype=u64)):
            _r("Component at index 1 exceeds its leased allocation", ev, pair, bad_sz, sm_span)

        for h, q, md in [(16384, 8, "full"), (524288, 3, "sp"), (1048576, 3, "sp")]:
            r: Any = np.arange(q, dtype=np.int32)[:, None]
            b = ((idx * 17 + (r % 4) * 31) if md == "sp" else (idx + r * 2048)) % h
            if md != "full":
                b[:, -3:] = np.stack([np.full(q, -1, np.int32), h + r[:, 0], np.full(q, h - 1, np.int32)], 1)
            exp = np.unique(b[:, None][(b[:, None] >= 0) & (b[:, None] < h)])
            assert (md != "sp" or (len(exp) <= 8192 and len(exp) * 10 < h)) and (md != "full" or len(exp) == h)
            assert h != 1048576 or (1048575 in exp and 1048576 not in exp)
            ids, st = _run_union(ops, b[:, None], [h], [0], [q], max(128, ((len(exp) + 127) // 128) * 128))
            assert st[0, :5].tolist() == [len(exp), q, len(exp) + q, 0, 0] and np.all(ids[0, len(exp) :] == -7)
            ae(ids[0, : len(exp)], exp)

        # Dual-request heterogeneous union_plan sharing one scratch buffer without cross-request leakage.
        dual: Any = np.full((6, 1, 2048), -1, np.int32)
        dual[:2, 0, :5] = [[10, 20, 20, -1, 100], [100, 30, 10, -1, 16384]]
        r2 = [[20000, 20001, -1, 32768], [20001, 20002, 20003, -1], [20004, 20000, -1, -1], [20005, -1, -1, -1]]
        dual[2:, 0, :4] = r2
        d_ids, d_st = _run_union(ops, dual, [16384, 32768], [0, 2], [2, 4], 1024)
        exp0: Any = np.array([10, 20, 30, 100], np.int32)
        exp1: Any = np.arange(20000, 20006, dtype=np.int32)
        assert d_st[:, :5].tolist() == [[4, 2, 6, 0, 0], [6, 4, 10, 0, 0]]
        ae(d_ids[0, :4], exp0)
        ae(d_ids[1, :6], exp1)
        assert np.all(d_ids[0, 4:] == -7) and np.all(d_ids[1, 6:] == -7)

        full: Any = np.zeros((2, 1, 2048), np.int32)
        full[:, 0] = [idx, idx + 2048]
        o_i, o_s = _run_union(ops, full, [32768], [0], [2], 2048)
        oob, oob_neg, oob_1m = (_z(4, 1, 2048).numpy() for _ in range(3))
        oob[2, 0, 99], oob_neg[1, 0, 5], oob_1m[3, 0, 17] = 65540, -2, 1048580
        b_i, b_s = _run_union(ops, oob, [65536], [0], [4], 1024)
        n_i, n_s = _run_union(ops, oob_neg, [65536], [0], [4], 1024)
        m_i, m_s = _run_union(ops, oob_1m, [1048576], [0], [4], 1024)
        assert (o_s[0, 0], o_s[0, 4]) == (4096, 1) and b_s[0, 3] != 0 and n_s[0, 3] != 0 and m_s[0, 3] != 0
        assert np.all(o_i == -7) and np.all(b_i == -7) and np.all(n_i == -7) and np.all(m_i == -7)
        u_bad = _args()[0]
        _r("Invalid selected H/Q span", ops.selected_kv_union_plan, *u_bad[:3], [1048577], [0], [4], 1024)
        _r("Invalid union geometry", ops.selected_kv_union_plan, *u_bad[:3], [1048576], [0], [4], 1048704)
    finally:
        sim[1].selected_kv_sim_reset()


@pytest.mark.parametrize("case", ["topk_dtype", "topk_dim", "topk_noncontig", "overlap_starts", "tbl_stride"])
def test_torch_check_input_contracts(sim, case):
    ops = sim[0]
    u, c, _ = _args()
    if case == "overlap_starts":
        _r("Invalid selected H/Q span", ops.selected_kv_union_plan, *u[:3], [256, 256], [2, 1], [2, 2], 1024)
    elif case == "tbl_stride":
        _r("unit column stride", ops.selected_kv_inplace_copy, *c[:4], _z(4, 8)[:, ::2], *c[5:])
    else:
        bad = {"topk_dtype": u[0].to(torch.int64), "topk_dim": u[0].view(4, 2048)}.get(case)
        _r(
            "Invalid selected TopK",
            ops.selected_kv_union_plan,
            _z(4, 2, 2048)[:, :1, :] if bad is None else bad,
            *u[1:],
        )


@pytest.mark.parametrize("capacity", [256, 262144, 1048576])
def test_inplace_copy_double_buffer_traps_and_li_gather(sim, capacity):
    ops, lib = sim
    ic, li = ops.selected_kv_inplace_copy, ops.selected_kv_li_gather
    lib.selected_kv_sim_reset()
    try:
        h, q, rb, mf, np_ = 512, 16, 656, 2048, 12
        omitted, cache = {3, 4, 5}, GA((np_, 128, 1, rb), np.int8, 0x2A)
        pool, bases = _pool_bases(ops, lib, np_, (st_rb := mf + 128 * rb) + 512, 7, omitted, st_rb)
        chosen = np.array([17, 128, 129, 130, *range(140, 200, 5), 255, *range(260, 335, 5), 390, 391], np.int32)
        t6 = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [6, 7, 3, 8, 9], [9, 4, 10, 11, 1], [1, 2, 3, 4, 5], [7, 5, 8, 9, 10]]
        sels = [chosen, chosen[:5], chosen[5:12], chosen[:3], np.empty(0, np.int32), chosen]
        _copy_run(ops, pool, bases, cache, np.array(t6, np.int32), sels, omitted, capacity, h, q, mf, rb)
        assert lib.selected_kv_sim_peak_queue() == 2
        if capacity != 256:
            return
        _, bases = (lib.selected_kv_sim_reset(), _pool_bases(ops, lib, np_, st_rb + 512, 7, omitted, st_rb))[1]
        ti = lambda v: torch.tensor(v, dtype=torch.int32)
        ids_t, st_t = torch.full((1, 128), -1, dtype=torch.int32), ti([[17, 16, 18, 0, 0, 0, 0, 0]])
        ids_t[0, :17], ids_t[0, :3] = 0, ti([5, 7, 260])
        _r("destination domain", ic, ids_t, st_t, bases, ti([[0, 512, 0, 16]]), ti([[3, 3, 99, 3]]), cache.tensor, mf)
        assert lib.selected_kv_sim_trap_count() >= 1 and lib.selected_kv_sim_peak_queue() >= 1

        # Inject status[3]=1 and status[4]=1 under otherwise-valid cv/q metadata and verify Trap + clean cache.
        pat, mt, tbl = "Invalid original-page copy metadata or union status", ti([[0, 512, 0, 16]]), ti([[3, 3, 3, 3]])
        for err_col in (3, 4):
            lib.selected_kv_sim_reset()
            _, bases = _pool_bases(ops, lib, np_, st_rb + 512, 7, omitted, st_rb)
            cache_snap = cache.value.copy()
            v_ids, v_st = torch.full((1, 128), -1, dtype=torch.int32), ti([[1, 16, 2, 0, 0, 0, 0, 0]])
            v_ids[0, 0], v_st[0, err_col] = 5, 1
            _r(pat, ic, v_ids, v_st, bases, mt, tbl, cache.tensor, mf)
            assert lib.selected_kv_sim_trap_count() >= 1
            ae(cache.value, cache_snap)
        np_l, d_off, s_off, om_l = 32, 84000, 84000 + 16384, {1, 2, 4, 7, 17, 18, 20, 23, 28, 31}
        lib.selected_kv_sim_reset()
        pool_l, bases_l = _pool_bases(ops, lib, np_l, s_off + 384, 99, om_l, s_off + 256)
        dst_k, dst_s = GA((np_l, 128, 128), np.int8, 0x11), GA((np_l, 128), np.float16, 1.5)
        exp_k, exp_s = dst_k.value.copy(), dst_s.value.copy()
        for p in om_l:
            exp_k[p] = pool_l.value[p, d_off : d_off + 16384].view(np.int8).reshape(128, 128)
            exp_s[p] = pool_l.value[p, s_off : s_off + 256].view(np.float16)
        li(bases_l, dst_k.tensor, dst_s.tensor, d_off, s_off)
        assert (lib.selected_kv_sim_peak_queue(), lib.selected_kv_sim_trap_count()) == (2, 0)
        _ = ae(dst_k.value, exp_k), ae(dst_s.value.view(np.uint16), exp_s.view(np.uint16))
        bases_l.numpy()[16] = (1 << 64) - 10
        _r("LI component address overflows", li, bases_l, dst_k.tensor, dst_s.tensor, d_off, s_off)
        assert lib.selected_kv_sim_trap_count() >= 1 and GA.assert_all(pool_l, dst_k, dst_s) is None
    finally:
        lib.selected_kv_sim_reset()


def test_inplace_copy_1m_columns_and_multi_span_per_lane(sim):
    ops, lib = sim
    lib.selected_kv_sim_reset()
    try:
        # 1M history loads 8192 page-table columns; 70000 IDs over 8 lanes -> 8750/lane > MAX_LOCAL_IDS (8192).
        # sel[8191]=8192 and sel[8192]=8193 sit on page 64 rows 0..1 across the S_MTE2 multi-span boundary.
        h, q, rb, mf, np_ = 1048576, 4, 656, 2048, 550
        omitted, cache = set(range(548)), GA((np_, 128, 1, rb), np.int8, 0x2A)
        pool, bases = _pool_bases(ops, lib, np_, (st_rb := mf + 128 * rb) + 512, 13, omitted, st_rb)
        tbl_rows: Any = np.full((1, 8192), 549, dtype=np.int32)
        tbl_rows[0, :547], tbl_rows[0, 8191] = np.arange(547, dtype=np.int32), 547
        sel = np.concatenate([np.arange(1, 70000, dtype=np.int32), np.array([1048575], dtype=np.int32)])
        _copy_run(ops, pool, bases, cache, tbl_rows, [sel], omitted, 131072, h, q, mf, rb)
        assert (lib.selected_kv_sim_peak_queue(), lib.selected_kv_sim_trap_count()) == (2, 0)
    finally:
        lib.selected_kv_sim_reset()


def test_bind_hybm_dlopen_lifecycle(sim, tmp_path):
    ops, lib = sim
    cc = shutil.which("c++") or pytest.skip("A host C++ compiler is required")
    lib.selected_kv_sim_reset()
    try:

        def mk(tag: str, code: str) -> str:
            (s := tmp_path / f"f_{tag}.cpp").write_text(code)
            subprocess.run(
                [cc, "-std=c++17", "-shared", "-fPIC", str(s), "-o", o := str(tmp_path / f"libf_{tag}.so")], check=True
            )
            return o

        h_ty = '#include <cstdint>\nextern "C" int32_t hybm_gva_to_va(uint64_t g, uint32_t, uint64_t* o)'
        so_a = mk("a", f"{h_ty} {{ *o = g + 0x10000000ULL; return 0; }}\n")
        so_b = mk("b", f"{h_ty} {{ *o = g + 0x20000000ULL; return 0; }}\n")
        so_empty = mk("empty", 'extern "C" int unused_hybm_stub() { return 0; }\n')
        _r("candidate HyBM library paths", ops.selected_kv_bind_hybm, [])
        _r("already initialized native HyBM SDK", ops.selected_kv_bind_hybm, [so_a])

        _h_e, _h_a = ctypes.CDLL(so_empty), ctypes.CDLL(so_a)
        _ = ops.selected_kv_bind_hybm(["", so_empty, so_a]), ops.selected_kv_bind_hybm([so_a])
        gvas, sz = torch.tensor([0x100000, 0x900000], dtype=u64), torch.tensor([4096, 4096], dtype=u64)
        exp_bases = lambda: ops.selected_kv_export_bases(gvas, sz, 4096).tolist()
        assert exp_bases() == [0x10100000, 0x10900000]
        _h_b = ctypes.CDLL(so_b)
        _r("conflicts with an already bound SDK instance", ops.selected_kv_bind_hybm, [so_b])
        assert exp_bases() == [0x10100000, 0x10900000]
        lib.selected_kv_sim_reset()
        _r("requires an initialized HyBM mapper", exp_bases)
    finally:
        lib.selected_kv_sim_reset()
