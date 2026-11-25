# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

if not (pytest.importorskip("torch_npu") and torch.npu.is_available()):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)


@pytest.mark.parametrize("h,q", [(65536, 1), (65536, 128), (65537, 125), (65663, 127), (262144, 2048), (1048576, 4)])
@torch.inference_mode()
def test_selected_kv_ops_preserve_exact_sfa_and_li_pages(h, q):
    from vllm_ascend.device.device_config import get_ascend_device_type
    from vllm_ascend.device.hardware import AscendDeviceType
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.selected_kv_plan import (
        SelectedWorkspaceSpec,
    )
    from vllm_ascend.utils import enable_custom_op

    if get_ascend_device_type() != AscendDeviceType.A3:
        pytest.skip("A3 selected-KV kernel")
    t, dev, rows, moff, tc = torch, torch.device("npu:0"), 128, 4096, torch.testing.assert_close
    reqs, sel_rows = (4, (0, 2, 3)) if h <= 262144 else (3, (0, 2))
    cols, gen, _ = (h + q + rows - 1) // rows, t.Generator().manual_seed(31031), enable_custom_op()
    npages, sh = reqs * cols + 2, (reqs * cols + 2, rows, 1)
    table_cpu = t.randperm(reqs * cols, generator=gen).reshape(reqs, cols).to(t.int32) + 1
    table_cpu[list(sel_rows), 0] = table_cpu[1, 0]
    p_n = t.randint(-8, 8, (*sh, 512), generator=gen, dtype=t.int8)
    p_r = t.randn((*sh, 64), generator=gen).to(t.bfloat16).view(t.int8)
    ref_cpu = t.cat((p_n, p_r, (t.rand((*sh, 4), generator=gen) * 0.05 + 0.05).view(t.int8)), dim=-1)
    ref, src, dst_cpu = ref_cpu.to(dev), ref_cpu.to(dev).clone(), ref_cpu.clone()
    om = sorted({p for r in sel_rows for p in table_cpu[r, 1 : h // rows].tolist()})
    keep, dst_cpu[om] = [p for p in range(npages) if p not in set(om)], -1
    exp_cpu, dst, d_off, s_off = dst_cpu.clone(), dst_cpu.to(dev), 96 * 1024, 96 * 1024 + 16384
    li_src = t.randint(-128, 127, (npages, s_off + 256), generator=gen, dtype=t.int8).to(dev)
    b_fn = lambda s, st: t.tensor([s + p * st if p in om else 0 for p in range(npages)], dtype=t.uint64).to(dev)
    bases, li_bases = b_fn(src.data_ptr() - moff, rows * 656), b_fn(li_src.data_ptr(), s_off + 256)
    z_dev = lambda s, dt: t.zeros(s, dtype=dt, device=dev)
    li_k, li_s = z_dev((npages, 128, 128), t.int8), z_dev((npages, 128), t.float16)
    lq, base = t.arange(q, dtype=t.int32)[:, None], t.arange(2048, dtype=t.int32)[None, :] * 17
    topk_cpu = t.cat([t.cat([((base + lq * 79 + r * 11) % h)[:, :-2], h + lq, lq * 0 - 1], 1) for r in range(reqs)], 0)
    topk_cpu, exp_ids, om_t = topk_cpu.reshape(reqs * q, 1, 2048), [], t.tensor(om, dtype=t.int32)
    for row in sel_rows:
        flat = topk_cpu[row * q : (row + 1) * q].flatten()
        exp_ids.append(ids_i := t.unique(flat[(flat >= 0) & (flat < h)], sorted=True))
        pages, mask = table_cpu[row, ids_i // rows], t.isin(table_cpu[row, ids_i // rows], om_t)
        exp_cpu[pages[mask], (ids_i % rows)[mask]] = ref_cpu[pages[mask], (ids_i % rows)[mask]]
    assert all(not t.equal(a, b) for i, a in enumerate(exp_ids) for b in exp_ids[i + 1 :])
    ops, n_sel = t.ops._C_ascend, len(sel_rows)
    spec = SelectedWorkspaceSpec.from_bounds(reqs * q, n_sel, h)
    sc, out = ([t.empty(s, dtype=t.int32, device=dev) for s in sp] for sp in (spec.scratch_shapes, spec.output_shapes))
    padded, topk = z_dev((reqs, cols + 3), t.int32), topk_cpu.to(dev)
    padded[:, :cols].copy_(table_cpu.to(dev))
    meta = t.tensor([[r, h, r * q, q] for r in sel_rows], dtype=t.int32, device=dev)
    ops.selected_kv_union_plan(topk, sc, out, [h] * n_sel, [r * q for r in sel_rows], [q] * n_sel, spec.union_capacity)
    ops.selected_kv_inplace_copy(out[0], out[1], bases, meta, padded[:, :cols], dst, moff)
    _ = ops.selected_kv_li_gather(li_bases, li_k, li_s, d_off, s_off), t.npu.synchronize()
    for i, exp in enumerate(exp_ids):
        assert out[1].cpu()[i, :2].tolist() == [len(exp), q] and out[1].cpu()[i, 3:5].tolist() == [0, 0]
        tc(out[0].cpu()[i, : len(exp)], exp, atol=0, rtol=0)
    _ = tc(dst.cpu(), exp_cpu, atol=0, rtol=0), tc(src.cpu(), ref_cpu, atol=0, rtol=0)
    tc(li_k[om].cpu(), li_src[om, d_off : d_off + 16384].cpu().view(-1, 128, 128))
    tc(li_s[om].view(t.int16).cpu(), li_src[om, s_off : s_off + 256].view(t.int16).cpu())
    assert t.all(li_k[keep].cpu() == 0) and t.all(li_s[keep].view(t.int16).cpu() == 0)
    ql, kl = t.arange(1, reqs + 1, dtype=t.int32, device=dev) * q, t.full((reqs,), h + q, dtype=t.int32, device=dev)
    kw = {"query": t.randn((reqs * q, 16, 576), generator=gen).to(t.bfloat16).to(dev), "sparse_indices": topk}
    kw |= {"scale_value": 576**-0.5, "sparse_block_size": 1, "block_table": table_cpu.to(dev), "layout_query": "TND"}
    kw |= {"layout_kv": "PA_BSND", "sparse_mode": 3, "actual_seq_lengths_query": ql, "actual_seq_lengths_kv": kl}
    kw |= {"attention_mode": 2, "tile_size": 128, "quant_scale_repo_mode": 1, "key_quant_mode": 2}
    kw |= {"value_quant_mode": 2, "rope_head_dim": 64, "return_softmax_lse": False}
    fn = ops.npu_kv_quant_sparse_flash_attention_vllm
    sfa = lambda c: r[0] if isinstance(r := fn(key=c, value=c, **kw), tuple) else r
    full_out, sel_out = sfa(ref), sfa(dst)
    # Perturb omitted pages in ref so non-triviality fails if SFA reads zero/garbage from unselected slots.
    pert = ref.clone()
    pert[om] = ~pert[om]
    pert_out = sfa(pert)
    t.npu.synchronize()
    assert t.isfinite(sel_out.cpu()).all()
    assert not t.equal(pert_out.cpu(), full_out.cpu())
    tc(sel_out.cpu(), full_out.cpu(), atol=0, rtol=0)

    import torch_npu

    if hasattr(torch_npu, "npu_kv_quant_sparse_flash_attention"):
        cann_kw = {k: v for k, v in kw.items() if k not in ("query", "sparse_indices", "return_softmax_lse")}
        cann_fn = torch_npu.npu_kv_quant_sparse_flash_attention
        cann_sfa = lambda c: r[0] if isinstance(r := cann_fn(kw["query"], c, c, topk, **cann_kw), tuple) else r
        cann_full, cann_sel = cann_sfa(ref), cann_sfa(dst)
        t.npu.synchronize()
        assert t.isfinite(cann_sel.cpu()).all()
        tc(cann_sel.cpu(), cann_full.cpu(), atol=0, rtol=0)
