# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
import time
from collections import namedtuple
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from math import prod
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.ascend_config import (
    SELECTED_KV_DEFAULT_MAX_QUERY_RATIO as DEFAULT_MAX_QUERY_RATIO,
)
from vllm_ascend.ascend_config import (
    SELECTED_KV_DEFAULT_MIN_HISTORY_TOKENS as DEFAULT_MIN_HISTORY_TOKENS,
)
from vllm_ascend.ascend_config import (
    SELECTED_KV_MAX_BATCHED_TOKENS as MAX_BATCHED_TOKENS,
)
from vllm_ascend.ascend_config import (
    SELECTED_KV_MAX_HISTORY_TOKENS as MAX_HISTORY_TOKENS,
)
from vllm_ascend.ascend_config import (
    SELECTED_KV_MAX_REQUESTS as MAX_REQUESTS,
)
from vllm_ascend.ascend_config import (
    get_ascend_config,
    resolve_ascend_config,
    selected_kv_enabled,
)
from vllm_ascend.ascend_config import (
    valid_selected_kv_max_query_ratio as valid_max_query_ratio,
)
from vllm_ascend.ascend_config import (
    valid_selected_kv_min_history_tokens as valid_min_history_tokens,
)
from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import cache_planner as cp
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import layerwise_cache_layout as lcl

PAGE_ROWS, MAIN_ROW_BYTES, KV_LORA_RANK, QK_ROPE_HEAD_DIM, TOPK_WIDTH = 128, 656, 512, 64, 2048
INDEXER_DATA_PAGE_BYTES, INDEXER_SCALE_PAGE_BYTES = PAGE_ROWS * 128, PAGE_ROWS * 2
NATIVE_ALIGNMENT, KERNEL_CORES, COUNTER_STRIDE = 2 * 1024 * 1024, 32, 8
BITMAP_INTS, RING_DEPTH, OBJECT_BASE_BYTES_PER_BLOCK = MAX_HISTORY_TOKENS // 32, 2, 16
MAIN_PAGE_BYTES, MEMCACHE_SSD_HIT_GVA = PAGE_ROWS * MAIN_ROW_BYTES, (1 << 64) - 1
LI_PAGE_BYTES = INDEXER_DATA_PAGE_BYTES + INDEXER_SCALE_PAGE_BYTES
ExporterFn = Callable[[NDArray[np.uint64], NDArray[np.uint64], int], NDArray[np.uint64]]
SCRATCH_SHAPES = ((MAX_HISTORY_TOKENS,), (BITMAP_INTS,), (2 * KERNEL_CORES, COUNTER_STRIDE))
SelectedRequest = namedtuple("SelectedRequest", "request_id history_tokens query_tokens selected")
MainComponent = namedtuple("MainComponent", "layer_name layer_id object_offset")
IndexerComponent = namedtuple("IndexerComponent", "offsets caches")
TrimmedPlan = namedtuple("TrimmedPlan", "ids gvas sizes replacements selected_req_ids domain")
_WS = namedtuple("_WS", "union_capacity max_requests meta_shape scratch_shapes output_shapes fixed_bytes")
_KB = namedtuple(
    "_KB",
    "available_bytes native_page_bytes native_padding_bytes fixed_bytes bytes_per_block maximum_blocks "
    "native_budget_bytes",
)


def classify_request(
    request_id: str,
    target_token_len: int,
    query_tokens: int,
    max_query_ratio: float = DEFAULT_MAX_QUERY_RATIO,
    min_history_tokens: int = DEFAULT_MIN_HISTORY_TOKENS,
) -> SelectedRequest:
    """Select cached prefill requests with H > min_history_tokens and Q / (H + Q) < max_query_ratio."""
    vq = type(query_tokens) is int and 0 < query_tokens <= target_token_len
    h, q = (target_token_len - query_tokens, query_tokens) if vq else (target_token_len, 0)
    ok = vq and valid_max_query_ratio(max_query_ratio) and valid_min_history_tokens(min_history_tokens)
    sel = ok and int(min_history_tokens) < h <= MAX_HISTORY_TOKENS and q / (h + q) < float(max_query_ratio)
    return SelectedRequest(request_id, h, q, sel)


def stop_transfer_threads(worker: Any, timeout: float, err_msg: str) -> list[Any]:
    """Stop send/recv threads with a bounded monotonic deadline before device sync.

    Must be called outside step.lock so a RecvingThread blocked on save_finished_events[0]
    wakes via stop() rather than deadlocking torch.npu.synchronize() or queue.join().
    """
    threads = [t for n in ("kv_send_thread", "kv_recv_thread") if (t := getattr(worker, n, None)) is not None]
    for t in threads:
        t.stop()
    deadline = time.monotonic() + timeout
    for t in threads:
        if t.is_alive():
            t.join(timeout=max(0.0, deadline - time.monotonic()))
    if any(t.is_alive() for t in threads):
        raise RuntimeError(err_msg)
    return threads


class SelectedWorkspaceSpec(_WS):
    @classmethod
    def from_bounds(cls, q_cap: int, max_seqs: int, mlen: int = MAX_HISTORY_TOKENS) -> SelectedWorkspaceSpec:
        if not 0 < q_cap <= MAX_BATCHED_TOKENS or max_seqs <= 0:
            raise ValueError("Selected KV workspace bounds exceed kernel limits")
        reqs, raw = min(MAX_REQUESTS, max_seqs, q_cap), min(mlen, MAX_HISTORY_TOKENS, TOPK_WIDTH * q_cap)
        cap = min(MAX_HISTORY_TOKENS, max(128, ((raw + 127) // 128) * 128))
        outs = ((reqs, cap), (reqs, COUNTER_STRIDE))
        return cls(cap, reqs, (reqs, 4), SCRATCH_SHAPES, outs, 4 * sum(map(prod, (*SCRATCH_SHAPES, *outs))) + 32 * reqs)


class SelectedKVBudget(_KB):
    @classmethod
    def reserve(cls, avail: int, page_bytes: int, fixed: int, per_block: int, pad: int = 0) -> SelectedKVBudget:
        if avail <= 0 or page_bytes <= 0 or fixed < 0 or per_block < 0:
            raise ValueError("Workspace budgets must be positive with nonnegative overhead bytes")
        max_blocks = (avail - pad) // page_bytes
        native_budget = avail - pad - fixed - per_block * max_blocks
        if max_blocks <= 0 or native_budget < page_bytes:
            raise ValueError("Insufficient physical memory for native KV capacity and selected-KV workspace")
        return cls(avail, page_bytes, pad, fixed, per_block, max_blocks, native_budget)

    def validate(self, num_blocks: int, native_storage: int | None = None) -> None:
        if type(num_blocks) is not int or not 0 < num_blocks <= self.maximum_blocks:
            raise ValueError("Final native block count exceeds the selected-KV workspace reservation")
        nat = (
            num_blocks * self.native_page_bytes + self.native_padding_bytes
            if native_storage is None
            else native_storage
        )
        if nat + self.fixed_bytes + self.bytes_per_block * num_blocks > self.available_bytes:
            raise ValueError("Native KV storage plus selected-KV workspace exceeds the physical memory budget")


def reserve_selected_kv_workspace(worker: Any, available_bytes: Any = None) -> Any:
    avail = worker.available_kv_cache_memory_bytes if available_bytes is None else available_bytes
    acfg = resolve_ascend_config(worker, get_ascend_config)
    if not selected_kv_enabled(acfg):
        return avail
    if get_ascend_device_type() != AscendDeviceType.A3:
        raise ValueError("Selected KV kernels currently require an A3 build")
    vcfg, cc = worker.vllm_config, worker.cache_config
    ac = getattr(vcfg, "attention_config", None)
    c8 = (acfg.enable_sparse_sfa_c8, acfg.enable_sparse_li_c8, acfg.enable_sparse_li_c4) == (True, True, False)
    dt_ok = (getattr(cc, "cache_dtype", None), getattr(ac, "indexer_kv_dtype", None)) == ("int8", "int8")
    excl = (
        acfg.kvpp_config.size != 1
        or acfg.sparse_kv_offload_config.enabled
        or getattr(ac, "hisparse_config", None) is not None
        or getattr(getattr(acfg, "xlite_graph_config", None), "enabled", False)
    )
    if not (c8 and dt_ok) or excl:
        raise ValueError(
            "Selected KV requires int8 C8 main and indexer caches without KVPP, HiSparse, Xlite, or offload"
        )
    runner, s = worker.model_runner, vcfg.scheduler_config
    names, page_bytes, pad = native_geometry(runner, worker.get_kv_cache_spec())
    topk_owners(runner, names)
    mlen = vcfg.model_config.max_model_len
    spec = SelectedWorkspaceSpec.from_bounds(s.max_num_batched_tokens, s.max_num_seqs, mlen)
    budget = SelectedKVBudget.reserve(int(avail), page_bytes, spec.fixed_bytes, OBJECT_BASE_BYTES_PER_BLOCK, pad)
    if (ovr := getattr(cc, "num_gpu_blocks_override", None)) is not None:
        budget.validate(ovr)
    runner.selected_kv_budget = budget
    return budget.native_budget_bytes


def probe_reuse_layout(vllm_config: Any, specs: Mapping[str, Any], num_blocks: int = 1) -> lcl.LayerwiseReuseLayout:
    rcfg, rules = cp.get_layerwise_offload_config(vllm_config.kv_transfer_config), cp.get_spec_reuse_rules(vllm_config)
    if rcfg is None or rules is None or not specs:
        raise ValueError("Selected KV requires the native layerwise reuse connector configuration")
    tensors = [
        KVCacheTensor(size=num_blocks * s.page_size_bytes, layers=[k], layer_stride=0, block_stride=s.page_size_bytes)
        for k, s in specs.items()
    ]
    group = KVCacheGroupSpec(list(specs), UniformTypeKVCacheSpecs(block_size=PAGE_ROWS, kv_cache_specs=dict(specs)))
    probe = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=tensors, kv_cache_groups=[group])
    return cp.LayerwiseCachePlanner(probe, vllm_config, rcfg, spec_reuse_rules=rules).build()


def ordered_layer_specs(
    reuse: Any, specs: Mapping[str, Any], expected_layers: int
) -> list[tuple[str, Any, str | None, Any]]:
    r_ok = reuse is not None and reuse.has_layer_reuse and not reuse.independent_layers and not reuse.draft_entry_names
    if not (r_ok and tuple(reuse.layers) == tuple(range(expected_layers))):
        raise ValueError("Selected KV requires two native reuse slots and no independent or draft layers")
    slot_names = [n for t in reuse.kv_cache_tensors for n in t.layers]
    if len(slot_names) != len(set(slot_names)) or set(slot_names) != set(specs):
        raise ValueError("Selected KV must account for every native main and indexer cache")
    mains: dict[int, list[tuple[str, Any]]] = {}
    lis: dict[int, list[tuple[str, Any]]] = {}
    for name, sp in specs.items():
        if not 0 <= (lid := reuse.cache_layer_indices.get(name, -1)) < expected_layers:
            raise ValueError("Selected KV cache layer index is out of range")
        if not isinstance(sp, (AscendSFAIndexerCacheSpec, MLAAttentionSpec)):
            raise ValueError("Selected KV only supports MLA main and SFA indexer cache specs")
        (lis if isinstance(sp, AscendSFAIndexerCacheSpec) else mains).setdefault(lid, []).append((name, sp))
    rows: list[tuple[str, Any, str | None, Any]] = []
    exp_m = (torch.int8, PAGE_ROWS, MAIN_PAGE_BYTES)
    exp_l = (torch.int8, torch.float16, PAGE_ROWS, LI_PAGE_BYTES)
    for lid in range(expected_layers):
        m_l, l_l = mains.get(lid, ()), lis.get(lid, ())
        if len(m_l) != 1 or (m_l[0][1].dtype, m_l[0][1].block_size, m_l[0][1].page_size_bytes) != exp_m:
            raise ValueError("Selected KV requires int8 packed C8 main page geometry (128 x 656B)")
        if len(l_l) > 1 or (
            l_l
            and (
                l_l[0][1].dtype,
                getattr(l_l[0][1], "scale_dtype", None),
                l_l[0][1].block_size,
                l_l[0][1].page_size_bytes,
            )
            != exp_l
        ):
            raise ValueError("Selected KV requires int8 C8 128-row indexer pages (16640 bytes/page) within each slot")
        rows.append((m_l[0][0], m_l[0][1], l_l[0][0] if l_l else None, l_l[0][1] if l_l else None))
    if not any(li_n is not None for _, _, li_n, _ in rows):
        raise ValueError("Selected KV requires native layerwise indexer pages")
    m_set, l_set = {m_n for m_n, _, _, _ in rows}, {li_n for _, _, li_n, _ in rows if li_n is not None}
    m_s = [t for t in reuse.kv_cache_tensors if t.layers and set(t.layers) <= m_set]
    l_s = [t for t in reuse.kv_cache_tensors if t.layers and set(t.layers) <= l_set]
    st_ok = all(t.layer_stride == 0 and t.block_stride == MAIN_PAGE_BYTES for t in m_s) and all(
        t.layer_stride == 0 and t.block_stride == LI_PAGE_BYTES for t in l_s
    )
    if len(m_s) != 2 or not 1 <= len(l_s) <= 2 or len(m_s) + len(l_s) != len(reuse.kv_cache_tensors) or not st_ok:
        raise ValueError("Selected KV requires two native reuse slots and 1..2 indexer slots")
    return rows


def native_geometry(runner: Any, specs: dict[str, Any]) -> tuple[tuple[str, ...], int, int]:
    layers = runner.model_config.get_num_layers(runner.parallel_config)
    reuse = probe_reuse_layout(runner.vllm_config, specs, num_blocks=1)
    rows = ordered_layer_specs(reuse, specs, layers)
    return tuple(m_n for m_n, _, _, _ in rows), reuse.kv_cache_tensors[0].size, NATIVE_ALIGNMENT


def _topk_buf_ptr(impl: Any) -> int | None:
    buf = getattr(impl, "topk_indices_buffer", None)
    return int(buf.data_ptr()) if buf is not None else None


def topk_owners(runner: Any, names: Sequence[str]) -> dict[str, str]:
    ctx, owners, prev, prev_impl, dims = (
        runner.compilation_config.static_forward_context,
        {},
        None,
        None,
        (KV_LORA_RANK, QK_ROPE_HEAD_DIM),
    )
    for name in names:
        impl = ctx[name].impl
        if (getattr(impl, "kv_lora_rank", None), getattr(impl, "qk_rope_head_dim", None)) != dims:
            raise ValueError("Selected KV kernels require kv_lora_rank=512 and qk_rope_head_dim=64")
        if not impl.skip_topk:
            if getattr(getattr(impl, "indexer", None), "topk_output_width", None) != TOPK_WIDTH:
                raise ValueError("Selected KV kernels require each TopK owner to output 2048 indices")
            prev, prev_impl = name, impl
        elif prev is None or prev_impl is None:
            raise ValueError("Selected KV followers require an earlier TopK owner")
        elif hasattr(impl, "topk_indices_buffer") or hasattr(prev_impl, "topk_indices_buffer"):
            f_ptr, o_ptr = _topk_buf_ptr(impl), _topk_buf_ptr(prev_impl)
            if f_ptr is None or f_ptr != o_ptr or getattr(prev_impl, "use_index_cache", True) is False:
                raise ValueError("Selected KV followers must share topk_indices_buffer with the preceding TopK owner")
        owners[name] = prev
    return owners


def discover_components(worker: Any) -> tuple[tuple[MainComponent, ...], tuple[IndexerComponent | None, ...]]:
    mode = (worker.use_layerwise_transfer, worker.layerwise_offload, worker.use_block_key_layerwise, worker.load_async)
    topo = (worker.num_kv_cache_groups, worker.grouped_block_size, worker.pp_size, worker.dcp_size)
    inv = hasattr(worker, "_invalid_block_ids") and hasattr(worker, "_invalid_block_ids_lock")
    kcfg = getattr(worker, "kv_cache_config", None)
    recv_ok = getattr(worker, "dspark_prefix_cache", None) is None and worker.kv_recv_thread is not None
    if mode != (True, True, False, False) or topo != (1, [PAGE_ROWS], 1, 1) or not (inv and recv_ok and kcfg):
        raise ValueError("Selected main reads require synchronous native GVA layerwise geometry and recv thread")
    specs = lcl.get_layerwise_kv_cache_specs(kcfg)
    rows, recv = ordered_layer_specs(worker._layerwise_reuse_layout, specs, worker.num_layers), worker.kv_recv_thread
    bld = recv.group_builders[0] if recv.group_builders else recv.layer_batch_builder
    blen, baddrs, bstride = bld._block_len_np, bld._kv_caches_base_addr_np, bld._block_stride_np
    eoffs, boff = bld._layer_cache_entry_offsets_np, int(bld.layer_byte_offset)

    def locate(lid: int, cache: torch.Tensor, expected: int) -> int:
        m = np.flatnonzero(baddrs[(f := int(eoffs[lid])) : int(eoffs[lid + 1])] == cache.data_ptr())
        if m.size != 1:
            raise ValueError("Cache identity does not uniquely match native registration")
        if (int(blen[idx := f + int(m[0])]), int(bstride[idx])) != (expected, expected):
            raise ValueError("Registered page extent differs from native C8 geometry")
        return int(boff + blen[:idx].sum())

    mains: list[MainComponent] = []
    indexers: list[IndexerComponent | None] = []
    nb = worker.num_blocks
    msig = (torch.int8, (nb, PAGE_ROWS, 1, MAIN_ROW_BYTES), True)
    lspecs = ((torch.int8, INDEXER_DATA_PAGE_BYTES), (torch.float16, INDEXER_SCALE_PAGE_BYTES))
    exp_li = [(d, True, nb, nb * n) for d, n in lspecs]
    for lid, (m_name, _, li_name, _) in enumerate(rows):
        c = worker._as_cache_tuple(worker.kv_caches[m_name])
        if len(c) != 1 or (c[0].dtype, tuple(c[0].shape), c[0].is_contiguous()) != msig:
            raise ValueError("Selected main reads require the original packed C8 cache view")
        mains.append(MainComponent(m_name, lid, locate(lid, c[0], MAIN_PAGE_BYTES)))
        if li_name is None:
            indexers.append(None)
            continue
        li = worker._as_cache_tuple(worker.kv_caches[li_name])
        if [(t.dtype, t.is_contiguous(), t.shape[0], t.nbytes) for t in li] != exp_li:
            raise ValueError("Selected LI copies require original C8 data and FP16 scale pages")
        offs = (locate(lid, li[0], lspecs[0][1]), locate(lid, li[1], lspecs[1][1]))
        indexers.append(IndexerComponent(offs, (li[0], li[1])))
    return tuple(mains), tuple(indexers)


def _safe_resolve(p: Path) -> Path | None:
    try:
        return p.resolve(strict=True)
    except OSError:
        return None


def _mod_dirs(mod: Any) -> set[Path]:
    if not isinstance(f := getattr(mod, "__file__", None), str) or not f:
        return set()
    p, rp = Path(f), _safe_resolve(Path(f))
    return {p.parent, *((rp.parent,) if rp is not None else ())}


def _store_targets(raw: Any) -> tuple[str, ...]:
    m0 = m if isinstance(m := getattr(type(raw), "__module__", None), str) else ""
    own = m0.partition(".")[0]
    if not own or own == "builtins":
        return ()
    own_dirs = _mod_dirs(sys.modules.get(m0)) | _mod_dirs(sys.modules.get(own))
    owners = [own]
    if own_dirs:
        for k, mod in tuple(sys.modules.items()):
            pk = k.partition(".")[0]
            if pk and pk != "builtins" and hasattr(mod, "__path__") and (_mod_dirs(mod) & own_dirs):
                owners.append(pk)
    allowed = set(owners) | {"memfabric_hybrid", "memcache_hybrid"}
    found = [m0, *owners]
    if hasattr(raw, "__dict__"):
        for v in vars(raw).values():
            if v is not None and isinstance(vm := getattr(type(v), "__module__", None), str):
                if vm.partition(".")[0] in allowed:
                    found.append(vm)
    for key in tuple(dict.fromkeys(found)):
        if (m_obj := sys.modules.get(key)) is not None and hasattr(m_obj, "__dict__"):
            for val in vars(m_obj).values():
                attr = "__name__" if isinstance(val, type(sys)) else "__module__"
                if isinstance(cand := getattr(val, attr, None), str) and cand.partition(".")[0] in allowed:
                    found.append(cand)
    return tuple(dict.fromkeys(found))


def _hybm_candidate_paths(targets: Sequence[str]) -> list[str]:
    ext_paths: list[str] = []
    pkg_roots: list[Path] = []
    target_dirs: list[Path] = []
    loaded = tuple(sys.modules.items())
    for target in targets:
        if not (pkg := target.partition(".")[0]) or pkg == "builtins":
            continue
        mods = sorted(
            ((k, m) for k, m in loaded if k == pkg or k.startswith(f"{pkg}.")),
            key=lambda kv: (kv[0] != target, kv[0] != pkg, kv[0]),
        )
        for k, mod in mods:
            if not isinstance(f := getattr(mod, "__file__", None), str) or not f:
                continue
            p, rp = Path(f), _safe_resolve(Path(f))
            if p.suffix in (".so", ".dylib") or ".so." in p.name:
                ext_paths.extend((f, *((str(rp),) if rp is not None else ())))
            dirs = (p.parent, *((rp.parent,) if rp is not None else ()))
            if k == pkg:
                pkg_roots.extend(dirs)
            elif k == target:
                target_dirs.extend(dirs)
    rels = ("lib/libmf_hybm_core.so", "libmf_hybm_core.so")
    lib_paths = [
        s
        for d in dict.fromkeys((*pkg_roots, *target_dirs))
        for r in rels
        for cp_p in (d / r,)
        for rp in (_safe_resolve(cp_p),)
        for s in (str(cp_p), *((str(rp),) if rp is not None else ()))
    ]
    return list(dict.fromkeys((*ext_paths, *lib_paths)))


def load_default_exporter(worker: Any = None) -> ExporterFn:
    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()
    if worker is not None:
        if (raw := getattr(getattr(worker, "m_store", None), "store", None)) is None:
            raise RuntimeError("Selected KV requires an initialized MemcacheBackend store")
        targets: Sequence[str] = _store_targets(raw)
    else:
        targets = ("memcache_hybrid", "memfabric_hybrid")
    if not (cands := _hybm_candidate_paths(targets)):
        raise RuntimeError("Selected KV requires the already initialized native HyBM SDK (libmf_hybm_core.so)")
    torch.ops._C_ascend.selected_kv_bind_hybm(cands)
    op = torch.ops._C_ascend.selected_kv_export_bases
    return lambda g, s, sp: op(torch.from_numpy(g), torch.from_numpy(s), sp).numpy()


def _hit_tokens(req: Any) -> int:
    if (ls := req.load_spec) is None:
        return 0
    return int(ls.kvpool_cached_tokens if ls.kvpool_store_skip_tokens is None else ls.kvpool_store_skip_tokens)


def _request_gvas(req: Any, hist: int, cnt: int) -> NDArray[np.uint64] | None:
    locs = getattr(req, "load_block_gvas_by_group_np", None)
    raw = locs[0] if locs else getattr(req, "load_block_gvas_np", None)
    if raw is None or (arr := np.asarray(raw)).ndim != 1 or arr.dtype.kind not in "iu" or cnt > arr.size:
        return None
    gvas = arr[:cnt].astype(np.uint64, copy=True)
    pt = getattr(req, "partial_load_gva_per_group", None)
    if hist % PAGE_ROWS and gvas[-1] == 0 and pt and (_hit_tokens(req) - 1) // PAGE_ROWS == cnt - 1:
        gvas[-1] = np.uint64(pt[0]) if int(pt[0]) > 0 else np.uint64(0)
    return gvas


def trim_load_plan(reqs: Any, plans: Any, leased: dict[int, int], span: int, comps: Any, worker: Any) -> Any:
    """Identify exclusive fully-cached history blocks whose native Main-KV SDMA load can be omitted."""
    loads, saves = worker.layer_load_tasks, worker.layer_save_tasks
    if len(loads) != len(comps) or len(saves) != len(comps):
        return None
    refs, full_ids, s_ids, s_gvas, sel_reqs = [], [], [], [], set()
    for req, plan in zip(reqs, plans):
        grps = getattr(req, "block_ids_by_group_np", None) or getattr(req, "block_ids_by_group", None)
        pages = (req.target_token_len + PAGE_ROWS - 1) // PAGE_ROWS
        tbl = np.asarray(grps[0]) if grps else np.empty(0, dtype=np.int64)
        if tbl.ndim != 1 or tbl.dtype.kind not in "iu" or pages > tbl.size:
            return None
        blocks: NDArray[np.int64] = tbl[:pages].astype(np.int64, copy=False)
        if np.any(blocks <= 0) or np.any(blocks >= worker.num_blocks):
            return None
        refs.append(np.unique(blocks))
        cnt = (plan.history_tokens + PAGE_ROWS - 1) // PAGE_ROWS
        gvas = _request_gvas(req, plan.history_tokens, cnt) if plan.selected and cnt > 0 else None
        ok = req.load_spec is not None and req.load_spec.can_load and _hit_tokens(req) >= plan.history_tokens
        ok = ok and gvas is not None and int(getattr(req, "load_gva_block_offset", 0)) == 0
        took = gvas is not None and ok and all(leased.get(int(g), 0) >= span for g in gvas)
        if took:
            sel_reqs.add(plan.request_id)
            s_ids.append(blocks[:cnt])
            s_gvas.append(gvas)
        # Keep unaligned partial-history tail blocks and unselected requests on native SDMA.
        full_ids.append(blocks[cnt - bool(plan.history_tokens % PAGE_ROWS) if took else 0 :])
    if not s_ids:
        return None
    ord_i = np.argsort(cat_i := np.concatenate(s_ids), kind="stable")
    ids_s, gvas_s = cat_i[ord_i], np.concatenate(s_gvas)[ord_i]
    if np.any((same := ids_s[1:] == ids_s[:-1]) & (gvas_s[1:] != gvas_s[:-1])):
        return None
    sel_ids, sel_gvas = ids_s[u := np.concatenate(([True], ~same))], gvas_s[u]

    entries, layer_sets = {}, []
    for comp, tasks in zip(comps, loads):
        objs = set()
        for t in tasks:
            sh = t.shared_block_data
            sig = None if sh is None else (t.group_id, t.layer_idx_in_group, t.use_key_major_ranges, sh.block_keys)
            if sh is None or sig != (0, comp.layer_id, False, None) or sh.block_gvas_arr is None:
                return None
            key = id(sh)
            if key not in entries:
                bi, bg = np.asarray(sh.block_ids_arr), np.asarray(sh.block_gvas_arr)
                if bi.ndim != 1 or bg.ndim != 1 or bi.shape != bg.shape:
                    return None
                pgs = bi[take := np.isin(bi, sel_ids)]
                pos = cast(NDArray[np.intp], np.searchsorted(sel_ids, pgs))
                if np.any(bg[take].astype(np.uint64) != sel_gvas[pos]):
                    return None
                entries[key] = (sh, take, pos, pgs)
            objs.add(key)
        layer_sets.append(frozenset(objs))
    for ks in set(layer_sets):
        if not ks or not np.array_equal(np.unique(np.concatenate([entries[k][3] for k in ks])), sel_ids):
            return None

    # Blocks shared across requests (counts > 1), scheduled for save, or in full_ids stay native;
    # their entries in object_bases remain 0 so ResolveRow skips redundant Main-KV row copies.
    s_sh = (t.shared_block_data for ts in saves for t in ts if t.shared_block_data is not None)
    sv = [np.asarray(sh.block_ids_arr) for sh in s_sh if sh.block_ids_arr is not None]
    saved = np.unique(np.concatenate(sv)) if sv else np.empty(0, dtype=np.int64)
    ref_ids, counts = np.unique(np.concatenate(refs), return_counts=True)
    omitted = ~np.isin(sel_ids, np.concatenate([ref_ids[counts > 1], saved, *full_ids]))
    if not omitted.any():
        return None
    repls = {}
    for oid, (sh, matched, pos, _) in entries.items():
        (keep := np.ones(matched.shape, dtype=np.bool_))[matched] = ~omitted[pos]
        if keep.all():
            repls[oid] = sh
        else:
            ids_k, gvas_k = sh.block_ids_arr[keep], cast(np.ndarray, sh.block_gvas_arr)[keep]
            repls[oid] = replace(sh, block_ids_arr=ids_k, block_gvas_arr=gvas_k)
    fi, fg = sel_ids[omitted], sel_gvas[omitted]
    sz = np.array([leased[int(g)] for g in fg], dtype=np.uint64)
    return TrimmedPlan(fi, fg, sz, repls, frozenset(sel_reqs), int(sel_ids.max()) + 1)


def build_trimmed_load_tasks(loads: list[list[Any]], repls: dict[int, Any]) -> Any:
    l01, cands = [False, False], []
    for lid, tasks in enumerate(loads):
        nt, has_b = [], False
        for t in tasks:
            tr = repls[id(t.shared_block_data)]
            nt.append(t if tr is t.shared_block_data else replace(t, shared_block_data=tr))
            has_b = has_b or tr.block_ids_arr.size > 0
        if lid < 2:
            l01[lid] = has_b
        if not has_b and tasks and hasattr(tasks[0], "block_ranges"):
            # Layer 1 retains an empty task sentinel so _submit_ready_layer_loads still registers defer_indexer(1, 0).
            nt = [replace(tasks[0], block_ranges=[], shared_block_data=None)] if lid == 1 else []
        cands.append(nt)
    return cands, (l01[0], l01[1])
