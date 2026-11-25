# SPDX-License-Identifier: Apache-2.0
"""Heterogeneous stream runtime for on-demand Main-KV prefill over GVA layerwise AscendStore.

Coordinates ``compute_stream``, ``_h2d_stream``, ``_li_stream``, and ``_copy_stream`` across a
``RING_DEPTH = 2`` metadata/event ring: C8 Lightning Indexer pages gather on ``_li_stream`` behind
preceding Main-KV gathers, while 32-core UB-bitset ``selected_kv_union_plan`` and
``selected_kv_inplace_copy`` run on ``compute_stream`` and ``_copy_stream``. HyBM bindings resolve
via ``RTLD_NOLOAD`` and ``selected_kv_export_bases`` probes every ``2 MiB`` slab boundary per step;
``ddr_done`` fences cross-step read-lease refcounting in ``SelectedKVPoolWorker``, and post-commit
failures stop transfer threads and synchronize NPU streams before releasing leases.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
import vllm.envs as envs_vllm
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.logger import logger
from vllm.v1.worker.gpu.kv_connector import KVConnector

from vllm_ascend.ascend_config import selected_kv_enabled
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import layerwise_cache_layout as lcl
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import selected_kv_plan as pl
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.selected_kv_plan import (
    discover_components,
    stop_transfer_threads,
)

SELFTEST_PAD_BYTES = 64


def _same_table(a: Any, b: Any) -> bool:
    return a is b or (
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and (a.device, a.dtype, a.shape, a.stride(), a.data_ptr())
        == (b.device, b.dtype, b.shape, b.stride(), b.data_ptr())
    )


class AscendSFASelectedKVImpl(AscendSFAImpl):
    selected_kv_runtime: Any = None

    def _execute_sparse_flash_attention_process(
        self,
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table=None,
    ):
        rt, name = self.selected_kv_runtime, self.layer_name or ""
        active = rt is not None and rt.has_active_step and name in rt.owner_by_layer
        if active:
            if block_table is not None and not _same_table(block_table, getattr(attn_metadata, "block_table", None)):
                raise ValueError("Selected loading does not support overriding attn_metadata.block_table")
            rt.load_layer(name, topk_indices, (kv_cache[0],), attn_metadata)
        a = (ql_nope, q_pe, kv_cache, topk_indices, attn_metadata, actual_seq_lengths_query, actual_seq_lengths_key)
        out = super()._execute_sparse_flash_attention_process(*a, block_table=block_table)
        if active:
            rt.prefetch_next(name)
        return out


@dataclass
class _RingSlot:
    """Per-step ring slot isolating async H2D staging and cross-step stream fences."""

    host_bases: torch.Tensor
    host_meta: torch.Tensor
    dev_bases: torch.Tensor
    dev_meta: torch.Tensor
    li_evs: list[Any]
    main_evs: list[Any]
    attn_evs: list[Any]
    h2d_ev: Any
    entry_ev: Any
    compute_done: Any
    ddr_done: Any
    ddr_fence: Any = None


@dataclass
class _ActiveStep:
    """Mutable state for one committed selected-KV forward pass."""

    slot: int
    ring: _RingSlot
    object_bases: torch.Tensor
    request_metadata: torch.Tensor
    prev_compute_done: Any
    l01_native_sdma: tuple[bool, bool]
    total_tokens: int
    histories: list[int]
    starts: list[int]
    queries: list[int]
    max_sel_row: int
    max_hist_pages: int
    compute_stream: Any
    failed: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)
    block_table: torch.Tensor | None = None
    current_owner: str | None = None
    current_plan: tuple[torch.Tensor, torch.Tensor] | None = None
    submitted: set[str] = field(default_factory=set)
    native_ready: set[int] = field(default_factory=set)
    slots_ready: set[int] = field(default_factory=set)
    pending_indexer: set[int] = field(default_factory=set)
    prefetched: dict[str, Any] = field(default_factory=dict)
    pending_main: dict[int, Callable[[], None]] = field(default_factory=dict)
    main_events: dict[int, Any] = field(default_factory=dict)
    indexer_dep: dict[int, int] = field(default_factory=dict)
    indexer_events: dict[int, Any] = field(default_factory=dict)


def _make_ring_slot(blocks: int, mshape: tuple[int, int], nlayers: int, dev: torch.device, h2d: Any) -> _RingSlot:
    pin, ev = dev.type != "cpu", torch.npu.Event
    hb = torch.empty(blocks, dtype=torch.uint64, pin_memory=pin)
    hm = torch.empty(mshape, dtype=torch.int32, pin_memory=pin)
    with torch.npu.stream(h2d):
        db = torch.empty(blocks, dtype=torch.uint64, device=dev)
        dm = torch.empty(mshape, dtype=torch.int32, device=dev)
    li, mn, at = ([ev() for _ in range(nlayers)] for _ in range(3))
    return _RingSlot(hb, hm, db, dm, li, mn, at, ev(), ev(), ev(), ev())


def build_selftest_source() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct a CPU reference blob with non-zero offsets and unique per-row fingerprints."""
    rows, cols = pl.PAGE_ROWS, pl.MAIN_ROW_BYTES
    mb, db, sb = pl.MAIN_PAGE_BYTES, pl.INDEXER_DATA_PAGE_BYTES, pl.INDEXER_SCALE_PAGE_BYTES
    idx = torch.arange(rows)
    main_ref = (((torch.arange(cols) % 61) + 1).expand(rows, cols)).clone().to(torch.int8)
    main_ref[idx, idx] = 109
    li_k_ref = -((torch.arange(rows * 128).view(1, rows, 128) % 113) + 1).to(torch.int8)
    li_s_ref = (0x3800 + idx.to(torch.int16)).view(1, rows)
    src = torch.full((SELFTEST_PAD_BYTES + mb + db + sb,), -0x33, dtype=torch.int8)
    m0, d0, s0 = SELFTEST_PAD_BYTES, SELFTEST_PAD_BYTES + mb, SELFTEST_PAD_BYTES + mb + db
    src[m0:d0] = main_ref.reshape(-1)
    src[d0:s0] = li_k_ref.reshape(-1)
    src[s0 : s0 + sb] = li_s_ref.view(torch.int8).reshape(-1)
    return src, main_ref, li_k_ref, li_s_ref


class SelectedKVRuntime:
    THREAD_JOIN_TIMEOUT_S = 5.0

    @property
    def has_active_step(self) -> bool:
        return self.step is not None and not self._closed

    def __init__(
        self,
        worker: Any,
        *,
        device: Any,
        max_num_batched_tokens: int,
        max_num_seqs: int,
        owner_by_layer: dict[str, str],
        main_caches: dict[str, torch.Tensor],
        max_model_len: int = pl.MAX_HISTORY_TOKENS,
        max_query_ratio: float = pl.DEFAULT_MAX_QUERY_RATIO,
        min_history_tokens: int = pl.DEFAULT_MIN_HISTORY_TOKENS,
        exporter: pl.ExporterFn | None = None,
    ) -> None:
        from vllm_ascend.utils import enable_custom_op

        if not pl.valid_max_query_ratio(max_query_ratio):
            raise ValueError("Selected KV max_query_ratio must be a finite number in [0, 1]")
        if not pl.valid_min_history_tokens(min_history_tokens):
            raise ValueError("Selected KV min_history_tokens must be an int in [0, 1048576]")
        enable_custom_op()
        self.worker, self.max_query_ratio, self.min_history_tokens = (
            worker,
            float(max_query_ratio),
            int(min_history_tokens),
        )
        self.components, self.indexers = discover_components(worker)
        self.layer_ids = {c.layer_name: c.layer_id for c in self.components}
        li_extents = (pl.INDEXER_DATA_PAGE_BYTES, pl.INDEXER_SCALE_PAGE_BYTES)
        li_span = max((o + n for x in self.indexers if x for o, n in zip(x.offsets, li_extents)), default=0)
        self.required_span = max(*(c.object_offset + pl.MAIN_PAGE_BYTES for c in self.components), li_span)
        self.exporter = exporter if exporter is not None else pl.load_default_exporter(worker)
        spec = pl.SelectedWorkspaceSpec.from_bounds(max_num_batched_tokens, max_num_seqs, max_model_len)
        self.device, self.query_capacity = torch.device(device), max_num_batched_tokens
        self.max_requests, self.capacity = spec.max_requests, spec.union_capacity
        self.owner_by_layer, self.main_caches = dict(owner_by_layer), dict(main_caches)
        self._main_offsets = {c.layer_name: c.object_offset for c in self.components}
        names = [c.layer_name for c in self.components]
        self._next_follower = {
            a: b for a, b in zip(names, names[1:]) if self.owner_by_layer[b] == self.owner_by_layer[a]
        }
        self.scratch = [torch.empty(s, dtype=torch.int32, device=self.device) for s in spec.scratch_shapes]
        self.outputs = [torch.empty(s, dtype=torch.int32, device=self.device) for s in spec.output_shapes]
        self._h2d_stream, self._li_stream, self._copy_stream = (torch.npu.Stream(device=self.device) for _ in range(3))
        nl, nb, hs = len(self.components), worker.num_blocks, self._h2d_stream
        self._ring = [_make_ring_slot(nb, spec.meta_shape, nl, self.device, hs) for _ in range(pl.RING_DEPTH)]
        self.step: _ActiveStep | None = None
        self._completed_fence: tuple[_RingSlot, Any] | None = None
        self._prev_compute_done: Any = None
        self._prev_was_selected = self._accepting_leases = self._lease_invalid = self._aborted = False
        self._poisoned = self._closed = False
        self._step_seq = 0
        self._leased_sizes: dict[int, int] = {}
        if self.device.type == "npu":
            self._verify_kernels()

    def _verify_kernels(self, ops: Any = None, base_va: int | None = None) -> None:
        dev, kops, rows = self.device, ops or torch.ops._C_ascend, pl.PAGE_ROWS
        m0 = SELFTEST_PAD_BYTES
        d0, s0 = m0 + pl.MAIN_PAGE_BYTES, m0 + pl.MAIN_PAGE_BYTES + pl.INDEXER_DATA_PAGE_BYTES
        src_cpu, main_ref, li_k_ref, li_s_ref = build_selftest_source()
        src = src_cpu.to(dev) if base_va is None else None
        ptr = src.data_ptr() if src is not None else int(base_va)  # type: ignore[arg-type]
        topk = torch.full((1, 1, pl.TOPK_WIDTH), -1, dtype=torch.int32, device=dev)
        topk[0, 0, :2] = 3
        ids = torch.full((1, rows), -7, dtype=torch.int32, device=dev)
        st = torch.full((1, pl.COUNTER_STRIDE), -9, dtype=torch.int32, device=dev)
        bases = torch.tensor([ptr], dtype=torch.uint64).to(dev)
        meta = torch.tensor([[0, rows, 0, 1]], dtype=torch.int32).to(dev)
        tbl = torch.zeros((1, 2), dtype=torch.int32, device=dev)
        dm = torch.zeros((1, rows, 1, pl.MAIN_ROW_BYTES), dtype=torch.int8, device=dev)
        dk = torch.zeros((1, rows, 128), dtype=torch.int8, device=dev)
        ds = torch.zeros((1, rows), dtype=torch.float16, device=dev)
        kops.selected_kv_union_plan(topk, self.scratch, [ids, st], [rows], [0], [1], rows)
        kops.selected_kv_inplace_copy(ids, st, bases, meta, tbl, dm, m0)
        kops.selected_kv_li_gather(bases, dk, ds, d0, s0)
        torch.npu.synchronize()
        mc = dm.cpu()
        plan_ok = st.cpu()[0, :5].tolist() == [1, 1, 2, 0, 0] and int(ids.cpu()[0, 0]) == 3
        main_ok = torch.equal(mc[0, 3, 0], main_ref[3]) and not bool(mc[0, :3].any()) and not bool(mc[0, 4:].any())
        li_ok = torch.equal(dk.cpu(), li_k_ref) and torch.equal(ds.cpu().view(torch.int16), li_s_ref)
        if not (plan_ok and main_ok and li_ok):
            raise RuntimeError("Selected KV kernel binary registration or execution self-test failed")

    def _end_leases(self) -> None:
        self._accepting_leases = False
        self._leased_sizes.clear()

    def fail_handoff(self) -> None:
        self._end_leases()
        if not self._closed:
            self._aborted = self._poisoned = True

    def _fail_step(self, step: _ActiveStep | None = None) -> None:
        if (target := step or self.step) is not None:
            with target.lock:
                target.failed = True
                target.pending_main.clear()
                target.pending_indexer.clear()
        self.step = self._completed_fence = None
        self._aborted = True

    @contextmanager
    def _guard_step(self, step: _ActiveStep) -> Iterator[None]:
        try:
            yield
        except BaseException:
            self._fail_step(step)
            raise

    def record_leased_infos(self, keys: Sequence[str], infos: Sequence[Any] | None) -> None:
        if not self._accepting_leases:
            return
        if infos is None or len(keys) != len(infos):
            self._lease_invalid = True
            return
        for info in (x for x in infos if x is not None):
            for raw in info.gva_list() or ():
                if 0 < int(raw) < pl.MEMCACHE_SSD_HIT_GVA:
                    self._leased_sizes[int(raw)] = int(info.size())

    def begin_step(self) -> None:
        if self._closed:
            raise RuntimeError("Selected KV runtime is closed")
        self.fence_aborted_step()
        if self._poisoned:
            raise RuntimeError("Selected KV runtime is poisoned after a mid-step failure")
        self._leased_sizes.clear()
        self._lease_invalid, self._accepting_leases = False, True

    def bind(self, scheduler_output: Any, input_batch: Any) -> bool:
        if self._closed or self._poisoned:
            raise RuntimeError("Selected KV runtime is closed or poisoned")
        try:
            meta = getattr(scheduler_output, "kv_connector_metadata", None)
            if meta is None or input_batch is None or self._lease_invalid or self.worker.next_layer_to_submit != 0:
                return False
            with self.worker._invalid_block_ids_lock:
                if self.worker._invalid_block_ids:
                    return False
            sched = dict(scheduler_output.num_scheduled_tokens)
            reqs, ids = tuple(meta.requests), tuple(input_batch.req_ids)
            req_ids = {r.req_id for r in reqs}
            id_ok = 0 < len(ids) <= self.max_requests and len(set(ids)) == len(ids) == len(reqs)
            if not id_ok or req_ids != {rid for rid, q in sched.items() if q > 0} or set(ids) != req_ids:
                return False
            starts = tuple(int(v) for v in input_batch.query_start_loc_np[: len(ids) + 1])
            b_hist = tuple(int(v) for v in input_batch.num_computed_tokens_np[: len(ids)])
            if len(starts) != len(ids) + 1 or starts[0] != 0 or not 0 < starts[-1] <= self.query_capacity:
                return False
            b_q = tuple(e - s for s, e in zip(starts, starts[1:]))
            by_id = {name: (h, q) for name, h, q in zip(ids, b_hist, b_q)}
            mr, mh = self.max_query_ratio, self.min_history_tokens
            plans = tuple(pl.classify_request(r.req_id, r.target_token_len, sched[r.req_id], mr, mh) for r in reqs)
            if not any(p.selected for p in plans) or any(
                by_id[p.request_id] != (p.history_tokens, p.query_tokens) for p in plans
            ):
                return False
            tr = pl.trim_load_plan(reqs, plans, self._leased_sizes, self.required_span, self.components, self.worker)
            if tr is None:
                return False
            sel_idx = [i for i, name in enumerate(ids) if name in tr.selected_req_ids]
            hi, cap = pl.MAX_HISTORY_TOKENS, self.capacity
            if not sel_idx or any(
                not mh < b_hist[i] <= hi or min(b_hist[i], pl.TOPK_WIDTH * b_q[i]) > cap for i in sel_idx
            ):
                return False
            if (refs := self.exporter(tr.gvas, tr.sizes, self.required_span)).shape != tr.gvas.shape:
                return False
            hists = [b_hist[i] for i in sel_idx]
            sel_starts, qs = [starts[i] for i in sel_idx], [b_q[i] for i in sel_idx]
            meta_rows = np.column_stack((sel_idx, hists, sel_starts, qs)).astype(np.int32)
            cand_tasks, l01 = pl.build_trimmed_load_tasks(self.worker.layer_load_tasks, tr.replacements)
        except Exception as exc:
            logger.warning_once("Selected KV prefill declined step and fell back to native SDMA load: %s", exc)
            return False
        finally:
            self._end_leases()

        slot, ring, n_sel = self._step_seq % pl.RING_DEPTH, self._ring[self._step_seq % pl.RING_DEPTH], len(sel_idx)
        if ring.ddr_fence is not None:
            try:
                ring.ddr_fence.synchronize()
            except BaseException:
                self.fail_handoff()
                raise
            ring.ddr_fence = None
        # Mark _aborted before async H2D staging so any mid-commit failure triggers full device quiescence.
        self._step_seq, self._aborted = self._step_seq + 1, True
        hb, db = ring.host_bases[: tr.domain], ring.dev_bases[: tr.domain]
        hm, dm = ring.host_meta[:n_sel], ring.dev_meta[:n_sel]
        hb.zero_()
        hb.numpy()[tr.ids], hm.numpy()[:] = refs, meta_rows
        with torch.npu.stream(self._h2d_stream):
            db.copy_(hb, non_blocking=True)
            dm.copy_(hm, non_blocking=True)
            ring.h2d_ev.record(self._h2d_stream)
        cur = torch.npu.current_stream()
        prev_done = self._prev_compute_done if self._prev_was_selected else ring.entry_ev
        if not self._prev_was_selected:
            prev_done.record(cur)
        max_pg = max((h + pl.PAGE_ROWS - 1) // pl.PAGE_ROWS for h in hists)
        step = _ActiveStep(
            slot=slot,
            ring=ring,
            object_bases=db,
            request_metadata=dm,
            prev_compute_done=prev_done,
            l01_native_sdma=l01,
            total_tokens=starts[-1],
            histories=hists,
            starts=sel_starts,
            queries=qs,
            max_sel_row=max(sel_idx),
            max_hist_pages=max_pg,
            compute_stream=cur.npu_stream,
        )
        self.worker.layer_load_tasks = cand_tasks
        self.step, self._aborted = step, False
        return True

    def defer_indexer(self, step: _ActiveStep, layer_id: int, after_layer_id: int) -> None:
        with step.lock:
            seen = layer_id in step.indexer_events or layer_id in step.pending_indexer
            if step.failed or self.indexers[layer_id] is None or seen:
                return
            if not 0 <= after_layer_id < layer_id or layer_id in step.indexer_dep:
                raise ValueError("LI prefetch must have one preceding main-copy dependency")
            step.indexer_dep[layer_id] = after_layer_id
            step.main_events.setdefault(after_layer_id, None)

    def _release_main_slot(self, step: _ActiveStep, layer_id: int) -> None:
        if step.failed or self.step is not step:
            return
        if layer_id in (0, 1) and step.l01_native_sdma[layer_id] and step.prev_compute_done is not None:
            step.prev_compute_done.synchronize()
        step.slots_ready.add(layer_id)
        if (submit := step.pending_main.pop(layer_id, None)) is not None:
            with self._guard_step(step):
                submit()

    def _enqueue_indexer(self, step: _ActiveStep, layer_id: int, dep: Any) -> None:
        idx = self.indexers[layer_id]
        assert idx is not None
        # Emit device ops outside step.lock only where safe; here _li_stream records per-layer events.
        with self._guard_step(step), torch.npu.stream(self._li_stream):
            s = torch.npu.current_stream()
            first_wave = layer_id < 2 or dep is None
            for ev in (step.ring.h2d_ev if first_wave else None, step.prev_compute_done if first_wave else None, dep):
                if ev is not None:
                    s.wait_event(ev)
            torch.ops._C_ascend.selected_kv_li_gather(step.object_bases, *idx.caches, *idx.offsets)
            (done := step.ring.li_evs[layer_id]).record(s)
            step.indexer_events[layer_id] = done

    def on_slot_released(self, layer_id: int) -> None:
        if (step := self.step) is not None:
            with step.lock:
                # KVCacheStoreLayerRecvingThread._wait_for_reuse_saves notifies each saved source layer
                # before notifying the target layer; completed source layers are already in native_ready.
                if layer_id in step.native_ready:
                    return
                self._release_main_slot(step, layer_id)
                if step.failed or self.step is not step or self.indexers[layer_id] is None:
                    return
                if layer_id in step.indexer_events or layer_id in step.pending_indexer:
                    raise RuntimeError("LI load was already requested for this layer")
                after = step.indexer_dep.get(layer_id)
                dep = step.main_events[after] if after is not None else None
                if after is not None and dep is None:
                    step.pending_indexer.add(layer_id)
                else:
                    self._enqueue_indexer(step, layer_id, dep)
        elif layer_id in (0, 1) and (prev := self._prev_compute_done) is not None:
            # Guard unselected follow-up steps so native L0/L1 SDMA waits for the prior selected step's tail layers.
            try:
                if layer_id < len(self.worker.layer_load_tasks) and self.worker.layer_load_tasks[layer_id]:
                    prev.synchronize()
                    self._prev_compute_done = None
                elif prev.query():
                    self._prev_compute_done = None
            except BaseException:
                self.fail_handoff()
                raise

    def on_native_layer_ready(self, layer_id: int) -> None:
        if (step := self.step) is None:
            if self._aborted:
                raise RuntimeError("Selected KV step has failed")
            return
        with step.lock:
            if step.failed:
                raise RuntimeError("Selected KV step has failed")
            if layer_id in step.native_ready:
                return
            self._release_main_slot(step, layer_id)
            if self.indexers[layer_id] is not None:
                stream = torch.npu.current_stream()
                if stream.npu_stream != step.compute_stream:
                    self._fail_step(step)
                    raise RuntimeError("Selected execution must stay on its bound compute stream")
                if (ev := step.indexer_events.get(layer_id)) is None:
                    self._fail_step(step)
                    raise RuntimeError("Selected LI completion event is missing")
                stream.wait_event(ev)
            step.native_ready.add(layer_id)

    def _run_main_copy(
        self, step: _ActiveStep, plan: tuple[torch.Tensor, torch.Tensor], name: str, stream: Any, *, force: bool = False
    ) -> Any:
        assert step.block_table is not None
        op, cache, off = torch.ops._C_ascend.selected_kv_inplace_copy, self.main_caches[name], self._main_offsets[name]
        op(plan[0], plan[1], step.object_bases, step.request_metadata, step.block_table, cache, off)
        lid = self.layer_ids[name]
        with step.lock:
            if not force and lid not in step.main_events:
                return None
            (ev := step.ring.main_evs[lid]).record(stream)
            step.main_events[lid] = ev
            for target in sorted(step.pending_indexer):
                if step.indexer_dep[target] == lid:
                    self._enqueue_indexer(step, target, ev)
                    step.pending_indexer.discard(target)
            return ev

    def load_layer(self, name: str, topk: torch.Tensor, cache: Sequence[torch.Tensor], metadata: Any) -> None:
        s = self.step
        if self._closed or s is None or name not in self.owner_by_layer or name in s.submitted:
            raise RuntimeError("Selected attention layer is inactive, missing, or duplicated")
        with self._guard_step(s):
            if len(cache) != 1 or cache[0].data_ptr() != self.main_caches[name].data_ptr():
                raise ValueError("Selected layer received an unexpected main cache view")
            tk_ok = (topk.device, topk.dtype, topk.is_contiguous(), topk.ndim) == (self.device, torch.int32, True, 3)
            if not tk_ok or topk.shape[1:] != (1, pl.TOPK_WIDTH) or topk.shape[0] < s.total_tokens:
                raise ValueError("Selected loading requires a contiguous [T, 1, 2048] int32 TopK tensor")
            tbl = getattr(metadata, "block_table", None)
            m_ok = (metadata.num_actual_tokens, metadata.block_size) == (s.total_tokens, pl.PAGE_ROWS)
            if not s.submitted:
                if (
                    not m_ok
                    or not isinstance(tbl, torch.Tensor)
                    or (tbl.device, tbl.dtype, tbl.ndim) != (self.device, torch.int32, 2)
                    or tbl.shape[0] <= s.max_sel_row
                    or tbl.shape[1] < s.max_hist_pages
                    or tbl.stride(1) != 1
                ):
                    raise ValueError("Selected loading requires matching token count and 128-row int32 block_table")
                s.block_table = tbl
            elif not (m_ok and _same_table(tbl, s.block_table)):
                raise ValueError("Selected loading requires a stable block_table across layers")

            stream = torch.npu.current_stream()
            if stream.npu_stream != s.compute_stream:
                raise RuntimeError("Selected execution must stay on its bound compute stream")
            if not s.submitted:
                stream.wait_event(s.ring.h2d_ev)
                if s.prev_compute_done is not None:
                    stream.wait_event(s.prev_compute_done)

            owner = self.owner_by_layer[name]
            if owner == name:
                cnt, uop = len(s.histories), torch.ops._C_ascend.selected_kv_union_plan
                uop(topk, self.scratch, self.outputs, s.histories, s.starts, s.queries, self.capacity)
                s.current_plan, s.current_owner = (self.outputs[0][:cnt], self.outputs[1][:cnt]), name
                self._run_main_copy(s, s.current_plan, name, stream)
            else:
                with s.lock:
                    if s.current_owner != owner or s.current_plan is None or name not in s.prefetched:
                        raise RuntimeError("Selected follower ran before its owner plan or prefetch")
                    pref_ev = s.prefetched[name]
                    if pref_ev is None:
                        s.pending_main.pop(self.layer_ids[name], None)
                if pref_ev is not None:
                    stream.wait_event(pref_ev)
                else:
                    self._run_main_copy(s, s.current_plan, name, stream)
            s.submitted.add(name)
            if len(s.submitted) == len(self.owner_by_layer):
                s.ring.ddr_done.record(stream)

    def prefetch_next(self, name: str) -> None:
        if self._closed or (step := self.step) is None:
            raise RuntimeError("Stale or inactive selected attention step")
        with self._guard_step(step):
            if (following := self._next_follower.get(name)) is None:
                return
            stream = torch.npu.current_stream()
            if stream.npu_stream != step.compute_stream or step.current_plan is None or step.block_table is None:
                raise RuntimeError("Follower prefetch requires an active owner plan on the bound compute stream")
            attn_done = step.ring.attn_evs[self.layer_ids[name]]
            foll_id, plan = self.layer_ids[following], step.current_plan
            attn_done.record(stream)
            step.prefetched[following] = None

            def submit() -> None:
                cs = self._copy_stream
                with torch.npu.stream(cs):
                    cs.wait_event(attn_done)
                    if foll_id < 2 and step.prev_compute_done is not None:
                        cs.wait_event(step.prev_compute_done)
                    step.prefetched[following] = self._run_main_copy(step, plan, following, cs, force=True)

            with step.lock:
                if foll_id in step.pending_main:
                    raise RuntimeError("Duplicate selected main prefetch")
                step.pending_main[foll_id] = submit
                if foll_id in step.slots_ready:
                    self._release_main_slot(step, foll_id)

    def finish(self, *, failed: bool = False) -> None:
        if (step := self.step) is None:
            if not failed and self._aborted:
                raise RuntimeError("Selected KV step has failed")
            return
        if failed or step.submitted != self.owner_by_layer.keys():
            self._fail_step(step)
            if not failed:
                raise RuntimeError("Selected execution did not submit every layer")
            return
        with self._guard_step(step):
            stream = torch.npu.current_stream()
            if stream.npu_stream != step.compute_stream:
                raise RuntimeError("Selected completion must stay on its bound compute stream")
            with step.lock:
                if any(x is not None and i not in step.native_ready for i, x in enumerate(self.indexers)):
                    raise RuntimeError("Selected LI loads were not joined by every consumer")
                if step.pending_main:
                    raise RuntimeError("Selected main prefetches were never released")
                step.ring.compute_done.record(stream)
                self._completed_fence, self.step = (step.ring, step.ring.compute_done), None

    def _stop_worker_threads(self, *, timeout: float = THREAD_JOIN_TIMEOUT_S) -> bool:
        if getattr(self.worker, "kv_send_thread", None) is getattr(self.worker, "kv_recv_thread", None) is None:
            return False
        self._poisoned = True
        msg = "KV transfer thread did not exit during selected-KV abort"
        return bool(stop_transfer_threads(self.worker, timeout, msg))

    def _drain_if_ready(self) -> None:
        w, k = self.worker, "_drain_attention_transfers"
        if callable(d := getattr(w, k, None)) and (k in vars(w) or getattr(w, "layerwise_backend", None) is not None):
            d(full=True)

    def _clear_fences(self) -> None:
        self._completed_fence = self._prev_compute_done = None
        self._prev_was_selected = self._aborted = False
        for r in self._ring:
            r.ddr_fence = None

    def take_step_fence(self, *, closing: bool = False) -> tuple[bool, Any]:
        self._end_leases()
        if self.step is not None or self._aborted:
            self._fail_step()
            if closing or self._poisoned or getattr(self.worker, "next_layer_to_submit", 0) > 0:
                if not self._stop_worker_threads():
                    self._drain_if_ready()
            elif getattr(self.worker, "kv_recv_thread", None) is None:
                self._drain_if_ready()
            torch.npu.synchronize()
            self._clear_fences()
            return False, None
        done, self._completed_fence = self._completed_fence, None
        if done is None:
            self._prev_was_selected = False
            return not self._poisoned, None
        ring, compute_done = done
        ring.ddr_fence = ring.ddr_done
        self._prev_compute_done, self._prev_was_selected = compute_done, True
        return True, ring.ddr_done

    def fence_aborted_step(self) -> None:
        # Invoked from begin_step() and wait_for_previous_save() (which ActiveKVConnector.pre_forward
        # calls inside handle_preemptions before bind_connector_metadata). Preserve _accepting_leases
        # when no prior step aborted so post-lease GVA recording stays open during pre_forward.
        if self.step is not None or self._aborted:
            self.take_step_fence()
        elif self._completed_fence is not None:
            torch.npu.synchronize()
            self._clear_fences()

    def abort_step(self) -> None:
        self.fence_aborted_step()
        self._end_leases()
        if not self._poisoned and callable(reclaim := getattr(self.worker, "reclaim_aborted_step_leases", None)):
            reclaim()

    def close(self) -> None:
        if self._closed:
            return
        if self.step is not None or self._aborted:
            self.take_step_fence(closing=True)
        else:
            self._end_leases()
            for fence in (*(r.ddr_fence for r in self._ring), self._prev_compute_done):
                if fence is not None:
                    fence.synchronize()
            torch.npu.synchronize()
            self._clear_fences()
        for bufs in (self.scratch, self.outputs, self._ring):
            bufs.clear()
        self._closed = True


class SelectedKVConnector(KVConnector):
    def __init__(self, connector: KVConnector, runtime: SelectedKVRuntime) -> None:
        self._connector, self._runtime, self._disabled = connector, runtime, False

    def set_disabled(self, disabled: bool) -> None:
        self._disabled = disabled
        if disabled:
            self._runtime.abort_step()
        self._connector.set_disabled(disabled)

    def reset_capture_state(self) -> None:
        self._runtime.abort_step()
        self._connector.reset_capture_state()

    def pre_forward(self, scheduler_output: Any, *args: Any, **kwargs: Any) -> None:
        if not self._disabled:
            self._runtime.begin_step()
        try:
            self._connector.pre_forward(scheduler_output, *args, **kwargs)
        except BaseException:
            if not self._disabled:
                self._runtime.abort_step()
            raise
        if not self._disabled:
            batch = kwargs.get("input_batch") if "input_batch" in kwargs else (args[0] if args else None)
            self._runtime.bind(scheduler_output, batch)

    def finish_forward(self) -> None:
        try:
            if not self._disabled:
                self._runtime.finish()
        finally:
            self._connector.finish_forward()

    def post_forward(self, finished_req_ids: set[str], *args: Any, **kwargs: Any) -> Any:
        return self._connector.post_forward(finished_req_ids, *args, **kwargs)

    def finalize_step(self) -> None:
        self._connector.finalize_step()

    def no_forward(self, scheduler_output: Any) -> Any:
        self._runtime.abort_step()
        return cast("Any", self._connector.no_forward(scheduler_output))

    def close(self) -> None:
        try:
            self._runtime.close()
        finally:
            if callable(inner_close := getattr(self._connector, "close", None)):
                inner_close()


def _effective_seconds(key: str, default: float) -> float:
    in_vars = [lambda: vars(envs_vllm)[key]] if key in vars(envs_vllm) else []
    for fn in (*in_vars, lambda: getattr(envs_vllm, key), lambda: os.getenv(key)):
        try:
            if (val := fn()) is not None and str(val).strip():
                return float(val)
        except Exception:
            continue
    return default


def start_selected_kv(runner: Any, kv_cache_config: Any) -> None:
    if not selected_kv_enabled(runner.ascend_config):
        return
    budget = getattr(runner, "selected_kv_budget", None)
    if getattr(runner, "_selected_kv_runtime", None) is not None or budget is None:
        raise RuntimeError("Selected KV must be budgeted once and started exactly once")
    pw = get_kv_transfer_group().connector_worker
    if not callable(getattr(pw, "attach_selected_runtime", None)) or pw.selected_kv_runtime is not None:
        raise RuntimeError("Selected KV requires an unattached selected-KV layerwise pool worker")
    ctx = runner.compilation_config.static_forward_context
    kv_caches = {k: v.kv_cache for k, v in ctx.items() if getattr(v, "kv_cache", None) is not None}
    tensors = (t for vals in kv_caches.values() for t in (vals if isinstance(vals, (tuple, list)) else (vals,)))
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors}
    budget.validate(kv_cache_config.num_blocks, sum(storages.values()))
    specs = lcl.get_layerwise_kv_cache_specs(kv_cache_config)
    names = [m_n for m_n, _, _, _ in pl.ordered_layer_specs(pw._layerwise_reuse_layout, specs, pw.num_layers)]
    owners, impls = pl.topk_owners(runner, names), [ctx[name].impl for name in names]
    if any(not isinstance(impl, AscendSFASelectedKVImpl) for impl in impls):
        raise RuntimeError("Selected KV requires AscendSFASelectedKVImpl on every main attention layer")

    s, mlen = runner.vllm_config.scheduler_config, runner.vllm_config.model_config.max_model_len
    backend = getattr(getattr(runner.vllm_config, "parallel_config", None), "distributed_executor_backend", None)
    exec_t = _effective_seconds("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", 300.0)
    total_t = exec_t + _effective_seconds("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", 5.0)
    if backend == "uni" or total_t >= 300.0:
        why = "has no RPC timeout on uni" if backend == "uni" else "is >= 300s"
        msg = "Selected KV uses 300s MemCache read leases while effective executor timeout "
        msg += "(execute+shutdown=%gs, executor=%s) %s; ensure external watchdog or server lease TTL covers hangs."
        logger.warning(msg, total_t, backend or "default", why)
    mains = {name: pw._as_cache_tuple(kv_caches[name])[0] for name in names}
    scfg = getattr(runner.ascend_config, "selected_kv_config", None)
    ratio = pl.DEFAULT_MAX_QUERY_RATIO if (mr := getattr(scfg, "max_query_ratio", None)) is None else mr
    min_h = pl.DEFAULT_MIN_HISTORY_TOKENS if (mh := getattr(scfg, "min_history_tokens", None)) is None else mh
    rt = SelectedKVRuntime(
        pw,
        device=runner.device,
        max_num_batched_tokens=s.max_num_batched_tokens,
        max_num_seqs=s.max_num_seqs,
        max_model_len=mlen,
        max_query_ratio=ratio,
        min_history_tokens=min_h,
        owner_by_layer=owners,
        main_caches=mains,
    )
    pw.attach_selected_runtime(rt)
    for impl in impls:
        impl.selected_kv_runtime = rt
    runner._selected_kv_runtime, runner.kv_connector = rt, SelectedKVConnector(runner.kv_connector, rt)


def stop_selected_kv(runner: Any) -> None:
    if (rt := getattr(runner, "_selected_kv_runtime", None)) is not None:
        rt.close()
        runner._selected_kv_runtime = None
