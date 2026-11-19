"""EPLB D2D + staging + async D2H writeback for the v18port offload.

Per-layer EPLB updates reuse the ORIGINAL NPU D2D exchange (HCCL P2P
isend/irecv via batch_isend_irecv) with three offload-specific
adaptations:

1. SEND SIDE — static prefetch pipeline. The stock isend source
   (weight_list views into the shared slot storage) is NOT
   authoritative: by the time EPLB updates layer L (iteration head),
   L's slot has already been overwritten by the wrap-around preload.
   Instead, a dedicated static send buffer (one [E, ...] tensor per
   mirrored param, same layout as the slot destination) is kept
   prefetched with the NEXT layer to update — one full layer ahead —
   on a dedicated send stream. The prefetch window is a whole
   iteration (~2s), so the mirror->send_buf H2D is fully hidden; a
   stream event guards the isend. The buffer layout stays the mirror
   (old) layout; isend slices per-expert views at the OLD local
   positions, exactly like the stock weight_list indexing.

2. RECV SIDE — unchanged from stock: irecv into the adaptor's
   buffer_tensor_list (independent NPU-resident buffers, allocated
   once at adaptor init).

3. FORWARD_END — instead of the stock in-place weight update, the
   received experts (raw D2D from the recv buffers) plus the
   locally-kept experts (raw H2D from the CPU mirror) are assembled
   into a per-layer NPU STAGING buffer holding the complete NEW
   layout. The staging buffer then:
     - serves as the preload source (raw D2D staging -> slot, faster
       than mirror H2D), and
     - is written back into the CPU mirror asynchronously on a
       dedicated D2H stream (so future cycles see the new layout).
   Routing maps are switched right after the D2D wait — same
   "data in place before map" ordering as stock, with the staging
   buffer being the in-place copy for the next slot load.

All staging/send transfers are raw `acl.rt.memcpy_async` byte copies
(NZ raw-bytes semantics, no format conversion). pyacl constants:
H2D=1, D2H=2, D2D=3.
"""

import logging
import weakref
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

# NOTE: `acl.rt` / `torch_npu` are imported lazily at their use sites so
# that CPU-only environments (e.g. x86 UT runners without torch_npu) can
# import this module without crashing.

logger = logging.getLogger(__name__)

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3


@dataclass
class _StagingState:
    """Per-layer NPU staging buffer state (one EPLB update)."""

    layer_prefix: str
    layer_id: int
    # Per-param uint8 1-D NPU flats, same raw bytes as the mirror flats.
    npu_flats: list
    # copy-stream event: kept-block H2D (mirror -> staging) submitted.
    ev_h2d: object = None
    # compute-stream event: recv buffers -> staging copies submitted.
    ev_recv: object = None
    # d2h-stream event: staging -> mirror writeback submitted.
    ev_d2h: object = None
    consumed: bool = False
    new_l2g: list | None = None
    expert_bytes: list | None = None
    kept: int = 0
    recv: int = 0


class EplbD2DManager:
    """Manages the send-buffer prefetch pipeline, the per-layer staging
    buffers and the async D2H writeback. One instance, composed into the
    MoEOffloadManager (created lazily by the EPLB patch after
    warmup; before that the offload runs the stock H2D path with zero
    EPLB overhead)."""

    def __init__(self, offload_manager):
        self._mgr = weakref.ref(offload_manager)
        # Dedicated streams. NEVER share: the send stream must not
        # serialize behind the offload copy stream, and the D2H stream
        # must not block (or be blocked by) either of them.
        self._send_stream = torch.npu.Stream()
        self._d2h_stream = torch.npu.Stream()
        # Static send buffer: PER-EXPERT independent NPU tensors
        # ([expert][param], allocated via empty_like of the slot dst's
        # per-expert views). HCCL P2P rejects internal-format (NZ)
        # tensors with nonzero storage_offset, so isend sources MUST be
        # independent tensors (offset 0) — never views into a shared
        # [E, ...] flat. Total size equals one full layer (~612MiB).
        self._send_experts: list | None = None
        self._send_prefetch_ev = None
        self._prefetched_prefix: str | None = None
        # Static double-buffered staging pool, allocated ONCE. Never
        # allocate/free per update: under expandable_segments the
        # VA remap churn of per-update torch.empty/free raced in-flight
        # device accesses (SMMU fault, EZ9999) once the second cycle
        # started recycling buffers. Reuse safety: the kept-block H2D
        # and the staging->slot preload both run on the copy stream
        # (same-stream ordering); the recv->staging D2D on the compute
        # stream explicitly waits the slot's last preload event, and
        # acquiring a slot waits its last D2H event.
        self._staging: dict = {}
        self._staging_pool: list = []
        self._staging_alloc_done = False
        self._cycles_done = 0

    # ------------------------------------------------------------------ #
    @property
    def mgr(self):
        m = self._mgr()
        if m is None:
            raise RuntimeError("offload manager was garbage collected")
        return m

    # ------------------------------------------------------------------ #
    # init (called from the EPLB patch after warm_up_eplb, when the
    # manager is finalized, primed and slot dsts exist)
    # ------------------------------------------------------------------ #
    def init_send_buffer_and_prime(self):
        mgr = self.mgr
        prefix = mgr.layer_prefixes[0]
        module = mgr._live_module(prefix)
        names = mgr.param_names.get(prefix, ["w13_weight", "w2_weight"])
        entry = mgr.full_layer_expert_params[prefix]
        E = entry.expert_count
        # Template: the per-expert list views the quant method already
        # built (`layer.<name>_list`). NB: `_offload_<name>` flats are
        # [E, ...]-shaped for weights/scales but 1-D for the fused
        # scales, so the *_list views are the only uniform per-expert
        # source. empty_like(view) creates an independent tensor
        # (storage_offset 0) with the same shape/dtype/format — exactly
        # what HCCL P2P requires and what the recv buffers look like.
        send = []
        for e in range(E):
            per: list[Any] = []
            for name in names:
                lst = getattr(module, f"{name}_list", None)
                if lst is None or e >= len(lst):
                    per.append(None)
                else:
                    per.append(torch.empty_like(lst[e]))
            send.append(per)
        # Sanity: each per-expert tensor's raw storage must match the
        # mirror's per-expert byte size.
        expert_bytes = [t.numel() // E if t is not None else 0 for t in entry.params]
        for e in range(E):
            for pid, t in enumerate(send[e]):
                if t is None:
                    continue
                if mgr._raw_storage_size(t) != expert_bytes[pid]:
                    raise RuntimeError(
                        "send buffer size mismatch: "
                        f"{prefix} e={e} pid={pid} "
                        f"send={mgr._raw_storage_size(t)} "
                        f"mirror={expert_bytes[pid]}"
                    )
        self._send_experts = send
        # Allocate the staging pool NOW (warmup, before any serving):
        # the previous lazy first-use allocation inside the first
        # complete_layer_update correlated with the first batch of
        # copy-stream kept H2Ds never landing (layer-0 kept experts
        # kept stale torch.empty content -> baked into the mirror by
        # the writeback). Allocating at a deterministic pre-serving
        # point removes that first-use window entirely.
        if not self._staging_alloc_done:
            self._alloc_staging_pool()
        # Prime: prefetch the first rotating layer.
        self._prefetch_prefix(prefix)
        logger.info("[EPLB-D2D] send buffer ready: %d experts x %d params, prime=%s", E, len(names), prefix)

    # ------------------------------------------------------------------ #
    # send-side prefetch (whole-layer, dedicated stream, event-guarded)
    # ------------------------------------------------------------------ #
    def _prefetch_prefix(self, layer_prefix):
        import acl.rt

        mgr = self.mgr
        entry = mgr.full_layer_expert_params[layer_prefix]
        E = entry.expert_count
        expert_bytes = [t.numel() // E if t is not None else 0 for t in entry.params]
        send_experts = self._send_experts
        assert send_experts is not None, "send buffer not primed"
        with torch.npu.stream(self._send_stream):
            for e in range(E):
                for pid, src_t in enumerate(entry.params):
                    dst_t = send_experts[e][pid]
                    if src_t is None or dst_t is None:
                        continue
                    eb = expert_bytes[pid]
                    if (e + 1) * eb > src_t.numel():
                        raise RuntimeError(
                            f"[EPLB-D2D-SENTINEL] prefetch OOB {layer_prefix} "
                            f"pid={pid} e={e} eb={eb} "
                            f"mirror_numel={src_t.numel()}"
                        )
                    ret = acl.rt.memcpy_async(
                        dst_t.data_ptr(),
                        eb,
                        src_t.data_ptr() + e * eb,
                        eb,
                        ACL_MEMCPY_HOST_TO_DEVICE,
                        self._send_stream.npu_stream,
                    )
                    if ret != 0:
                        raise RuntimeError(
                            "send-buffer prefetch H2D failed: "
                            f"layer={layer_prefix} e={e} pid={pid} "
                            f"bytes={eb} acl_error={ret}"
                        )
        self._send_prefetch_ev = self._send_stream.record_event()
        self._prefetched_prefix = layer_prefix

    def _prefetch_next(self, just_updated_prefix):
        mgr = self.mgr
        prefixes = mgr.layer_prefixes
        idx = prefixes.index(just_updated_prefix)
        self._prefetch_prefix(prefixes[(idx + 1) % len(prefixes)])

    # ------------------------------------------------------------------ #
    # forward_before (offload-managed layer): build the P2P op list and
    # launch. isend sources = send-buffer per-expert views at the OLD
    # map positions; irecv targets = stock adaptor buffers.
    # ------------------------------------------------------------------ #
    def launch_d2d_exchange(self, layer_id, expert_send_info, expert_recv_info, updator):
        mgr = self.mgr
        prefix = mgr.layer_to_prefix[layer_id + mgr.moe_layer_offset]
        # Fallback: if the pipeline got desynced, prefetch now — the
        # compute stream waits for the event either way.
        if self._prefetched_prefix != prefix:
            logger.warning(
                "[EPLB-D2D] send buffer desynced (has %s, need %s); prefetching synchronously",
                self._prefetched_prefix,
                prefix,
            )
            self._prefetch_prefix(prefix)
        compute = torch.npu.current_stream()
        if self._send_prefetch_ev is not None:
            compute.wait_event(self._send_prefetch_ev)

        comm_group = updator.comm_group
        old_map = updator.adaptor.expert_map_per_layer_cpu[layer_id]
        send_experts = self._send_experts
        assert send_experts is not None, "send buffer not primed before exchange"
        ops = []
        n_send = 0
        for dst_rank, g in expert_send_info:
            local_pos = int(old_map[int(g)].item())
            if local_pos < 0 or local_pos >= len(send_experts):
                raise RuntimeError(
                    "[EPLB-D2D-SENTINEL] send pos OOB "
                    f"layer_id={layer_id} g={int(g)} "
                    f"local_pos={local_pos} "
                    f"E={len(send_experts)}"
                )
            for t in send_experts[local_pos]:
                if t is None:
                    continue
                ops.append(
                    dist.P2POp(
                        dist.isend,
                        t,
                        comm_group.ranks[int(dst_rank)],
                        group=comm_group.device_group,
                    )
                )
                n_send += 1
        key = updator.adaptor.expert_weight_key_per_layer[layer_id]
        n_recv = 0
        for buffer_id, (src_rank, _g) in enumerate(expert_recv_info):
            for buf in updator.adaptor.buffer_tensor_list[key][buffer_id]:
                ops.append(
                    dist.P2POp(
                        dist.irecv,
                        buf,
                        comm_group.ranks[int(src_rank)],
                        group=comm_group.device_group,
                    )
                )
                n_recv += 1
        reqs = []
        if ops:
            reqs = dist.batch_isend_irecv(ops)
        return reqs

    # ------------------------------------------------------------------ #
    # forward_end (offload-managed layer): wait D2D, build staging,
    # submit D2H writeback, switch maps, prefetch next layer.
    # ------------------------------------------------------------------ #
    def complete_layer_update(
        self,
        layer_id,
        expert_send_info,
        expert_recv_info,
        updated_expert_map,
        updator,
        reqs,
        apply_fn,
        log2phy_map_t=None,
    ):
        import time as _t

        import acl.rt

        _t0 = _t.monotonic()

        mgr = self.mgr
        prefix = mgr.layer_to_prefix[layer_id + mgr.moe_layer_offset]
        # 1) wait for the D2D exchange (same blocking point as stock).
        for r in reqs or []:
            r.wait()

        entry = mgr.full_layer_expert_params[prefix]
        E = entry.expert_count
        old_l2g = mgr.local_to_global.get(prefix)
        if old_l2g is None:
            raise RuntimeError(f"{prefix}: missing local_to_global")
        old_pos_of = {g: p for p, g in enumerate(old_l2g) if g >= 0}

        new_map = updated_expert_map
        new_l2g = [-1] * E
        for g in range(len(new_map)):
            p = int(new_map[g].item())
            if 0 <= p < E:
                new_l2g[p] = g

        # Identity short-circuit: no experts move in/out of this rank
        # and the local layout is unchanged — skip staging/D2H/slot
        # invalidation entirely (the slot data is already correct);
        # just apply the maps and advance the prefetch pipeline.
        if not expert_send_info and not expert_recv_info and new_l2g == list(old_l2g):
            apply_fn()
            self._prefetch_next(prefix)
            return

        # 2) staging buffer: acquire from the static pool (no alloc).
        slot = self._acquire_staging()
        flats = slot["flats"]
        expert_bytes = [t.numel() // E if t is not None else 0 for t in entry.params]

        # 2a) locally-kept experts: raw H2D mirror -> staging
        #     (copy stream; overlaps with 2b on the compute stream —
        #     disjoint flat regions, no conflict).
        copy_stream, _ = mgr._copy_stream()
        compute = torch.npu.current_stream()
        kept = [(npos, old_pos_of[g]) for npos, g in enumerate(new_l2g) if g >= 0 and g in old_pos_of]
        for _pid0, _mt in enumerate(entry.params):
            if _mt is not None and _mt.numel() % E != 0:
                raise RuntimeError(
                    f"[EPLB-D2D-SENTINEL] {prefix} pid={_pid0} mirror numel={_mt.numel()} not divisible by E={E}"
                )
        with torch.npu.stream(copy_stream):
            for npos, opos in kept:
                for pid, src_t in enumerate(entry.params):
                    if src_t is None:
                        continue
                    eb = expert_bytes[pid]
                    if (opos + 1) * eb > src_t.numel() or (npos + 1) * eb > flats[pid].numel():
                        raise RuntimeError(
                            f"[EPLB-D2D-SENTINEL] kept OOB {prefix} pid={pid} "
                            f"opos={opos} npos={npos} eb={eb} "
                            f"mirror_numel={src_t.numel()} "
                            f"staging_numel={flats[pid].numel()}"
                        )
                    ret = acl.rt.memcpy_async(
                        flats[pid].data_ptr() + npos * eb,
                        eb,
                        src_t.data_ptr() + opos * eb,
                        eb,
                        ACL_MEMCPY_HOST_TO_DEVICE,
                        copy_stream.npu_stream,
                    )
                    if ret != 0:
                        raise RuntimeError(
                            "staging kept-block H2D failed: "
                            f"layer={prefix} pid={pid} pos={npos} "
                            f"bytes={eb} acl_error={ret}"
                        )
            ev_h2d = copy_stream.record_event()

        # 2b) received experts: raw D2D recv buffers -> staging
        #     (compute stream, in-context).
        key = updator.adaptor.expert_weight_key_per_layer[layer_id]
        _kept_npos = {n for n, _ in kept}
        for buffer_id, (_src_rank, g) in enumerate(expert_recv_info):
            local_pos = int(new_map[int(g)].item())
            bufs = updator.adaptor.buffer_tensor_list[key][buffer_id]
            if local_pos < 0 or local_pos >= E:
                raise RuntimeError(f"[EPLB-D2D-SENTINEL] recv pos OOB {prefix} g={g} local_pos={local_pos} E={E}")
            if local_pos in _kept_npos:
                logger.warning("[EPLB-D2D] OVERLAP %s g=%s local_pos=%s in kept_npos!", prefix, int(g), local_pos)
            for pid, buf in enumerate(bufs):
                flat = flats[pid]
                if flat is None or buf is None:
                    continue
                eb = expert_bytes[pid]
                buf_bytes = buf.untyped_storage().nbytes()
                if (local_pos + 1) * eb > flat.numel() or buf_bytes != eb:
                    raise RuntimeError(
                        f"[EPLB-D2D-SENTINEL] recv OOB {prefix} pid={pid} "
                        f"local_pos={local_pos} eb={eb} buf_bytes={buf_bytes} "
                        f"staging_numel={flat.numel()}"
                    )
                ret = acl.rt.memcpy_async(
                    flat.data_ptr() + local_pos * eb,
                    eb,
                    buf.data_ptr(),
                    eb,
                    ACL_MEMCPY_DEVICE_TO_DEVICE,
                    compute.npu_stream,
                )
                if ret != 0:
                    raise RuntimeError(
                        f"staging recv D2D failed: layer={prefix} pid={pid} pos={local_pos} bytes={eb} acl_error={ret}"
                    )
        # [BUFFER-REUSE BARRIER] The recv fills above read the adaptor's
        # SHARED buffer pool asynchronously on the compute stream. The
        # next layer's forward_before re-posts irecv into the same
        # buffers from the host as soon as this forward_end returns.
        # Without a barrier those HCCL writes can overwrite a buffer
        # before the compute stream has consumed it (observed as ~2%
        # corrupted received experts under heavy rearrangement; kept
        # experts are unaffected because the mirror is never reused).
        # Synchronize so the buffers are fully consumed before return.
        compute.synchronize()
        ev_recv = compute.record_event()

        # 3) async D2H writeback: staging (complete new layout) ->
        #    mirror, on the dedicated D2H stream. Waits for both
        #    staging-fill events first (copy-stream H2D + compute-
        #    stream recv copies) — the writeback must not race the
        #    fills.
        with torch.npu.stream(self._d2h_stream):
            self._d2h_stream.wait_event(ev_h2d)
            self._d2h_stream.wait_event(ev_recv)
            for pid, flat in enumerate(flats):
                m = entry.params[pid]
                if flat is None or m is None:
                    continue
                nbytes = m.numel()
                ret = acl.rt.memcpy_async(
                    m.data_ptr(),
                    nbytes,
                    flat.data_ptr(),
                    nbytes,
                    ACL_MEMCPY_DEVICE_TO_HOST,
                    self._d2h_stream.npu_stream,
                )
                if ret != 0:
                    raise RuntimeError(
                        f"staging D2H writeback failed: layer={prefix} pid={pid} bytes={nbytes} acl_error={ret}"
                    )
            ev_d2h = self._d2h_stream.record_event()

        state = _StagingState(
            layer_prefix=prefix,
            layer_id=layer_id,
            npu_flats=flats,
            ev_h2d=ev_h2d,
            ev_recv=ev_recv,
            ev_d2h=ev_d2h,
            new_l2g=new_l2g,
            expert_bytes=expert_bytes,
            kept=len(kept),
            recv=len(expert_recv_info),
        )
        self._staging[prefix] = state
        slot["last_d2h_ev"] = ev_d2h

        # 4) switch the routing maps NOW (compute thread; the D2D data
        #    is in the staging buffer, the mirror follows async — the
        #    manager's D2H guard keeps mirror reads ordered).
        apply_fn()
        mgr.local_to_global[prefix] = new_l2g

        # 5) invalidate the layer's slot + any in-flight preload of the
        #    OLD layout. Only clear the slot entry if it still holds
        #    THIS layer (the slot may already belong to another layer).
        slot_idx = mgr._slot_idx(prefix)
        if mgr.slot_loaded_layers.get(slot_idx) == prefix:
            mgr.slot_loaded_layers[slot_idx] = None
        mgr.pending_transfers.pop(prefix, None)
        mgr._eplb_pending_d2h.add(prefix)

        # 6) advance the send-buffer prefetch pipeline.
        self._prefetch_next(prefix)

        logger.info(
            "[EPLB-D2D] update done: %s kept=%s recv=%s send=%s wall=%.1fms",
            prefix,
            len(kept),
            len(expert_recv_info),
            len(expert_send_info),
            (_t.monotonic() - _t0) * 1000,
        )
        if layer_id == updator.adaptor.num_moe_layers - 1:
            self._cycles_done += 1
            logger.info(
                "[EPLB-D2D] full expert_map update cycle completed, total_layers=%s cycles_done=%s",
                updator.adaptor.num_moe_layers,
                self._cycles_done,
            )

    # ------------------------------------------------------------------ #
    # staging lifecycle helpers (manager hooks)
    # ------------------------------------------------------------------ #
    def _alloc_staging_pool(self):
        mgr = self.mgr
        prefix = mgr.layer_prefixes[0]
        entry = mgr.full_layer_expert_params[prefix]
        # ONE slot: by the time the next layer's update builds its
        # staging (next iteration's forward_end), this one has always
        # been consumed by the mid-iteration preload; the defensive
        # event waits in _acquire_staging cover the in-flight tail.
        flats = [
            torch.empty(t.numel(), dtype=torch.uint8, device="npu") if t is not None else None for t in entry.params
        ]
        self._staging_pool.append({"flats": flats, "last_d2h_ev": None, "last_preload_ev": None})
        self._staging_alloc_done = True
        logger.info("[EPLB-D2D] static staging pool ready (1 slot)")

    def _acquire_staging(self):
        if not self._staging_alloc_done:
            self._alloc_staging_pool()
        slot = self._staging_pool[0]
        # Drop the _staging dict entries that still point at this slot:
        # the slot is being rewritten for a new layer, and a stale entry
        # would make get_staging_for_preload(old_layer) return the NEW
        # layer's content (wrong data for a later preload of old_layer).
        for p in list(self._staging.keys()):
            if self._staging[p].npu_flats is slot["flats"]:
                del self._staging[p]
                # its D2H writeback is ordered by the ev_d2h sync below,
                # so the mirror is already current — drop the guard too
                self.mgr._eplb_pending_d2h.discard(p)
        # Defensive ordering against the slot's previous life: wait its
        # last D2H writeback (host sync; long done in practice) and let
        # the compute stream wait its last staging->slot preload.
        ev = slot["last_d2h_ev"]
        if ev is not None and not ev.query():
            ev.synchronize()
        ev = slot["last_preload_ev"]
        if ev is not None and not ev.query():
            torch.npu.current_stream().wait_event(ev)
        return slot

    def get_staging_for_preload(self, layer_prefix):
        """Returns the per-param NPU staging flats for the layer (new
        layout) if available, else None (fall back to mirror H2D)."""
        state = self._staging.get(layer_prefix)
        if state is None:
            return None
        return state.npu_flats

    def get_staging_recv_event(self, layer_prefix):
        """Event covering the compute-stream staging fills (the recv ->
        staging copies). The copy-stream fills are covered by same-
        stream ordering inside _enqueue_layer_copy."""
        state = self._staging.get(layer_prefix)
        if state is None:
            return None
        return state.ev_recv

    def get_d2h_event(self, layer_prefix):
        """Event covering the staging -> mirror writeback (None if the
        writeback has already been observed complete)."""
        state = self._staging.get(layer_prefix)
        if state is None or state.ev_d2h is None:
            return None
        if state.ev_d2h.query():
            # Writeback complete: self-clear the pending guard.
            self.mgr._eplb_pending_d2h.discard(layer_prefix)
            return None
        return state.ev_d2h

    def note_staging_preload(self, layer_prefix, copy_done_event):
        """Called by the manager when a staging->slot preload is
        submitted: records the copy_done event on the owning slot so
        the slot's next acquire can order against it."""
        state = self._staging.get(layer_prefix)
        if state is None:
            return
        for slot in self._staging_pool:
            if slot["flats"] is state.npu_flats:
                slot["last_preload_ev"] = copy_done_event
                break

    def on_layer_loaded(self, layer_prefix):
        """Called by the manager after a staging->slot load completes:
        marks the staging consumed and clears the pending-D2H guard
        (the slot now serves the new layout). The dict entry stays only
        until the slot is re-acquired for another layer (cleared in
        _acquire_staging); from now on this layer loads from the
        (eventually written-back) mirror."""
        state = self._staging.get(layer_prefix)
        if state is not None:
            state.consumed = True
        self.mgr._eplb_pending_d2h.discard(layer_prefix)
