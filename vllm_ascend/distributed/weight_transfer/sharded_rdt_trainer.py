# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend sharded_rdt trainer: local NPU IPC publish, remote HIXL READ.

The gather/ownership/serve-slot protocol is synchronized from upstream vLLM
24420b7a341af37a9bc3186e6fc6062512e0aa58. Ray transports descriptors only.
Consumer sessions must drain and disconnect before producer shutdown.
"""

import contextlib
import threading
import time
import uuid

import ray
import torch
from torch.multiprocessing.reductions import (
    StorageWeakRef,
)
from torch_npu.multiprocessing.reductions import rebuild_npu_tensor
from vllm.distributed.weight_transfer.sharded_rdt_trainer import (
    _EXPORT_RING_MAX_BYTES,
    DEFAULT_STALL_TIMEOUT_S,
    ShardedRDTTrainerWeightTransferEngine,
    _RDTProducerServer,
    _SharedPack,
)

from vllm_ascend.distributed.weight_transfer.hixl_session import HixlSession


class _HixlProducerServer(_RDTProducerServer):
    def __init__(
        self,
        *,
        num_rdt_buffers: int,
        device_index: int,
        buffer_presize_gb: float,
        gather_lookahead: int,
        served_names: list[str] | None = None,
        stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S,
        workers_per_replica: int = 0,
    ) -> None:
        import gc

        torch.npu.set_device(device_index)
        self._device_index = device_index
        # name -> rebuilt NPU-IPC tensor (or view); guarded by _cache_cond.
        self._cache: dict[str, torch.Tensor] = {}
        self._cache_cond = threading.Condition()
        self._gather_error: BaseException | None = None

        # Monotonic stamp of the last forward step of the
        # publish -> serve -> free loop. The waits below poll it instead of
        # blocking forever, so a dead consumer fails the sync with a real error.
        self._stall_timeout = float(stall_timeout_s)
        self._last_progress = time.monotonic()

        # Names this producer publishes, so a misrouted pull fails loudly
        # instead of blocking forever in the cache wait. None = serve anything.
        self._served_names = set(served_names) if served_names is not None else None

        # Every live consumer signals free_group(gi) at every
        # owner, once per sync; the group frees when the count reaches
        # begin_sync's live total — one uniform target, no routed per-producer
        # ones. Signals may precede the publish, which completes them.
        self._live_count = 1
        self._free_counts: dict[int, int] = {}
        # gi -> the names this producer published for it (what release drops).
        self._group_names: dict[int, list[str]] = {}

        # Published-but-unfreed groups, plus freed ones not
        # yet handed back. The memory gate lives in the ENGINE's gather loop; this
        # side only accounts — free_group moves a group from _inflight_groups to
        # _freed_pending, which the engine collects via wait_freed / end_sync to
        # drop its storage refs. _lookahead is engine-enforced, unused here.
        self._lookahead = max(0, gather_lookahead)
        self._inflight_groups: list[int] = []
        self._freed_pending: list[int] = []

        # Ring of packed serve buffers, one ring per SLOT-SHARING GROUP (one
        # consumer per group unless several deployments share); the slot within
        # it is the consumer's ``seq % nring``.
        self._nring = max(1, num_rdt_buffers)
        self._serve_rings: dict[int, list[torch.Tensor | None]] = {}
        self._serve_lock = threading.Lock()
        # registerMem on a shared HIXL agent is not concurrency-safe; serialize.
        self._reg_lock = threading.Lock()
        self._buffer_presize = int(buffer_presize_gb * (1 << 30))
        self._serve_device = torch.device("npu", self._device_index)

        self._hixl = HixlSession(self._serve_device)
        self._hixl_consumers: set[str] = set()
        self._hixl_closing = False

        # Slot-sharing state, all guarded by _cache_cond, so the rendezvous
        # waits ride the stall watchdog like every other wait here.
        # ``_share_width`` = consumers per deployment; 0 means every group is a
        # singleton and this degenerates to the unshared serve path.
        self._share_width = max(0, int(workers_per_replica))
        self._live_ids: set[int] | None = None
        self._sharing_active = False
        self._sharers: dict[int, frozenset] = {}  # group -> live members
        self._gens: dict[int, dict] = {}  # group -> {seq: _SharedPack}
        # group -> {cid: plan digest}. Init-time state, not cleared per sync: it
        # is what proves the sharers of a group pull the same chunks.
        self._plan_digests: dict[int, dict[int, str]] = {}

        # (sharing group, ring idx, packed layout) ->
        # (buffer data_ptr, destination views). Keyed on the buffer pointer too, so
        # a ring regrow invalidates rather than writing into a freed buffer. See
        # the serve path for why the layout, not the spec names, is the key.
        self._pack_dsts: dict[tuple, tuple[int, list[torch.Tensor]]] = {}
        # [RDT-EXPORT-RING] handle -> rebuilt base. The trainer reuses buffers
        # across groups, so the same handle arrives on many publishes; without
        # this we reopen it every group. Cleared per sync so no IPC mapping into
        # trainer memory outlives the sync that made it.
        self._base_cache: dict[tuple, torch.Tensor] = {}

        # Freeze the static post-init object graph so gen-2 GC never stops the
        # world mid-serve, which shows up as serve stragglers.
        gc.collect()
        gc.freeze()

    def hixl_add_consumer(self, endpoint: str) -> None:
        with self._cache_cond:
            if self._hixl_closing:
                raise RuntimeError("HIXL producer is shutting down")
            self._hixl_consumers.add(endpoint)

    def hixl_remove_consumer(self, endpoint: str) -> None:
        # Called only after the consumer drained and disconnected its session.
        with self._cache_cond:
            self._hixl_consumers.discard(endpoint)

    def publish_group(self, group_idx: int, entries: tuple) -> None:
        """Rebuild one gather group's NPU-IPC tensors into the serve cache.

        NEVER blocks: a gathered group is serveable immediately. The memory bound
        lives in the engine's gather loop, which stops GATHERING (not publishing)
        past ``gather_lookahead`` unfreed groups.

        ``entries`` is ``(storages, views)``: one NPU-IPC export per storage,
        plus per-name ``(sid, dtype_name, shape, stride, storage_offset)`` rebuilt
        here as ``as_strided`` views.

        Signals can arrive BEFORE their publish — a consumer pulling nothing of a
        group signals it as its plan starts — so a group whose barrier is already
        satisfied is released here. Freed groups reach the engine only through
        ``wait_freed`` / ``end_sync``: a freed notice riding an unharvested async
        publish result would wedge the loop.
        """
        torch.npu.set_device(self._device_index)
        storages, views = entries
        bases: dict[int, torch.Tensor] = {}
        for sid, reduce_args in storages.items():
            cached = self._base_cache.get(reduce_args)
            if cached is not None:
                bases[sid] = cached
                continue
            list_args = list(reduce_args)
            # Index 6 of reduce_tensor's args is the exporter's device index;
            # rebuild on this server's device (same physical NPU as the rank).
            list_args[6] = self._device_index
            bases[sid] = self._base_cache[reduce_args] = rebuild_npu_tensor(*list_args)
        rebuilt: dict[str, torch.Tensor] = {}
        for name, (sid, dtype_name, shape, stride, storage_offset) in views.items():
            typed = bases[sid].view(getattr(torch, dtype_name))
            rebuilt[name] = torch.as_strided(typed, shape, stride, storage_offset)
        del bases

        with self._cache_cond:
            self._cache.update(rebuilt)
            self._inflight_groups.append(group_idx)
            self._group_names[group_idx] = list(rebuilt)
            if self._free_counts.get(group_idx, 0) >= self._live_count:
                self._release_group_locked(group_idx)
            self._note_progress_locked()
            self._cache_cond.notify_all()

    def _new_serve_buffer(self, nbytes: int) -> torch.Tensor:
        """Allocate + HIXL-register one serve slot: the single allocation seam,
        so registration cannot be skipped on either of the two paths that make
        buffers (the init-time reservation and the serve-path backstop)."""
        with self._reg_lock:
            t = self._hixl.allocate(nbytes)
        return t

    def _serve_slot(self, sg: int, idx: int, need: int) -> torch.Tensor:
        with self._serve_lock:
            buffer = self._serve_rings.get(sg, [None] * self._nring)[idx]
        if buffer is None or buffer.numel() < need:
            raise RuntimeError("HIXL serve ring cannot grow after initialization")
        return buffer

    def rdt_produce_weights_batched(self, specs: list, consumer_id: int = 0, seq: int = -1):
        # Reuse upstream validation, rendezvous and failure propagation.
        # Its virtual pack hook retains our NPU completion-before-publish rule.
        torch.npu.set_device(self._device_index)
        [blob] = super().rdt_produce_weights_batched(specs, consumer_id, seq)
        torch.npu.current_stream(self._serve_device).synchronize()
        return self._hixl.endpoint, blob.data_ptr(), blob.numel()

    def _pack_shared(self, gen: _SharedPack, specs: list) -> torch.Tensor:
        # Upstream publishes a pack as soon as copies are enqueued. Pack into a
        # private generation so other consumers cannot read it until NPU work
        # completes, while reusing upstream's op validation and layout cache.
        torch.npu.set_device(self._serve_device)
        blob = super()._pack_shared(_SharedPack(group=gen.group, slot=gen.slot), specs)
        torch.npu.current_stream(self._serve_device).synchronize()
        with self._cache_cond:
            gen.blob = blob
            gen.done = True
            self._note_progress_locked()
            self._cache_cond.notify_all()
        return blob

    def shutdown(self) -> None:
        with self._cache_cond:
            if self._hixl_consumers:
                raise RuntimeError("Close inference HIXL sessions before the producer")
            self._hixl_closing = True
        self._hixl.close()
        with self._cache_cond:
            self._cache.clear()
            self._base_cache.clear()
        with self._cache_cond:
            # Generations hold blob views into the rings, so they go first.
            self._gens.clear()
            self._sharers.clear()
        with self._serve_lock:
            self._serve_rings.clear()
            # Must go with the rings: these are views INTO them, and the
            # data_ptr guard that normally invalidates them cannot tell a freed
            # buffer from a new one recycled at the same address.
            self._pack_dsts.clear()


class ShardedRDTHixlTrainerWeightTransferEngine(ShardedRDTTrainerWeightTransferEngine):
    def _pack_group_for_export(self, held: list, slot_idx: int) -> tuple:
        """Copy ``held`` into ring slot ``slot_idx``; return ``(storages, views,
        refs)`` in the same shape the per-storage path returns, but with ONE
        storage. Views are built exactly as ``publish_group`` rebuilds them, so
        the two sides cannot disagree about layout.

        Slot safety is not the credit gate alone: that bounds how MANY groups
        are unfreed, not the order they free in, and barriers do complete out of
        order. The caller takes ``slot_idx`` from a pool keyed by live group
        (``_slot_of_group`` / ``_free_slots``); ``_drop_inflight`` returns a slot
        only once its group is freed everywhere.
        """
        offsets: list[int] = []
        cur = 0
        for _name, t in held:
            cur = (cur + 15) & ~15  # 16B keeps every element size we ship aligned
            offsets.append(cur)
            cur += t.numel() * t.element_size()
        need = max(16, (cur + 15) & ~15)

        slot = self._export_ring[slot_idx]
        device = held[0][1].device
        if slot is None or slot.numel() < need or slot.device != device:
            self._export_ring[slot_idx] = self._export_ring_args[slot_idx] = None
            slot = torch.empty(need, dtype=torch.uint8, device=device)
            self._export_ring[slot_idx] = slot

        ust = slot.untyped_storage()
        sid = ust.data_ptr()
        reduce_args = self._export_ring_args[slot_idx]
        if reduce_args is None:
            base = torch.empty(0, dtype=torch.uint8, device=device)
            base.set_(ust, 0, (ust.nbytes(),))
            _rebuild, reduce_args = _reduce_npu_tensor(base)
            self._export_ring_args[slot_idx] = reduce_args

        views: dict[str, tuple] = {}
        refs: dict[str, torch.Tensor] = {}
        for (name, t), off in zip(held, offsets):
            shape, esz = tuple(t.shape), t.element_size()
            stride = self._contig_stride(shape)
            dst = torch.as_strided(slot.view(t.dtype), shape, stride, off // esz)
            dst.copy_(t)
            refs[name] = dst
            views[name] = (
                sid,
                str(t.dtype).split(".")[-1],
                list(shape),
                list(stride),
                off // esz,
            )
        # The producer sidecar reads this slot from another process over NPU
        # IPC, while these copies are merely enqueued on this rank's stream;
        # nothing orders the two. Without this sync it packs bytes the copy has
        # not written yet -- silent corruption, and only for ring-packed tensors.
        if slot.device.type == "npu":
            torch.accelerator.current_stream(slot.device).synchronize()
        return {sid: reduce_args}, views, refs

    def _publish_async(self, group_idx: int, entries):
        # The sidecar opens NPU IPC storage on a different stream/process.
        torch.npu.current_stream().synchronize()
        return super()._publish_async(group_idx, entries)

    def _spawn_server(self, served_names: list[str]) -> None:
        import os

        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        ii = self._init_info
        self._server_name = f"wx_rdt_producer_{uuid.uuid4().hex[:12]}_rk{ii.rank}"
        env_vars = {
            key: value
            for key, value in os.environ.items()
            if key
            in (
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "PATH",
                "PYTHONPATH",
                "ASCEND_HOME_PATH",
                "ASCEND_OPP_PATH",
                "ASCEND_RT_VISIBLE_DEVICES",
                "HCCL_IF_IP",
            )
        }
        # Preserve the trainer's visible-device mapping; the sidecar uses the
        # same logical index and never reserves an additional accelerator.
        env_vars["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"
        server_cls = ray.remote(_HixlProducerServer).options(
            name=self._server_name,
            namespace=ii.trainer_actor_namespace,
            num_cpus=0,
            num_gpus=0,
            max_concurrency=ii.num_consumers * ii.num_rdt_buffers + 4,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(), soft=False
            ),
            runtime_env={"env_vars": env_vars},
        )
        self._server = server_cls.remote(
            device_index=torch.npu.current_device(),
            served_names=served_names,
            num_rdt_buffers=ii.num_rdt_buffers,
            buffer_presize_gb=ii.buffer_presize_gb,
            gather_lookahead=ii.gather_lookahead,
            stall_timeout_s=ii.stall_timeout_s,
            workers_per_replica=ii.workers_per_replica,
        )
        ray.get(self._server.ping.remote())

    def _run_gather_loop(self, update_future, live_count: int, live_ids: list[int] | None = None) -> None:
        """Gather this rank's weights group-by-group and publish each into the
        server over NPU IPC. A gathered group is published — serveable —
        immediately; the loop gates BEFORE the next gather while more than
        `gather_lookahead` groups are unfreed (the per-group free barrier: a
        credit releases when every live consumer has signaled the group). So
        the loop self-paces to the consumers' pull rate with at most
        `gather_lookahead + 1` groups resident. Runs on every rank; only the
        sender has an `update_future` to fail fast on."""
        assert self.source is not None  # guaranteed by trainer_init
        # The live IDS ride along with the count: the free barrier needs only
        # the count, but the slot-sharing rendezvous needs to know WHICH
        # consumers are live.
        self._rpc("begin_sync", live_count, live_ids)
        # Every owner reaches a group in the same (metadata) order, so their
        # shared gathers line up.
        groups = self._iter_owned_groups()
        # Publishes fire without an inline ray.get and are harvested this window
        # deep, so the RPC and server-side rebuild overlap the NEXT group's
        # gather/export. The window also surfaces a server-side error at depth 2
        # rather than at end_sync.
        _PUBLISH_WINDOW = 2
        pending_publish: list = []
        # The memory bound: gate BEFORE gathering while more
        # than `bound` groups are unfreed, so at most bound + 1 are resident. The
        # count is exact — `_inflight` gains its entry right after each gather and
        # only wait_freed/end_sync shrinks it, so a group freed server-side but not
        # yet collected still holds its refs and still counts. The publish window
        # drains first, so every resident group is pullable while we wait.
        #
        # 0 is legal and means NO pipelining: the loop waits for group N to be
        # freed by every live consumer before gathering N+1, so exactly one group
        # is resident and the sync costs the full serialized
        # gather + publish + pull + free RTT per group. 1 (the default) overlaps
        # N+1's gather with N's pulls.
        bound = max(0, self._init_info.gather_lookahead)

        # [RDT-EXPORT-RING] `bound + 1` slots is exactly the residency the credit
        # gate enforces. Small storages are packed into a slot and exported once
        # per slot instead of once per storage (518 -> 7 cudaIpcGetMemHandle
        # calls per rank per sync at 235B, each 0.5-1.5ms).
        nslots = bound + 1
        # data_ptr -> (storage weakref, reduce_args), this sync only. `storages`
        # is per-group, so a source that reuses a buffer across groups would
        # otherwise re-export it every group. The weakref is load-bearing: a
        # freed allocation can be reissued at the same address, and serving that
        # from a stale handle is silently wrong bytes.
        handle_cache: dict[int, tuple] = {}
        if len(getattr(self, "_export_ring", [])) != nslots:
            self._export_ring = [None] * nslots
            self._export_ring_args = [None] * nslots
        # Slots are handed out per group and returned by _drop_inflight, so the
        # pool starts full every sync.
        self._slot_of_group = {}
        self._free_slots = list(range(nslots))

        try:
            for gi in self._owned_idx:
                while len(self._inflight) > bound:
                    while pending_publish:
                        self._await_publish(pending_publish.pop(0))
                    self._drop_inflight(self._rpc("wait_freed"))
                names, tensors = next(groups)
                # Share each unique STORAGE once and describe every name as an
                # as_strided view onto it.
                storages: dict[int, tuple] = {}
                views: dict[str, tuple] = {}
                refs: dict[str, torch.Tensor] = {}
                small: list = []
                for name, tensor in zip(names, tensors):
                    tensor = tensor.detach()
                    if tensor.device.type != "npu":
                        tensor = tensor.npu()
                    # Threshold on the STORAGE, not the tensor: a source may hand
                    # out many small views of one large stacked buffer, and a
                    # per-tensor test would pack the whole thing, costing +1
                    # group of peak memory for a storage already coalesced.
                    if tensor.untyped_storage().nbytes() <= _EXPORT_RING_MAX_BYTES:
                        small.append((name, tensor))
                        continue
                    tensor = tensor.contiguous()
                    refs[name] = tensor  # keep the export alive
                    ust = tensor.untyped_storage()
                    sid = ust.data_ptr()
                    if sid not in storages:
                        cached = handle_cache.get(sid)
                        if cached is not None and not cached[0].expired():
                            reduce_args = cached[1]
                        else:
                            base = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                            base.set_(ust, 0, (ust.nbytes(),))
                            _rebuild, reduce_args = _reduce_npu_tensor(base)
                            handle_cache[sid] = (StorageWeakRef(ust), reduce_args)
                        storages[sid] = reduce_args
                    views[name] = (
                        sid,
                        str(tensor.dtype).split(".")[-1],
                        list(tensor.shape),
                        list(tensor.stride()),
                        tensor.storage_offset(),
                    )
                if small:
                    # Take a slot no live group holds. The gate above already
                    # guarantees one exists, so this is a backstop -- but it
                    # must drain publishes first, like the credit gate.
                    while not self._free_slots:
                        while pending_publish:
                            self._await_publish(pending_publish.pop(0))
                        self._drop_inflight(self._rpc("wait_freed"))
                    slot_idx = self._free_slots.pop(0)
                    self._slot_of_group[gi] = slot_idx
                    st, vw, rf = self._pack_group_for_export(small, slot_idx)
                    storages.update(st)
                    views.update(vw)
                    refs.update(rf)
                del small
                del tensors
                # The loop leaves its last `tensor` / `ust` / `base` bound
                # through the credit gate and into the next gather, and `base`
                # is a whole-storage view -- so one tensor and its entire
                # allocation stay alive past `_drop_inflight`, unseen by the
                # `_inflight` accounting the residency bound rests on.
                # Assignment, not `del`: unbound if no direct-path tensor.
                tensor = ust = base = None
                if not refs:
                    # A group with nothing held here cannot occur (every group
                    # carries replicated names), but publishing an empty group
                    # would park a credit nobody's pull is waiting on; skip.
                    continue
                # Hold our refs before publishing; drop them only when the
                # server reports the group freed (IPC export must outlive import).
                self._inflight[gi] = refs
                pending_publish.append(self._publish_async(gi, (storages, views)))
                while len(pending_publish) >= _PUBLISH_WINDOW:
                    self._await_publish(pending_publish.pop(0))
                if update_future is not None and update_future.done():
                    # update_weights returned/failed early — surface now instead
                    # of blocking further gathers.
                    update_future.result()
            while pending_publish:
                self._await_publish(pending_publish.pop(0))
            freed = self._rpc("end_sync")
            self._drop_inflight(freed)
        except BaseException as e:
            with contextlib.suppress(Exception):
                self._rpc("set_gather_error", repr(e))
            self._inflight.clear()
            raise
        finally:
            # Keep the slots, drop the handles: the sidecar's base cache is
            # cleared per sync, so a handle must not outlive the sync either.
            self._export_ring_args = [None] * len(getattr(self, "_export_ring", []))

    def shutdown(self) -> None:
        if self._server is None:
            return
        import ray

        ray.get(self._server.shutdown.remote(), timeout=300)
        ray.kill(self._server)
        self._server = None
        self._inflight.clear()


def _reduce_npu_tensor(tensor):
    # torch_npu replaces this function during initialization. Resolve it at
    # call time so a previously imported CUDA reducer cannot escape the patch.
    rebuild, args = torch.multiprocessing.reductions.reduce_tensor(tensor)
    if rebuild is not rebuild_npu_tensor:
        raise RuntimeError("torch_npu NPU IPC reduction is not active")
    return rebuild, args
