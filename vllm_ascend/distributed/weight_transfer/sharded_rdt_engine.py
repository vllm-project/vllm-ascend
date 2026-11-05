# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend sharded_rdt consumer: Ray control plane and HIXL READ data plane.

Planning and layerwise replay are synchronized from upstream vLLM commit
24420b7a341af37a9bc3186e6fc6062512e0aa58. Unlike upstream, actor calls return
an endpoint/address/size descriptor, never a tensor through Ray RDT. Receive
rings are registered once with HIXL; NPU events guard their scatter lifetime.
"""

from typing import Any

import torch
from vllm.distributed.weight_transfer.sharded_rdt_common import (
    RdtRouter,
    buffer_alloc_bytes,
)
from vllm.distributed.weight_transfer.sharded_rdt_engine import (
    ShardedRDTWeightTransferEngine,
    ShardedRDTWeightTransferInitInfo,
    _Chunk,
    _PendingPull,
    _plan_digest,
    _ProcItem,
)
from vllm.distributed.weight_transfer.sharded_rdt_fake import (
    FetchKey,
)
from vllm.logger import logger

from vllm_ascend.distributed.weight_transfer.hixl_session import HixlSession


class ShardedRDTHixlWeightTransferEngine(ShardedRDTWeightTransferEngine):
    def init_transfer_engine(self, init_info: ShardedRDTWeightTransferInitInfo) -> None:
        """Configure the ring, bind the producers, bake the replay plan, and
        pre-register every HIXL buffer -- in that order, because each step depends
        on the previous one.

        The bake drives ``model.load_weights`` and the pre-registration blocks on
        RPCs to the producers, so this is a heavyweight one-off; every later
        ``update_weights`` is pure replay.
        """
        # Read the attribute directly: if a vLLM bump renames it, this guard
        # must fail loudly at init rather than silently disappear.
        if self.parallel_config.enable_eplb:
            raise RuntimeError(
                "sharded_rdt does not support EPLB (enable_eplb=true): dynamic "
                "expert rearrangement invalidates the baked replay plan, which "
                "records each expert's destination slot once at init — after a "
                "rearrangement the replay would silently load weights into the "
                "wrong expert slots. Disable EPLB or use another weight-transfer "
                "backend."
            )
        # The worker is often a different install from whatever spawned it, so
        # check here too and not only where the actor options are set.
        self._hixl = HixlSession(self.device)
        self._configure_ring(init_info)
        self._resolve_producers(init_info)

        # A pure dry run: the trainer's gather cache is empty at init, so nothing
        # can (or does) get pulled -- we only record how each slice is fetched and
        # where it lands, then restore the model.
        self._bake(init_info)
        self._build_static_plan(init_info)
        # Register ALL HIXL memory now, while the fabric is idle: dma-buf NPUDirect
        # registration concurrent with in-flight RDMA intermittently fails
        # (ibv_reg_mr 'Bad address'), which bites under M:N fan-in. Sizes come from
        # the static plan, so this is exact.
        self._preregister_at_init()
        # Start the background post-processing worker (pull/process pipelining).
        self._ensure_proc_worker()

    def _bake(self, init_info: ShardedRDTWeightTransferInitInfo) -> None:
        from vllm.model_executor.model_loader.reload.meta import SKIP_LOAD_TENSORS, SKIP_TENSORS

        # Ascend's EP dispatcher registers this runtime routing buffer on the
        # expert layer. It has no checkpoint entry: counting it makes a complete
        # expert load look partial, and materializing it would corrupt routing.
        # Extend the upstream process-wide skip registries, just as _expert_map
        # is excluded there; do not temporarily mutate them around worker threads.
        SKIP_LOAD_TENSORS.add("expert_ids_per_ep_rank")
        SKIP_TENSORS.add("expert_ids_per_ep_rank")
        super()._bake(init_info)

    def _resolve_producers(self, init_info: ShardedRDTWeightTransferInitInfo) -> None:
        """Work out this worker's consumer identity, build the router, and bind
        EVERY producer actor.

        Pull routing is M:N: each chunk goes to ONE producer holding every name
        in it (see ``RdtRouter``), so one producer serves a whole pull. All
        producers are bound regardless, because the per-group ``free_group``
        signal fans out to every owner of a group, including producers this
        worker never pulls from.
        """
        try:
            import ray
        except ImportError as e:
            raise RuntimeError(
                "Ray is required for the 'sharded_rdt' weight transfer "
                "backend. Install Ray and run workers as Ray actors "
                "(distributed_executor_backend='ray')."
            ) from e

        producer_names = list(init_info.trainer_actor_names)
        if not producer_names:
            raise RuntimeError("Sharded RDT engine requires a trainer producer: set init_info.trainer_actor_names.")

        consumer_id = self._resolve_consumer_id(init_info)
        name_owner_class = list(init_info.name_owner_class or [])
        if name_owner_class and len(name_owner_class) != len(init_info.names):
            raise RuntimeError(
                f"Sharded RDT engine: {len(name_owner_class)} owner-class entries for {len(init_info.names)} names."
            )
        router = RdtRouter(
            len(producer_names),
            self._num_consumers(),
            list(init_info.owner_sets) or None,
            name_owner_class or None,
            list(init_info.names),
            list(init_info.group_lens),
            # Carve the block per deployment so every replica's worker w pulls
            # each name from the same producer, which is what lets that producer
            # serve all of them from one slot.
            workers_per_replica=self._workers_per_replica(init_info),
        )
        router.validate()

        # A fresh router per init, never appended to: every owner index is a
        # position in these lists, so a rejoining engine's second
        # init_transfer_engine must not shift them.
        actors: list[Any] = []
        methods: list[Any] = []

        for chosen_name in producer_names:
            try:
                actor = ray.get_actor(
                    chosen_name,
                    namespace=init_info.trainer_actor_namespace,
                )
            except ValueError as e:
                raise RuntimeError(
                    f"Sharded RDT engine could not find trainer actor "
                    f"{chosen_name!r} (namespace="
                    f"{init_info.trainer_actor_namespace!r})."
                ) from e
            ray.get(actor.hixl_add_consumer.remote(self._hixl.endpoint), timeout=300)
            actors.append(actor)
            methods.append(getattr(actor, init_info.produce_method_name))
        router.bind(actors, methods, consumer_id)
        self._router = router
        logger.info(
            "Sharded RDT engine (consumer %d) bound to all %d producers (batched method %r, %d owner class(es))",
            consumer_id,
            len(producer_names),
            init_info.produce_method_name,
            len(init_info.owner_sets) or 1,
        )

    def _preregister_at_init(self) -> None:
        """Register every HIXL buffer this worker will use at init, before any
        transfer runs, so nothing registers during the sync-0 RDMA churn.

        Both sides are sized from the static plan: receive buffers are
        ``ring_depth`` slots at the largest chunk's ``pack_bytes``, and each bound
        producer is asked to pre-register a serve ring at the max bytes this
        consumer will pull from it. A no-op when this worker has no chunks."""
        plan = self._cached_plan
        if plan is None or not plan.chunks:
            return
        # (a) consumer receive buffers — one per ring slot, at the largest chunk.
        max_pack = max(c.pack_bytes for c in plan.chunks)
        alloc = buffer_alloc_bytes(max_pack, self._buffer_presize)
        for slot in range(self._ring_depth):
            buffer = self._dest_buffers[slot].get(torch.uint8)
            if buffer is None or buffer.numel() < alloc:
                buffer = self._hixl.allocate(alloc)
                self._dest_buffers[slot][torch.uint8] = buffer

        # (b) producer serve rings — max bytes this consumer pulls from each
        # bound producer, plus the per-producer plan digest a producer sharing
        # one ring across deployments checks them against.
        assert self._router is not None
        serve_bytes = [0] * self._router.num_producers
        per_owner_keys: list[list] = [[] for _ in range(self._router.num_producers)]
        for c in plan.chunks:
            serve_bytes[c.owner] = max(serve_bytes[c.owner], c.pack_bytes)
            per_owner_keys[c.owner].append(c.keys)
        digests = [_plan_digest(keys) for keys in per_owner_keys]
        import ray

        refs = self._router.reserve_serve_buffers(serve_bytes, digests)
        if refs:
            ray.get(refs)  # block until every serve ring is registered
        logger.info(
            "[RDT-PLAN] pre-registered %d receive slots (%.0f MiB each) + serve rings on %d producer(s) %s",
            self._ring_depth,
            alloc / (1 << 20),
            len(refs),
            [nb // (1 << 20) for nb in serve_bytes],
        )

    def _issue_pull(self, chunk: "_Chunk", slot: int) -> "_PendingPull":
        if self._slot_read_done:
            with self._slot_cv:
                while self._slot_done[slot] < self._slot_queued[slot]:
                    self._slot_cv.wait(timeout=1.0)
            self._raise_proc_error()
            self._slot_read_done[slot].synchronize()
        buffer = self._dest_buffers[slot][torch.uint8]
        if buffer.numel() < chunk.pack_bytes:
            raise RuntimeError("HIXL receive ring cannot grow after initialization")
        assert self._router is not None
        ref = self._router.pull(chunk.owner, chunk.keys, chunk.seq)
        return _PendingPull(
            ref=ref,
            keys=chunk.keys,
            targets=self._pull_targets(chunk, slot, buffer),
            blob=buffer[: chunk.pack_bytes],
            slot=slot,
        )

    def _complete_pull(self, pending: "_PendingPull") -> "dict[FetchKey, torch.Tensor]":
        import ray

        endpoint, address, nbytes = ray.get(pending.ref, timeout=300)
        if nbytes != pending.blob.numel():
            raise RuntimeError("HIXL producer and consumer packed sizes disagree")
        request = self._hixl.read(endpoint, pending.blob, address, nbytes)
        self._hixl.wait(request)
        return dict(zip(pending.keys, pending.targets))

    def _ensure_proc_worker(self) -> None:
        """Lazily create the per-slot events, the background NPU stream, the work
        queue, and the single processing thread. Idempotent."""
        if self._proc_thread is not None:
            return
        import queue
        import threading

        self._slot_read_done = [torch.npu.Event() for _ in range(self._ring_depth)]
        # Generation handshake for the events above: the RPC thread counts items
        # queued per slot, this thread counts records, and a pull may synchronize()
        # only once done[slot] has caught up with queued[slot]. Without it the
        # synchronize can bind to a stale record and pass, letting the next RDMA
        # overwrite a slot under a pending scatter. See the doc's "The slot
        # generation handshake".
        self._slot_queued = [0] * self._ring_depth
        self._slot_done = [0] * self._ring_depth
        self._slot_cv = threading.Condition()
        self._proc_stream = torch.npu.Stream(device=self.device)
        self._proc_queue = queue.Queue()
        self._proc_error = None
        t = threading.Thread(target=self._proc_worker_loop, name="rdt-postprocess", daemon=True)
        self._proc_thread = t
        t.start()
        self._quant_stream = torch.npu.Stream(device=self.device)
        self._quant_queue = queue.Queue()
        qt = threading.Thread(target=self._quant_worker_loop, name="rdt-quant", daemon=True)
        self._quant_thread = qt
        qt.start()

    def _process_item(self, item: "_ProcItem") -> None:
        """Scatter-thread half: materialize this chunk's first-seen modules,
        scatter its slices on the process stream, publish the slot, then hand the
        modules it COMPLETES to the quant thread.

        Mirrors ``_layerwise_process`` minus the loader replay. Once every scatter
        reading ``item.slot`` is enqueued, records the slot's read-done event so
        the RPC thread can block on it before overwriting the slot.
        """
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
        )
        from vllm.model_executor.model_loader.reload.meta import materialize_layer

        results = item.results
        chunk = item.chunk
        with (
            torch.accelerator.device_index(self.device.index),
            torch.npu.stream(self._proc_stream),
            torch.device(self.device),
        ):
            # PASS 1 — slot readers: materialize the modules whose FIRST scatter
            # is in this chunk (empty HF params, once per module by construction)
            # then scatter this chunk's copies. The scatter copies are the ONLY
            # reads of the receive buffer; quant and kernel-copy operate on the
            # scattered params. Releasing the slot right after the scatters lets
            # the NEXT chunk's RDMA overwrite the buffer while quant still runs.
            try:
                for layer in chunk.materialize:
                    info = LAYERWISE_INFO.get(layer)
                    if info is None or not info.can_load():
                        raise RuntimeError(
                            f"Baked replay: layer {type(layer).__name__} "
                            "was not set up for reload this sync "
                            "(start_weight_update must run before "
                            "update_weights)."
                        )
                    materialize_layer(layer, info)
                for sc in chunk.scatters:
                    param = getattr(sc.layer, sc.param_name)
                    # Through .data, exactly like default_weight_loader: params
                    # that keep requires_grad=True (plain nn.Parameters like
                    # GLM's router bias) reject an in-place copy through an
                    # autograd view of the leaf.
                    dst = param.data.as_strided(sc.shape, sc.stride, sc.offset)
                    with torch._C.DisableTorchFunctionSubclass():
                        dst.copy_(results[sc.src])
                # All reads of this slot's buffer are now enqueued on the process
                # stream; record + publish so the RPC thread can reuse the slot.
                self._slot_read_done[item.slot].record(self._proc_stream)
            finally:
                # Publish even on error so the RPC thread's generation wait
                # unblocks (the failure itself surfaces via _raise_proc_error).
                self._mark_slot_done(item.slot)

            # PASS 2 — param readers only: quant / kernel-copy / reset for the
            # modules whose LAST scatter is in this chunk. Handed to the
            # DEDICATED quant thread (own NPU stream, event-chained after this
            # chunk's scatters) so it never delays this thread's next pass-1.
            # Running it in order here stalls the RPC thread's slot handshake by
            # ~0.5-0.75s/iter, since every group's quant delays the next publish.
            if chunk.quant:
                ready = torch.npu.Event()
                ready.record(self._proc_stream)
                if self._quant_queue is None:
                    self._run_quant(chunk.quant, ready)
                else:
                    self._quant_queue.put((chunk.quant, ready))

    def _run_quant(self, layers: "list[Any]", ready: "torch.Event") -> None:
        """Quant/kernel-copy/reset the given COMPLETED leaf modules, exactly as
        _layerwise_process. Runs on the quant thread's own stream, ordered after
        the modules' scatters via ``ready``; touches only the scattered params
        (never a receive slot), so it can overlap subsequent chunks' RDMA and
        scatters. ``info.reset()`` is what makes finalize skip the layer —
        drain_pending joins the quant queue before finalize runs."""
        from vllm.model_executor.layers.quantization.base_config import (
            QuantizeMethodBase,
        )
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
            _copy_and_restore_kernel_tensors,
        )

        stream = self._quant_stream or self._proc_stream
        assert stream is not None  # created in _ensure_proc_worker before use
        with (
            torch.accelerator.device_index(self.device.index),
            torch.npu.stream(stream),
            torch.device(self.device),
        ):
            stream.wait_event(ready)
            for layer in layers:
                info = LAYERWISE_INFO.get(layer)
                assert info is not None  # completed leaf module is set up for reload
                quant_method = getattr(layer, "quant_method", None)
                if isinstance(quant_method, QuantizeMethodBase):
                    if hasattr(layer, "_already_called_process_weights_after_loading"):
                        delattr(layer, "_already_called_process_weights_after_loading")
                    quant_method.process_weights_after_loading(layer)
                # Copy into persistent kernel storage (preserves cudagraph refs).
                if info.kernel_tensors is not None:
                    _copy_and_restore_kernel_tensors(layer, info)
                # Reset so finalize_layerwise_reload skips this (loaded) layer.
                info.reset()

    def shutdown(self) -> None:
        # Propagate drain failures and retain live storage instead of dropping
        # buffers underneath outstanding device work.
        self.drain_pending()
        for thread, work_queue in (
            (self._proc_thread, self._proc_queue),
            (self._quant_thread, self._quant_queue),
        ):
            if thread is not None:
                work_queue.put(None)
                thread.join(timeout=30)
                if thread.is_alive():
                    raise TimeoutError("NPU weight processing thread did not stop")
        if hasattr(self, "_hixl"):
            self._hixl.close()
            if self._router is not None:
                import ray

                ray.get(
                    [actor.hixl_remove_consumer.remote(self._hixl.endpoint) for actor in self._router._actors],
                    timeout=300,
                )
        self._proc_thread = self._quant_thread = None
        self._proc_queue = self._quant_queue = None
        self._proc_stream = self._quant_stream = None
        self._slot_read_done = []
        self._router = None
        self._name_to_plan.clear()
        self._name_meta.clear()
        self._live_names.clear()
        self._cached_plan = None
        self._dest_buffers = [{} for _ in range(self._ring_depth)]
