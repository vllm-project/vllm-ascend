# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend sharded_rdt consumer: Ray control plane and HIXL READ data plane.

Planning and layerwise replay are inherited from vLLM. HIXL actor calls return
an endpoint/address/size descriptor, never a tensor through Ray RDT. Receive
rings are registered once with HIXL; NPU events guard their scatter lifetime.
"""

from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.weight_transfer.sharded_rdt_common import (
    buffer_alloc_bytes,
)
from vllm.distributed.weight_transfer.sharded_rdt_engine import (
    ShardedRDTWeightTransferEngine,
    ShardedRDTWeightTransferInitInfo,
    _Chunk,
    _PendingPull,
    _ProcItem,
)
from vllm.distributed.weight_transfer.sharded_rdt_fake import (
    FetchKey,
)

from vllm_ascend.distributed.weight_transfer.hixl_session import HixlSession

if TYPE_CHECKING:
    import queue
    import threading

    from vllm.distributed.weight_transfer.sharded_rdt_common import RdtRouter
    from vllm.distributed.weight_transfer.sharded_rdt_engine import _CallPlan
    from vllm.distributed.weight_transfer.sharded_rdt_fake import _Scatter


class _ConsumerConnection:
    """The receive registrations and producer leases of one trainer connection."""

    def __init__(self, device: torch.device) -> None:
        self.session = HixlSession(device)
        self.producers: list[Any] = []

    def attach(self, actor: Any) -> None:
        import ray

        # Track the actor before waiting, so a lost reply also triggers removal.
        self.producers.append(actor)
        ray.get(actor.hixl_add_consumer.remote(self.session.endpoint), timeout=300)

    def close(self) -> None:
        import ray

        self.session.close()
        while self.producers:
            actor = self.producers[-1]
            ray.get(actor.hixl_remove_consumer.remote(self.session.endpoint), timeout=300)
            self.producers.pop()


class ShardedRDTHixlWeightTransferEngine(ShardedRDTWeightTransferEngine):
    _router: "RdtRouter | None"
    # Initialized by the upstream constructor. Declare the types here because
    # CI skips imported implementations and cannot infer their instance state.
    _cached_plan: "_CallPlan | None"
    _name_to_plan: "dict[str, list[_Scatter]]"
    _dest_buffers: list[dict[torch.dtype, torch.Tensor]]
    _slot_read_done: list[Any]
    _slot_cv: Any
    _slot_done: list[int]
    _slot_queued: list[int]
    _proc_thread: "threading.Thread | None"
    _quant_thread: "threading.Thread | None"
    _proc_queue: "queue.Queue[_ProcItem | None] | None"
    _quant_queue: "queue.Queue[tuple[list[Any], Any] | None] | None"
    _proc_stream: Any
    _quant_stream: Any

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._connection: _ConsumerConnection | None = None
        self._retired_connections: list[_ConsumerConnection] = []

    def init_transfer_engine(self, init_info: ShardedRDTWeightTransferInitInfo) -> None:
        if getattr(init_info, "staging_device", "gpu") != "gpu":
            raise ValueError("HIXL READ requires NPU receive buffers")
        previous = self._connection
        if previous is not None:
            self.drain_pending()
        self._close_retired_connections()
        connection = _ConsumerConnection(self.device)
        self._connection = connection
        # v0.34 builds directly into the engine. Save its planning fields for
        # rollback. Newer vLLM stages these fields in its own init candidate.
        legacy = not hasattr(ShardedRDTWeightTransferEngine, "_alloc_receive_buffer")
        saved = {}
        if legacy:
            saved = {
                key: getattr(self, key)
                for key in (
                    "_num_consumers_override",
                    "_ring_depth",
                    "_dest_buffers",
                    "_targets_cache",
                    "_buffer_presize",
                    "_router",
                    "_cached_plan",
                    "_name_meta",
                    "_live_names",
                    "_name_to_plan",
                )
            }
            self._name_to_plan = {}
        try:
            super().init_transfer_engine(init_info)
        except Exception:
            self._connection = previous
            self.__dict__.update(saved)
            self._retire_connection(connection)
            raise
        if previous is not None:
            self._retire_connection(previous)

    def _retire_connection(self, connection: _ConsumerConnection) -> None:
        self._retired_connections.append(connection)
        self._close_retired_connections()

    def _close_retired_connections(self) -> None:
        while self._retired_connections:
            self._retired_connections[-1].close()
            self._retired_connections.pop()

    def _bake(self, init_info: ShardedRDTWeightTransferInitInfo, *args: Any) -> None:
        from vllm.model_executor.model_loader.reload.meta import SKIP_LOAD_TENSORS, SKIP_TENSORS

        # The EP dispatcher owns this runtime buffer. It has no checkpoint entry.
        SKIP_LOAD_TENSORS.add("expert_ids_per_ep_rank")
        SKIP_TENSORS.add("expert_ids_per_ep_rank")
        super()._bake(init_info, *args)

    def _resolve_producers(self, init_info: ShardedRDTWeightTransferInitInfo, *args: Any) -> None:
        super()._resolve_producers(init_info, *args)
        state = args[0] if args else self
        assert state._router is not None and self._connection is not None
        for actor in state._router._actors:
            # HIXL returns a descriptor through the ordinary Ray control plane.
            actor._ray_enable_tensor_transport = False
            self._connection.attach(actor)

    def _alloc_receive_buffer(self, nbytes: int, **kwargs: Any) -> torch.Tensor:
        assert self._connection is not None
        return self._connection.session.allocate(nbytes)

    def _preregister_at_init(self, *args: Any) -> None:
        if args:
            # A new endpoint owns its own registrations, including on re-init.
            state = args[0]
            state._dest_buffers = [{} for _ in state._dest_buffers]
        elif (plan := self._cached_plan) is not None and plan.chunks:
            # v0.34 has no allocation hook. Seed the slots before delegating the
            # sizing checks, per-owner digests and producer reservations.
            nbytes = buffer_alloc_bytes(max(chunk.pack_bytes for chunk in plan.chunks), self._buffer_presize)
            for slot in self._dest_buffers:
                slot[torch.uint8] = self._alloc_receive_buffer(nbytes)
        super()._preregister_at_init(*args)

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
        assert self._connection is not None
        session = self._connection.session
        request = session.read(endpoint, pending.blob, address, nbytes)
        session.wait(request)
        return dict(zip(pending.keys, pending.targets))

    def _ensure_proc_worker(self, *args: Any) -> None:
        from vllm_ascend.worker.v2.utils import torch_cuda_wrapper

        # The shared wrapper installs process-wide mappings before either
        # background thread starts. Both replay methods can be inherited.
        with torch_cuda_wrapper():
            super()._ensure_proc_worker(*args)

    def shutdown(self) -> None:
        # Propagate drain failures and retain live storage instead of dropping
        # buffers underneath outstanding device work.
        self.drain_pending()
        for thread, work_queue in (
            (self._proc_thread, self._proc_queue),
            (self._quant_thread, self._quant_queue),
        ):
            if thread is not None:
                assert work_queue is not None
                work_queue.put(None)
                thread.join(timeout=30)
                if thread.is_alive():
                    raise TimeoutError("NPU weight processing thread did not stop")
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._close_retired_connections()
        self._proc_thread = self._quant_thread = None
        self._proc_queue = self._quant_queue = None
        self._proc_stream = self._quant_stream = None
        self._slot_read_done = []
        self._router = None
        self._name_to_plan.clear()
        self._name_meta.clear()
        self._live_names.clear()
        self._cached_plan = None
        self._targets_cache.clear()
        self._dest_buffers = [{} for _ in self._dest_buffers]
