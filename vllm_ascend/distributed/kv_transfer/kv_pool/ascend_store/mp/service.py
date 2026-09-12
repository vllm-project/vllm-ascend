"""Child-local KV transfer service and device-resource ownership."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast

import msgspec

from ..mooncake_session_tracker import MooncakeSessionTracker

if TYPE_CHECKING:
    from ..kv_transfer import KVCacheStoreRecvingThread, KVTransferThread
    from ..metadata import ChunkedTokenDatabase, LayerTransferTask
    from .npu_ipc import ImportedKVCache


def _restore_layer_tasks(rows: list[dict[str, Any]]) -> list[LayerTransferTask]:
    """Rebuild domain tasks from their wire-safe snapshots."""
    import numpy as np

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
        LayerBlockRange,
        LayerTransferTask,
        ReqMeta,
        SharedBlockData,
    )

    tasks = []
    for row in rows:
        shared_row = row["shared"]
        shared = None
        if shared_row is not None:
            shared = SharedBlockData(
                block_ids_arr=np.asarray(shared_row["block_ids"], dtype=np.int64),
                block_gvas_arr=(
                    None if shared_row["block_gvas"] is None else np.asarray(shared_row["block_gvas"], dtype=np.int64)
                ),
                req_ids=shared_row["req_ids"],
                is_last_chunks=shared_row["is_last_chunks"],
                block_keys=shared_row["block_keys"],
                save_keys=shared_row["save_keys"],
                load_keys=shared_row["load_keys"],
            )
        block_ranges = [
            LayerBlockRange(
                request=ReqMeta(**block_range["request"]),
                start_block=block_range["start_block"],
                end_block=block_range["end_block"],
                partial_block_index=block_range["partial_block_index"],
            )
            for block_range in row["block_ranges"]
        ]
        tasks.append(
            LayerTransferTask(
                layer_id=row["layer_id"],
                block_ranges=block_ranges,
                shared_block_data=shared,
                group_id=row["group_id"],
                layer_idx_in_group=row["layer_idx_in_group"],
                write_finish_keys=row["write_finish_keys"],
                use_key_major_ranges=row["use_key_major_ranges"],
            )
        )
    return tasks


class _ProcessMooncakeSessionTracker(MooncakeSessionTracker):
    """Report child-side put outcomes to the Worker-owned session tracker."""

    def __init__(self) -> None:
        super().__init__()
        self.committed: list[str] = []
        self.revoked: list[str] = []

    def reset(self) -> None:
        self.committed.clear()
        self.revoked.clear()

    def commit_put_keys(self, keys: Iterable[str]) -> None:
        self.committed.extend(keys)

    def revoke_put_keys(self, keys: Iterable[str]) -> None:
        self.revoked.extend(keys)


class TransferService:
    """Own child-local transfer handlers, backend and imported NPU resources.

    Socket transport and process lifetime remain owned by ``TransferServer``.
    """

    def __init__(self, config: dict[str, Any]):
        # Import device code only inside the fresh child, never at entry-point
        # discovery time (CPU lifecycle tests provide their own runtime).
        import torch

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer_backend import (
            create_transfer_backend,
        )

        self.config = config
        self.device_index = config["device_index"]
        torch.npu.set_device(self.device_index)
        self.backend = create_transfer_backend(
            config["backend"], self.device_index, config["global_rank"], config.get("lazy_init", False)
        )
        self.cache: ImportedKVCache | None = None
        self.sender: KVTransferThread | None = None
        self.receiver: KVTransferThread | None = None
        self._layerwise = False
        self._operation_stats = threading.local()
        self._mooncake_put_outcomes = _ProcessMooncakeSessionTracker()
        self._send = ThreadPoolExecutor(max_workers=1, initializer=self.backend.set_device)
        self._recv = ThreadPoolExecutor(max_workers=1, initializer=self.backend.set_device)

    # TransferServer serializes these control operations on its control lane.
    def execute(self, operation: str, payload: Any) -> Any:
        self.backend.set_device()
        if operation == "register":
            return self.register(payload)
        if operation == "exists":
            return self.backend.exists(payload)
        if operation == "batch_is_exist":
            return self.backend.batch_is_exist(payload)
        if operation == "batch_get_key_info":
            return [
                (int(info.size()), [int(gva) for gva in info.gva_list()])
                for info in self.backend.batch_get_key_info(payload)
            ]
        if operation == "batch_alloc":
            keys, sizes, lease_ttl_ms = payload
            return self.backend.batch_alloc(keys, sizes, lease_ttl_ms)
        if operation == "batch_add_lease":
            keys, lease_ttl_ms = payload
            return self.backend.batch_add_lease(keys, lease_ttl_ms)
        if operation == "batch_remove_lease":
            return self.backend.batch_remove_lease(payload)
        if operation == "batch_put_start":
            keys, sizes = payload
            return self.backend.batch_put_start(keys, sizes)
        if operation == "batch_get_start":
            return self.backend.batch_get_start(payload)
        if operation == "batch_get_end":
            return self.backend.batch_get_end(payload)
        if operation == "ensure_ready":
            ensure = getattr(self.backend, "ensure_initialized", None)
            if ensure is not None:
                ensure()
            return None
        if operation == "validate_layerwise_support":
            self.backend.validate_layerwise_support()
            return None
        if operation == "get_ranges":
            if self.cache is None:
                raise RuntimeError("KV caches are not registered")
            keys, ranges = payload
            addresses, sizes = [], []
            for row in ranges:
                addresses.append([self.cache.resolve_range(*item) for item in row])
                sizes.append([item[2] for item in row])
            return self.backend.get(keys, addresses, sizes)
        raise ValueError(f"Unknown KV transfer operation: {operation}")

    # Registration restores child-owned state before data-plane calls arrive.
    def register(self, payload: dict[str, Any]) -> None:
        database, addresses = self._restore_registration(payload)
        layerwise = payload.get("layerwise")
        if layerwise is not None:
            self._create_layerwise_handlers(payload, database, layerwise)
        else:
            self._create_block_handlers(payload, database, addresses)

    def _restore_registration(self, payload: dict[str, Any]) -> tuple[ChunkedTokenDatabase, dict[int, list[int]]]:
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase, KeyMetadata
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.npu_ipc import (
            WorkerKVCacheSpec,
            import_worker_kv_caches,
        )

        if self.cache is not None:
            raise RuntimeError("KV caches are already registered")
        cache = import_worker_kv_caches(
            msgspec.convert(payload["cache"], WorkerKVCacheSpec), device_index=self.device_index
        )
        self.cache = cache
        if cache.device_index != self.device_index:
            raise ValueError("Imported KV caches do not match the transfer backend device")
        addresses = {
            int(group): [cache.resolve_range(*item) for item in ranges]
            for group, ranges in payload["group_ranges"].items()
        }
        database = ChunkedTokenDatabase(
            [KeyMetadata(**item) for item in payload["metadata"]],
            payload["block_sizes"],
            payload["partitions"],
            hash_block_size=payload["hash_block_size"],
        )
        database.set_group_buffers(
            addresses,
            payload["block_lengths"],
            payload["block_strides"],
            group_cache_families=payload["families"],
            group_num_layers=payload["num_layers"],
            group_layer_cache_entry_offsets=payload["entry_offsets"],
        )
        ranges = payload["registered_ranges"]
        self.backend.register_buffer([cache.resolve_range(*item) for item in ranges], [item[2] for item in ranges])
        return database, addresses

    def _create_layerwise_handlers(
        self,
        payload: dict[str, Any],
        database: ChunkedTokenDatabase,
        layerwise: dict[str, Any],
    ) -> None:
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
            KVCacheStoreKeyLayerRecvingThread,
            KVCacheStoreKeyLayerSendingThread,
            KVCacheStoreLayerRecvingThread,
            KVCacheStoreLayerSendingThread,
            LayerBatchBuilder,
        )

        self._layerwise = True
        num_layers = layerwise["num_layers"]
        ready_event = threading.Event()
        get_event = threading.Event()
        load_finished = [threading.Event() for _ in range(num_layers)]
        save_finished = [threading.Event() for _ in range(num_layers)]
        sync_save_events: list[Any] = [None for _ in range(num_layers)]
        common = dict(
            m_store=self.backend,
            token_database=database,
            block_size=payload["block_sizes"] if layerwise["use_gva"] else layerwise["block_size"],
            tp_rank=self.config["tp_rank"],
            tp_size=self.config["tp_size"],
            dcp_size=self.config["dcp_size"],
        )
        can_save = self.config["kv_role"] in ("kv_producer", "kv_both") or layerwise["consumer_is_to_put"]
        use_range_handlers = layerwise["use_gva"] or layerwise["use_key_major_ranges"]
        if use_range_handlers:
            group_builders = [
                LayerBatchBuilder(
                    database,
                    sum(payload["block_lengths"][group_id]),
                    payload["num_layers"][group_id],
                    group_id=group_id,
                )
                for group_id in sorted(payload["block_lengths"])
            ]
            if can_save:
                self.sender = KVCacheStoreLayerSendingThread(
                    **common,
                    page_size_bytes=layerwise["page_size_bytes"],
                    ready_event=ready_event,
                    layer_save_finished_events=save_finished,
                    sync_save_events=sync_save_events,
                    num_layers=num_layers,
                    max_transfer_blocks=layerwise["max_transfer_blocks"],
                    max_transfer_bytes=layerwise["max_transfer_bytes"],
                    group_builders=group_builders,
                    session_tracker=(self._mooncake_put_outcomes if layerwise["use_key_major_ranges"] else None),
                )
            self.receiver = KVCacheStoreLayerRecvingThread(
                **common,
                page_size_bytes=layerwise["page_size_bytes"],
                ready_event=ready_event,
                get_event=get_event,
                layer_load_finished_events=load_finished,
                layer_save_finished_events=save_finished,
                sync_save_events=sync_save_events,
                num_layers=num_layers,
                # The parent adapter owns the attention/slot-release sequence,
                # so it applies this delay immediately before releasing the slot.
                h2d_stagger_us=0,
                max_transfer_blocks=layerwise["max_transfer_blocks"],
                max_transfer_bytes=layerwise["max_transfer_bytes"],
                group_builders=group_builders,
                save_failure_checker=self.sender.raise_if_failed if self.sender is not None else None,
            )
        else:
            if can_save:
                self.sender = KVCacheStoreKeyLayerSendingThread(
                    **common,
                    put_step=self.config["put_step"],
                    ready_event=ready_event,
                    layer_save_finished_events=save_finished,
                    sync_save_events=sync_save_events,
                    num_layers=num_layers,
                )
            self.receiver = KVCacheStoreKeyLayerRecvingThread(
                **common,
                ready_event=ready_event,
                get_event=get_event,
                layer_load_finished_events=load_finished,
                layer_save_finished_events=save_finished,
                num_layers=num_layers,
            )

    def _create_block_handlers(
        self,
        payload: dict[str, Any],
        database: ChunkedTokenDatabase,
        addresses: dict[int, list[int]],
    ) -> None:
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
            KVCacheStoreRecvingThread,
            KVCacheStoreSendingThread,
        )

        transfer_worker = None
        tp_mismatch = payload["tp_mismatch"]
        if tp_mismatch is not None:
            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

            # The existing transfer handlers delegate this mode to KVPoolWorker.
            # Rebuild only the state read by those methods instead of copying
            # their key, address, and backend logic into the process layer.
            transfer_worker = KVPoolWorker.__new__(KVPoolWorker)
            transfer_worker.tp_mismatch = True
            transfer_worker.m_store = self.backend
            transfer_worker.token_database = database
            transfer_worker.group_kv_caches_base_addr = addresses
            transfer_worker.group_block_len = payload["block_lengths"]
            transfer_worker.group_block_stride = payload["block_strides"]
            transfer_worker.block_size = tp_mismatch["block_size"]
            transfer_worker.num_sub_keys = tp_mismatch["num_sub_keys"]
            transfer_worker.sub_size_bytes = tp_mismatch["sub_size_bytes"]
            transfer_worker.tp_rank = self.config["tp_rank"]
            transfer_worker.enable_kv_events = self.config["enable_kv_events"]
            transfer_worker._record_kv_connector_operation = (  # type: ignore[method-assign]
                self._record_kv_connector_operation
            )
        common = dict(
            m_store=self.backend,
            token_database=database,
            block_size=payload["block_sizes"],
            tp_rank=self.config["tp_rank"],
            tp_size=self.config["tp_size"],
            dcp_size=self.config["dcp_size"],
        )
        self.sender = KVCacheStoreSendingThread(
            **common,
            put_step=self.config["put_step"],
            kv_role=self.config["kv_role"],
            group_uses_align_state=payload["align_state"],
            enable_kv_event=self.config["enable_kv_events"],
            worker=transfer_worker,
        )
        self.receiver = KVCacheStoreRecvingThread(
            **common,
            worker=transfer_worker,
            record_operation=self._record_kv_connector_operation,
        )
        if transfer_worker is not None:
            transfer_worker.kv_send_thread = self.sender
            transfer_worker._invalid_block_ids = self.receiver._invalid_block_ids
            transfer_worker._invalid_block_ids_lock = self.receiver._invalid_block_ids_lock

    # Independent single-worker lanes preserve send and receive ordering.
    def submit(self, operation: str, payload: Any) -> Future:
        if self.cache is None:
            raise RuntimeError("KV caches are not registered")
        worker = self.sender if operation == "store" else self.receiver
        if worker is None:
            raise RuntimeError(f"KV transfer process cannot {operation} with the configured role")
        executor = self._send if operation == "store" else self._recv
        handler = self._layer_transfer if self._layerwise else self._transfer
        return executor.submit(handler, operation, payload)

    def _transfer(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LoadSpec, ReqMeta

        self._operation_stats.entries = []
        if payload["load_spec"] is not None:
            payload["load_spec"] = LoadSpec(**payload["load_spec"])
        request = ReqMeta(**payload)
        worker = self.sender if operation == "store" else self.receiver
        assert worker is not None
        if operation == "store":
            worker.add_stored_request(request.req_id)
        self._run_handler(worker, request)
        finished = worker.get_and_clear_finished_requests()
        result = {
            "finished": request.req_id in finished,
            "events": worker.get_kv_events(),
            "invalid_blocks": [],
            "operations": self._operation_stats.entries,
        }
        if operation == "load":
            assert self.receiver is not None
            receiver = cast("KVCacheStoreRecvingThread", self.receiver)
            with receiver._invalid_block_ids_lock:
                result["invalid_blocks"] = list(receiver._invalid_block_ids)
                receiver._invalid_block_ids.clear()
        return result

    def _record_kv_connector_operation(self, operation: str, duration_seconds: float, num_keys: int) -> None:
        self._operation_stats.entries.append((operation, duration_seconds, num_keys))

    def _layer_transfer(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LayerLoadTask

        layer_id = payload["layer_id"]
        tasks = _restore_layer_tasks(payload["tasks"])
        request_ids = []
        for task in tasks:
            if task.shared_block_data is not None:
                request_ids.extend(task.shared_block_data.req_ids)
            else:
                request_ids.extend(block_range.request.req_id for block_range in task.block_ranges)

        worker: Any = self.sender if operation == "store" else self.receiver
        assert worker is not None
        if operation == "store":
            self._mooncake_put_outcomes.reset()
            worker.layer_save_finished_events[layer_id].clear()
            for req_id in request_ids:
                worker.add_stored_request(req_id)
            self._run_handler(worker, tasks)
        else:
            if layer_id == 0 and hasattr(worker, "_load_abort_event"):
                worker._load_abort_event.clear()
            worker.layer_load_finished_events[layer_id].clear()
            self._run_handler(worker, LayerLoadTask(None, tasks, layer_id))

        finished = worker.get_and_clear_finished_requests()
        result: dict[str, Any] = {
            "completed_req_ids": request_ids,
            "finished_req_ids": list(finished),
            "events": worker.get_kv_events(),
        }
        if operation == "store":
            result["committed_keys"] = list(self._mooncake_put_outcomes.committed)
            result["revoked_keys"] = list(self._mooncake_put_outcomes.revoked)
        elif hasattr(worker, "_load_abort_event"):
            receiver = worker
            with receiver._invalid_block_ids_lock:
                result["invalid_blocks"] = list(receiver._invalid_block_ids)
                receiver._invalid_block_ids.clear()
            result["load_aborted"] = receiver._load_abort_event.is_set()
        else:
            result["invalid_blocks"] = []
            result["load_aborted"] = False
        return result

    @staticmethod
    def _run_handler(worker: KVTransferThread, request: Any) -> None:
        # Child handlers are not started as threads, but _handle_request()
        # still calls task_done(). Consume one queued item synchronously so
        # its task counter matches the in-process thread loop.
        worker.request_queue.put(request)
        queued_request = worker.request_queue.get_nowait()
        worker._handle_request(queued_request)

    def close(self) -> None:
        """Drain transfer lanes before releasing backend and NPU resources."""
        self._send.shutdown(wait=True)
        self._recv.shutdown(wait=True)
        self.backend.set_device()
        # Unregister before releasing imported allocations. If cleanup fails,
        # keep those references until the interpreter exits.
        self.backend.close()
        if self.cache is not None:
            self.cache.close()
