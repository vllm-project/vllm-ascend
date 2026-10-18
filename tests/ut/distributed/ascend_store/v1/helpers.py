"""Small fixtures expressed in the public v1 execution vocabulary."""

from __future__ import annotations

import threading
from typing import Any

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import KeyMetadata
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    BackendSpec,
    BufferRegistration,
    LayerwiseAccessKind,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.mooncake.sessions import MooncakeSessions
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.results import (
    integer_results,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.reachability import (
    HybridReachability,
    UnitaryReachability,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.resources import KVPoolResources
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    KVTransferStep,
    LoadCommandBatch,
    StoreCommand,
    StoreCommandBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.topology import (
    KVPoolGroupTopology,
    KVPoolLayerTopology,
    KVPoolTopology,
    TPPartitionSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk import (
    BulkWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise import (
    GVALayerwiseWorker,
    KeyRangeLayerwiseWorker,
)


class FakeBackend:
    requires_exists_before_put = False

    def __init__(self) -> None:
        self.presence: list[int] | None = None
        self.presence_results: list[list[int]] = []
        self.get_result: object = None
        self.put_result: object = None
        self.session_start_result: list[int] | None = None
        self.session_copy_result: object = None
        self.session_end_result = 0
        self.store_session_start_result: list[int] | None = None
        self.store_session_copy_result: object = None
        self.store_session_commit_result: list[int] | None = None
        self.store_session_revoke_result: list[int] | None = None
        self.calls: list[tuple[Any, ...]] = []
        self.closed = False

    def create_sessions(self) -> MooncakeSessions:
        return MooncakeSessions(self)

    def exists(self, keys):
        self.calls.append(("exists", tuple(keys)))
        if self.presence_results:
            return self.presence_results.pop(0)
        return self.presence if self.presence is not None else [1] * len(keys)

    def get(self, keys, addresses, sizes):
        self.calls.append(("get", tuple(keys), tuple(map(tuple, addresses)), tuple(map(tuple, sizes))))
        return self.get_result if self.get_result is not None else [0] * len(keys)

    def put(self, keys, addresses, sizes):
        self.calls.append(("put", tuple(keys), tuple(map(tuple, addresses)), tuple(map(tuple, sizes))))
        if isinstance(self.put_result, BaseException):
            raise self.put_result
        return self.put_result if self.put_result is not None else [0] * len(keys)

    def load(self, keys, addresses, sizes):
        return integer_results("fake load", keys, self.get(keys, addresses, sizes))

    def store(self, keys, addresses, sizes):
        return integer_results("fake store", keys, self.put(keys, addresses, sizes))

    def set_device(self) -> None:
        self.calls.append(("set_device",))

    def register_buffer(self, addresses, sizes) -> BufferRegistration:
        self.calls.append(("register_buffer", tuple(addresses), tuple(sizes)))
        registration = BufferRegistration(self._unregister_buffer)
        for address, size in zip(addresses, sizes, strict=True):
            registration.acquire(address, size)
        return registration

    def _unregister_buffer(self, address: int, size: int) -> None:
        self.calls.append(("unregister_buffer", address, size))

    def validate_layerwise_support(self) -> None:
        self.calls.append(("validate_layerwise_support",))

    def validate_key_range_support(self) -> None:
        self.validate_layerwise_support()

    def validate_gva_support(self) -> None:
        self.validate_layerwise_support()

    def batch_get_start(self, keys):
        self.calls.append(("batch_get_start", tuple(keys)))
        return self.session_start_result if self.session_start_result is not None else [0] * len(keys)

    def start_key_range_load(self, keys):
        return integer_results("batch_get_start", keys, self.batch_get_start(keys))

    def batch_copy_get(self, keys, addresses, sizes, offsets):
        self.calls.append(
            (
                "batch_copy_get",
                tuple(keys),
                tuple(map(tuple, addresses)),
                tuple(map(tuple, sizes)),
                tuple(map(tuple, offsets)),
            )
        )
        if isinstance(self.session_copy_result, BaseException):
            raise self.session_copy_result
        return self.session_copy_result if self.session_copy_result is not None else [0] * len(keys)

    def copy_key_range_load(self, keys, addresses, sizes, offsets):
        return integer_results("batch_copy_get", keys, self.batch_copy_get(keys, addresses, sizes, offsets))

    def batch_get_end(self, keys):
        self.calls.append(("batch_get_end", tuple(keys)))
        return self.session_end_result

    def finish_key_range_load(self, keys):
        result = self.batch_get_end(keys)
        if result != 0:
            raise RuntimeError(f"batch_get_end failed with result code {result}")

    def batch_put_start(self, keys, object_sizes):
        self.calls.append(("batch_put_start", tuple(keys), tuple(object_sizes)))
        return self.store_session_start_result if self.store_session_start_result is not None else [0] * len(keys)

    def start_key_range_store(self, keys, object_sizes):
        return integer_results("batch_put_start", keys, self.batch_put_start(keys, object_sizes))

    def batch_copy_put(self, keys, addresses, sizes, offsets):
        self.calls.append(
            (
                "batch_copy_put",
                tuple(keys),
                tuple(map(tuple, addresses)),
                tuple(map(tuple, sizes)),
                tuple(map(tuple, offsets)),
            )
        )
        if isinstance(self.store_session_copy_result, BaseException):
            raise self.store_session_copy_result
        return self.store_session_copy_result if self.store_session_copy_result is not None else [0] * len(keys)

    def copy_key_range_store(self, keys, addresses, sizes, offsets):
        return integer_results("batch_copy_put", keys, self.batch_copy_put(keys, addresses, sizes, offsets))

    def batch_commit(self, keys):
        self.calls.append(("batch_commit", tuple(keys)))
        return self.store_session_commit_result if self.store_session_commit_result is not None else [0] * len(keys)

    def commit_key_range_store(self, keys):
        return integer_results("batch_commit", keys, self.batch_commit(keys))

    def batch_revoke(self, keys):
        self.calls.append(("batch_revoke", tuple(keys)))
        return self.store_session_revoke_result if self.store_session_revoke_result is not None else [0] * len(keys)

    def revoke_key_range_store(self, keys):
        return integer_results("batch_revoke", keys, self.batch_revoke(keys))

    def close(self) -> None:
        self.calls.append(("backend_close",))
        self.closed = True


def make_backend_spec(
    *,
    name: str = "fake",
    layerwise_access: LayerwiseAccessKind | None = LayerwiseAccessKind.KEY_RANGE,
    requires_exists_before_put: bool = False,
) -> BackendSpec:
    return BackendSpec(
        name,
        FakeBackend,
        layerwise_access,
        requires_exists_before_put,
    )


def make_reachability(
    topology: KVPoolTopology, max_model_len: int = 64, *, retention_interval: int | None = None
) -> UnitaryReachability | HybridReachability:
    groups = topology.transfer_groups
    if len(groups) == 1 and not groups[0].uses_align_state:
        return UnitaryReachability(groups[0].group_id, max_model_len, topology.cache_transfer_granularity)
    return HybridReachability(
        groups,
        topology.cache_transfer_granularity,
        topology.hash_block_size,
        max_model_len,
        retention_interval=retention_interval,
    )


def make_topology(
    *,
    group_ids: tuple[int, ...] = (0,),
    physical_layers: tuple[int, ...] = (0, 1),
    tp_mismatch: bool = False,
    tp_rank: int = 0,
    tp_size: int = 1,
    pp_size: int = 1,
    pp_rank: int = 0,
    pcp_rank: int = 0,
    pcp_size: int = 1,
    dcp_rank: int = 0,
    dcp_size: int = 1,
    put_step: int = 1,
    head_or_tp_rank: int | None = None,
    key_rank_count: int | None = None,
    key_slices_per_rank: int | None = None,
    consumer_pipeline_partitions: tuple[int, ...] | None = None,
) -> KVPoolTopology:
    head_or_tp_rank = (tp_rank if tp_mismatch else 0) if head_or_tp_rank is None else head_or_tp_rank
    groups = tuple(
        KVPoolGroupTopology(
            group_id,
            FullAttentionSpec(
                block_size=4,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
            tuple(
                KVPoolLayerTopology(layer_id, (f"layers.{layer_id}.group.{group_id}",)) for layer_id in physical_layers
            ),
            KeyMetadata("model", head_or_tp_rank, dcp_rank, pp_rank, group_id),
        )
        for group_id in range(max(group_ids) + 1)
    )
    return KVPoolTopology(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_size=pp_size,
        pcp_rank=pcp_rank,
        pcp_size=pcp_size,
        dcp_size=dcp_size,
        put_step=put_step,
        cache_transfer_granularity=4,
        hash_block_size=4,
        tp_partition=TPPartitionSpec(
            tp_mismatch,
            key_rank_count if key_rank_count is not None else (2 if tp_mismatch else 1),
            key_slices_per_rank if key_slices_per_rank is not None else (2 if tp_mismatch else 1),
        ),
        groups=groups,
        transfer_group_ids=group_ids,
        consumer_pipeline_partitions=consumer_pipeline_partitions,
    )


class FakeEvent:
    def __init__(self) -> None:
        self.recorded = False
        self.synchronized = False

    def record(self) -> None:
        self.recorded = True

    def synchronize(self) -> None:
        self.synchronized = True


class FakeResources(KVPoolResources):
    def register_kv_buffers(self):
        pass

    def __init__(
        self,
        backend: FakeBackend,
        backend_spec: BackendSpec,
        topology: KVPoolTopology,
    ) -> None:
        super().__init__(backend, backend_spec, 8, topology.transfer_groups)
        self._topology = topology
        self.closed = False

    def prepare_kv_caches(self, kv_caches):
        # This fixture publishes synthetic addresses without backing tensor storage.
        self.kv_caches = (
            {
                name: torch.empty_strided((8, 4, 2, 1), (16, 2, 1, 1), dtype=torch.float32)
                for group in self._topology.transfer_groups
                for name in group.layer_names
            }
            if self._topology.tp_partition.tp_mismatch
            else None
        )
        base_addresses = {}
        block_lengths = {}
        block_strides = {}
        layer_entry_offsets = {}
        object_sizes = {}
        object_offsets = {}
        for group in self._topology.transfer_groups:
            group_id = group.group_id
            base_addresses[group_id] = [
                1000 + group_id * 10000 + layer.physical_layer_id * 1000 for layer in group.layers
            ]
            block_lengths[group_id] = [32] * len(group.layers)
            block_strides[group_id] = [64] * len(group.layers)
            layer_entry_offsets[group_id] = list(range(len(group.layers) + 1))
            object_sizes[group_id] = 32 * len(group.layers)
            object_offsets[group_id] = 0
        registration: dict[str, Any] = {
            "base_addresses": base_addresses,
            "block_lengths": block_lengths,
            "block_strides": block_strides,
            "layer_entry_offsets": layer_entry_offsets,
        }
        return registration

    def close(self) -> None:
        if self._backend_sessions is not None:
            self._backend_sessions.close()
        self.closed = True
        self.kv_caches = None


def make_worker(
    backend: FakeBackend | None = None,
    *,
    topology: KVPoolTopology | None = None,
    layerwise: bool = False,
    async_load: bool = False,
    store: bool = True,
    physical_layers: tuple[int, ...] = (0, 1),
    backend_name: str = "fake",
    requires_exists_before_put: bool = False,
    source_ready_event_factory=None,
    start_gate_factory=None,
    bind: bool = True,
):
    backend = backend or FakeBackend()
    topology = topology or make_topology(physical_layers=physical_layers)
    backend_spec = make_backend_spec(
        name=backend_name,
        layerwise_access=LayerwiseAccessKind.KEY_RANGE if layerwise else None,
        requires_exists_before_put=requires_exists_before_put,
    )
    resources = FakeResources(backend, backend_spec, topology)
    if layerwise:
        worker_type = (
            GVALayerwiseWorker if backend_spec.layerwise_access is LayerwiseAccessKind.GVA else KeyRangeLayerwiseWorker
        )
        worker = worker_type(
            topology,
            resources,
            reachability=make_reachability(topology),
            store_enabled=store,
            **({"start_gate_factory": start_gate_factory} if start_gate_factory is not None else {}),
            source_ready_event_factory=source_ready_event_factory or FakeEvent,
        )
    else:
        worker = BulkWorker(
            topology,
            resources,
            reachability=make_reachability(topology),
            store_enabled=store,
            load_async=async_load,
            source_ready_event_factory=source_ready_event_factory or FakeEvent,
        )
    if bind:
        worker.bind_kv_caches({"cache": object()})
    return worker, resources, backend


def begin_step(worker, *, load=(), store=()) -> None:
    worker.begin_step(
        KVTransferStep(
            LoadCommandBatch(tuple(load)),
            StoreCommandBatch(tuple(store)),
        )
    )


def store_one(worker, command: StoreCommand):
    begin_step(worker, store=(command,))
    worker.finish_step()
    (completion,) = worker.fence_previous_store()
    worker.end_step()
    return completion


class FakeLoadStartGate:
    def __init__(self, opened: bool = False) -> None:
        self._opened = threading.Event()
        if opened:
            self._opened.set()

    def open(self) -> None:
        self._opened.set()

    def cancel(self) -> None:
        self._opened.set()

    def wait(self, timeout) -> bool:
        return self._opened.wait(timeout)
