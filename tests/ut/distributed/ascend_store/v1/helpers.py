"""Small fixtures expressed in the public v1 execution vocabulary."""

from __future__ import annotations

from typing import Any

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import KeyMetadata
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    BackendSpec,
    BufferRegistration,
    LayerwiseAccessKind,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection import (
    BulkProjectionBinder,
    GVALayerwiseProjectionBinder,
    KeyRangeLayerwiseProjectionBinder,
    LayerwiseProjectionBinder,
    compile_bulk_projection_binder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.route import KVPoolRouteSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.topology import (
    KVPoolGroupTopology,
    KVPoolLayerTopology,
    KVPoolTopology,
    TPPartitionSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import (
    AsynchronousBulkWorker,
    GVALayerwiseWorker,
    KeyRangeLayerwiseWorker,
    SynchronousBulkWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.state import (
    select_store_candidate_objects as _select_store_candidate_objects,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.state import (
    store_candidate_keys as _store_candidate_keys,
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
        return self.get(keys, addresses, sizes)

    def store(self, keys, addresses, sizes):
        return self.put(keys, addresses, sizes)

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
        return tuple(self.batch_get_start(keys))

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
        return tuple(self.batch_copy_get(keys, addresses, sizes, offsets))

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
        return tuple(self.batch_put_start(keys, object_sizes))

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
        return tuple(self.batch_copy_put(keys, addresses, sizes, offsets))

    def batch_commit(self, keys):
        self.calls.append(("batch_commit", tuple(keys)))
        return self.store_session_commit_result if self.store_session_commit_result is not None else [0] * len(keys)

    def commit_key_range_store(self, keys):
        return tuple(self.batch_commit(keys))

    def batch_revoke(self, keys):
        self.calls.append(("batch_revoke", tuple(keys)))
        return self.store_session_revoke_result if self.store_session_revoke_result is not None else [0] * len(keys)

    def revoke_key_range_store(self, keys):
        return tuple(self.batch_revoke(keys))

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
    head_or_tp_rank: int = 0,
    key_rank_count: int | None = None,
    key_slices_per_rank: int | None = None,
    consumer_pipeline_partitions: tuple[int, ...] | None = None,
) -> KVPoolTopology:
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


class FakeResources:
    def __init__(
        self,
        backend: FakeBackend,
        backend_spec: BackendSpec,
        topology: KVPoolTopology,
    ) -> None:
        self.backend = backend
        self.backend_spec = backend_spec
        self._topology = topology
        self.num_blocks = 8
        self.kv_caches = None
        self.closed = False

    def bind_kv_caches(self, kv_caches):
        self.kv_caches = kv_caches
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
        registration = {
            "base_addresses": base_addresses,
            "block_lengths": block_lengths,
            "block_strides": block_strides,
            "layer_entry_offsets": layer_entry_offsets,
        }
        if self.backend_spec.layerwise_access is LayerwiseAccessKind.GVA:
            registration["object_sizes"] = object_sizes
            registration["object_offsets"] = object_offsets
        return registration

    def close(self) -> None:
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
    route_spec = KVPoolRouteSpec(topology, "fake", 64, use_layerwise=layerwise)
    full_key = lambda group, value, head, stage: f"g{group}:p{stage}:h{head}:{value}"
    projection_binder: LayerwiseProjectionBinder | BulkProjectionBinder
    if not layerwise:
        projection_binder = compile_bulk_projection_binder(topology, route_spec.max_model_len)
    else:
        assert backend_spec.layerwise_access is not None
        binder_type = (
            GVALayerwiseProjectionBinder
            if backend_spec.layerwise_access is LayerwiseAccessKind.GVA
            else KeyRangeLayerwiseProjectionBinder
        )
        projection_binder = binder_type(topology, route_spec.max_model_len, full_key)
    if isinstance(projection_binder, GVALayerwiseProjectionBinder):
        worker = GVALayerwiseWorker(
            topology,
            projection_binder,
            resources,  # type: ignore[arg-type]
            store_enabled=store,
            **({"start_gate_factory": start_gate_factory} if start_gate_factory is not None else {}),
            source_ready_event_factory=source_ready_event_factory or FakeEvent,
        )
    elif isinstance(projection_binder, KeyRangeLayerwiseProjectionBinder):
        worker = KeyRangeLayerwiseWorker(
            topology,
            projection_binder,
            resources,  # type: ignore[arg-type]
            store_enabled=store,
            **({"start_gate_factory": start_gate_factory} if start_gate_factory is not None else {}),
            source_ready_event_factory=source_ready_event_factory or FakeEvent,
        )
    else:
        worker_type = AsynchronousBulkWorker if async_load else SynchronousBulkWorker
        worker = worker_type(
            topology,
            projection_binder,
            resources,  # type: ignore[arg-type]
            store_enabled=store,
            source_ready_event_factory=source_ready_event_factory or FakeEvent,
        )
    if bind:
        worker.bind_kv_caches({"cache": object()})
    return worker, resources, backend


def build_admitted_store_batch(worker, commands, *, layerwise: bool = False):
    """Keep pre-Slice-4 lowering tests out of Worker's public mainline."""

    candidates = worker._build_store_candidates(commands)
    candidate_keys = _store_candidate_keys(candidates)
    accepted, claim_once = worker._admitted_store_keys(candidate_keys)
    if accepted is not None and not accepted:
        return None
    selected_objects = (
        None if accepted is None else _select_store_candidate_objects(candidate_keys, accepted, claim_once=claim_once)
    )
    return worker._materialize_store_candidates(
        commands,
        candidates,
        selected_objects,
        prepare_layerwise=layerwise,
    )


def worker_backend_io(worker):
    """Return the concrete Backend boundary for lower-level migration guards."""

    return worker._backend_io
