"""Registered CPU buffers and failure-injected resource ownership."""

from __future__ import annotations

import torch

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    make_backend_spec,
    make_topology,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    BufferRegistration,
    rollback_buffer_registration,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection import (
    compile_bulk_projection_binder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk import (
    SynchronousBulkWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.resources import (
    KVPoolResources,
)


class RecordingBackend(FakeBackend):
    def __init__(
        self,
        *,
        fail_registration_index: int | None = None,
        fail_unregister_count: int = 0,
        fail_close_count: int = 0,
    ) -> None:
        super().__init__()
        self.fail_registration_index = fail_registration_index
        self.fail_unregister_count = fail_unregister_count
        self.fail_close_count = fail_close_count

    def register_buffer(self, addresses, sizes) -> BufferRegistration:
        registration = BufferRegistration(self._unregister_buffer)
        try:
            for index, (address, size) in enumerate(zip(addresses, sizes, strict=True)):
                self.calls.append(("register_region", address, size))
                if index == self.fail_registration_index:
                    raise RuntimeError(f"registration failed at region {index}")
                registration.acquire(address, size)
        except BaseException as error:
            rollback_buffer_registration(registration, error)
            raise
        return registration

    def _unregister_buffer(self, address: int, size: int) -> None:
        self.calls.append(("unregister_region", address, size))
        if self.fail_unregister_count:
            self.fail_unregister_count -= 1
            raise RuntimeError("unregister failed")

    def close(self) -> None:
        self.calls.append(("backend_close",))
        if self.fail_close_count:
            self.fail_close_count -= 1
            raise RuntimeError("backend close failed")
        self.closed = True


def make_caches(topology, *, shared_storage: bool = False):
    layer_names = topology.groups[0].layer_names
    if not shared_storage:
        return {layer_name: torch.empty((8, 4), dtype=torch.float32) for layer_name in layer_names}
    storage = torch.empty((8 * len(layer_names), 4), dtype=torch.float32)
    return {layer_name: storage[index * 8 : (index + 1) * 8] for index, layer_name in enumerate(layer_names)}


def make_resources(backend: RecordingBackend, *, physical_layers=(0, 1)):
    topology = make_topology(physical_layers=physical_layers)
    resources = KVPoolResources(
        backend,
        make_backend_spec(layerwise_access=None),
        8,
        topology.groups,
    )
    return topology, resources


def run_backend_resource_lifecycle(backend, backend_spec) -> None:
    topology = make_topology(physical_layers=(0,))
    resources = KVPoolResources(backend, backend_spec, 8, topology.groups)
    projection_binder = compile_bulk_projection_binder(topology, 64)
    worker = SynchronousBulkWorker(
        topology,
        projection_binder,
        resources,
        store_enabled=False,
    )
    worker.bind_kv_caches(make_caches(topology))
    worker.close()
