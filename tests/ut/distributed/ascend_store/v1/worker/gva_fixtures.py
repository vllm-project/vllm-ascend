"""GVA object allocation, leases, publication, and Worker binding fixtures."""

from __future__ import annotations

from typing import Any

import numpy as np

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeEvent,
    FakeResources,
    make_topology,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    BackendSpec,
    GVARegion,
    LayerwiseAccessKind,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection import (
    GVALayerwiseProjection,
    GVALayerwiseProjectionBinder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection.layerwise.gva import gva_local_keys
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.io import (
    GVABackendIO,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise import (
    GVALayerwiseWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.batch import (
    KVGroupBatch,
    KVTransferBatch,
)


class FakeKeyInfo:
    def __init__(self, region: tuple[int, int, bool] | None) -> None:
        self._region = region

    def size(self):
        return self._region[1] if self._region is not None else 0

    def gva_list(self):
        return [self._region[0]] if self._region is not None else []


class FakeGVAStore:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[int, int, bool]] = {}
        self.leases: set[str] = set()
        self.calls: list[tuple[Any, ...]] = []
        self.copy_result: object = 0
        self.lease_result: list[int] | None = None
        self.allocation_result: list[int] | None = None
        self.commit_result: list[int] | None = None
        self.remove_result = 0

    def batch_get_key_info(self, keys, flag):
        self.calls.append(("query", tuple(keys), flag))
        regions = [self.objects.get(key) for key in keys]
        return [FakeKeyInfo(region if flag == 0 or region is None or region[2] else None) for region in regions]

    def batch_add_lease(self, keys):
        self.calls.append(("lease", tuple(keys)))
        codes = self.lease_result
        if codes is None:
            codes = [0 if self.objects.get(key, (0, 0, False))[2] else -7 for key in keys]
        self.leases.update(key for key, code in zip(keys, codes, strict=True) if code == 0)
        return codes

    def batch_remove_lease(self, keys):
        self.calls.append(("release", tuple(keys)))
        if self.remove_result == 0:
            self.leases.difference_update(keys)
        return self.remove_result

    def batch_alloc(self, keys, sizes):
        self.calls.append(("alloc", tuple(keys), tuple(sizes)))
        if self.allocation_result is not None:
            for key, size, gva in zip(keys, sizes, self.allocation_result, strict=True):
                if gva > 0:
                    self.objects[key] = (gva, size, False)
            return self.allocation_result

        gvas = []
        for key, size in zip(keys, sizes, strict=True):
            if key in self.objects:
                gvas.append(0)
                continue
            gva = 10_000 + 1000 * len(self.objects)
            self.objects[key] = (gva, size, False)
            gvas.append(gva)
        return gvas

    def batch_copy(self, gvas, addresses, sizes, direction):
        self.calls.append(("copy", tuple(gvas), tuple(addresses), tuple(sizes), direction))
        if isinstance(self.copy_result, BaseException):
            raise self.copy_result
        return self.copy_result

    def batch_write_finish(self, keys, results):
        self.calls.append(("publish", tuple(keys), tuple(results)))
        codes = [0] * len(keys) if self.commit_result is None else self.commit_result
        for key, result, code in zip(keys, results, codes, strict=True):
            if code != 0:
                continue
            base, size, _ = self.objects[key]
            if result == 0:
                self.objects[key] = (base, size, True)
            else:
                del self.objects[key]
        return codes


class FakeGVABackend(FakeBackend):
    requires_exists_before_put = True

    def __init__(self) -> None:
        super().__init__()
        self.native_store: FakeGVAStore | None = FakeGVAStore()

    def validate_gva_support(self) -> None:
        if self.native_store is None:
            raise RuntimeError("Memcache store is unavailable for GVA")
        for method in (
            "batch_get_key_info",
            "batch_add_lease",
            "batch_remove_lease",
            "batch_alloc",
            "batch_copy",
            "batch_write_finish",
        ):
            if not callable(getattr(self.native_store, method, None)):
                raise RuntimeError(f"Memcache GVA requires native {method}")

    def query_gva_regions(self, keys):
        assert self.native_store is not None
        infos = self.native_store.batch_get_key_info(keys, 1)
        return tuple(
            None if not info.gva_list() or info.size() <= 0 else GVARegion(info.gva_list()[0], info.size())
            for info in infos
        )

    def add_gva_leases(self, keys):
        assert self.native_store is not None
        return tuple(self.native_store.batch_add_lease(keys))

    def remove_gva_leases(self, keys):
        assert self.native_store is not None
        result = self.native_store.batch_remove_lease(keys)
        if result != 0:
            raise RuntimeError(f"batch_remove_lease failed with result {result}")

    def allocate_gva(self, keys, sizes):
        assert self.native_store is not None
        return tuple(self.native_store.batch_alloc(keys, sizes))

    def copy_from_gva(self, remote_addresses, local_addresses, sizes):
        assert self.native_store is not None
        return self.native_store.batch_copy(remote_addresses, local_addresses, sizes, 1)

    def copy_to_gva(self, remote_addresses, local_addresses, sizes):
        assert self.native_store is not None
        return self.native_store.batch_copy(remote_addresses, local_addresses, sizes, 0)

    def publish_gva(self, keys):
        assert self.native_store is not None
        return tuple(self.native_store.batch_write_finish(keys, [0] * len(keys)))


def make_gva_spec() -> BackendSpec:
    return BackendSpec(
        "memcache",
        FakeGVABackend,
        LayerwiseAccessKind.GVA,
        True,
    )


def make_gva_worker():
    backend = FakeGVABackend()
    topology = make_topology()
    backend_spec = make_gva_spec()
    resources = FakeResources(backend, backend_spec, topology)
    projection_binder = GVALayerwiseProjectionBinder(
        topology,
        64,
        lambda group, value, head, stage: f"g{group}:p{stage}:h{head}:{value}",
    )
    worker = GVALayerwiseWorker(
        topology,
        projection_binder,
        resources,  # type: ignore[arg-type]
        source_ready_event_factory=FakeEvent,
    )
    worker.bind_kv_caches({"cache": object()})
    assert backend.native_store is not None
    return worker, resources, backend.native_store


def make_gva_binding(block_ids: tuple[int, ...] = (1, 3)):
    backend = FakeGVABackend()
    topology = make_topology()
    backend_spec = make_gva_spec()
    resources = FakeResources(backend, backend_spec, topology)
    binder = GVALayerwiseProjectionBinder(
        topology,
        64,
        lambda group, value, head, stage: f"g{group}:p{stage}:h{head}:{value}",
    )
    projection = binder.bind(**resources.bind_kv_caches({"cache": object()}))
    assert isinstance(projection, GVALayerwiseProjection)
    backend_io = GVABackendIO(backend, backend_spec)
    backend_io.bind_projection(projection)

    hashes = tuple(f"h{index}" for index in range(len(block_ids)))
    ids = np.asarray(block_ids, dtype=np.uint64)
    counts: np.ndarray = np.full(len(block_ids), 4, dtype=np.uint64)
    request_splits = np.asarray([0, len(block_ids)], dtype=np.intp)
    for values in (ids, counts, request_splits):
        values.flags.writeable = False
    group = KVGroupBatch(
        0,
        ids,
        counts,
        gva_local_keys(projection.groups[0], hashes),
        request_splits,
        (0, 1),
        projection.groups[0].object_size,
    )
    assert backend.native_store is not None
    return backend_io, KVTransferBatch(("request",), (group,), (17,)), backend.native_store
