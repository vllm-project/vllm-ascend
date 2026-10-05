"""Exercise the GVA adapter without reconstructing bound transfer structure."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import backend as backend_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    BackendSpec,
    GVARegion,
    LayerwiseAccessKind,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection import (
    GVALayerwiseProjection,
    GVALayerwiseProjectionBinder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection.layerwise.gva import gva_local_keys
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    KVTransferStep,
    RangeStoreCommand,
    StoreCommandBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.backend import (
    GVABackendIO,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.backend import (
    arguments as arguments_module,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.batch import (
    KVGroupBatch,
    KVTransferBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.resources import (
    GVAObjectLayout,
    KVPoolResources,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import (
    GVALayerwiseWorker,
)

from .v1.helpers import (
    FakeBackend,
    FakeEvent,
    FakeResources,
    make_topology,
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


def test_backend_selection_fixes_access_kind_once(monkeypatch) -> None:
    classes = {class_name: FakeBackend for _, class_name in backend_module.BACKEND_IMPORTS.values()}
    monkeypatch.setattr(backend_module.importlib, "import_module", lambda _: SimpleNamespace(**classes))
    expected = {
        "mooncake": LayerwiseAccessKind.KEY_RANGE,
        "memcache": LayerwiseAccessKind.GVA,
    }
    assert {name: backend_module.resolve_backend_spec(name).layerwise_access for name in expected} == expected


def test_backend_registry_loads_only_v1_owned_adapters() -> None:
    expected_modules = {
        "mooncake": f"{backend_module.__name__}.mooncake",
        "memcache": f"{backend_module.__name__}.memcache",
    }

    assert {
        name: backend_module.resolve_backend_spec(name).backend_factory.__module__ for name in expected_modules
    } == expected_modules
    with pytest.raises(ValueError, match="Unsupported AscendStore v1 backend"):
        backend_module.resolve_backend_spec("yuanrong")


def test_gva_initialization_validates_native_capabilities_and_registered_layout(monkeypatch) -> None:
    backend = FakeGVABackend()
    assert backend.native_store is not None
    monkeypatch.setattr(backend.native_store, "batch_write_finish", None)
    monkeypatch.setattr(backend, "batch_write_finish", lambda *args: [0], raising=False)
    with pytest.raises(RuntimeError, match="native batch_write_finish"):
        GVABackendIO(backend, make_gva_spec())

    unavailable = FakeGVABackend()
    unavailable.native_store = None
    with pytest.raises(RuntimeError, match="store is unavailable"):
        GVABackendIO(unavailable, make_gva_spec())

    topology = make_topology(physical_layers=(4, 5))
    registration_backend = FakeBackend()
    resources = KVPoolResources(
        registration_backend,
        make_gva_spec(),
        8,
        topology.groups,
        gva_layout=GVAObjectLayout(
            pipeline_layer_start=4,
            total_hidden_layers=4,
            dcp_rank=1,
            dcp_size=2,
            put_step=2,
        ),
    )
    caches = {layer_name: torch.empty((8, 4), dtype=torch.float32) for layer_name in topology.groups[0].layer_names}
    registration = resources.bind_kv_caches(caches)

    assert registration["base_addresses"] == {0: [cache.data_ptr() for cache in caches.values()]}
    assert registration["block_lengths"] == {0: [16, 16]}
    assert registration["block_strides"] == {0: [16, 16]}
    assert registration["layer_entry_offsets"] == {0: [0, 1, 2]}
    assert registration["object_offsets"] == {0: 2 * 1024 * 1024 + 64}
    assert registration["object_sizes"] == {0: 4 * 1024 * 1024}
    assert registration_backend.calls[-1][0] == "register_buffer"


def test_gva_worker_close_unregisters_exact_region_before_backend_close() -> None:
    backend = FakeGVABackend()
    topology = make_topology(physical_layers=(0,))
    backend_spec = make_gva_spec()
    resources = KVPoolResources(
        backend,
        backend_spec,
        8,
        topology.groups,
        gva_layout=GVAObjectLayout(0, 1, 0, 1, 1),
    )
    projection_binder = GVALayerwiseProjectionBinder(
        topology,
        64,
        lambda group, value, head, stage: f"g{group}:p{stage}:h{head}:{value}",
    )
    worker = GVALayerwiseWorker(
        topology,
        projection_binder,
        resources,
        source_ready_event_factory=FakeEvent,
    )
    cache = torch.empty((8, 4), dtype=torch.float32)

    worker.bind_kv_caches({topology.groups[0].layer_names[0]: cache})
    worker.close()

    lifecycle = [call for call in backend.calls if call[0] in ("register_buffer", "unregister_buffer", "backend_close")]
    assert lifecycle == [
        ("register_buffer", (cache.data_ptr(),), (cache.nbytes,)),
        ("unregister_buffer", cache.data_ptr(), cache.nbytes),
        ("backend_close",),
    ]


def test_gva_readability_changes_only_after_publication() -> None:
    backend_io, batch, store = make_gva_binding((1,))
    key = batch.selected_keys()[0]

    assert backend_io.start_store_sessions([key], [64]) == (0,)
    assert not backend_io.exists([key])[0]
    assert backend_io.store_batch(batch, 0)[0].evidence.succeeded
    assert not backend_io.exists([key])[0]
    assert backend_io.commit_store_sessions([key]) == (0,)
    assert backend_io.exists([key])[0]
    assert not backend_io._store_sessions
    assert store.objects[key][2]


def test_gva_load_admission_resolves_after_lease_and_releases_only_owned_keys(monkeypatch) -> None:
    backend_io, batch, store = make_gva_binding()
    first, second = batch.selected_keys()
    store.objects.update(
        {
            first: (1000, 64, True),
            second: (2000, 64, True),
            "wrong-size": (3000, 8, True),
        }
    )
    store.lease_result = [0, -7, 0]
    original_add_lease = store.batch_add_lease

    def acquire(keys):
        store.objects[first] = (4000, 64, True)
        return original_add_lease(keys)

    monkeypatch.setattr(store, "batch_add_lease", acquire)
    assert backend_io.start_load_sessions([first, second, "wrong-size"], [64, 64, 64]) == (0, -7, -1)
    completion = backend_io.load_batch(batch.select_keys({first}), 0)[0]
    assert completion.transfer_evidence[0].result_code == 0
    assert backend_io._load_plan is not None
    assert backend_io._load_plan.object_bases_by_group is not None
    assert backend_io._load_plan.object_bases_by_group[0].tolist() == [4000]
    assert next(call for call in store.calls if call[0] == "copy")[1] == (4000,)
    backend_io.finish_load_sessions([first, second, "wrong-size"])
    assert store.calls[-1] == ("release", (first,))
    assert not store.leases
    assert backend_io._load_plan is None


def test_gva_layerwise_load_reuses_rows_and_leased_bases_across_layers(monkeypatch) -> None:
    backend_io, batch, store = make_gva_binding()
    first, second = batch.selected_keys()
    store.objects.update(
        {
            first: (1000, 64, True),
            second: (2000, 64, True),
        }
    )
    assert backend_io.start_load_sessions([first, second], [64, 64]) == (0, 0)
    backend_io.prepare_load_layers(batch)

    prepared_plan = backend_io._load_plan
    assert prepared_plan is not None
    assert prepared_plan.groups_by_layer[0][0] is prepared_plan.groups_by_layer[1][0]
    assert prepared_plan.keys_by_layer[0] is prepared_plan.keys_by_layer[1]
    assert prepared_plan.object_bases_by_group is not None
    resolved_bases = prepared_plan.object_bases_by_group[0]

    direct_calls = []
    direct_projection = arguments_module.gva_layer_ranges

    def record_direct_projection(group, block_ids, token_counts, object_bases, *, layer_id):
        direct_calls.append(layer_id)
        return direct_projection(group, block_ids, token_counts, object_bases, layer_id=layer_id)

    def reject_rebuild(*_args, **_kwargs):
        raise AssertionError("Layerwise GVA Load rebuilt cross-layer execution state")

    monkeypatch.setattr(arguments_module, "gva_layer_ranges", record_direct_projection)
    monkeypatch.setattr(backend_io, "_resolve_load_bases", reject_rebuild)
    monkeypatch.setattr(KVTransferBatch, "for_layer", reject_rebuild)
    monkeypatch.setattr(KVGroupBatch, "selected_keys", reject_rebuild)

    assert all(item.result_code == 0 for item in backend_io.load_layer(0)[0].transfer_evidence)
    assert all(item.result_code == 0 for item in backend_io.load_layer(1)[0].transfer_evidence)
    assert direct_calls == [0, 1]
    assert backend_io._load_plan is prepared_plan
    assert backend_io._load_plan.object_bases_by_group[0] is resolved_bases
    assert [call[1] for call in store.calls if call[0] == "copy"] == [
        (1000, 2000),
        (1032, 2032),
    ]

    backend_io.finish_load_sessions([first, second])
    assert backend_io._load_plan is None


def test_gva_load_copy_exception_becomes_unknown_source_evidence() -> None:
    backend_io, batch, store = make_gva_binding(block_ids=(1,))
    key = batch.selected_keys()[0]
    store.objects[key] = (1000, 64, True)

    assert backend_io.start_load_sessions([key], [64]) == (0,)
    store.copy_result = RuntimeError("GVA copy failed")
    completion = backend_io.load_batch(batch, 0)[0]

    assert len(completion.transfer_evidence) == 1
    evidence = completion.transfer_evidence[0]
    assert evidence.result_code is None
    assert (
        evidence.source.group_id,
        evidence.source.block_id,
        evidence.source.physical_layer_ids,
    ) == (0, 1, (0,))
    backend_io.finish_load_sessions([key])
    assert not store.leases


def test_gva_store_admission_owns_only_successful_allocations() -> None:
    backend_io, batch, store = make_gva_binding()
    first, second = batch.selected_keys()
    store.allocation_result = [1000, 0]

    assert backend_io.start_store_sessions([first, second], [64, 64]) == (0, -1)
    assert backend_io._store_sessions == {first: (1000, 64)}
    assert store.calls == [("alloc", (first, second), (64, 64))]
    assert backend_io.revoke_store_sessions([first, second]) == (-1, 0)
    assert not backend_io._store_sessions


def test_gva_copy_normalizes_batch_result_and_unknown_evidence() -> None:
    for native_result, expected_codes, succeeded, released in (
        (0, [0, 0], True, True),
        (-9, [-9, -9], False, False),
        (None, [None, None], False, False),
        (True, [None, None], False, False),
        ([0], [None, None], False, False),
        (RuntimeError("copy failed"), [None, None], False, False),
    ):
        backend_io, batch, store = make_gva_binding()
        keys = list(batch.selected_keys())
        store.copy_result = native_result
        assert backend_io.start_store_sessions(keys, [64, 64]) == (0, 0)
        evidence = backend_io.store_batch(batch, 0)[0].evidence
        assert [item.result_code for item in evidence.transfer_evidence] == expected_codes
        assert all(item.source_release_confirmed is released for item in evidence.transfer_evidence)
        assert evidence.succeeded is succeeded
        assert evidence.source_release_confirmed is released


def test_gva_publication_failure_does_not_hide_copy_source_release() -> None:
    backend_io, batch, store = make_gva_binding((1,))
    store.commit_result = [-8]
    key = batch.selected_keys()[0]

    assert backend_io.start_store_sessions([key], [64]) == (0,)
    assert backend_io.store_batch(batch, 0)[0].evidence.source_release_confirmed
    assert backend_io.commit_store_sessions([key]) == (-8,)
    assert not store.objects[key][2]
    assert backend_io.revoke_store_sessions([key]) == (-1,)


def test_gva_worker_publishes_after_all_layers_and_reports_job_release() -> None:
    worker, resources, store = make_gva_worker()
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, store_job_id=17)
    worker.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))

    worker.save_layer("layers.0.group.0")
    assert worker._layerwise_store_timeline is not None
    worker._layerwise_store_timeline._executor._queue.join()
    backend_io = worker._gva_backend_io
    assert backend_io is not None
    prepared_plan = backend_io._store_plan
    assert prepared_plan is not None
    assert prepared_plan.groups_by_layer[0][0] is prepared_plan.groups_by_layer[1][0]
    assert prepared_plan.keys_by_layer[0] is prepared_plan.keys_by_layer[1]
    assert prepared_plan.object_bases_by_group is not None
    resolved_bases = prepared_plan.object_bases_by_group[0]
    assert prepared_plan.object_bases_by_group[0] is resolved_bases
    assert [call[0] for call in store.calls].count("copy") == 1
    assert not any(call[0] == "publish" for call in store.calls)

    worker.save_layer("layers.1.group.0")
    worker._layerwise_store_timeline._executor._queue.join()
    assert backend_io._store_plan is prepared_plan
    assert backend_io._store_plan.object_bases_by_group[0] is resolved_bases
    worker.finish_step()
    assert [call[0] for call in store.calls if call[0] in ("alloc", "copy", "publish")] == [
        "alloc",
        "copy",
        "copy",
        "publish",
    ]
    assert all(region[2] for region in store.objects.values())
    assert backend_io._store_plan is None
    assert worker.take_released_store_job_ids() == {17}
    worker.close()
    assert resources.closed


def test_gva_worker_distinguishes_copy_failure_from_publication_failure() -> None:
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, store_job_id=17)

    copy_worker, copy_resources, copy_store = make_gva_worker()
    copy_store.copy_result = -9
    copy_worker.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
    copy_worker.save_layer("layers.0.group.0")
    copy_worker.save_layer("layers.1.group.0")
    with pytest.raises(RuntimeError, match="Store failed"):
        copy_worker.finish_step()
    assert copy_worker.take_released_store_job_ids() == set()
    assert copy_worker._pending_store_batch is not None
    assert not any(call[0] == "publish" for call in copy_store.calls)
    with pytest.raises(RuntimeError, match="previous Store failure"):
        copy_worker.close()
    assert not copy_resources.closed

    publish_worker, publish_resources, publish_store = make_gva_worker()
    publish_store.commit_result = [-8]
    publish_worker.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
    publish_worker.save_layer("layers.0.group.0")
    publish_worker.save_layer("layers.1.group.0")
    with pytest.raises(RuntimeError, match="Store failed"):
        publish_worker.finish_step()
    assert publish_worker.take_released_store_job_ids() == {17}
    assert publish_worker._pending_store_batch is None
    assert not any(region[2] for region in publish_store.objects.values())
    with pytest.raises(RuntimeError, match="previous Store failure"):
        publish_worker.close()
    assert publish_resources.closed
