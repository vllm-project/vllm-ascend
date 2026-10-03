"""Exercise the GVA adapter without reconstructing bound transfer structure."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import backend as backend_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    BackendSpec,
    LayerwiseAccessKind,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.spec.compilation import (
    KVPoolCompilationSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.spec.schedule import (
    KVPoolSchedule,
    LoadScheduleKind,
    StoreScheduleKind,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    KVTransferStep,
    RangeStoreCommand,
    StoreCommandBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.rules import compile_kv_pool_rules
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.backend import (
    GVABackendIO,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.resources import (
    GVAObjectLayout,
    KVPoolResources,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.runtime import (
    KVPoolRuntime,
)

from .v1.helpers import (
    FakeBackend,
    FakeEvent,
    FakeResources,
    make_memory_geometry,
    make_topology,
    make_work,
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
        gvas = []
        for key, size in zip(keys, sizes, strict=True):
            region = self.objects.setdefault(key, (10_000 + 1000 * len(self.objects), size, False))
            gvas.append(region[0])
        return gvas if self.allocation_result is None else self.allocation_result

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
        self.store = FakeGVAStore()

    def ensure_initialized(self) -> None:
        pass


def make_gva_spec() -> BackendSpec:
    directions = SimpleNamespace(COPY_L2G=SimpleNamespace(value=0), COPY_G2L=SimpleNamespace(value=1))
    return BackendSpec(
        "memcache",
        FakeGVABackend,
        SimpleNamespace(MmcDirect=directions),
        LayerwiseAccessKind.GVA,
        True,
    )


def make_gva_runtime():
    backend = FakeGVABackend()
    topology = make_topology()
    schedule = KVPoolSchedule(LoadScheduleKind.LAYERWISE, StoreScheduleKind.LAYERWISE, 2)
    backend_spec = make_gva_spec()
    resources = FakeResources(backend, backend_spec, make_memory_geometry(topology))
    spec = KVPoolCompilationSpec(topology, "memcache", schedule, 64)
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.rules.rules.resolve_backend_spec",
        return_value=backend_spec,
    ):
        rule_binder = compile_kv_pool_rules(
            spec,
            layerwise_full_key=lambda group, value, head, stage: f"g{group}:p{stage}:h{head}:{value}",
        )
    runtime = KVPoolRuntime(
        spec,
        rule_binder,
        resources,
        source_ready_event_factory=FakeEvent,
    )
    runtime.bind_kv_caches({"cache": object()})
    return runtime, resources, backend.store


def test_backend_selection_fixes_access_kind_once(monkeypatch) -> None:
    classes = {class_name: FakeBackend for _, class_name in backend_module.BACKEND_IMPORTS.values()}
    monkeypatch.setattr(backend_module.importlib, "import_module", lambda _: SimpleNamespace(**classes))
    expected = {
        "mooncake": LayerwiseAccessKind.KEY_RANGE,
        "memcache": LayerwiseAccessKind.GVA,
        "yuanrong": None,
    }
    assert {name: backend_module.resolve_backend_spec(name).layerwise_access for name in expected} == expected


def test_gva_initialization_validates_native_capabilities_and_registered_layout(monkeypatch) -> None:
    backend = FakeGVABackend()
    monkeypatch.setattr(backend.store, "batch_write_finish", None)
    monkeypatch.setattr(backend, "batch_write_finish", lambda *args: [0], raising=False)
    with pytest.raises(RuntimeError, match="native batch_write_finish"):
        GVABackendIO(backend, make_gva_spec())

    unavailable = FakeGVABackend()
    monkeypatch.setattr(unavailable, "store", None)
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
    caches = {
        layer_name: torch.empty((8, 4), dtype=torch.float32)
        for layer_name in topology.groups[0].layer_names
    }
    registration = resources.bind_kv_caches(caches)

    assert registration["base_addresses"] == {0: [cache.data_ptr() for cache in caches.values()]}
    assert registration["block_lengths"] == {0: [16, 16]}
    assert registration["block_strides"] == {0: [16, 16]}
    assert registration["layer_entry_offsets"] == {0: [0, 1, 2]}
    assert registration["object_offsets"] == {0: 2 * 1024 * 1024 + 64}
    assert registration["object_sizes"] == {0: 4 * 1024 * 1024}
    assert registration_backend.calls[-1][0] == "register_buffer"


def test_gva_readability_changes_only_after_publication() -> None:
    backend = FakeGVABackend()
    backend_io = GVABackendIO(backend, make_gva_spec())
    work = make_work(physical_layer_id=0, block_ids=(1,))
    remote_object = work.remote_objects[0]
    key = remote_object.key

    assert backend_io.start_store_sessions([key], [64]) == (0,)
    assert not backend_io.observe_objects((remote_object,))[0].readable
    assert backend_io.store(work).succeeded
    assert not backend_io.observe_objects((remote_object,))[0].readable
    assert backend_io.commit_store_sessions([key]) == (0,)
    assert backend_io.observe_objects((remote_object,))[0].readable
    assert not backend_io._store_sessions


def test_gva_load_admission_resolves_after_lease_and_releases_only_owned_keys(monkeypatch) -> None:
    backend = FakeGVABackend()
    backend_io = GVABackendIO(backend, make_gva_spec())
    work = make_work(physical_layer_id=0)
    first, second = (item.key for item in work.remote_objects)
    backend.store.objects.update(
        {
            first: (1000, 64, True),
            second: (2000, 64, True),
            "wrong-size": (3000, 8, True),
        }
    )
    backend.store.lease_result = [0, -7, 0]
    original_add_lease = backend.store.batch_add_lease

    def acquire(keys):
        backend.store.objects[first] = (4000, 64, True)
        return original_add_lease(keys)

    monkeypatch.setattr(backend.store, "batch_add_lease", acquire)
    assert backend_io.start_load_sessions([first, second, "wrong-size"], [64, 64, 64]) == (0, -7, -1)
    assert backend_io.load(work.select_keys({first}))[0].result_code == 0
    assert len(backend_io._resolved_load_sessions) == 1
    assert len(backend_io._resolved_load_batches) == 1
    assert next(call for call in backend.store.calls if call[0] == "copy")[1] == (4000,)
    backend_io.finish_load_sessions([first, second, "wrong-size"])
    assert backend.store.calls[-1] == ("release", (first,))
    assert not backend.store.leases
    assert not backend_io._resolved_load_sessions
    assert not backend_io._resolved_load_batches


def test_gva_store_admission_keeps_only_compatible_owned_allocations() -> None:
    backend = FakeGVABackend()
    backend_io = GVABackendIO(backend, make_gva_spec())
    work = make_work(physical_layer_id=0)
    first, second = (item.key for item in work.remote_objects)
    backend.store.objects[first] = (1000, 8, True)
    backend.store.allocation_result = [1000, 0]

    assert backend_io.start_store_sessions([first, second], [64, 64]) == (-1, -1)
    assert not backend_io._store_sessions
    assert backend_io.store(work).source_release_confirmed
    assert not any(call[0] in ("copy", "publish") for call in backend.store.calls)
    assert backend_io.revoke_store_sessions([first, second]) == (0, 0)
    assert backend.store.objects[first] == (1000, 8, True)


def test_gva_copy_normalizes_batch_result_and_unknown_evidence() -> None:
    work = make_work(physical_layer_id=0)
    keys = [item.key for item in work.remote_objects]

    for native_result, expected_codes, succeeded, released in (
        (0, [0, 0], True, True),
        (-9, [-9, -9], False, False),
        (None, [None, None], False, False),
        (True, [None, None], False, False),
        ([0], [None, None], False, False),
        (RuntimeError("copy failed"), [None, None], False, False),
    ):
        backend = FakeGVABackend()
        backend.store.copy_result = native_result
        backend_io = GVABackendIO(backend, make_gva_spec())
        assert backend_io.start_store_sessions(keys, [64, 64]) == (0, 0)
        result = backend_io.store(work)
        assert [item.result_code for item in result.transfer_evidence] == expected_codes
        assert all(item.source_release_confirmed is released for item in result.transfer_evidence)
        assert result.succeeded is succeeded
        assert result.source_release_confirmed is released


def test_gva_publication_failure_does_not_hide_copy_source_release() -> None:
    backend = FakeGVABackend()
    backend.store.commit_result = [-8]
    backend_io = GVABackendIO(backend, make_gva_spec())
    work = make_work(physical_layer_id=0, block_ids=(1,))
    key = work.remote_objects[0].key

    assert backend_io.start_store_sessions([key], [64]) == (0,)
    assert backend_io.store(work).source_release_confirmed
    assert backend_io.commit_store_sessions([key]) == (-8,)
    assert not backend.store.objects[key][2]
    assert backend_io.revoke_store_sessions([key]) == (-1,)


def test_gva_runtime_publishes_after_all_layers_and_reports_job_release() -> None:
    runtime, resources, store = make_gva_runtime()
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, store_job_id=17)
    runtime.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))

    runtime.save_layer("layers.0.group.0")
    runtime._timeline.store._executor._queue.join()
    backend_io = runtime._backend_io
    assert len(backend_io._rule_store_bases) == 1
    resolved_bases = next(iter(backend_io._rule_store_bases.values()))
    assert [call[0] for call in store.calls].count("copy") == 1
    assert not any(call[0] == "publish" for call in store.calls)

    runtime.save_layer("layers.1.group.0")
    runtime._timeline.store._executor._queue.join()
    assert next(iter(backend_io._rule_store_bases.values())) is resolved_bases
    runtime.finish_step()
    assert [call[0] for call in store.calls if call[0] in ("alloc", "copy", "publish")] == [
        "alloc",
        "copy",
        "copy",
        "publish",
    ]
    assert all(region[2] for region in store.objects.values())
    assert not backend_io._rule_store_bases
    assert runtime.take_released_store_job_ids() == {17}
    runtime.close()
    assert resources.closed


def test_gva_runtime_distinguishes_copy_failure_from_publication_failure() -> None:
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, store_job_id=17)

    copy_runtime, copy_resources, copy_store = make_gva_runtime()
    copy_store.copy_result = -9
    copy_runtime.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
    copy_runtime.save_layer("layers.0.group.0")
    copy_runtime.save_layer("layers.1.group.0")
    with pytest.raises(RuntimeError, match="Store failed"):
        copy_runtime.finish_step()
    assert copy_runtime.take_released_store_job_ids() == set()
    assert copy_runtime._pending_store_batch is not None
    assert not any(call[0] == "publish" for call in copy_store.calls)
    with pytest.raises(RuntimeError, match="previous Store failure"):
        copy_runtime.close()
    assert not copy_resources.closed

    publish_runtime, publish_resources, publish_store = make_gva_runtime()
    publish_store.commit_result = [-8]
    publish_runtime.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
    publish_runtime.save_layer("layers.0.group.0")
    publish_runtime.save_layer("layers.1.group.0")
    with pytest.raises(RuntimeError, match="Store failed"):
        publish_runtime.finish_step()
    assert publish_runtime.take_released_store_job_ids() == {17}
    assert publish_runtime._pending_store_batch is None
    assert not any(region[2] for region in publish_store.objects.values())
    with pytest.raises(RuntimeError, match="previous Store failure"):
        publish_runtime.close()
    assert publish_resources.closed
