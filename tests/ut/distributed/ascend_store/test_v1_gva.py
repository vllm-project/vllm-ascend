"""Exercise GVA session contracts and their existing KV Pool runtime path."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import backend as backend_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import BackendSpec, LayerwiseAccessKind
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program import compiler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.compiler import compile_kv_pool_program
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.spec.compilation import KVPoolCompilationSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.spec.schedule import (
    KVPoolSchedule,
    LoadScheduleKind,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.stages.region import LayerwiseRegionProjection
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    LoadCommand,
    LoadCommandBatch,
    RangeStoreCommand,
    StoreCommandBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.backend import GVABackendIO

from . import test_v1_domain_model as domain_model


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
            region = self.objects.setdefault(key, (10000 + 1000 * len(self.objects), size, False))
            gvas.append(region[0])
        return gvas if self.allocation_result is None else self.allocation_result

    def batch_copy(self, gvas, addresses, sizes, direction):
        self.calls.append(("copy", tuple(gvas), tuple(addresses), tuple(sizes), direction))
        if isinstance(self.copy_result, Exception):
            raise self.copy_result
        return self.copy_result

    def batch_write_finish(self, keys, results):
        self.calls.append(("publish", tuple(keys), tuple(results)))
        codes = [0] * len(keys) if self.commit_result is None else self.commit_result
        for key, result, code in zip(keys, results, codes, strict=True):
            if code == 0:
                base, size, _ = self.objects[key]
                if result == 0:
                    self.objects[key] = (base, size, True)
                else:
                    del self.objects[key]
        return codes


class FakeGVABackend(domain_model.FakeBackend):
    requires_exists_before_put = True

    def __init__(self) -> None:
        super().__init__()
        self.store = FakeGVAStore()

    def ensure_initialized(self):
        pass


def make_gva_spec(*args, **kwargs) -> BackendSpec:
    directions = SimpleNamespace(COPY_L2G=SimpleNamespace(value=0), COPY_G2L=SimpleNamespace(value=1))
    return BackendSpec("memcache", FakeGVABackend, SimpleNamespace(MmcDirect=directions), LayerwiseAccessKind.GVA, True)


def make_gva_runtime(monkeypatch, *, store=False, groups=(0,)):
    backend = FakeGVABackend()
    monkeypatch.setattr(domain_model, "make_backend_spec", make_gva_spec)
    if store:
        runtime, resources = domain_model.make_layerwise_store_runtime(backend)
    else:
        runtime, resources = domain_model.make_layerwise_load_runtime(backend, groups=groups)
    return runtime, resources, backend.store


@pytest.mark.parametrize(
    ("name", "access"),
    [("mooncake", LayerwiseAccessKind.KEY_RANGE), ("memcache", LayerwiseAccessKind.GVA), ("yuanrong", None)],
)
def test_backend_selection_fixes_the_access_kind(monkeypatch, name, access) -> None:
    classes = {class_name: domain_model.FakeBackend for _, class_name in backend_module.BACKEND_IMPORTS.values()}
    monkeypatch.setattr(backend_module.importlib, "import_module", lambda _: SimpleNamespace(**classes))
    assert backend_module.resolve_backend_spec(name).layerwise_access is access


def test_gva_uses_the_existing_layerwise_projection(monkeypatch) -> None:
    monkeypatch.setattr(compiler, "resolve_backend_spec", make_gva_spec)
    spec = KVPoolCompilationSpec(
        domain_model.make_topology(), "memcache", KVPoolSchedule(LoadScheduleKind.LAYERWISE, None, 2), 64
    )
    program = compile_kv_pool_program(spec)
    assert isinstance(program._transfer_region_projection, LayerwiseRegionProjection)


def test_gva_requires_native_publication_not_the_old_wrapper_fallback(monkeypatch) -> None:
    backend = FakeGVABackend()
    monkeypatch.setattr(backend.store, "batch_write_finish", None)
    monkeypatch.setattr(backend, "batch_write_finish", lambda *args: [0], raising=False)
    with pytest.raises(RuntimeError, match="requires native batch_write_finish"):
        GVABackendIO(backend, make_gva_spec())


def test_gva_rejects_an_unavailable_native_store(monkeypatch) -> None:
    backend = FakeGVABackend()
    monkeypatch.setattr(backend, "store", None)
    with pytest.raises(RuntimeError, match="Memcache store is unavailable for GVA"):
        GVABackendIO(backend, make_gva_spec())


def test_gva_observes_readability_not_allocated_presence() -> None:
    backend = FakeGVABackend()
    backend_io = GVABackendIO(backend, make_gva_spec())
    binding = domain_model.make_binding_batch().bindings[0]
    assert backend_io.start_store_sessions(["key"], [16]) == (0,)
    assert not backend_io.observe_objects((binding.remote_object,))[0].readable
    assert backend_io.store((domain_model.make_binding_batch(),)).succeeded
    assert not backend_io.observe_objects((binding.remote_object,))[0].readable
    assert backend_io.commit_store_sessions(["key"]) == (0,)
    assert backend_io.observe_objects((binding.remote_object,))[0].readable
    assert not backend_io._store_sessions
    assert [call[2] for call in backend.store.calls if call[0] == "query"] == [0, 1, 1, 1]


def test_gva_load_resolves_addresses_after_acquiring_the_lease(monkeypatch) -> None:
    backend = FakeGVABackend()
    backend.store.objects["key"] = (1000, 16, True)
    backend_io = GVABackendIO(backend, make_gva_spec())
    binding = domain_model.make_binding_batch().bindings[0]
    assert backend_io.observe_objects((binding.remote_object,))[0].readable
    original_add_lease = backend.store.batch_add_lease

    def acquire(keys):
        backend.store.objects["key"] = (2000, 16, True)
        return original_add_lease(keys)

    monkeypatch.setattr(backend.store, "batch_add_lease", acquire)
    assert backend_io.start_load_sessions(["key"], [16]) == (0,)
    assert backend_io.load((binding,))[0].result_code == 0
    assert next(call for call in backend.store.calls if call[0] == "copy")[1] == (2000,)
    backend_io.finish_load_sessions(["key"])
    assert not backend.store.leases


def test_gva_partial_lease_failure_releases_only_acquired_keys() -> None:
    backend = FakeGVABackend()
    backend.store.objects.update(a=(1000, 16, True), b=(2000, 16, True))
    backend.store.lease_result = [0, -7]
    backend_io = GVABackendIO(backend, make_gva_spec())
    assert backend_io.start_load_sessions(["a", "b"], [16, 16]) == (0, -7)
    backend_io.finish_load_sessions(["a", "b"])
    assert backend.store.calls[-1] == ("release", ("a",))
    assert not backend.store.leases


def test_gva_partial_allocation_does_not_submit_failed_keys() -> None:
    backend = FakeGVABackend()
    backend.store.allocation_result = [10000, 0]
    backend_io = GVABackendIO(backend, make_gva_spec())
    assert backend_io.start_store_sessions(["key", "other"], [16, 16]) == (0, -1)
    assert tuple(backend_io._store_sessions) == ("key",)
    assert backend_io.store((domain_model.make_binding_batch(),)).succeeded
    assert backend_io.commit_store_sessions(["key"]) == (0,)
    assert [call[1] for call in backend.store.calls if call[0] == "publish"] == [("key",)]


def test_gva_read_lease_release_failure_is_visible() -> None:
    backend = FakeGVABackend()
    backend.store.objects["key"] = (1000, 16, True)
    backend_io = GVABackendIO(backend, make_gva_spec())
    assert backend_io.start_load_sessions(["key"], [16]) == (0,)
    backend.store.remove_result = -4
    with pytest.raises(RuntimeError, match="batch_remove_lease failed"):
        backend_io.finish_load_sessions(["key"])
    assert backend.store.leases == {"key"}
    assert "key" in backend_io._load_sessions


def test_gva_rejects_wrong_object_size_and_releases_its_read_lease() -> None:
    backend = FakeGVABackend()
    backend.store.objects["key"] = (1000, 8, True)
    backend_io = GVABackendIO(backend, make_gva_spec())
    assert backend_io.start_load_sessions(["key"], [16]) == (-1,)
    assert not backend_io._load_sessions
    assert not backend.store.leases
    assert not any(call[0] == "copy" for call in backend.store.calls)


def test_gva_rejects_duplicate_allocation_with_incompatible_size() -> None:
    backend = FakeGVABackend()
    backend.store.objects["key"] = (1000, 8, True)
    backend_io = GVABackendIO(backend, make_gva_spec())
    assert backend_io.start_store_sessions(["key"], [16]) == (-1,)
    result = backend_io.store((domain_model.make_binding_batch(),))
    assert not result.succeeded
    assert result.source_release_confirmed
    assert not any(call[0] in ("copy", "publish") for call in backend.store.calls)


@pytest.mark.parametrize("result_code", [0, -9])
def test_gva_copy_returns_batch_scope_evidence(result_code) -> None:
    backend = FakeGVABackend()
    backend.store.copy_result = result_code
    backend_io = GVABackendIO(backend, make_gva_spec())
    batch = domain_model.make_binding_batch()
    binding = batch.bindings[0]
    other = replace(binding, remote_object=replace(binding.remote_object, key="other"))
    batches = (replace(batch, bindings=(binding, other)),)
    assert backend_io.start_store_sessions(["key", "other"], [16, 16]) == (0, 0)
    result = backend_io.store(batches)
    assert [item.result_code for item in result.binding_evidence] == [result_code, result_code]
    assert result.succeeded is (result_code == 0)
    assert result.source_release_confirmed is (result_code == 0)


@pytest.mark.parametrize("copy_result", [RuntimeError("copy failed"), None, True, [0]])
def test_gva_copy_error_does_not_invent_source_release(copy_result) -> None:
    backend = FakeGVABackend()
    backend.store.copy_result = copy_result
    backend_io = GVABackendIO(backend, make_gva_spec())
    assert backend_io.start_store_sessions(["key"], [16]) == (0,)
    result = backend_io.store((domain_model.make_binding_batch(),))
    assert not result.succeeded
    assert not result.source_release_confirmed
    assert result.binding_evidence[0].result_code is None
    assert result.error is not None


def test_gva_abort_does_not_delete_a_duplicate_object() -> None:
    backend = FakeGVABackend()
    backend.store.objects["key"] = (1000, 16, True)
    backend_io = GVABackendIO(backend, make_gva_spec())
    assert backend_io.start_store_sessions(["key"], [16]) == (0,)
    assert backend_io.revoke_store_sessions(["key", "failed_allocation"]) == (-1, 0)
    assert backend.store.objects["key"] == (1000, 16, True)
    assert not any(call[0] == "publish" for call in backend.store.calls)


def test_gva_publication_failure_is_not_hidden_by_copy_success() -> None:
    backend = FakeGVABackend()
    backend.store.commit_result = [-8]
    backend_io = GVABackendIO(backend, make_gva_spec())
    assert backend_io.start_store_sessions(["key"], [16]) == (0,)
    assert backend_io.store((domain_model.make_binding_batch(),)).source_release_confirmed
    assert backend_io.commit_store_sessions(["key"]) == (-8,)
    assert not backend.store.objects["key"][2]
    assert backend_io.revoke_store_sessions(["key"]) == (-1,)


def test_gva_layerwise_store_uses_existing_runtime_and_publishes_after_all_layers(monkeypatch) -> None:
    monkeypatch.setattr(torch.npu, "Event", domain_model.FakeEvent)
    runtime, resources, store = make_gva_runtime(monkeypatch, store=True)
    commands = StoreCommandBatch(
        (RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, store_job_id=17),)
    )
    try:
        domain_model.begin_kv_pool_step(runtime, store=commands)
        runtime.save_layer("layers.0.group.0")
        runtime.save_layer("layers.1.group.0")
        runtime.finish_step()
        runtime.fence_previous_store()
        copies = [call for call in store.calls if call[0] == "copy"]
        assert copies == [("copy", (10000,), (1064,), (32,), 0), ("copy", (10032,), (2064,), (32,), 0)]
        assert [call[0] for call in store.calls if call[0] in ("alloc", "copy", "publish")] == [
            "alloc",
            "copy",
            "copy",
            "publish",
        ]
        assert all(region[2] for region in store.objects.values())
        assert runtime.take_released_store_job_ids() == {17}
    finally:
        runtime.close()
    assert resources.closed


def test_gva_runtime_publication_failure_still_releases_copied_sources(monkeypatch) -> None:
    monkeypatch.setattr(torch.npu, "Event", domain_model.FakeEvent)
    runtime, resources, store = make_gva_runtime(monkeypatch, store=True)
    store.commit_result = [-8]
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, store_job_id=17)
    domain_model.begin_kv_pool_step(runtime, store=StoreCommandBatch((command,)))
    runtime.save_layer("layers.0.group.0")
    runtime.save_layer("layers.1.group.0")
    with pytest.raises(RuntimeError, match="Store failed"):
        runtime.finish_step()
    assert runtime.take_released_store_job_ids() == {17}
    assert runtime._pending_store_batch is None
    assert not any(region[2] for region in store.objects.values())
    assert all(all(code == 0 for code in call[2]) for call in store.calls if call[0] == "publish")
    with pytest.raises(RuntimeError, match="previous Store failure"):
        runtime.close()
    assert resources.closed


def test_gva_runtime_copy_failure_keeps_source_resources_and_does_not_publish(monkeypatch) -> None:
    monkeypatch.setattr(torch.npu, "Event", domain_model.FakeEvent)
    runtime, resources, store = make_gva_runtime(monkeypatch, store=True)
    store.copy_result = -9
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, store_job_id=17)
    domain_model.begin_kv_pool_step(runtime, store=StoreCommandBatch((command,)))
    runtime.save_layer("layers.0.group.0")
    runtime.save_layer("layers.1.group.0")
    with pytest.raises(RuntimeError, match="Store failed"):
        runtime.finish_step()
    assert not runtime.take_released_store_job_ids()
    assert runtime._pending_store_batch is not None
    assert not any(call[0] == "publish" for call in store.calls)
    with pytest.raises(RuntimeError, match="previous Store failure"):
        runtime.close()
    assert not resources.closed


def test_gva_runtime_unreached_layer_does_not_publish_an_incomplete_object(monkeypatch) -> None:
    monkeypatch.setattr(torch.npu, "Event", domain_model.FakeEvent)
    runtime, resources, store = make_gva_runtime(monkeypatch, store=True)
    commands = StoreCommandBatch((RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4),))
    domain_model.begin_kv_pool_step(runtime, store=commands)
    runtime.save_layer("layers.0.group.0")
    with pytest.raises(RuntimeError, match="Store failed"):
        runtime.finish_step()
    assert not any(call[0] == "publish" for call in store.calls)
    assert not any(region[2] for region in store.objects.values())
    with pytest.raises(RuntimeError, match="previous Store failure"):
        runtime.close()
    assert resources.closed


def test_gva_layerwise_hybrid_load_aggregates_group_bindings_and_releases_leases(monkeypatch) -> None:
    runtime, resources, store = make_gva_runtime(monkeypatch, groups=(0, 1))
    commands = LoadCommandBatch((LoadCommand("request", TokenRange(0, 4), ((1,), (2,)), (b"a",)),))
    transfers = runtime._program.select_load_transfers(commands.commands)
    keys = tuple(dict.fromkeys(binding.remote_object.key for transfer in transfers for binding in transfer.traversal))
    store.objects.update((key, (10000 + index * 1000, 64, True)) for index, key in enumerate(keys))
    try:
        domain_model.begin_kv_pool_step(runtime, load=commands)
        runtime.start_load()
        runtime.wait_for_layer_load("layers.0.group.0")
        runtime.wait_for_layer_load("layers.1.group.0")
        result = runtime.collect_load_result()
        assert not result.failed_request_ids
        assert not result.failed_block_ids
        assert not store.leases
        copies = [call for call in store.calls if call[0] == "copy"]
        assert len(copies) == 2
        assert copies[0][1] == (10000, 11000)
        assert copies[1][1] == (10032, 11032)
        assert all(call[4] == 1 for call in copies)
    finally:
        runtime.close()
    assert resources.closed


def test_gva_hybrid_load_failure_aborts_acquired_group_leases(monkeypatch) -> None:
    runtime, resources, store = make_gva_runtime(monkeypatch, groups=(0, 1))
    commands = LoadCommandBatch((LoadCommand("request", TokenRange(0, 4), ((1,), (2,)), (b"a",)),))
    transfers = runtime._program.select_load_transfers(commands.commands)
    key = transfers[0].traversal[0].remote_object.key
    store.objects[key] = (10000, 64, True)
    domain_model.begin_kv_pool_step(runtime, load=commands)
    with pytest.raises(RuntimeError, match="Hybrid KV Load failed"):
        runtime.start_load()
    assert not store.leases
    assert not any(call[0] == "copy" for call in store.calls)
    runtime.close()
    assert resources.closed


def test_gva_load_open_exception_cleans_up_known_leases(monkeypatch) -> None:
    runtime, resources, store = make_gva_runtime(monkeypatch)
    commands = LoadCommandBatch((LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",)),))
    key = runtime._program.select_load_transfers(commands.commands)[0].traversal[0].remote_object.key
    store.objects[key] = (10000, 64, True)
    monkeypatch.setattr(store, "batch_get_key_info", lambda *args: [])
    domain_model.begin_kv_pool_step(runtime, load=commands)
    with pytest.raises(RuntimeError, match="Layerwise Load timeline terminated"):
        runtime.start_load()
    assert not store.leases
    with pytest.raises(RuntimeError, match="Layerwise Load timeline terminated"):
        runtime.close()
    assert resources.closed
