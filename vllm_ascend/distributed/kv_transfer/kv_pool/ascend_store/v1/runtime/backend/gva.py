"""Execute existing KV bindings through Memcache GVA sessions."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import TYPE_CHECKING, cast

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend

from ...backend import BackendSpec
from ...program.values.evidence import BindingEvidence, RemoteObjectObservation, StoreEvidence
from ...program.values.representation import BindingBatch, KVBinding, RemoteKVObject
from .io import BackendIO

if TYPE_CHECKING:
    from ....backend.memcache_backend import MemcacheBackend

READABLE_GVA_QUERY = 1
GVA_SESSION_FAILURE = -1


class GVABackendIO(BackendIO):
    """Keep leased GVA addresses inside the Backend boundary, not the Program."""

    def __init__(self, backend: Backend, backend_spec: BackendSpec) -> None:
        super().__init__(backend, backend_spec)
        memcache_backend = cast("MemcacheBackend", backend)
        memcache_backend.ensure_initialized()
        store = memcache_backend.store
        if store is None:
            raise RuntimeError("Memcache store is unavailable for GVA")
        self._store = store
        self._load_sessions: dict[str, tuple[int, int] | None] = {}
        self._store_sessions: dict[str, tuple[int, int]] = {}
        self._load_direction = backend_spec.backend_module.MmcDirect.COPY_G2L.value
        self._store_direction = backend_spec.backend_module.MmcDirect.COPY_L2G.value
        self.validate_support()

    def validate_support(self) -> None:
        required = ("batch_get_key_info", "batch_add_lease", "batch_remove_lease", "batch_alloc", "batch_copy")
        for method in (*required, "batch_write_finish"):
            if not callable(getattr(self._store, method, None)):
                raise RuntimeError(f"Memcache GVA requires native {method}; upgrade the Backend library")

    def observe_objects(self, remote_objects: tuple[RemoteKVObject, ...]) -> tuple[RemoteObjectObservation, ...]:
        if not remote_objects:
            return ()
        regions = self._query_regions([remote_object.key for remote_object in remote_objects])
        return tuple(
            RemoteObjectObservation(remote_object, region is not None)
            for remote_object, region in zip(remote_objects, regions, strict=True)
        )

    def start_load_sessions(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]:
        if not keys:
            return ()
        if any(key in self._load_sessions for key in keys):
            raise RuntimeError("Previous GVA Load sessions have not ended")
        result_codes = list(self._require_result_codes("batch_add_lease", keys, self._store.batch_add_lease(keys)))
        leased_keys = [key for key, code in zip(keys, result_codes, strict=True) if code == 0]
        self._load_sessions.update((key, None) for key in leased_keys)
        # Resolve after acquiring the lease: an earlier Lookup address may already have been evicted.
        regions = dict(zip(leased_keys, self._query_regions(leased_keys), strict=True))
        rejected_keys = []
        for index, (key, object_size) in enumerate(zip(keys, object_sizes, strict=True)):
            if result_codes[index] != 0:
                continue
            region = regions[key]
            if region is None or region[1] != object_size:
                result_codes[index] = GVA_SESSION_FAILURE
                rejected_keys.append(key)
            else:
                self._load_sessions[key] = region
        self.finish_load_sessions(rejected_keys)
        return tuple(result_codes)

    def finish_load_sessions(self, keys: list[str]) -> None:
        leased_keys = [key for key in keys if key in self._load_sessions]
        if not leased_keys:
            return
        result_code = self._store.batch_remove_lease(leased_keys)
        if isinstance(result_code, bool) or not isinstance(result_code, Integral) or result_code != 0:
            raise RuntimeError(f"batch_remove_lease failed with result {result_code!r}")
        for key in leased_keys:
            del self._load_sessions[key]

    def start_store_sessions(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]:
        if not keys:
            return ()
        if any(key in self._store_sessions for key in keys):
            raise RuntimeError("Previous GVA Store sessions have not ended")
        gvas = self._require_result_codes("batch_alloc", keys, self._store.batch_alloc(keys, object_sizes))
        allocated_keys = []
        result_codes = []
        for key, gva, object_size in zip(keys, gvas, object_sizes, strict=True):
            result_codes.append(0 if gva > 0 else GVA_SESSION_FAILURE)
            if gva > 0:
                self._store_sessions[key] = (gva, object_size)
                allocated_keys.append(key)
        # Allocation can return an existing object; its real size must still match the binding.
        regions = dict(zip(allocated_keys, self._query_regions(allocated_keys, flag=0), strict=True))
        for index, key in enumerate(keys):
            if result_codes[index] == 0 and regions[key] != self._store_sessions[key]:
                result_codes[index] = GVA_SESSION_FAILURE
                del self._store_sessions[key]
        return tuple(result_codes)

    def load(self, bindings: tuple[KVBinding, ...]) -> tuple[BindingEvidence, ...]:
        if not bindings:
            return ()
        copy_args = self._copy_arguments(bindings, self._load_sessions)
        result_code = self._store.batch_copy(*copy_args, self._load_direction)
        if isinstance(result_code, bool) or not isinstance(result_code, Integral):
            raise RuntimeError("GVA batch_copy returned a non-integer result")
        # The native API reports the whole batch, not independently completed keys.
        return tuple(BindingEvidence(binding, int(result_code)) for binding in bindings)

    def store(self, batches: tuple[BindingBatch, ...]) -> StoreEvidence:
        bindings = tuple(binding for batch in batches for binding in batch.bindings)
        if not bindings:
            return StoreEvidence((), True, source_release_confirmed=True)
        source_handed_off = False
        try:
            copy_args = self._copy_arguments(bindings, self._store_sessions)
            source_handed_off = True
            result_code = self._store.batch_copy(*copy_args, self._store_direction)
            if isinstance(result_code, bool) or not isinstance(result_code, Integral):
                raise RuntimeError("GVA batch_copy returned a non-integer result")
        except Exception as error:
            return StoreEvidence(self._unknown_binding_evidence(bindings), False, not source_handed_off, error)
        succeeded = result_code == 0
        evidence = tuple(BindingEvidence(binding, int(result_code)) for binding in bindings)
        return StoreEvidence(evidence, succeeded, source_release_confirmed=succeeded)

    def commit_store_sessions(self, keys: list[str]) -> tuple[int, ...]:
        if not keys:
            return ()
        if any(key not in self._store_sessions for key in keys):
            raise RuntimeError("GVA commit has no owning Store session")
        native_result = self._store.batch_write_finish(keys, [0] * len(keys))
        result_codes = self._require_result_codes("batch_write_finish", keys, native_result)
        for key, code in zip(keys, result_codes, strict=True):
            if code == 0:
                del self._store_sessions[key]
        return result_codes

    def revoke_store_sessions(self, keys: list[str]) -> tuple[int, ...]:
        # Native failed-write notification deletes by key, even after a duplicate allocation.
        # Do not delete another writer's object; leave uncertain write leases to Backend expiry.
        return tuple(GVA_SESSION_FAILURE if self._store_sessions.pop(key, None) is not None else 0 for key in keys)

    def _query_regions(self, keys: list[str], flag: int = READABLE_GVA_QUERY) -> tuple[tuple[int, int] | None, ...]:
        if not keys:
            return ()
        infos = tuple(self._store.batch_get_key_info(keys, flag))
        if len(infos) != len(keys):
            raise RuntimeError(f"batch_get_key_info returned {len(infos)} results for {len(keys)} keys")
        regions: list[tuple[int, int] | None] = []
        for info in infos:
            size = info.size()
            gvas = tuple(info.gva_list())
            if isinstance(size, bool) or not isinstance(size, Integral):
                raise RuntimeError("batch_get_key_info returned a non-integer object size")
            if size <= 0 or not gvas:
                regions.append(None)
                continue
            if len(gvas) != 1 or isinstance(gvas[0], bool) or not isinstance(gvas[0], Integral):
                raise RuntimeError("GVA sessions require exactly one integer address per object")
            regions.append((int(gvas[0]), int(size)) if gvas[0] > 0 else None)
        return tuple(regions)

    @staticmethod
    def _copy_arguments(
        bindings: tuple[KVBinding, ...], sessions: Mapping[str, tuple[int, int] | None]
    ) -> tuple[list[int], list[int], list[int]]:
        gvas, addresses, sizes = [], [], []
        for binding in bindings:
            key = binding.remote_object.key
            region = sessions.get(key)
            layout = binding.remote_layout
            if region is None or region[1] != layout.object_size:
                raise RuntimeError(f"GVA binding has no matching session for key {key!r}")
            memory = binding.local_region.memory
            for offset, address, size in zip(layout.offsets, memory.addresses, memory.sizes, strict=True):
                gvas.append(region[0] + offset)
                addresses.append(address)
                sizes.append(size)
        return gvas, addresses, sizes
