"""GVA object allocation, leases, publication, and Worker binding fixtures."""

from typing import Any

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    BackendSpec,
    GVARegion,
    LayerwiseAccessKind,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.memcache.backend import GVA_REVOKE_UNCONFIRMED
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.memcache.sessions import MemcacheSessions
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.results import integer_result, integer_results


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

    def create_sessions(self) -> MemcacheSessions:
        return MemcacheSessions(self)

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
        return integer_results("batch_add_lease", keys, self.native_store.batch_add_lease(keys))

    def remove_gva_leases(self, keys):
        assert self.native_store is not None
        result = self.native_store.batch_remove_lease(keys)
        if result != 0:
            raise RuntimeError(f"batch_remove_lease failed with result {result}")

    def allocate_gva(self, keys, sizes):
        assert self.native_store is not None
        return integer_results("batch_alloc", keys, self.native_store.batch_alloc(keys, sizes))

    def copy_from_gva(self, remote_addresses, local_addresses, sizes):
        assert self.native_store is not None
        return integer_result("batch_copy", self.native_store.batch_copy(remote_addresses, local_addresses, sizes, 1))

    def copy_to_gva(self, remote_addresses, local_addresses, sizes):
        assert self.native_store is not None
        return integer_result("batch_copy", self.native_store.batch_copy(remote_addresses, local_addresses, sizes, 0))

    def publish_gva(self, keys):
        assert self.native_store is not None
        return integer_results("batch_write_finish", keys, self.native_store.batch_write_finish(keys, [0] * len(keys)))

    def revoke_gva(self, keys):
        return (GVA_REVOKE_UNCONFIRMED,) * len(keys)


def make_gva_spec() -> BackendSpec:
    return BackendSpec("memcache", FakeGVABackend, LayerwiseAccessKind.GVA, True)
