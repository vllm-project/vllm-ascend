"""Translate KV Pool values to Backend calls and normalize their evidence."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral
from typing import TYPE_CHECKING, Any, cast

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend

from ...backend import BackendSpec
from ...program.values.evidence import RemoteObjectObservation, StoreEvidence, TransferEvidence
from ...program.values.representation import RemoteKVObject
from ...program.values.selection import TransferWork
from ...program.values.transfer import TransferSource
from .arguments import materialize_ranges

if TYPE_CHECKING:
    from ....backend.memcache_backend import MemcacheBackend
    from ....backend.mooncake_backend import MooncakeBackend
    from ....backend.yuanrong_backend import YuanrongBackend


class BackendIO:
    """Own the v1 boundary between domain batches and one AscendStore Backend."""

    def __init__(self, backend: Backend, backend_spec: BackendSpec) -> None:
        self._backend = backend
        self._backend_spec = backend_spec

    def initialize_thread(self) -> None:
        self._backend.set_device()

    def observe_objects(self, remote_objects: tuple[RemoteKVObject, ...]) -> tuple[RemoteObjectObservation, ...]:
        if not remote_objects:
            return ()
        presence = tuple(self._backend.exists([remote_object.key for remote_object in remote_objects]))
        if len(presence) != len(remote_objects):
            raise ValueError(f"Backend returned {len(presence)} results for {len(remote_objects)} remote objects")
        if any(value not in (0, 1) for value in presence):
            raise ValueError("Backend returned object states other than 0 or 1")
        return tuple(
            RemoteObjectObservation(remote_object, value == 1)
            for remote_object, value in zip(remote_objects, presence, strict=True)
        )

    def load(self, work: TransferWork) -> tuple[TransferEvidence, ...]:
        arguments = materialize_ranges(work)
        sources = work.sources
        if not sources:
            return ()
        native_result = self._backend.get(
            arguments.keys,
            arguments.addresses,
            arguments.sizes,
        )
        if native_result is None:
            return self._unknown_transfer_evidence(sources)
        try:
            result_codes = tuple(native_result)
        except (TypeError, ValueError):
            return self._unknown_transfer_evidence(sources)
        if len(result_codes) != len(sources):
            return self._unknown_transfer_evidence(sources)
        if any(isinstance(code, bool) or not isinstance(code, Integral) for code in result_codes):
            return self._unknown_transfer_evidence(sources)
        return tuple(TransferEvidence(source, int(code)) for source, code in zip(sources, result_codes, strict=True))

    def store(self, work: TransferWork) -> StoreEvidence:
        if work.empty:
            return StoreEvidence((), True, source_release_confirmed=True)

        source_addresses_handed_off = False
        try:
            arguments = materialize_ranges(work)
            native_put, native_args = self._prepare_put(
                arguments.keys,
                arguments.addresses,
                arguments.sizes,
            )
            source_addresses_handed_off = True
            native_result = native_put(*native_args)
        except Exception as error:
            return StoreEvidence(
                self._unknown_transfer_evidence(work.sources, not source_addresses_handed_off),
                False,
                not source_addresses_handed_off,
                error,
            )
        if self._backend_spec.name == "yuanrong":
            return StoreEvidence(
                self._unknown_transfer_evidence(work.sources, True),
                True,
                source_release_confirmed=True,
            )
        return self._store_evidence_from_aligned_results(work.sources, native_result)

    def _prepare_put(
        self,
        keys: list[str],
        addresses: list[list[int]],
        sizes: list[list[int]],
    ) -> tuple[Callable[..., Any], tuple[Any, ...]]:
        if self._backend_spec.name == "mooncake":
            mooncake_backend = cast("MooncakeBackend", self._backend)
            mooncake_backend.ensure_initialized()
            if mooncake_backend.store is None:
                raise RuntimeError("Mooncake store is unavailable for put")
            replicate_config = mooncake_backend._build_replicate_config()
            return mooncake_backend.store.batch_put_from_multi_buffers, (keys, addresses, sizes, replicate_config)
        if self._backend_spec.name == "memcache":
            memcache_backend = cast("MemcacheBackend", self._backend)
            memcache_backend.ensure_initialized()
            if memcache_backend.store is None:
                raise RuntimeError("Memcache store is unavailable for put")
            direction = self._backend_spec.backend_module.MmcDirect.COPY_L2G.value
            return memcache_backend.store.batch_put_from_layers, (keys, addresses, sizes, direction)
        if self._backend_spec.name == "yuanrong":
            yuanrong_backend = cast("YuanrongBackend", self._backend)
            if yuanrong_backend.store is None:
                raise RuntimeError("Yuanrong store is unavailable for put")
            return yuanrong_backend.store.mset_d2h_from_multi_buffers, (
                keys,
                addresses,
                sizes,
                yuanrong_backend._ds_set_param,
            )
        return self._backend.put, (keys, addresses, sizes)

    def _store_evidence_from_aligned_results(
        self,
        sources: tuple[TransferSource, ...],
        native_result: Any,
    ) -> StoreEvidence:
        result_codes, error = self._interpret_store_results(len(sources), native_result)
        succeeded = result_codes is not None and all(code == 0 for code in result_codes)
        if result_codes is None:
            transfer_evidence = self._unknown_transfer_evidence(sources, False)
        else:
            transfer_evidence = tuple(
                TransferEvidence(source, code, succeeded) for source, code in zip(sources, result_codes, strict=True)
            )
        return StoreEvidence(
            transfer_evidence,
            succeeded,
            source_release_confirmed=succeeded,
            error=error,
        )

    def _interpret_store_results(
        self, expected_count: int, native_result: Any
    ) -> tuple[tuple[int, ...] | None, Exception | None]:
        try:
            result_codes = None if native_result is None else tuple(native_result)
        except Exception as error:
            return None, error
        if result_codes is None:
            return None, RuntimeError(f"{self._backend_spec.name} Store returned no per-key results")
        if len(result_codes) != expected_count:
            result_error = RuntimeError(
                f"{self._backend_spec.name} Store returned {len(result_codes)} results for {expected_count} keys"
            )
            return None, result_error
        if any(isinstance(code, bool) or not isinstance(code, Integral) for code in result_codes):
            return None, RuntimeError(f"{self._backend_spec.name} Store returned non-integer results")
        return tuple(int(code) for code in result_codes), None

    @staticmethod
    def _unknown_transfer_evidence(
        sources: tuple[TransferSource, ...],
        source_release_confirmed: bool | None = None,
    ) -> tuple[TransferEvidence, ...]:
        return tuple(TransferEvidence(source, None, source_release_confirmed) for source in sources)

    @staticmethod
    def _require_result_codes(operation: str, keys: list[str], native_result: Any) -> tuple[int, ...]:
        try:
            result_codes = tuple(native_result)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"{operation} returned non-integer results") from error
        if len(result_codes) != len(keys):
            raise RuntimeError(f"{operation} returned {len(result_codes)} results for {len(keys)} keys")
        if any(isinstance(code, bool) or not isinstance(code, Integral) for code in result_codes):
            raise RuntimeError(f"{operation} returned non-integer results")
        return tuple(int(code) for code in result_codes)
