# SPDX-License-Identifier: Apache-2.0
"""Per-QoS Store clients sharing one master/key namespace (upstream Ascend).

Each client owns an engine with immutable GlobalResourceConfig. Only the default
client contributes pool capacity; all clients register the SAME caller buffers.
No request-time environment changes and no QoS suffix is added to cache keys.
"""

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import ExitStack
from threading import RLock
from typing import Any

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.qos_engine import TransferEngine, initialize_engine


class QosStorePool:
    def __init__(
        self,
        *,
        qos_values: Iterable[int],
        default_qos: int,
        setup_kwargs: Mapping[str, Any],
        resource_config: Mapping[str, Any] | None = None,
        engine_factory: Callable[[], TransferEngine] | None = None,
        store_factory: Callable[[], Any] | None = None,
    ) -> None:
        values = list(qos_values)
        if not values or any(type(q) is not int or not 0 <= q <= 7 for q in values):
            raise ValueError("QoS values must be integers in 0..7")
        values = sorted(set(values))
        if type(default_qos) is not int or default_qos not in values:
            raise ValueError("default_qos must be in qos_values")
        if setup_kwargs.get("protocol") != "ascend":
            raise ValueError("QoS lanes require protocol=ascend")
        if setup_kwargs.get("enable_ssd_offload") or os.getenv(envs.MOONCAKE_FABRIC_MEMORY_ENV, "0") == "1":
            raise ValueError("QoS lanes currently require direct memory transfer without SSD/fabric memory")
        if engine_factory is None:
            from mooncake.engine import TransferEngine as NativeEngine  # type: ignore[import-not-found, import-untyped]

            engine_factory = NativeEngine
        if store_factory is None:
            from mooncake.store import MooncakeDistributedStore  # type: ignore[import-not-found, import-untyped]

            store_factory = MooncakeDistributedStore
        self.default_qos = default_qos
        self._lanes: dict[int, tuple[TransferEngine, Any, str, RLock]] = {}
        self._buffers: dict[int, int] = {}
        self._closed = False
        self._broken = False
        self._rollback_pending: list[tuple[TransferEngine, int]] = []
        try:
            for q in values:
                host = setup_kwargs["local_hostname"]
                engine = initialize_engine(
                    engine_factory, host, setup_kwargs.get("rdma_devices", ""), q, resource_config
                )
                store = store_factory()
                segment = f"{host}:{engine.get_rpc_port()}"
                self._lanes[q] = (engine, store, segment, RLock())
                kw = dict(setup_kwargs, local_hostname=segment, engine=engine.get_engine())
                if q != default_qos:
                    kw.update(global_segment_size=0, local_buffer_size=0)
                rc = store.setup(**kw)
                if rc != 0:
                    raise RuntimeError(f"QoS {q}: Store setup failed: {rc}")
        except BaseException as cause:
            try:
                self.close()
            except Exception as cleanup:  # noqa: BLE001 - report both setup and cleanup failures
                raise RuntimeError(f"Store setup failed; cleanup also failed: {cleanup}") from cause
            raise

    @property
    def default_store(self) -> Any:
        return self._lanes[self.default_qos][1]

    @property
    def default_segment(self) -> str:
        return self._lanes[self.default_qos][2]

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("QoS pool is closed")
        if self._broken:
            raise RuntimeError("QoS pool is broken after registration rollback")

    def register_buffers(self, ptrs: Sequence[int], lengths: Sequence[int]) -> None:
        if len(ptrs) != len(lengths):
            raise ValueError("buffer pointers and lengths have different counts")
        with ExitStack() as locks:
            for lane in self._lanes.values():
                locks.enter_context(lane[3])
            self._require_open()
            added: list[tuple[int, list[TransferEngine]]] = []
            try:
                for ptr, size in zip(ptrs, lengths):
                    if type(ptr) is not int or type(size) is not int or ptr <= 0 or size <= 0:
                        raise ValueError("buffer pointer and size must be positive integers")
                    if ptr in self._buffers and self._buffers[ptr] != size:
                        raise ValueError("a registered pointer cannot change size")
                    if ptr in self._buffers:
                        continue
                    if ptr + size > (1 << 64) or any(
                        ptr < base + n and base < ptr + size for base, n in self._buffers.items()
                    ):
                        raise ValueError("overlapping or overflowing registration range")
                    done: list[TransferEngine] = []
                    added.append((ptr, done))
                    for engine, _, _, _ in self._lanes.values():
                        rc = engine.register_memory(ptr, size)
                        if rc != 0:
                            raise RuntimeError(f"memory registration failed: {rc}")
                        done.append(engine)
                    self._buffers[ptr] = size
            except BaseException as cause:
                for ptr, engines in reversed(added):
                    for engine in reversed(engines):
                        try:
                            failed = engine.unregister_memory(ptr) != 0
                        except Exception:  # noqa: BLE001 - preserve cleanup failures until all lanes are handled
                            failed = True
                        if failed:
                            self._rollback_pending.append((engine, ptr))
                    self._buffers.pop(ptr, None)
                if self._rollback_pending:
                    self._broken = True
                    raise RuntimeError("registration rollback failed; pool is broken") from cause
                raise

    def transfer(
        self,
        qos: int,
        operation: str,
        keys: Sequence[str],
        addrs: Sequence[Sequence[int]],
        sizes: Sequence[Sequence[int]],
        replicate_config: Any = None,
    ) -> list[int]:
        if type(qos) is not int or qos not in self._lanes:
            raise ValueError(f"QoS {qos} has no lane")
        if operation not in ("get", "put"):
            raise ValueError("operation must be get or put")
        if len(keys) != len(addrs) or len(keys) != len(sizes):
            raise ValueError("key/buffer batch shape mismatch")
        for aa, ss in zip(addrs, sizes):
            if len(aa) != len(ss) or not aa or any(type(n) is not int or n <= 0 for n in ss):
                raise ValueError("invalid multi-buffer shape")
        _, store, _, lock = self._lanes[qos]
        with lock:
            self._require_open()
            for aa, ss in zip(addrs, sizes):
                for addr, size in zip(aa, ss):
                    if not any(base <= addr and addr + size <= base + length for base, length in self._buffers.items()):
                        raise ValueError("transfer range is not registered in every lane")
            if not keys:
                return []
            if operation == "get":
                result = list(store.batch_get_into_multi_buffers(keys, addrs, sizes))
            elif replicate_config is None:
                # Upstream pybind supplies its default config only when omitted.
                result = list(store.batch_put_from_multi_buffers(keys, addrs, sizes))
            else:
                result = list(store.batch_put_from_multi_buffers(keys, addrs, sizes, replicate_config))
            if len(result) != len(keys) or any(r < 0 for r in result):
                raise RuntimeError(f"QoS {qos} {operation} failed: {result}")
            expected = [sum(parts) for parts in sizes] if operation == "get" else [0] * len(keys)
            if result != expected:
                raise RuntimeError(f"QoS {qos} {operation}: unexpected byte count/status {result}, expected {expected}")
            return [0] * len(result)

    def close(self) -> None:
        with ExitStack() as locks:
            for lane in self._lanes.values():
                locks.enter_context(lane[3])
            if self._closed:
                return
            self._closed = True
            errors: list[str] = []
            # Stop Store clients before unregistering caller-owned buffers.
            for engine, store, _, _ in reversed(list(self._lanes.values())):
                try:
                    rc = store.close()
                    if rc != 0:
                        errors.append(f"store close: {rc}")
                except Exception as e:  # noqa: BLE001 - aggregate native cleanup failures
                    errors.append(str(e))
                for ptr in self._buffers:
                    try:
                        rc = engine.unregister_memory(ptr)
                        if rc != 0:
                            errors.append(f"unregister: {rc}")
                    except Exception as e:  # noqa: BLE001 - aggregate native cleanup failures
                        errors.append(str(e))
            pending = []
            for engine, ptr in self._rollback_pending:
                try:
                    failed = engine.unregister_memory(ptr) != 0
                except Exception:  # noqa: BLE001 - preserve cleanup failures until all lanes are handled
                    failed = True
                if failed:
                    errors.append("rollback unregister failed")
                    pending.append((engine, ptr))
            self._rollback_pending = pending
            self._buffers.clear()
            self._lanes.clear()
            if errors:
                raise RuntimeError("; ".join(errors))
