# SPDX-License-Identifier: Apache-2.0
"""P/D direct READ lanes. Engines and registrations live for the worker lifetime.

Uses upstream initialization through a shared startup-only environment scope.
Requests do not change environment variables, buffer ownership or cache keys.
"""

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from threading import RLock
from typing import Any

from vllm_ascend.distributed.kv_transfer.qos_engine import TransferEngine, initialize_engine


class QosPDPool:
    def __init__(
        self,
        host: str,
        device_name: str,
        qos_values: Iterable[int],
        default_qos: int,
        resource_config: Mapping[str, Any] | None = None,
        engine_factory: Callable[[], TransferEngine] | None = None,
    ) -> None:
        values = list(qos_values)
        if not values or any(type(q) is not int or not 0 <= q <= 7 for q in values):
            raise ValueError("QoS must be integers in 0..7")
        values = sorted(set(values))
        if type(default_qos) is not int or default_qos not in values:
            raise ValueError("default_qos has no lane")
        if engine_factory is None:
            from mooncake.engine import TransferEngine as NativeEngine  # type: ignore[import-not-found, import-untyped]

            engine_factory = NativeEngine
        self.default_qos = default_qos
        self._engines: dict[int, TransferEngine] = {}
        self._buffers: dict[int, int] = {}
        self._locks = {q: RLock() for q in values}
        self._closed = False
        self._broken = False
        self._rollback_pending: list[tuple[TransferEngine, int]] = []
        for qos in values:
            self._engines[qos] = initialize_engine(engine_factory, host, device_name, qos, resource_config)
        ports = list(self.rpc_ports.values())
        if len(ports) != len(set(ports)):
            raise RuntimeError("QoS engines must advertise distinct RPC ports")

    @property
    def default_engine(self) -> TransferEngine:
        return self._engines[self.default_qos]

    @property
    def rpc_ports(self) -> dict[int, int]:
        return {q: e.get_rpc_port() for q, e in self._engines.items()}

    @contextmanager
    def _all_lanes(self) -> Iterator[None]:
        with ExitStack() as locks:
            for lock in self._locks.values():
                locks.enter_context(lock)
            yield

    def register_buffers(self, ptrs: Sequence[int], sizes: Sequence[int]) -> None:
        with self._all_lanes():
            if self._closed:
                raise RuntimeError("QoS pool is closed")
            if self._broken:
                raise RuntimeError("QoS pool is broken after registration rollback")
            if len(ptrs) != len(sizes):
                raise ValueError("registration shape mismatch")
            added: list[tuple[int, list[TransferEngine]]] = []
            try:
                for ptr, size in zip(ptrs, sizes):
                    if type(ptr) is not int or type(size) is not int or ptr <= 0 or size <= 0:
                        raise ValueError("invalid registration range")
                    if ptr in self._buffers:
                        if self._buffers[ptr] != size:
                            raise ValueError("registered pointer size changed")
                        continue
                    if ptr + size > (1 << 64) or any(
                        ptr < base + n and base < ptr + size for base, n in self._buffers.items()
                    ):
                        raise ValueError("overlapping or overflowing registration range")
                    completed: list[TransferEngine] = []
                    added.append((ptr, completed))
                    for engine in self._engines.values():
                        rc = engine.register_memory(ptr, size)
                        if rc != 0:
                            raise RuntimeError(f"registration failed: {rc}")
                        completed.append(engine)
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

    def read(self, qos: int, session: str, local: Sequence[int], remote: Sequence[int], sizes: Sequence[int]) -> int:
        # Separate lanes can progress concurrently. Same-lane reads and memory
        # lifetime changes are serialized; the request queue is not reordered.
        if type(qos) is not int or qos not in self._locks:
            raise ValueError("requested QoS lane is unavailable")
        with self._locks[qos]:
            if self._closed:
                raise RuntimeError("QoS pool is closed")
            if self._broken:
                raise RuntimeError("QoS pool is broken after registration rollback")
            if not (len(local) == len(remote) == len(sizes)) or not sizes:
                raise ValueError("empty or inconsistent transfer batch")
            for dst, src, size in zip(local, remote, sizes):
                if any(type(v) is not int or v <= 0 for v in (dst, src, size)):
                    raise ValueError("invalid transfer range")
                if not any(base <= dst and dst + size <= base + n for base, n in self._buffers.items()):
                    raise ValueError("local transfer range is not registered")
            rc = self._engines[qos].batch_transfer_sync_read(session, local, remote, sizes)
            if rc != 0:
                raise RuntimeError(f"QoS {qos}: P/D READ failed: {rc}")
            return rc

    def close(self) -> None:
        """Call only after receive threads stop and before releasing KV memory."""
        with self._all_lanes():
            if self._closed:
                return
            self._closed = True
            errors: list[str | int] = []
            for engine in self._engines.values():
                for ptr in self._buffers:
                    try:
                        if engine.unregister_memory(ptr) != 0:
                            errors.append(ptr)
                    except Exception:  # noqa: BLE001 - preserve cleanup failures until all lanes are handled
                        errors.append(ptr)
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
            self._engines.clear()
            if errors:
                raise RuntimeError(f"unregister failed for {len(errors)} ranges")
