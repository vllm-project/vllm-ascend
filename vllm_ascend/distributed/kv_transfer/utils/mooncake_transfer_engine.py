import threading


class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self._initialization_attempted = False
        self.is_register_buffer: bool = False
        self.registered_buffers: list[tuple[int, int]] = []
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()
        self._exclusive_owner = None
        self._untracked_user = False
        self._unregistered_indices: set[int] = set()
        self._registration_incomplete = False

    def get_transfer_engine(self, hostname: str, device_name: str | None, *, owner=None):
        # Ownership registration and initialization are serialized together.
        # Existing callers remain untracked; an exclusive user must fail closed
        # rather than release an engine which an unknown caller may still use.
        with self.transfer_engine_lock:
            if owner is None and self._exclusive_owner is not None:
                raise RuntimeError("Mooncake transfer engine is exclusively owned")
            if owner is not None and (
                self._untracked_user or (self._exclusive_owner is not None and self._exclusive_owner is not owner)
            ):
                raise RuntimeError("Mooncake transfer engine already has other users")
            if self.transfer_engine is None:
                if self._initialization_attempted:
                    raise RuntimeError(
                        "TransferEngine initialization previously failed or the engine was released. "
                        "Restart the process before retrying to avoid leaking native resources."
                    )
                try:
                    from mooncake.engine import TransferEngine  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "Please install mooncake by following the instructions at "
                        "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
                        "to run vLLM with MooncakeConnector."
                    ) from e
                self._initialization_attempted = True
                engine = TransferEngine()
                device_name = device_name if device_name is not None else ""
                ret_value = engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)
                if ret_value != 0:
                    raise RuntimeError(
                        f"TransferEngine initialization failed with ret_value: {ret_value}. "
                        "Restart the process before retrying."
                    )
                self.transfer_engine = engine
            if owner is None:
                self._untracked_user = True
            else:
                self._exclusive_owner = owner
            return self.transfer_engine

    def release_exclusive_engine(self, owner, engine):
        """Unregister and release only an explicitly exclusive engine owner."""
        with self.transfer_engine_lock:
            if self._exclusive_owner is not owner or self._untracked_user or self.transfer_engine is not engine:
                raise RuntimeError("Cannot release a shared Mooncake transfer engine")
            with self.register_buffer_lock:
                self._unregister_buffer_locked()
                if self.is_register_buffer:
                    raise RuntimeError("Cannot release Mooncake engine with unknown registered buffers")
                self.transfer_engine = None
                self._exclusive_owner = None
                self.registered_buffers.clear()
                self._unregistered_indices.clear()
                self._registration_incomplete = False
                # Reinitialization in the same process is intentionally blocked.

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            if self._exclusive_owner is None:
                if self.is_register_buffer:
                    return
                for ptr, size in zip(ptrs, sizes):
                    if self.transfer_engine.register_memory(ptr, size) != 0:
                        raise RuntimeError("Mooncake memory registration failed.")
                self.registered_buffers = list(zip(ptrs, sizes))
                self.is_register_buffer = True
                return
            if self._registration_incomplete:
                raise RuntimeError("Mooncake registration is incomplete; release the engine before retrying")
            if self.is_register_buffer:
                return
            if len(ptrs) != len(sizes):
                raise ValueError("Mooncake buffer pointers and sizes must have equal length")
            self.registered_buffers = []
            self._unregistered_indices.clear()
            try:
                for ptr, size in zip(ptrs, sizes):
                    ret_value = self.transfer_engine.register_memory(ptr, size)
                    if ret_value != 0:
                        raise RuntimeError("Mooncake memory registration failed.")
                    self.registered_buffers.append((ptr, size))
                    self.is_register_buffer = True
            except BaseException:
                self._registration_incomplete = True
                raise
            self.is_register_buffer = True

    def _unregister_buffer_locked(self):
        if not self.is_register_buffer or not self.registered_buffers:
            return
        assert self.transfer_engine is not None, "Transfer engine must be initialized"
        # Keep progress on partial failure so a retry does not unregister twice.
        for index, (ptr, _) in enumerate(self.registered_buffers):
            if index in self._unregistered_indices:
                continue
            ret_value = self.transfer_engine.unregister_memory(ptr)
            if ret_value != 0:
                raise RuntimeError(f"Mooncake memory unregistration failed for ptr={ptr:#x}, ret_value={ret_value}")
            self._unregistered_indices.add(index)
        self.is_register_buffer = False

    def unregister_buffer(self):
        with self.register_buffer_lock:
            if self._exclusive_owner is None:
                if not self.is_register_buffer or not self.registered_buffers:
                    return
                assert self.transfer_engine is not None, "Transfer engine must be initialized"
                for ptr, _ in self.registered_buffers:
                    ret_value = self.transfer_engine.unregister_memory(ptr)
                    if ret_value != 0:
                        raise RuntimeError(
                            f"Mooncake memory unregistration failed for ptr={ptr:#x}, ret_value={ret_value}"
                        )
                self.is_register_buffer = False
                return
            self._unregister_buffer_locked()

    def reregister_buffer(self):
        with self.register_buffer_lock:
            if self._exclusive_owner is None:
                if self.is_register_buffer or not self.registered_buffers:
                    return
                assert self.transfer_engine is not None, "Transfer engine must be initialized"
                for ptr, size in self.registered_buffers:
                    ret_value = self.transfer_engine.register_memory(ptr, size)
                    if ret_value != 0:
                        raise RuntimeError(
                            f"Mooncake memory re-registration failed for ptr={ptr:#x}, ret_value={ret_value}"
                        )
                self.is_register_buffer = True
                return
            if self._registration_incomplete:
                raise RuntimeError("Mooncake registration is incomplete; release the engine before retrying")
            if not self.registered_buffers or (self.is_register_buffer and not self._unregistered_indices):
                return
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            # On a partial unregister failure, restore only the missing regions.
            for index, (ptr, size) in enumerate(self.registered_buffers):
                if index not in self._unregistered_indices:
                    continue
                ret_value = self.transfer_engine.register_memory(ptr, size)
                if ret_value != 0:
                    raise RuntimeError(
                        f"Mooncake memory re-registration failed for ptr={ptr:#x}, ret_value={ret_value}"
                    )
                self._unregistered_indices.remove(index)
                self.is_register_buffer = True
            self.is_register_buffer = True


global_te = GlobalTE()
